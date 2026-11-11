"""Move analysis by simulation: estimated win % for each candidate move.

For every candidate move we imagine many possible versions of what the player can't
see (opponents' hidden cards, their tickets, the deck order) that are consistent
with what they *have* seen, play each one out to the end with the network, and
average the results. Every candidate is tried on the same imagined versions, so
the comparison between moves is fair.
"""

import copy
import math
import random
import time

import numpy as np
import torch

from ..data import CARDS_PER_COLOR, CITY_NAMES, COLORS, LOCO, NUM_LOCOS, ROUTES, TICKETS
from ..engine import CHOOSE_TICKETS, GAME_OVER, MAIN, SETUP_TICKETS, card_name
from ..planning import edge_costs, shortest_path
from .encode import decode

_FULL_COSTS = None
_BASE = {}
_WEIGHTS = {}


def _base_cost(t):
    """Trains needed for a ticket on an empty board."""
    global _FULL_COSTS
    if t not in _BASE:
        if _FULL_COSTS is None:
            from ..planning import EDGES
            _FULL_COSTS = {k: ROUTES[rids[0]].length for k, rids in EDGES.items()}
        a, b, _ = TICKETS[t]
        _BASE[t] = shortest_path(a, b, _FULL_COSTS)[0]
    return _BASE[t]


def ticket_weights(game, q, candidates):
    """How plausible each ticket is for player q, judging by the routes they built."""
    if not game.players[q].routes:
        return [1.0] * len(candidates)
    # Depends only on who owns which routes, so it's worked out once per board position
    # rather than once per imagined version of the game.
    key = (tuple(game.route_owner), game.num_players, q)
    w = _WEIGHTS.get(key)
    if w is None:
        if len(_WEIGHTS) > 512:
            _WEIGHTS.clear()
        costs, _ = edge_costs(game, q)
        w = []
        for t in range(len(TICKETS)):
            a, b, _ = TICKETS[t]
            now = shortest_path(a, b, costs)[0]
            covered = 0.0 if math.isinf(now) else max(0.0, 1 - now / _base_cost(t))
            w.append(math.exp(3 * covered))
        _WEIGHTS[key] = w
    return [w[t] for t in candidates]


def determinize(game, viewer, rng):
    """A copy of the game with everything `viewer` can't see re-dealt at random."""
    g = copy.deepcopy(game, {id(game.log): []})  # the move log isn't needed
    g.rng = random.Random(rng.random())
    me = g.players[viewer]

    # Cards: the viewer sees their hand, the face-up row, the discard pile, and
    # which cards opponents took face-up. Everything else is shuffled.
    pool = [CARDS_PER_COLOR] * 8 + [NUM_LOCOS]
    seen = list(me.hand)
    for c in g.face_up + g.discard:
        if c is not None:
            seen[c] += 1
    for q in g.players:
        if q.index != viewer:
            for c in range(9):
                seen[c] += q.known_hand[c]
    unknown = [c for c in range(9) for _ in range(max(0, pool[c] - seen[c]))]
    rng.shuffle(unknown)
    for q in g.players:
        if q.index == viewer:
            continue
        hand = list(q.known_hand)
        for _ in range(q.hand_size - sum(q.known_hand)):
            hand[unknown.pop()] += 1
        q.hand = hand
    g.deck = unknown

    # Tickets: the viewer knows their own; opponents' are guessed, favoring tickets
    # their routes seem to be building toward.
    hidden = [t for t in g.rules["tickets"] if t not in me.tickets and t not in me.pending]
    for q in g.players:
        if q.index == viewer:
            continue
        chosen = []
        weight = dict(zip(hidden, ticket_weights(g, q.index, hidden)))
        for _ in range(len(q.tickets)):
            t = rng.choices(hidden, weights=[weight[x] for x in hidden])[0]
            hidden.remove(t)
            chosen.append(t)
        q.tickets = chosen
        # Tickets they're still choosing from (e.g. their opening three) are unseen too.
        for k in range(len(q.pending)):
            q.pending[k] = hidden.pop(rng.randrange(len(hidden)))
    rng.shuffle(hidden)
    g.ticket_deck = hidden
    return g


def playout(g, net):
    """Play to the end with the network choosing every move (best move each time)."""
    while g.phase != GAME_OVER:
        pi = g.current
        obs, mask = net.encode(g, pi)
        if mask.sum() == 1:
            idx = int(mask.argmax())
        else:
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
            idx = int(logits[0].argmax())
        g.step(decode(g, pi, idx), validate=False)
    return g


def simulate(game, viewer, idx, net, seeds):
    """Win share and score margin for move `idx` over the given imagined versions."""
    wins, margins = 0.0, 0.0
    for s in seeds:
        g = determinize(game, viewer, random.Random(s))
        g.step(decode(g, viewer, idx), validate=False)
        playout(g, net)
        scores = [r["total"] for r in g.result["scores"]]
        winners = g.result["winners"]
        if viewer in winners:
            wins += 1 / len(winners)
        margins += scores[viewer] - max(s2 for i, s2 in enumerate(scores) if i != viewer)
    return wins, margins


# ---- a bot that thinks before it moves

def outcome(g, seat):
    """A finished game on the network's value scale: +1 win / -1 loss, plus margin / 100."""
    scores = [r["total"] for r in g.result["scores"]]
    winners = g.result["winners"]
    win = (1.0 if len(winners) == 1 else 0.0) if seat in winners else -1.0
    return win + (scores[seat] - max(s for i, s in enumerate(scores) if i != seat)) / 100


@torch.no_grad()
def lookahead(net, games, viewer, depth):
    """Play every game forward with the network (best move each time) for at least `depth`
    moves, until it's `viewer`'s turn again, then score it from `viewer`'s point of view.
    All games advance together so the network sees them in one batch."""
    values = [0.0] * len(games)
    moves = [0] * len(games)
    active = list(range(len(games)))
    while active:
        rows, leaves = [], []
        for k in active:
            g = games[k]
            if g.phase == GAME_OVER:
                values[k] = outcome(g, viewer)
            elif moves[k] >= depth and g.current == viewer and g.phase == MAIN:
                leaves.append(k)
            else:
                rows.append(k)
        if leaves:
            enc = [net.encode(games[k], viewer) for k in leaves]
            _, v = net(torch.from_numpy(np.stack([e[0] for e in enc])),
                       torch.from_numpy(np.stack([e[1] for e in enc])))
            for k, x in zip(leaves, v.tolist()):
                values[k] = x
        if rows:
            enc = [net.encode(games[k], games[k].current) for k in rows]
            logits, _ = net(torch.from_numpy(np.stack([e[0] for e in enc])),
                            torch.from_numpy(np.stack([e[1] for e in enc])))
            for k, idx in zip(rows, logits.argmax(-1).tolist()):
                g = games[k]
                g.step(decode(g, g.current, idx), validate=False)
                moves[k] += 1
        active = rows
    return values


class SearchBot:
    """The Expert network, plus thinking time.

    Before each move that isn't obvious it takes the network's favorite few moves (and,
    on a normal turn, the claim that looks most damaging to the opponent's likely
    tickets), tries each on the same imagined versions of the hidden cards and tickets,
    plays a few moves ahead with the network, and picks the move whose positions the
    network rates best on average.

    It thinks in rounds of `batch` imagined versions. After each round, moves that are
    clearly worse than the leader (by more than two standard errors, comparing them on
    the same imagined versions) are dropped, so the time goes to the close calls. With
    `think` set, it doesn't start a round that would likely end after that many seconds."""

    # Defaults from head-to-head tests against the plain network (74.5% over 400 games).
    def __init__(self, net=None, path=None, sims=32, top=6, depth=12, confident=0.98,
                 prior=0.05, block_min=5, batch=8, think=None, seed=None, name="master"):
        from .model import EXPERT_PATH, load
        self.net = net if net is not None else load(path or EXPERT_PATH)
        self.sims, self.top, self.depth = sims, top, depth
        self.confident, self.prior, self.block_min = confident, prior, block_min
        self.batch, self.think = batch, think
        self.rng = random.Random(seed)
        self.name = name
        self.stats = {"searched": 0, "changed": 0, "block_cands": 0, "blocks": 0}

    @torch.no_grad()
    def choose(self, game, pi):
        obs, mask = self.net.encode(game, pi)
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            return decode(game, pi, int(legal[0]))
        logits, _ = self.net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
        probs = torch.softmax(logits[0], -1).numpy()
        best = int(legal[np.argmax(probs[legal])])
        tickets = game.phase in (SETUP_TICKETS, CHOOSE_TICKETS)
        if not tickets and probs[best] >= self.confident:
            return decode(game, pi, best)

        order = legal[np.argsort(-probs[legal])]
        cands = [int(i) for i in (order if tickets else order[:self.top])]
        worlds = [determinize(game, pi, random.Random(self.rng.random())) for _ in range(self.sims)]
        block = self._block(game, pi, mask, logits[0], worlds) if game.phase == MAIN else None
        if block is not None:
            self.stats["block_cands"] += 1
            if block not in cands:
                cands.append(block)

        bonus = self.prior * np.log(np.maximum(probs[cands], 1e-6) / probs[best])
        start = time.monotonic()
        alive = list(range(len(cands)))
        values = np.zeros((len(cands), 0))
        while values.shape[1] < self.sims:
            batch = worlds[values.shape[1]:values.shape[1] + self.batch]
            games = []
            for k in alive:
                for w in batch:
                    g = copy.deepcopy(w, {id(w.log): []})
                    g.step(decode(g, pi, cands[k]), validate=False)
                    games.append(g)
            v = np.full((len(cands), len(batch)), np.nan)
            v[alive] = np.array(lookahead(self.net, games, pi, self.depth)).reshape(len(alive), len(batch))
            values = np.concatenate([values, v], 1)
            score = values[alive].mean(1) + bonus[alive]
            lead = alive[int(np.argmax(score))]
            d = values[alive] - values[lead] + (bonus[alive] - bonus[lead])[:, None]
            se = d.std(1, ddof=1) / np.sqrt(d.shape[1])
            alive = [k for k, m, s in zip(alive, d.mean(1), se) if m + 2 * s >= 0]
            # Stop if another round (judged by how long this one took) would run past `think`.
            spent = time.monotonic() - start
            if len(alive) == 1 or (self.think is not None and spent * (values.shape[1] + self.batch)
                                   / values.shape[1] > self.think):
                break
        pick = cands[max(alive, key=lambda k: values[k].mean() + bonus[k])]
        self.stats["searched"] += 1
        self.stats["changed"] += pick != best
        self.stats["blocks"] += pick == block and pick != best
        return decode(game, pi, pick)

    def _block(self, game, pi, mask, logits, worlds):
        """The claim that would hurt the opponents' tickets most, judged over the imagined
        versions of those tickets (None if no affordable claim averages block_min damage)."""
        from ..planning import ticket_damage
        from .encode import A_CLAIM, NC, ROUTE_KEY
        damage = {}
        for w in worlds:
            for q in range(w.num_players):
                if q != pi:
                    for key, (d, _) in ticket_damage(w, q).items():
                        damage[key] = damage.get(key, 0) + d / len(worlds)
        claims = np.flatnonzero(mask[A_CLAIM:]) + A_CLAIM
        best, best_d = None, self.block_min
        for idx in claims:
            d = damage.get(ROUTE_KEY[(idx - A_CLAIM) // NC], 0)
            if d > best_d or (d == best_d and best is not None and logits[idx] > logits[best]):
                best, best_d = int(idx), d
        if best is None:
            return None
        rid = (best - A_CLAIM) // NC
        pays = [i for i in claims if (i - A_CLAIM) // NC == rid]
        return int(max(pays, key=lambda i: logits[i]))


# ---- running simulations in worker processes

_worker_net = None


def worker_init(path):
    global _worker_net
    torch.set_num_threads(1)
    from .model import load
    _worker_net = load(path)


def worker_task(args):
    game, viewer, idx, seeds = args
    return idx, simulate(game, viewer, idx, _worker_net, seeds), len(seeds)


def analyze(game, viewer, net, pool, sims=48, chunk=12):
    """Instinct plus simulated win % / margin for each candidate move."""
    value, picks = instinct(game, viewer, net)
    seeds = [random.randrange(2**31) for _ in range(sims)]
    tasks = [(game, viewer, idx, seeds[k:k + chunk]) for idx, _ in picks
             for k in range(0, sims, chunk)]
    totals = {idx: [0.0, 0.0, 0] for idx, _ in picks}
    for idx, (wins, margins), n in pool.imap_unordered(worker_task, tasks):
        totals[idx][0] += wins
        totals[idx][1] += margins
        totals[idx][2] += n
    moves = []
    for idx, prior in picks:
        label, rid = describe(game, viewer, idx)
        w, m, n = totals[idx]
        moves.append({"label": label, "route": rid, "prior": prior,
                      "win": w / n, "margin": m / n, "sims": n})
    moves.sort(key=lambda mv: (-mv["win"], -mv["margin"]))
    return {"value": value, "moves": moves}


def instinct(game, viewer, net, top=6):
    """The network's immediate opinion: move preferences and how it rates the position."""
    obs, mask = net.encode(game, viewer)
    with torch.no_grad():
        logits, value = net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
    probs = torch.softmax(logits[0], -1).numpy()
    legal = np.flatnonzero(mask)
    order = legal[np.argsort(-probs[legal])]
    if game.phase in (SETUP_TICKETS, CHOOSE_TICKETS):
        picks = order  # at most 7 ticket choices: consider them all
    else:
        picks = order[:top]
    return float(value.item()), [(int(i), float(probs[i])) for i in picks]


def describe(game, pi, idx):
    """Plain-English label for an action index, plus the route it claims (if any)."""
    a = decode(game, pi, idx)
    kind = a[0]
    if kind == "draw_deck":
        return "Draw a card from the deck", None
    if kind == "draw_face_up":
        return f"Take the face-up {card_name(game.face_up[a[1]])}", None
    if kind == "draw_tickets":
        return "Draw destination tickets", None
    if kind == "pass":
        return "Pass", None
    if kind == "keep_tickets":
        names = [f"{CITY_NAMES[TICKETS[t][0]]}–{CITY_NAMES[TICKETS[t][1]]}" for t in a[1]]
        return "Keep " + ", ".join(names), None
    _, rid, color, n_locos = a
    r = ROUTES[rid]
    n = r.length - n_locos
    pay = f"{r.length} locomotives" if color == LOCO else \
        f"{n} {COLORS[color]}" + (f" + {n_locos} loco" if n_locos else "")
    return f"Claim {CITY_NAMES[r.a]}–{CITY_NAMES[r.b]} ({pay})", rid

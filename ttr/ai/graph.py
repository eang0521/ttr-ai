"""A board-independent network: it reads the board as a map of cities and routes.

The original network (model.PolicyNet) has one input slot per route and per ticket of the
USA board, so what it knows is tied to those exact routes. This one describes every
city and route by what it *is* (length, color, owner, whether it's on my plan, ...) and
passes information along the map's connections (a graph neural network). The same
weights then work for any number of routes, any tickets, and 2-5 players.

Moves use the same action menu as encode.py (so legality, decoding, search and training
are shared): one score per route and payment color comes from that route's state, the
draw moves come from the whole-board summary, and each ticket-keeping choice is scored
from the tickets in it.

Inputs added later (the 4th and 5th ticket on offer and the game-mode features, for the
Mega Game) sit at the end of the observation, so networks trained before them still load
(see upgrade_state) and play exactly as before.
"""

import numpy as np
import torch
from torch import nn

from ..data import ROUTES, TICKETS
from ..engine import CHOOSE_TICKETS, MAIN, SETUP_TICKETS
from ..planning import edge_costs, plan, unfinished
from .encode import ACTION_SIZE, MAXP, NC, PHASES, _network, legal_mask, route_gains

N_CITIES = 1 + max(max(r.a, r.b) for r in ROUTES)
N_ROUTES = len(ROUTES)
MAX_LEN = 8          # longest route length the static features allow for
HELD = 12            # held tickets described (more is very rare)
PEND = 5             # tickets on offer at once (3 in the standard game, up to 5 in the Mega Game)
PEND_OLD = 3         # the first three keep the place they had before the Mega Game

NODE_F = 12
EDGE_F = 12
GLOB_F = 9 + 9 + MAXP * 14 + 9 + 16 + MAXP + 2
MODE_F = 8           # game-mode features (Mega Game), see encode
HELD_F = 5           # + 2 endpoint city numbers
PEND_F = 4           # + 2 endpoint city numbers
OFF = {}
_o = 0
for _name, _n in [("node", N_CITIES * NODE_F), ("edge", N_ROUTES * EDGE_F), ("glob", GLOB_F),
                  ("held", HELD * (HELD_F + 2)), ("pend", PEND_OLD * (PEND_F + 2)),
                  ("pend2", (PEND - PEND_OLD) * (PEND_F + 2)), ("mode", MODE_F)]:
    OFF[_name] = _o
    _o += _n
OBS_SIZE = _o

STATIC_F = MAX_LEN + NC + 2


def board_tensors():
    """What never changes about the board: each route's two cities and its fixed traits."""
    ea = torch.tensor([r.a for r in ROUTES])
    eb = torch.tensor([r.b for r in ROUTES])
    static = torch.zeros(N_ROUTES, STATIC_F)
    for r in ROUTES:
        static[r.id, r.length - 1] = 1
        static[r.id, MAX_LEN + (8 if r.color < 0 else r.color)] = 1
        static[r.id, MAX_LEN + NC] = 1 if r.pair is not None else 0
        static[r.id, MAX_LEN + NC + 1] = r.length / 6
    return ea, eb, static


def encode(game, pi):
    """Return (observation float32[OBS_SIZE], legal-action mask bool[ACTION_SIZE])."""
    n = game.num_players
    me = game.players[pi]
    owner = game.route_owner
    node = np.zeros((N_CITIES, NODE_F), np.float32)
    edge = np.zeros((N_ROUTES, EDGE_F), np.float32)
    glob = np.zeros(GLOB_F, np.float32)
    held = np.zeros((HELD, HELD_F + 2), np.float32)
    pend = np.zeros((PEND, PEND_F + 2), np.float32)
    mode = np.zeros(MODE_F, np.float32)

    costs, best_rid = edge_costs(game, pi)
    todo = unfinished(game, pi)
    needed, total, feasible, _, tcost = plan(game, pi, todo, costs)
    claims = game.claimable(pi) if game.phase == MAIN else []

    # Routes: who owns them (seats counted from me), whether I can use them, my plan, ...
    hand = me.hand
    best_color = max(hand[:8])
    for r in ROUTES:
        w = owner[r.id]
        if w == -1:
            if game.route_open_to(r.id, pi):
                edge[r.id, 5] = 1
        else:
            edge[r.id, (w - pi) % n] = 1
        if r.pair is not None and owner[r.pair] != -1:
            edge[r.id, 11] = 1
        edge[r.id, 10] = (best_color if r.color < 0 else hand[r.color]) / 12
    for key in needed:
        rid = best_rid.get(key)
        if rid is not None:
            edge[rid, 6] = 1
            r = ROUTES[rid]
            node[r.a, 11] = node[r.b, 11] = 1
    for rid, _ in claims:
        edge[rid, 7] = 1

    # Cities: ticket ends, whose networks reach them, how far a path can run from them.
    for t in me.tickets:
        a, b, pts = TICKETS[t]
        for c in (a, b):
            if t in todo:
                node[c, 0] += pts / 22
                node[c, 1] += 1 / 3
            else:
                node[c, 2] = 1
    for t in me.pending:
        a, b, _ = TICKETS[t]
        node[a, 3] = node[b, 3] = 1
    longest = []
    for j in range(n):
        q = game.players[(pi + j) % n]
        for rid in q.routes:
            r = ROUTES[rid]
            node[r.a, 4 + j] = node[r.b, 4 + j] = 1
        best, gain = route_gains(game, q.index)
        longest.append(best)
        start = _network(tuple(q.routes))[1]
        col = 9 if j == 0 else 10
        for c, s in start.items():
            node[c, col] = max(node[c, col], s / 45)
        col = 8 if j == 0 else 9
        for rid, d in gain.items():
            edge[rid, col] = max(edge[rid, col], d / 10)

    # The whole table: my hand, the face-up cards, every player, the decks, the endgame.
    glob[0:9] = np.array(hand, np.float32) / 12
    for c in game.face_up:
        if c is not None:
            glob[9 + c] += 0.2
    o = 18
    for j in range(n):
        q = game.players[(pi + j) % n]
        b = o + j * 14
        glob[b] = 1
        glob[b + 1] = q.trains / 45
        glob[b + 2] = q.hand_size / 20
        glob[b + 3] = len(q.tickets) / 10
        glob[b + 4] = q.route_score / 100
        glob[b + 5:b + 14] = np.array(q.known_hand, np.float32) / 6
    o += MAXP * 14
    for key in needed:
        rid = best_rid.get(key)
        if rid is not None:
            r = ROUTES[rid]
            glob[o + (8 if r.color < 0 else r.color)] += r.length / 20
    o += 9
    glob[o + PHASES.index(game.phase)] = 1
    glob[o + 4] = len(game.deck) / 110
    glob[o + 5] = len(game.discard) / 110
    glob[o + 6] = len(game.ticket_deck) / 30
    glob[o + 7] = 1 if game.final_turns_left is not None else 0
    glob[o + 8] = (game.final_turns_left or 0) / 5
    glob[o + 9] = total / 45
    glob[o + 10] = (me.trains - total) / 45
    glob[o + 11 + (n - 2)] = 1
    glob[o + 15] = min(min(q.trains for q in game.players), 10) / 10
    o += 16
    for j, best in enumerate(longest):
        glob[o + j] = best / 45
    o += MAXP
    rival = max(longest[1:])
    glob[o] = np.clip((longest[0] - rival) / 20, -1, 1)
    glob[o + 1] = 1 if longest[0] >= rival and longest[0] > 0 else 0

    # Tickets as things in themselves: points, what's left to build, where they go.
    for k, t in enumerate(me.tickets[:HELD]):
        a, b, pts = TICKETS[t]
        done = t not in todo
        held[k, :HELD_F] = (1, pts / 22, done, 0 if done else min(1, tcost.get(t, 45) / 45),
                            0 if done or t in tcost else 1)
        held[k, HELD_F:] = (a, b)
    for k, t in enumerate(me.pending[:PEND]):
        a, b, pts = TICKETS[t]
        _, tot2, ok2, _, _ = plan(game, pi, todo + [t], costs)
        pend[k, :PEND_F] = (1, pts / 22, (tot2 - total) / 45, 1 if t in ok2 else 0)
        pend[k, PEND_F:] = (a, b)

    # Game mode: the Mega Game's Globetrotter bonus rewards completing many tickets.
    rules = game.rules
    universe = len(rules["tickets"])
    choosing = game.phase in (SETUP_TICKETS, CHOOSE_TICKETS)
    mode[:] = (rules["globetrotter"] > 0, rules["globetrotter"] / 15, (len(me.tickets) - len(todo)) / 10,
               len(todo) / 10, game.min_keep() / 3 if choosing else 0, len(me.pending) / 5,
               len(game.ticket_deck) / universe, universe / 100)

    obs = np.concatenate([node.ravel(), edge.ravel(), glob, held.ravel(), pend.ravel(), mode])
    return obs, legal_mask(game, pi, claims)


def _mlp(n_in, n_out, hidden):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


class _Layer(nn.Module):
    """One round of message passing. Each route updates from itself, its two cities and the
    table summary; each city from itself, its routes and the summary; the summary from
    everything. City states are projected once and then picked up by each route, which
    keeps the per-route work small."""

    def __init__(self, d):
        super().__init__()
        self.e_norm, self.n_norm, self.g_norm = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.e_self, self.e_out = nn.Linear(d, d), nn.Linear(d, d)
        self.e_sum, self.e_dif, self.e_glob = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.n_self, self.n_out = nn.Linear(d, d), nn.Linear(d, d)
        self.n_agg, self.n_glob = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.g_up = _mlp(3 * d, d, d)

    def forward(self, h, e, g, ea, eb, inv_deg):
        hn = self.n_norm(h)
        ps, pd = self.e_sum(hn), self.e_dif(hn)
        at = lambda x, i: torch.index_select(x, 1, i)
        x = (self.e_self(self.e_norm(e)) + at(ps, ea) + at(ps, eb) + (at(pd, ea) - at(pd, eb)).abs()
             + self.e_glob(g)[:, None])
        e = e + self.e_out(torch.relu(x))
        agg = torch.zeros_like(h).index_add_(1, ea, e).index_add_(1, eb, e) * inv_deg
        h = h + self.n_out(torch.relu(self.n_self(hn) + self.n_agg(agg) + self.n_glob(g)[:, None]))
        g = g + self.g_up(torch.cat([self.g_norm(g), h.mean(1), e.mean(1)], -1))
        return h, e, g


class GraphNet(nn.Module):
    """Message passing over the map: routes and cities repeatedly share what they know
    with their neighbors, alongside a summary of the whole table."""

    encode = staticmethod(encode)

    def __init__(self, d=96, layers=4):
        super().__init__()
        self.config = {"kind": "graph", "d": d, "layers": layers}
        ea, eb, static = board_tensors()
        self.register_buffer("ea", ea, persistent=False)
        self.register_buffer("eb", eb, persistent=False)
        self.register_buffer("static", static, persistent=False)
        deg = torch.zeros(N_CITIES).index_add_(0, ea, torch.ones(N_ROUTES)).index_add_(0, eb, torch.ones(N_ROUTES))
        self.register_buffer("inv_deg", (1 / deg.clamp(min=1)).view(1, -1, 1), persistent=False)
        subsets = torch.tensor([[(m >> k) & 1 for k in range(PEND)] for m in range(1, 2 ** PEND)],
                               dtype=torch.float32)
        self.register_buffer("subsets", subsets, persistent=False)

        self.node_in = nn.Linear(NODE_F, d)
        self.edge_in = nn.Linear(EDGE_F + STATIC_F, d)
        self.glob_in = nn.Linear(GLOB_F, d)
        self.mode_in = nn.Linear(MODE_F, d)
        self.layers = nn.ModuleList(_Layer(d) for _ in range(layers))
        self.held_in = _mlp(HELD_F + d, d, 2 * d)
        self.pend_in = _mlp(PEND_F + 2 * d, d, 2 * d)
        self.glob_out = nn.Sequential(nn.LayerNorm(2 * d), _mlp(2 * d, d, 2 * d))
        self.claim_e, self.claim_g = nn.Linear(d, d), nn.Linear(d, d, bias=False)
        self.claim_out = nn.Linear(d, NC)
        self.move_head = _mlp(d, 12, 2 * d)       # deck, 9 face-up, tickets, pass
        self.keep_head = _mlp(2 * d + PEND, 1, 2 * d)
        self.value_head = _mlp(3 * d, 1, 2 * d)

    def forward(self, obs, mask):
        B = obs.shape[0]
        node = obs[:, OFF["node"]:OFF["edge"]].view(B, N_CITIES, NODE_F)
        edge = obs[:, OFF["edge"]:OFF["glob"]].view(B, N_ROUTES, EDGE_F)
        glob = obs[:, OFF["glob"]:OFF["held"]]
        held = obs[:, OFF["held"]:OFF["pend"]].view(B, HELD, HELD_F + 2)
        pend = obs[:, OFF["pend"]:OFF["mode"]].view(B, PEND, PEND_F + 2)
        mode = obs[:, OFF["mode"]:]

        h = self.node_in(node)
        e = self.edge_in(torch.cat([edge, self.static.expand(B, -1, -1)], -1))
        g = self.glob_in(glob) + self.mode_in(mode)
        for layer in self.layers:
            h, e, g = layer(h, e, g, self.ea, self.eb, self.inv_deg)

        def ends(tokens, f):
            idx = tokens[:, :, f:f + 2].long()
            pick = lambda i: torch.gather(h, 1, i[:, :, None].expand(-1, -1, h.shape[-1]))
            return pick(idx[:, :, 0]) + pick(idx[:, :, 1])

        held_e = self.held_in(torch.cat([held[:, :, :HELD_F], ends(held, HELD_F)], -1)) * held[:, :, :1]
        g = g + self.glob_out(torch.cat([g, held_e.sum(1)], -1))
        pend_e = self.pend_in(torch.cat([pend[:, :, :PEND_F], ends(pend, PEND_F),
                                         g[:, None].expand(-1, PEND, -1)], -1)) * pend[:, :, :1]
        combo = torch.einsum("mk,bkd->bmd", self.subsets, pend_e)
        n_kept = self.subsets.expand(B, -1, -1)
        keep = self.keep_head(torch.cat([combo, g[:, None].expand(-1, combo.shape[1], -1), n_kept], -1))[..., 0]
        claim = self.claim_out(torch.relu(self.claim_e(e) + self.claim_g(g)[:, None])).reshape(B, -1)
        # Keep choices: with up to 3 tickets on offer they use the original slots (subsets
        # 1-7 involve only the first three tickets); with 4-5 the slots added at the end.
        logits = torch.cat([self.move_head(g), keep[:, :7], claim, keep], -1).masked_fill(~mask, -1e9)
        value = self.value_head(torch.cat([g, h.mean(1), e.mean(1)], -1))[:, 0]
        return logits, value


def upgrade_state(net, state):
    """Load weights saved before the Mega Game inputs: the new parts start at zero."""
    state = dict(state)
    w = state["keep_head.0.weight"]
    want = net.keep_head[0].weight.shape[1]
    if w.shape[1] < want:
        state["keep_head.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], want - w.shape[1])], 1)
    for k in ("mode_in.weight", "mode_in.bias"):
        if k not in state:
            state[k] = torch.zeros_like(net.state_dict()[k])
    return state


assert 12 + 7 + N_ROUTES * NC + (2 ** PEND - 1) == ACTION_SIZE

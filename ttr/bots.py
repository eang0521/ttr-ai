"""Rule-based bots at selectable difficulty.

Bots only look at what a human at the table could know: the board, the face-up
cards, their own hand and tickets, every player's train/card/ticket counts, and
the cards each opponent was seen taking from the face-up row (Player.known_hand).
The one exception is "blocker", a training-only bot that sees opponents' tickets.
"""

import random
from collections import defaultdict
from pathlib import Path

from .data import GRAY, LOCO
from .engine import CHOOSE_TICKETS, SECOND_DRAW, SETUP_TICKETS
from .planning import INF, edge_costs, plan, shortest_path, ticket_damage, unfinished


class RandomBot:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose(self, game, pi):
        return self.rng.choice(game.legal_actions())


class HeuristicBot:
    """Plans shortest paths for its tickets, collects matching cards, claims routes."""

    LEVELS = {
        # mistakes: chance of a random card-draw move; extra_tickets: draws more tickets
        # when done; ticket_risk: how strongly long ticket paths are penalized.
        # Feature flags: bottleneck/threat-aware claiming, taking face-up locomotives,
        # dumping cards into routes in the endgame, network-extending filler routes.
        "easy": dict(mistakes=0.3, extra_tickets=False, ticket_risk=1.0),
        # Feature choices come from bot-vs-bot testing; taking face-up locomotives
        # measured clearly worse, so it stays off.
        "medium": dict(mistakes=0.0, extra_tickets=True, ticket_risk=1.1,
                       bottleneck=False, threat=True, take_locos=False, endgame_dump=True,
                       network=True),
        # The competitive 2-player meta: hoard blind draws up to 44 cards, then claim
        # long routes every turn. Beats "medium" ~88% of games.
        "hard": dict(mistakes=0.0, extra_tickets=False, ticket_risk=1.1,
                     bottleneck=False, threat=True, take_locos=False, endgame_dump=True,
                     network=True, hoard=44),
        # Training sparring partner only (not offered in the game): plays like "hard"
        # but peeks at opponents' real tickets and claims the routes they depend on,
        # spending up to block_budget trains on routes it doesn't need itself.
        "blocker": dict(mistakes=0.0, extra_tickets=False, ticket_risk=1.1,
                        bottleneck=False, threat=True, take_locos=False, endgame_dump=True,
                        network=True, hoard=44, block_min=5, block_budget=20),
    }

    def __init__(self, level="medium", seed=None):
        self.level = level
        self.name = level
        self.cfg = self.LEVELS[level]
        self.rng = random.Random(seed)
        self.block_trains = 0  # trains spent so far on routes only an opponent needed

    # ---------------------------------------------------------------- entry

    def choose(self, game, pi):
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        if game.phase in (SETUP_TICKETS, CHOOSE_TICKETS):
            return self._choose_tickets(game, pi, legal)
        if game.phase == SECOND_DRAW:
            return self._draw_card(game, pi, legal, second=True)
        return self._main_turn(game, pi, legal)

    # ------------------------------------------------------------- planning

    def _edge_costs(self, game, pi):
        return edge_costs(game, pi)

    _shortest_path = staticmethod(shortest_path)

    def _plan(self, game, pi, tickets, costs=None):
        """(needed edge keys, total trains needed, feasible tickets, edge -> tickets)."""
        if costs is None:
            costs, _ = edge_costs(game, pi)
        return plan(game, pi, tickets, costs)[:4]

    def _unfinished(self, game, pi):
        return unfinished(game, pi)

    # ------------------------------------------------------------ main turn

    def _main_turn(self, game, pi, legal):
        p = game.players[pi]
        costs, best_rid = self._edge_costs(game, pi)
        needed, need_total, _, _ = self._plan(game, pi, self._unfinished(game, pi), costs)
        claims = defaultdict(list)
        for a in legal:
            if a[0] == "claim":
                claims[a[1]].append(a)

        endgame = min(q.trains for q in game.players) <= 6

        if self.cfg.get("block_min"):
            block = self._block(game, pi, claims, p.trains - need_total, needed, best_rid)
            if block:
                return block

        # Hoarding (competitive 2-player meta): draw blind until the hand is big
        # enough to lay most trains, then claim a route every turn, longest first.
        target = self.cfg.get("hoard")
        if target:
            hoarding = (p.hand_size < min(target, p.trains - 1) and not endgame
                        and ("draw_deck",) in legal
                        and min(q.trains for q in game.players if q.index != pi)
                        > self.cfg.get("hoard_until_opp", 12))
            if hoarding:
                return ("draw_deck",)
            options = []
            routes, edges = game.board.routes, game.board.edges
            for key in needed:
                for rid in edges[key]:
                    if claims.get(rid):
                        options.append((routes[rid].length + 10, rid))
            spare = p.trains - need_total
            for rid in claims:
                if routes[rid].length <= spare:
                    options.append((routes[rid].length, rid))
            if options:
                rid = max(options)[1]
                return self._best_payment(game, pi, claims[rid], needed, best_rid)

        # 1. Claim a route our tickets need.
        options = []
        for key in needed:
            for rid in game.board.edges[key]:
                if claims.get(rid):
                    options.append((self._claim_priority(game, pi, key, rid, costs, needed), rid))
        if options:
            options.sort(reverse=True)
            rid = options[0][1]
            return self._best_payment(game, pi, claims[rid], needed, best_rid)

        # 2. Nothing left to connect: take more tickets while there is time.
        spare = p.trains - need_total
        if (not needed and self.cfg["extra_tickets"] and ("draw_tickets",) in legal
                and spare >= 12 and not endgame):
            return ("draw_tickets",)

        # 3. Spend trains we won't need on points (optionally also in the endgame).
        filler = self._filler_claim(game, pi, claims, spare, needed, best_rid)
        if filler and (not needed or (self.cfg.get("endgame_dump") and endgame)):
            return filler

        # 4. Draw cards toward what we need.
        draw = self._draw_card(game, pi, legal, second=False, needed=needed, best_rid=best_rid)
        if draw:
            return draw
        if filler:
            return filler
        return self.rng.choice(legal)

    def _block(self, game, pi, claims, spare, needed, best_rid):
        """Claim the open route whose loss hurts an opponent's tickets most, if worth it."""
        budget = self.cfg["block_budget"] - self.block_trains
        best, best_dmg = None, self.cfg["block_min"] - 1
        for q in game.players:
            if q.index == pi:
                continue
            for key, (dmg, _) in ticket_damage(game, q.index).items():
                ours = key in needed
                for rid in game.board.edges[key]:
                    L = game.board.routes[rid].length
                    if not claims.get(rid) or (not ours and L > min(budget, spare)):
                        continue
                    if dmg > best_dmg:
                        best, best_dmg = (rid, ours), dmg
        if best is None:
            return None
        rid, ours = best
        if not ours:
            self.block_trains += game.board.routes[rid].length
        return self._best_payment(game, pi, claims[rid], needed, best_rid)

    def _claim_priority(self, game, pi, key, rid, costs, needed):
        r = game.board.routes[rid]
        prio = r.length
        if self.cfg.get("bottleneck"):
            # How much would losing this edge cost us?
            blocked = dict(costs)
            blocked[key] = INF
            _, base_total, base_ok, _ = self._plan(game, pi, self._unfinished(game, pi), costs)
            _, alt_total, alt_ok, _ = self._plan(game, pi, self._unfinished(game, pi), blocked)
            lost_points = sum(game.board.tickets[t][2] for t in base_ok if t not in alt_ok)
            prio += 2 * lost_points + 0.5 * max(0, alt_total - base_total)
        if self.cfg.get("threat") and r.color != GRAY:
            # An opponent visibly collecting this color.
            for q in game.players:
                if q.index != pi and q.known_hand[r.color] + q.known_hand[LOCO] >= r.length - 1:
                    prio += 5
        return prio

    def _best_payment(self, game, pi, actions, needed, best_rid):
        """Fewest locomotives; on gray routes use the color we need least elsewhere."""
        demand = self._color_demand(game, pi, needed, best_rid)
        hand = game.players[pi].hand

        def cost(a):
            _, rid, color, n_locos = a
            if color == LOCO:
                return (100, 0)
            leftover = hand[color] - (game.board.routes[rid].length - n_locos)
            return (n_locos * 10 + demand[color], leftover)

        return min(actions, key=cost)

    def _color_demand(self, game, pi, needed, best_rid):
        """Cards of each color still needed for planned colored routes."""
        need = [0] * 8
        routes = game.board.routes
        for key in needed:
            rid = best_rid.get(key)
            if rid is not None and routes[rid].color != GRAY:
                need[routes[rid].color] += routes[rid].length
        return need

    def _filler_claim(self, game, pi, claims, spare, needed, best_rid):
        """Longest claimable route we can afford without starving our tickets."""
        p = game.players[pi]
        routes = game.board.routes
        my_cities = {c for rid in p.routes for c in (routes[rid].a, routes[rid].b)}
        best, best_score = None, -1
        for rid, acts in claims.items():
            r = routes[rid]
            if r.length > spare:
                continue
            score = r.length * 10
            if self.cfg.get("network") and (r.a in my_cities or r.b in my_cities):
                score += 5  # extends our network toward longest path
            if score > best_score:
                best, best_score = rid, score
        if best is None:
            return None
        return self._best_payment(game, pi, claims[best], needed, best_rid)

    # --------------------------------------------------------- card drawing

    def _draw_card(self, game, pi, legal, second, needed=None, best_rid=None):
        draws = [a for a in legal if a[0] in ("draw_face_up", "draw_deck")]
        if not draws:
            return None
        if self.cfg.get("hoard") and ("draw_deck",) in draws:
            return ("draw_deck",)  # blind draws reveal nothing to the opponent
        if self.rng.random() < self.cfg["mistakes"]:
            return self.rng.choice(draws)
        if needed is None:
            costs, best_rid = self._edge_costs(game, pi)
            needed, _, _, _ = self._plan(game, pi, self._unfinished(game, pi), costs)
        hand = game.players[pi].hand
        demand = self._color_demand(game, pi, needed, best_rid)
        deficit = [max(0, demand[c] - hand[c]) for c in range(8)]
        routes, edges = game.board.routes, game.board.edges
        gray_len = max((routes[edges[k][0]].length for k in needed
                        if best_rid.get(k) is not None and routes[best_rid[k]].color == GRAY),
                       default=0)
        gray_color = max(range(8), key=lambda c: (hand[c] - demand[c]))

        best, best_val = None, 0
        for a in draws:
            if a[0] != "draw_face_up":
                continue
            c = game.face_up[a[1]]
            if c == LOCO:
                val = 2.5 if self.cfg.get("take_locos") and sum(deficit) >= 4 else 0
            elif deficit[c] > 0:
                val = 1 + deficit[c] / 6
            elif gray_len and c == gray_color and hand[c] < gray_len:
                val = 1
            else:
                val = 0
            if val > best_val:
                best, best_val = a, val
        if best:
            return best
        if ("draw_deck",) in draws:
            return ("draw_deck",)
        return draws[0]

    # -------------------------------------------------------------- tickets

    def _choose_tickets(self, game, pi, legal):
        if self.level == "easy":
            # Keep the cheapest tickets, the minimum allowed.
            smallest = min(len(a[1]) for a in legal)
            options = [a for a in legal if len(a[1]) == smallest]
            return min(options, key=lambda a: sum(self._solo_cost(game, pi, t) for t in a[1]))

        current = self._unfinished(game, pi)
        costs, _ = self._edge_costs(game, pi)
        _, base_total, base_ok, _ = self._plan(game, pi, current, costs)
        trains = game.players[pi].trains
        best, best_val = None, -INF
        for a in legal:
            keep = list(a[1])
            _, total, ok, _ = self._plan(game, pi, current + keep, costs)
            extra = total - base_total
            val = 0
            for t in keep:
                pts = game.board.tickets[t][2]
                val += pts if t in ok else -pts
            # Risk of not finishing grows with the trains still to lay.
            val -= self.cfg["ticket_risk"] * extra * (total / max(trains, 1))
            if total > trains - 5:
                val -= 15
            if val > best_val:
                best, best_val = a, val
        return best

    def _solo_cost(self, game, pi, t):
        costs, _ = self._edge_costs(game, pi)
        a, b, _ = game.board.tickets[t]
        return self._shortest_path(a, b, costs, game.board.neighbors)[0]


MODELS = Path(__file__).resolve().parent.parent / "models"


def model_for(board="usa"):
    """The network the Expert and Master play with on a board: the USA-trained Expert on
    the USA map, the board-independent graph network anywhere else."""
    if board == "usa":
        return MODELS / "expert.pt"
    from .data import BOARDS
    from .ai.graph import fits  # needs PyTorch, so only imported when used
    return MODELS / "graph.pt" if fits(BOARDS[board]) else None


def make_bot(kind, seed=None, board="usa"):
    if kind == "random":
        return RandomBot(seed)
    if kind == "expert":
        from .ai.model import NNBot  # needs PyTorch, so only imported when used
        return NNBot(path=model_for(board), seed=seed)
    if kind == "master":
        from .ai.search import SearchBot  # the Expert network plus look-ahead search
        return SearchBot(path=model_for(board), seed=seed, think=3.0)  # keep moves snappy
    return HeuristicBot(kind, seed)


def bot_levels(board="usa"):
    """Difficulty levels on offer; Expert and Master appear once a trained network exists."""
    try:
        path = model_for(board)
    except ImportError:  # no PyTorch: rule-based bots only
        path = None
    return ["easy", "medium", "hard"] + (["expert", "master"] if path and path.exists() else [])

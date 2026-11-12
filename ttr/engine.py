"""Rules engine for Ticket to Ride (2015 rules), on the USA map or another board.

Headless and deterministic for a given seed, so it can drive both the UI and
self-play training. Actions are plain tuples:

    ("draw_deck",)
    ("draw_face_up", slot)                 slot 0..4
    ("draw_tickets",)
    ("claim", route_id, color, n_locos)    color 0..7, or LOCO when paying only locomotives
    ("keep_tickets", (ticket_id, ...))
    ("pass",)                              only offered when nothing else is legal
"""

import random
from collections import defaultdict

from .data import (
    BOARDS, CARDS_PER_COLOR, COLORS, GRAY, LOCO, LONGEST_PATH_BONUS, NUM_CARD_TYPES, NUM_LOCOS,
    ROUTE_POINTS, ROUTES, TRAINS_PER_PLAYER, USA,
)

SETUP_TICKETS = "setup_tickets"  # choosing the opening tickets (keep >= 2; Mega: 3 of 5)
MAIN = "main"                    # start of a turn
SECOND_DRAW = "second_draw"      # drew one train card, must draw another
CHOOSE_TICKETS = "choose_tickets"  # drew tickets mid-game (keep >= 1; Mega: of 4)
GAME_OVER = "game_over"

CARD_NAMES = COLORS + ["locomotive"]


def card_name(c):
    return CARD_NAMES[c]


class Player:
    def __init__(self, index, trains=TRAINS_PER_PLAYER):
        self.index = index
        self.hand = [0] * NUM_CARD_TYPES
        self.trains = trains
        self.route_score = 0
        self.routes = []
        self.tickets = []
        self.pending = []  # tickets drawn but not yet kept/returned
        # Cards other players know this player holds (taken from the face-up row).
        self.known_hand = [0] * NUM_CARD_TYPES

    @property
    def hand_size(self):
        return sum(self.hand)


class Game:
    mode = "standard"  # games saved before modes existed are standard games
    board = USA        # ... and games saved before boards existed are on the USA map

    def __init__(self, num_players=2, seed=None, mode="standard", board="usa"):
        if not 2 <= num_players <= 5:
            raise ValueError("Ticket to Ride supports 2-5 players")
        if board not in BOARDS:
            raise ValueError(f"unknown board {board}")
        self.board = BOARDS[board]
        if mode not in self.board.modes:
            raise ValueError(f"unknown game mode {mode} for the {self.board.title} board")
        self.mode = mode
        self.num_players = num_players
        self.rng = random.Random(seed)

        self.deck = [c for c in range(8) for _ in range(CARDS_PER_COLOR)] + [LOCO] * NUM_LOCOS
        self.rng.shuffle(self.deck)
        self.discard = []
        self.face_up = [None] * 5

        self.ticket_deck = list(self.rules["tickets"])
        self.rng.shuffle(self.ticket_deck)

        self.route_owner = [-1] * len(self.board.routes)
        self.players = [Player(i, self.board.trains) for i in range(num_players)]
        for p in self.players:
            for _ in range(4):
                p.hand[self._draw_card()] += 1
        self._refill_face_up()
        for p in self.players:
            p.pending = self._draw_tickets(self.rules["setup_draw"])

        self.current = 0
        self.phase = SETUP_TICKETS
        self.final_turns_left = None
        self.consecutive_passes = 0
        self.turn_number = 0
        self.log = []
        self.result = None

    # ------------------------------------------------------------------ cards

    def _draw_card(self):
        if not self.deck:
            if not self.discard:
                return None
            self.deck = self.discard
            self.discard = []
            self.rng.shuffle(self.deck)
        return self.deck.pop()

    def _refill_face_up(self):
        for i in range(5):
            if self.face_up[i] is None:
                self.face_up[i] = self._draw_card()
        # Three or more face-up locomotives: discard all five and lay out a new row.
        for _ in range(10):
            if sum(1 for c in self.face_up if c == LOCO) < 3:
                return
            self.discard.extend(c for c in self.face_up if c is not None)
            self.face_up = [self._draw_card() for _ in range(5)]
            if len(self.deck) + len(self.discard) < 5:
                return  # not enough cards left to fix it; stop trying

    def _cards_available(self):
        return bool(self.deck or self.discard)

    # ---------------------------------------------------------------- tickets

    @property
    def rules(self):
        """The rules that differ by game mode (see data.GAME_MODES and Board.modes)."""
        return self.board.modes[self.mode]

    def min_keep(self):
        """Fewest of the tickets on offer that the current player may keep."""
        r = self.rules
        n = r["setup_keep"] if self.phase == SETUP_TICKETS else r["draw_keep"]
        return min(n, len(self.players[self.current].pending))

    def _draw_tickets(self, n):
        drawn = self.ticket_deck[:n]
        self.ticket_deck = self.ticket_deck[n:]
        return drawn

    # ------------------------------------------------------------- legality

    def route_open_to(self, rid, pi):
        """Can player pi claim this route, ignoring cards and trains?"""
        if self.route_owner[rid] != -1:
            return False
        pair = self.board.routes[rid].pair
        if pair is not None and self.route_owner[pair] != -1:
            if self.num_players <= 3 or self.route_owner[pair] == pi:
                return False
        return True

    def claim_options(self, pi, rid):
        """All legal (color, n_locos) payments for player pi on route rid."""
        p = self.players[pi]
        r = self.board.routes[rid]
        if p.trains < r.length or not self.route_open_to(rid, pi):
            return []
        L, locos = r.length, p.hand[LOCO]
        options = []
        colors = range(8) if r.color == GRAY else (r.color,)
        for c in colors:
            have = p.hand[c]
            if have == 0:
                continue
            for n in range(max(0, L - have), min(locos, L - 1) + 1):
                options.append((c, n))
        if locos >= L:
            options.append((LOCO, L))
        return options

    def claimable(self, pi):
        """Fast list of (route_id, color) the player can claim, color LOCO = all locomotives.

        Each pair stands for the cheapest payment in that color (fewest locomotives).
        """
        p = self.players[pi]
        hand, locos, trains = p.hand, p.hand[LOCO], p.trains
        owner = self.route_owner
        out = []
        for r in self.board.routes:
            L = r.length
            if L > trains or owner[r.id] != -1:
                continue
            if r.pair is not None and owner[r.pair] != -1 and (
                    self.num_players <= 3 or owner[r.pair] == pi):
                continue
            if r.color == GRAY:
                for c in range(8):
                    if hand[c] and hand[c] + locos >= L:
                        out.append((r.id, c))
            elif hand[r.color] and hand[r.color] + locos >= L:
                out.append((r.id, r.color))
            if locos >= L:
                out.append((r.id, LOCO))
        return out

    def cheapest_claim(self, pi, rid, color):
        """Engine action paying for route rid in `color` with the fewest locomotives."""
        L = self.board.routes[rid].length
        if color == LOCO:
            return ("claim", rid, LOCO, L)
        return ("claim", rid, color, max(0, L - self.players[pi].hand[color]))

    def legal_actions(self):
        if self.phase == GAME_OVER:
            return []
        pi = self.current
        p = self.players[pi]

        if self.phase in (SETUP_TICKETS, CHOOSE_TICKETS):
            if not p.pending:  # a small ticket deck ran out before this player was dealt any
                return [("keep_tickets", ())]
            return [("keep_tickets", keep) for keep in _subsets(p.pending, self.min_keep())]

        if self.phase == SECOND_DRAW:
            return self._second_draw_actions()

        actions = []
        if self._cards_available():
            actions.append(("draw_deck",))
        for slot, c in enumerate(self.face_up):
            if c is not None:
                actions.append(("draw_face_up", slot))
        if self.ticket_deck:
            actions.append(("draw_tickets",))
        for rid in range(len(self.board.routes)):
            for c, n in self.claim_options(pi, rid):
                actions.append(("claim", rid, c, n))
        if not actions:
            actions.append(("pass",))
        return actions

    def _second_draw_actions(self):
        actions = []
        if self._cards_available():
            actions.append(("draw_deck",))
        for slot, c in enumerate(self.face_up):
            if c is not None and c != LOCO:
                actions.append(("draw_face_up", slot))
        return actions

    # ------------------------------------------------------------- applying

    def step(self, action, validate=True):
        """Apply an action for the current player. Raises ValueError if illegal.

        validate=False skips the legality check; training code that only picks
        from legal moves uses it for speed.
        """
        action = tuple(action)
        if action and action[0] == "keep_tickets":
            # Accept the kept tickets in any order.
            wanted = set(action[1])
            action = ("keep_tickets", tuple(t for t in self.players[self.current].pending if t in wanted))
        if validate and action not in self.legal_actions():
            raise ValueError(f"Illegal action {action} in phase {self.phase}")
        pi = self.current
        p = self.players[pi]
        kind = action[0]

        if kind == "keep_tickets":
            keep = action[1]
            returned = [t for t in p.pending if t not in keep]
            p.tickets.extend(keep)
            self.ticket_deck.extend(returned)
            p.pending = []
            self._log(pi, f"kept {len(keep)} destination ticket{'s' if len(keep) != 1 else ''}")
            if self.phase == SETUP_TICKETS:
                if pi == self.num_players - 1:
                    self.current = 0
                    self.phase = MAIN
                else:
                    self.current = pi + 1
            else:
                self._end_turn()
            return

        if kind == "draw_tickets":
            p.pending = self._draw_tickets(self.rules["draw"])
            self.phase = CHOOSE_TICKETS
            self._log(pi, "drew destination tickets")
            return

        if kind == "claim":
            _, rid, color, n_locos = action
            r = self.board.routes[rid]
            n_color = r.length - n_locos
            if color != LOCO:
                p.hand[color] -= n_color
                p.known_hand[color] = max(0, p.known_hand[color] - n_color)
                self.discard.extend([color] * n_color)
            p.hand[LOCO] -= n_locos
            p.known_hand[LOCO] = max(0, p.known_hand[LOCO] - n_locos)
            self.discard.extend([LOCO] * n_locos)
            # If the deck had run dry, the spent cards refill empty face-up slots at once.
            self._refill_face_up()
            p.trains -= r.length
            p.route_score += ROUTE_POINTS[r.length]
            p.routes.append(rid)
            self.route_owner[rid] = pi
            self._log(pi, f"claimed {self.board.city_names[r.a]} - {self.board.city_names[r.b]} ({r.length})")
            self._end_turn()
            return

        if kind == "draw_face_up":
            slot = action[1]
            card = self.face_up[slot]
            self.face_up[slot] = None
            p.hand[card] += 1
            p.known_hand[card] += 1
            self._refill_face_up()
            self._log(pi, f"took a face-up {self.board.card_label(card)}")
            if card == LOCO or self.phase == SECOND_DRAW:
                self._end_turn()
            else:
                self.phase = SECOND_DRAW
                if not self._second_draw_actions():
                    self._end_turn()
            return

        if kind == "draw_deck":
            card = self._draw_card()
            p.hand[card] += 1
            self._log(pi, "drew a card from the deck")
            if self.phase == SECOND_DRAW:
                self._end_turn()
            else:
                self.phase = SECOND_DRAW
                if not self._second_draw_actions():
                    self._end_turn()
            return

        if kind == "pass":
            self._log(pi, "had no legal move and passed")
            self.consecutive_passes += 1
            if self.consecutive_passes >= self.num_players:
                self._finish()
            else:
                self._end_turn(passed=True)
            return

    def _end_turn(self, passed=False):
        if not passed:
            self.consecutive_passes = 0
        pi = self.current
        self.turn_number += 1
        if self.final_turns_left is not None:
            self.final_turns_left -= 1
            if self.final_turns_left <= 0:
                self._finish()
                return
        elif self.players[pi].trains <= 2:
            # Everyone, including this player, gets one final turn.
            self.final_turns_left = self.num_players
            self._log(pi, "is down to 2 or fewer trains - final round!")
        self.current = (pi + 1) % self.num_players
        self.phase = MAIN

    def _log(self, pi, text):
        self.log.append({"player": pi, "text": text, "turn": self.turn_number})

    # -------------------------------------------------------------- scoring

    def ticket_done(self, pi, tid):
        a, b, _ = self.board.tickets[tid]
        return connected(self.players[pi].routes, a, b, self.board.routes)

    def _finish(self):
        self.phase = GAME_OVER
        longest = [longest_path(p.routes, self.board.routes) for p in self.players]
        best = max(longest)
        completed = [[t for t in p.tickets if self.ticket_done(p.index, t)] for p in self.players]
        most = max(len(c) for c in completed)
        rows = []
        for p in self.players:
            done = completed[p.index]
            failed = [t for t in p.tickets if t not in done]
            tickets = self.board.tickets
            ticket_pts = sum(tickets[t][2] for t in done) - sum(tickets[t][2] for t in failed)
            bonus = LONGEST_PATH_BONUS if best > 0 and longest[p.index] == best else 0
            # Globetrotter (Mega Game): the most completed tickets, shared on a tie.
            globe = self.rules["globetrotter"] if most > 0 and len(done) == most else 0
            rows.append({
                "player": p.index,
                "route_points": p.route_score,
                "tickets_completed": done,
                "tickets_failed": failed,
                "ticket_points": ticket_pts,
                "longest_path": longest[p.index],
                "longest_bonus": bonus,
                "globetrotter_bonus": globe,
                "total": p.route_score + ticket_pts + bonus + globe,
            })
        # Winner: most points, then most completed tickets, then longest path bonus.
        key = lambda r: (r["total"], len(r["tickets_completed"]), r["longest_bonus"])
        top = max(key(r) for r in rows)
        self.result = {"scores": rows, "winners": [r["player"] for r in rows if key(r) == top]}
        self._log(-1, "Game over")

    def scores(self):
        """Final totals once the game is over, else current route points."""
        if self.result:
            return [r["total"] for r in self.result["scores"]]
        return [p.route_score for p in self.players]


# ------------------------------------------------------------------ helpers

def _subsets(items, min_size):
    n = len(items)
    out = []
    for mask in range(1, 1 << n):
        chosen = tuple(items[i] for i in range(n) if mask >> i & 1)
        if len(chosen) >= min_size:
            out.append(chosen)
    return out


def connected(route_ids, a, b, routes=ROUTES):
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for rid in route_ids:
        r = routes[rid]
        ra, rb = find(r.a), find(r.b)
        if ra != rb:
            parent[ra] = rb
    return find(a) == find(b)


def longest_path(route_ids, routes=ROUTES):
    """Length in trains of the longest continuous path (routes used once each)."""
    adj = defaultdict(list)
    for i, rid in enumerate(route_ids):
        r = routes[rid]
        adj[r.a].append((i, r.b, r.length))
        adj[r.b].append((i, r.a, r.length))
    used = [False] * len(route_ids)
    best = 0

    def dfs(city, total):
        nonlocal best
        if total > best:
            best = total
        for i, other, length in adj[city]:
            if not used[i]:
                used[i] = True
                dfs(other, total + length)
                used[i] = False

    for city in adj:
        dfs(city, 0)
    return best


def trails(route_ids, routes=ROUTES):
    """(longest path, {city: longest path starting at that city}) for a set of routes."""
    adj = defaultdict(list)
    for i, rid in enumerate(route_ids):
        r = routes[rid]
        adj[r.a].append((i, r.b, r.length))
        adj[r.b].append((i, r.a, r.length))
    used = [False] * len(route_ids)
    start = {}

    def dfs(city, total):
        nonlocal best
        if total > best:
            best = total
        for i, other, length in adj[city]:
            if not used[i]:
                used[i] = True
                dfs(other, total + length)
                used[i] = False

    for city in adj:
        best = 0
        dfs(city, 0)
        start[city] = best
    return max(start.values(), default=0), start

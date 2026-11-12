"""Turn a game position into a fixed-size vector, and moves into a fixed action menu.

Everything is from the point of view of the player to move, using only what that
player could know at the table. Seats are relative: slot 0 is "me", slot 1 the next
player, and so on, so one network plays any seat.

Action menu (ACTION_SIZE entries, illegal ones masked out):
    0           draw from the deck
    1..9        take a face-up card of type 0..8 (8 = locomotive)
    10          draw destination tickets
    11          pass (only when nothing else is legal)
    12..18      keep pending tickets, bitmask 1..7 over their positions
    19 + 9r + c claim route r paying with color c (8 = only locomotives),
                using the fewest locomotives possible
    then 31     keep pending tickets when 4 or 5 are on offer (Mega Game), bitmask 1..31;
                added at the end so networks trained before them keep their layout

This network's inputs know only the 30 standard tickets by name; in the Mega Game it
still sees what its tickets need (the planning features) but not which ones they are.
Its route inputs are tied to the USA map's route numbers, so it plays only the USA
board; the graph network (graph.py) reuses the shared pieces here on any board with the
same number of cities and routes.
"""

from functools import lru_cache

import numpy as np

from ..data import BOARDS, LOCO, ROUTES, STANDARD_TICKETS, USA
from ..engine import CHOOSE_TICKETS, MAIN, SECOND_DRAW, SETUP_TICKETS, longest_path, trails
from ..planning import edge_costs, plan, unfinished

MAXP = 5
NR = len(ROUTES)
NT = len(STANDARD_TICKETS)
NC = 9

A_DECK, A_FACE, A_TICKETS, A_PASS, A_KEEP, A_CLAIM = 0, 1, 10, 11, 12, 19
A_KEEP5 = A_CLAIM + NR * NC
ACTION_SIZE = A_KEEP5 + 31

ROUTE_KEY = [(min(r.a, r.b), max(r.a, r.b)) for r in ROUTES]

_sizes = [
    ("own", NR * 6),      # per route: owned by seat 0..4, or open to me
    ("plan", NR),         # route is on my planned ticket paths
    ("afford", NR),       # I can claim it right now
    ("hand", NC),
    ("face", NC),
    ("players", MAXP * 14),
    ("held", NT), ("done", NT), ("tcost", NT),
    ("pend", 3 * NT), ("pendf", 3 * 4),
    ("demand", 9),
    ("global", 16),
    # Longest Route bonus. Kept last so networks trained before it existed load with
    # zero weight on it (see model.load).
    ("lp", MAXP),         # each seat's longest continuous path
    ("lpx", NR),          # how much claiming this open route would lengthen my longest path
    ("lpo", NR),          # ... the longest opponent path (the most it lengthens any opponent's)
    ("lpg", 2),           # my longest path minus the best opponent's; I'd get the bonus now
]
OFF = {}
_o = 0
for _name, _n in _sizes:
    OFF[_name] = _o
    _o += _n
OBS_SIZE = _o
PHASES = [SETUP_TICKETS, MAIN, SECOND_DRAW, CHOOSE_TICKETS]


def encode(game, pi):
    """Return (observation float32[OBS_SIZE], legal-action mask bool[ACTION_SIZE])."""
    obs = np.zeros(OBS_SIZE, np.float32)
    n = game.num_players
    me = game.players[pi]
    owner = game.route_owner

    # Planning for my unfinished tickets.
    costs, best_rid = edge_costs(game, pi)
    todo = unfinished(game, pi)
    needed, total, feasible, _, tcost = plan(game, pi, todo, costs)

    board = game.board
    o = OFF["own"]
    for r in board.routes:
        w = owner[r.id]
        if w == -1:
            if game.route_open_to(r.id, pi):
                obs[o + r.id * 6 + 5] = 1
        else:
            obs[o + r.id * 6 + (w - pi) % n] = 1

    o = OFF["plan"]
    for key in needed:
        rid = best_rid.get(key)
        if rid is not None:
            obs[o + rid] = 1

    claims = game.claimable(pi) if game.phase == MAIN else []
    o = OFF["afford"]
    for rid, _ in claims:
        obs[o + rid] = 1

    obs[OFF["hand"]:OFF["hand"] + NC] = np.array(me.hand, np.float32) / 12
    o = OFF["face"]
    for c in game.face_up:
        if c is not None:
            obs[o + c] += 0.2

    o = OFF["players"]
    for j in range(n):
        q = game.players[(pi + j) % n]
        b = o + j * 14
        obs[b] = 1
        obs[b + 1] = q.trains / 45
        obs[b + 2] = q.hand_size / 20
        obs[b + 3] = len(q.tickets) / 10
        obs[b + 4] = q.route_score / 100
        obs[b + 5:b + 14] = np.array(q.known_hand, np.float32) / 6

    for t in me.tickets:
        if t >= NT or board is not USA:
            continue
        obs[OFF["held"] + t] = 1
        if t not in todo:
            obs[OFF["done"] + t] = 1
        elif t in tcost:
            obs[OFF["tcost"] + t] = tcost[t] / 45
        else:
            obs[OFF["tcost"] + t] = 1  # can't be completed any more

    if me.pending:
        for k, t in enumerate(me.pending[:3]):
            if t < NT and board is USA:
                obs[OFF["pend"] + k * NT + t] = 1
            _, tot2, ok2, _, _ = plan(game, pi, todo + [t], costs)
            b = OFF["pendf"] + k * 4
            obs[b] = 1
            obs[b + 1] = board.tickets[t][2] / 22
            obs[b + 2] = (tot2 - total) / 45
            obs[b + 3] = 1 if t in ok2 else 0

    # Cards still needed for planned routes, per color and for gray routes.
    o = OFF["demand"]
    for key in needed:
        rid = best_rid.get(key)
        if rid is None:
            continue
        r = board.routes[rid]
        obs[o + (8 if r.color < 0 else r.color)] += r.length / 20

    o = OFF["global"]
    obs[o + PHASES.index(game.phase)] = 1
    obs[o + 4] = len(game.deck) / 110
    obs[o + 5] = len(game.discard) / 110
    obs[o + 6] = len(game.ticket_deck) / 30
    obs[o + 7] = 1 if game.final_turns_left is not None else 0
    obs[o + 8] = (game.final_turns_left or 0) / 5
    obs[o + 9] = total / 45
    obs[o + 10] = (me.trains - total) / 45
    obs[o + 11 + (n - 2)] = 1
    obs[o + 15] = min(min(q.trains for q in game.players), 10) / 10

    longest = []
    for j in range(n):
        q = game.players[(pi + j) % n]
        best, gain = route_gains(game, q.index)
        longest.append(best)
        obs[OFF["lp"] + j] = best / 45
        o = OFF["lpx"] if j == 0 else OFF["lpo"]
        for rid, d in gain.items():
            if d / 10 > obs[o + rid]:
                obs[o + rid] = d / 10
    rival = max(longest[1:])
    obs[OFF["lpg"]] = np.clip((longest[0] - rival) / 20, -1, 1)
    obs[OFF["lpg"] + 1] = 1 if longest[0] >= rival and longest[0] > 0 else 0

    return obs, legal_mask(game, pi, claims)


def route_gains(game, q):
    """Player q's longest path, and how much each open route would lengthen it."""
    return _gains(game.board.key, tuple(game.players[q].routes), tuple(game.route_owner),
                  game.num_players, q)


@lru_cache(maxsize=4096)
def _gains(board, routes, owner, n, q):
    # Cached on the board's ownership, which only changes when someone claims a route.
    best, start, piece = _network(board, routes)
    routes = list(routes)
    gain = {}
    for r in BOARDS[board].routes:
        if owner[r.id] != -1 or (r.length <= best and r.a not in start and r.b not in start):
            continue
        # Double routes: with 2-3 players only one of the pair may be used, and never
        # both by the same player (Game.route_open_to).
        if r.pair is not None and owner[r.pair] != -1 and (n <= 3 or owner[r.pair] == q):
            continue
        if r.a in start and r.b in start and piece[r.a] == piece[r.b]:
            new = _longest(board, tuple(sorted(routes + [r.id])))  # closes a loop: count it exactly
        else:
            new = r.length + start.get(r.a, 0) + start.get(r.b, 0)
        if new > best:
            gain[r.id] = new - best
    return best, gain


@lru_cache(maxsize=65536)
def _longest(board, routes):
    return longest_path(list(routes), BOARDS[board].routes)


@lru_cache(maxsize=4096)
def _network(board, routes):
    """Longest path, longest path from each city, and each city's connected piece.
    A player's routes change only when they claim one, so this is cached."""
    best, start = trails(list(routes), BOARDS[board].routes)
    parent = {}

    def find(c):
        while parent.get(c, c) != c:
            c = parent[c]
        return c
    for rid in routes:
        r = BOARDS[board].routes[rid]
        a, b = find(r.a), find(r.b)
        if a != b:
            parent[a] = b
    return best, start, {c: find(c) for c in start}


def legal_mask(game, pi, claims=None):
    mask = np.zeros(ACTION_SIZE, bool)
    phase = game.phase
    me = game.players[pi]
    if phase in (SETUP_TICKETS, CHOOSE_TICKETS):
        k = len(me.pending)
        base = A_KEEP if k <= 3 else A_KEEP5
        min_keep = game.min_keep()
        for m in range(1, 1 << k):
            if bin(m).count("1") >= min_keep:
                mask[base + m - 1] = True
        return mask
    cards = bool(game.deck or game.discard)
    mask[A_DECK] = cards
    for c in game.face_up:
        if c is not None and not (phase == SECOND_DRAW and c == LOCO):
            mask[A_FACE + c] = True
    if phase == MAIN:
        mask[A_TICKETS] = bool(game.ticket_deck)
        if claims is None:
            claims = game.claimable(pi)
        for rid, c in claims:
            mask[A_CLAIM + rid * NC + c] = True
        if not mask.any():
            mask[A_PASS] = True
    return mask


def decode(game, pi, idx):
    """Action index -> engine action tuple."""
    if idx == A_DECK:
        return ("draw_deck",)
    if A_FACE <= idx < A_TICKETS:
        c = idx - A_FACE
        return ("draw_face_up", game.face_up.index(c))
    if idx == A_TICKETS:
        return ("draw_tickets",)
    if idx == A_PASS:
        return ("pass",)
    if A_KEEP <= idx < A_CLAIM or idx >= A_KEEP5:
        m = idx - (A_KEEP if idx < A_CLAIM else A_KEEP5) + 1
        pending = game.players[pi].pending
        return ("keep_tickets", tuple(t for i, t in enumerate(pending) if m >> i & 1))
    rid, c = divmod(idx - A_CLAIM, NC)
    return game.cheapest_claim(pi, rid, c)


def action_index(game, pi, action):
    """Engine action tuple -> action index (for learning from other bots)."""
    kind = action[0]
    if kind == "draw_deck":
        return A_DECK
    if kind == "draw_face_up":
        return A_FACE + game.face_up[action[1]]
    if kind == "draw_tickets":
        return A_TICKETS
    if kind == "pass":
        return A_PASS
    if kind == "keep_tickets":
        keep = set(action[1])
        pending = game.players[pi].pending
        m = sum(1 << i for i, t in enumerate(pending) if t in keep)
        return (A_KEEP if len(pending) <= 3 else A_KEEP5) + m - 1
    _, rid, color, _ = action
    return A_CLAIM + rid * NC + color

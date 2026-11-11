"""Route planning shared by the rule-based bots and the neural network's inputs."""

import heapq
from collections import defaultdict

from .data import GRAY, ROUTES, TICKETS

INF = float("inf")

# City pair -> route ids (a double route has two).
EDGES = defaultdict(list)
for _r in ROUTES:
    EDGES[(min(_r.a, _r.b), max(_r.a, _r.b))].append(_r.id)
EDGE_KEYS = list(EDGES)
NEIGHBORS = defaultdict(list)
for _k in EDGE_KEYS:
    NEIGHBORS[_k[0]].append((_k[1], _k))
    NEIGHBORS[_k[1]].append((_k[0], _k))


def edge_costs(game, pi):
    """Cost in trains for player pi to own each city-pair edge (0 if owned, inf if blocked).

    Also returns, per open edge, the route id worth planning for (of a double route,
    the half whose color pi holds most of).
    """
    costs, best_rid = {}, {}
    hand = game.players[pi].hand
    owner = game.route_owner
    for key, rids in EDGES.items():
        if any(owner[r] == pi for r in rids):
            costs[key] = 0
            continue
        open_ids = [r for r in rids if game.route_open_to(r, pi)]
        if not open_ids:
            costs[key] = INF
            continue
        costs[key] = ROUTES[open_ids[0]].length
        if len(open_ids) == 1:
            best_rid[key] = open_ids[0]
        else:
            best_rid[key] = max(open_ids, key=lambda r: (
                99 if ROUTES[r].color == GRAY else hand[ROUTES[r].color]))
    return costs, best_rid


def shortest_path(a, b, costs):
    """Dijkstra over city-pair edges. Returns (cost, [edge keys]) or (inf, None)."""
    dist = {a: 0}
    prev = {}
    heap = [(0, a)]
    while heap:
        d, city = heapq.heappop(heap)
        if city == b:
            path, c = [], b
            while c != a:
                key = prev[c]
                path.append(key)
                c = key[0] if key[1] == c else key[1]
            return d, path
        if d > dist.get(city, INF):
            continue
        for other, key in NEIGHBORS[city]:
            nd = d + costs[key]
            if nd < dist.get(other, INF):
                dist[other] = nd
                prev[other] = key
                heapq.heappush(heap, (nd, other))
    return INF, None


def plan(game, pi, tickets, costs):
    """Plan paths for tickets, biggest first, sharing track between them.

    Returns (needed edge keys, total trains needed, feasible tickets, edge -> tickets,
    ticket -> trains it adds to the plan).
    """
    costs = dict(costs)
    trains = game.players[pi].trains
    needed, used_by, ticket_cost = [], defaultdict(list), {}
    total, feasible = 0, []
    for t in sorted(tickets, key=lambda t: -TICKETS[t][2]):
        a, b, _ = TICKETS[t]
        cost, path = shortest_path(a, b, costs)
        if path is None or total + cost > trains:
            continue
        feasible.append(t)
        ticket_cost[t] = cost
        total += cost
        for key in path:
            used_by[key].append(t)
            if costs[key] > 0:
                needed.append(key)
                costs[key] = 0
    return needed, total, feasible, used_by, ticket_cost


def unfinished(game, pi):
    return [t for t in game.players[pi].tickets if not game.ticket_done(pi, t)]


def ticket_damage(game, q):
    """Open edge -> (damage, ticket points lost) if player q could no longer use it.

    Uses q's real tickets. A ticket made impossible counts twice its points; otherwise
    the damage is the extra trains of the detour.
    """
    costs, _ = edge_costs(game, q)
    trains = game.players[q].trains
    out = {}
    for t in unfinished(game, q):
        a, b, pts = TICKETS[t]
        base, path = shortest_path(a, b, costs)
        if path is None or base > trains:
            continue
        for key in path:
            if costs[key] == 0:
                continue
            blocked = dict(costs)
            blocked[key] = INF
            alt, _ = shortest_path(a, b, blocked)
            lost = pts if alt > trains else 0
            dmg, pts_lost = out.get(key, (0, 0))
            out[key] = (dmg + (2 * lost if lost else alt - base), pts_lost + lost)
    return out

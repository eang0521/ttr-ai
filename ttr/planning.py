"""Route planning shared by the rule-based bots and the neural network's inputs."""

import heapq
from collections import defaultdict

from .data import GRAY, USA

INF = float("inf")

# The USA board's city pairs -> route ids (a double route has two); every board carries
# its own (Board.edges, Board.neighbors), which the functions below use via game.board.
EDGES = USA.edges
EDGE_KEYS = USA.edge_keys
NEIGHBORS = USA.neighbors


def edge_costs(game, pi):
    """Cost in trains for player pi to own each city-pair edge (0 if owned, inf if blocked).

    Also returns, per open edge, the route id worth planning for (of a double route,
    the half whose color pi holds most of).
    """
    costs, best_rid = {}, {}
    hand = game.players[pi].hand
    owner = game.route_owner
    routes = game.board.routes
    for key, rids in game.board.edges.items():
        if any(owner[r] == pi for r in rids):
            costs[key] = 0
            continue
        open_ids = [r for r in rids if game.route_open_to(r, pi)]
        if not open_ids:
            costs[key] = INF
            continue
        costs[key] = routes[open_ids[0]].length
        if len(open_ids) == 1:
            best_rid[key] = open_ids[0]
        else:
            best_rid[key] = max(open_ids, key=lambda r: (
                99 if routes[r].color == GRAY else hand[routes[r].color]))
    return costs, best_rid


def shortest_path(a, b, costs, neighbors=NEIGHBORS):
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
        for other, key in neighbors[city]:
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
    board = game.board
    for t in sorted(tickets, key=lambda t: -board.tickets[t][2]):
        a, b, _ = board.tickets[t]
        cost, path = shortest_path(a, b, costs, board.neighbors)
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
    board = game.board
    for t in unfinished(game, q):
        a, b, pts = board.tickets[t]
        base, path = shortest_path(a, b, costs, board.neighbors)
        if path is None or base > trains:
            continue
        for key in path:
            if costs[key] == 0:
                continue
            blocked = dict(costs)
            blocked[key] = INF
            alt, _ = shortest_path(a, b, blocked, board.neighbors)
            lost = pts if alt > trains else 0
            dmg, pts_lost = out.get(key, (0, 0))
            out[key] = (dmg + (2 * lost if lost else alt - base), pts_lost + lost)
    return out

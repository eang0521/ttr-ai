"""Static map data for Ticket to Ride (USA base game, 2015 rules).

Cities carry real latitude/longitude so the UI can draw them on a real map
(Duluth sits at St. Paul's position, matching the official board).
Routes list both halves of a double route as separate entries; `pair` links them.
"""

# Card colors. Index 8 is the Locomotive (wild).
COLORS = ["purple", "white", "blue", "yellow", "orange", "black", "red", "green"]
LOCO = 8
NUM_CARD_TYPES = 9
GRAY = -1  # route color meaning "any single color"

CARDS_PER_COLOR = 12
NUM_LOCOS = 14
TRAINS_PER_PLAYER = 45
ROUTE_POINTS = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
LONGEST_PATH_BONUS = 10

# name -> (lat, lon)
CITIES = {
    "Vancouver": (49.28, -123.12),
    "Seattle": (47.61, -122.33),
    "Portland": (45.52, -122.68),
    "San Francisco": (37.77, -122.42),
    "Los Angeles": (34.05, -118.24),
    "Las Vegas": (36.17, -115.14),
    "Phoenix": (33.45, -112.07),
    "Salt Lake City": (40.76, -111.89),
    "Helena": (46.59, -112.04),
    "Calgary": (51.05, -114.07),
    "Winnipeg": (49.90, -97.14),
    "Duluth": (44.95, -93.09),  # drawn where St. Paul, MN is, as on the game board
    "Denver": (39.74, -104.99),
    "Santa Fe": (35.69, -105.94),
    "El Paso": (31.76, -106.49),
    "Omaha": (41.26, -95.93),
    "Kansas City": (39.10, -94.58),
    "Oklahoma City": (35.47, -97.52),
    "Dallas": (32.78, -96.80),
    "Houston": (29.76, -95.37),
    "Sault St. Marie": (46.50, -84.35),
    "Chicago": (41.88, -87.63),
    "Saint Louis": (38.63, -90.20),
    "Little Rock": (34.75, -92.29),
    "New Orleans": (29.95, -90.07),
    "Toronto": (43.65, -79.38),
    "Pittsburgh": (40.44, -80.00),
    "Nashville": (36.16, -86.78),
    "Atlanta": (33.75, -84.39),
    "Miami": (25.76, -80.19),
    "Montreal": (45.50, -73.57),
    "Boston": (42.36, -71.06),
    "New York": (40.71, -74.01),
    "Washington": (38.91, -77.04),
    "Raleigh": (35.78, -78.64),
    "Charleston": (32.78, -79.93),
}
CITY_NAMES = list(CITIES)
CITY_INDEX = {name: i for i, name in enumerate(CITY_NAMES)}

# (city A, city B, length, color or color pair for double routes).
# A tuple of two colors marks a double route; "gray" means any color.
_ROUTE_SPECS = [
    ("Vancouver", "Calgary", 3, "gray"),
    ("Vancouver", "Seattle", 1, ("gray", "gray")),
    ("Seattle", "Calgary", 4, "gray"),
    ("Seattle", "Helena", 6, "yellow"),
    ("Seattle", "Portland", 1, ("gray", "gray")),
    ("Portland", "San Francisco", 5, ("green", "purple")),
    ("Portland", "Salt Lake City", 6, "blue"),
    ("San Francisco", "Salt Lake City", 5, ("orange", "white")),
    ("San Francisco", "Los Angeles", 3, ("yellow", "purple")),
    ("Los Angeles", "Las Vegas", 2, "gray"),
    ("Los Angeles", "Phoenix", 3, "gray"),
    ("Los Angeles", "El Paso", 6, "black"),
    ("Calgary", "Winnipeg", 6, "white"),
    ("Calgary", "Helena", 4, "gray"),
    ("Helena", "Winnipeg", 4, "blue"),
    ("Helena", "Duluth", 6, "orange"),
    ("Helena", "Omaha", 5, "red"),
    ("Helena", "Denver", 4, "green"),
    ("Helena", "Salt Lake City", 3, "purple"),
    ("Salt Lake City", "Denver", 3, ("red", "yellow")),
    ("Salt Lake City", "Las Vegas", 3, "orange"),
    ("Phoenix", "Denver", 5, "white"),
    ("Phoenix", "Santa Fe", 3, "gray"),
    ("Phoenix", "El Paso", 3, "gray"),
    ("El Paso", "Santa Fe", 2, "gray"),
    ("El Paso", "Oklahoma City", 5, "yellow"),
    ("El Paso", "Dallas", 4, "red"),
    ("El Paso", "Houston", 6, "green"),
    ("Santa Fe", "Denver", 2, "gray"),
    ("Santa Fe", "Oklahoma City", 3, "blue"),
    ("Denver", "Omaha", 4, "purple"),
    ("Denver", "Kansas City", 4, ("black", "orange")),
    ("Denver", "Oklahoma City", 4, "red"),
    ("Winnipeg", "Duluth", 4, "black"),
    ("Winnipeg", "Sault St. Marie", 6, "gray"),
    ("Duluth", "Sault St. Marie", 3, "gray"),
    ("Duluth", "Toronto", 6, "purple"),
    ("Duluth", "Chicago", 3, "red"),
    ("Duluth", "Omaha", 2, ("gray", "gray")),
    ("Omaha", "Chicago", 4, "blue"),
    ("Omaha", "Kansas City", 1, ("gray", "gray")),
    ("Kansas City", "Saint Louis", 2, ("blue", "purple")),
    ("Kansas City", "Oklahoma City", 2, ("gray", "gray")),
    ("Oklahoma City", "Dallas", 2, ("gray", "gray")),
    ("Oklahoma City", "Little Rock", 2, "gray"),
    ("Dallas", "Little Rock", 2, "gray"),
    ("Dallas", "Houston", 1, ("gray", "gray")),
    ("Houston", "New Orleans", 2, "gray"),
    ("Little Rock", "Saint Louis", 2, "gray"),
    ("Little Rock", "Nashville", 3, "white"),
    ("Little Rock", "New Orleans", 3, "green"),
    ("New Orleans", "Atlanta", 4, ("yellow", "orange")),
    ("New Orleans", "Miami", 6, "red"),
    ("Saint Louis", "Chicago", 2, ("green", "white")),
    ("Saint Louis", "Nashville", 2, "gray"),
    ("Saint Louis", "Pittsburgh", 5, "green"),
    ("Chicago", "Pittsburgh", 3, ("orange", "black")),
    ("Chicago", "Toronto", 4, "white"),
    ("Sault St. Marie", "Toronto", 2, "gray"),
    ("Sault St. Marie", "Montreal", 5, "black"),
    ("Toronto", "Montreal", 3, "gray"),
    ("Toronto", "Pittsburgh", 2, "gray"),
    ("Montreal", "Boston", 2, ("gray", "gray")),
    ("Montreal", "New York", 3, "blue"),
    ("Boston", "New York", 2, ("yellow", "red")),
    ("New York", "Pittsburgh", 2, ("white", "green")),
    ("New York", "Washington", 2, ("orange", "black")),
    ("Pittsburgh", "Washington", 2, "gray"),
    ("Pittsburgh", "Nashville", 4, "yellow"),
    ("Pittsburgh", "Raleigh", 2, "gray"),
    ("Washington", "Raleigh", 2, ("gray", "gray")),
    ("Nashville", "Raleigh", 3, "black"),
    ("Nashville", "Atlanta", 1, "gray"),
    ("Atlanta", "Raleigh", 2, ("gray", "gray")),
    ("Atlanta", "Charleston", 2, "gray"),
    ("Atlanta", "Miami", 5, "blue"),
    ("Raleigh", "Charleston", 2, "gray"),
    ("Charleston", "Miami", 4, "purple"),
]


class Route:
    __slots__ = ("id", "a", "b", "length", "color", "pair")

    def __init__(self, rid, a, b, length, color, pair=None):
        self.id = rid
        self.a = a  # city index
        self.b = b
        self.length = length
        self.color = color  # 0..7 or GRAY
        self.pair = pair  # id of the other half of a double route, or None

    def to_dict(self):
        return {
            "id": self.id,
            "a": CITY_NAMES[self.a],
            "b": CITY_NAMES[self.b],
            "length": self.length,
            "color": "gray" if self.color == GRAY else COLORS[self.color],
            "pair": self.pair,
        }


def _color_index(name):
    return GRAY if name == "gray" else COLORS.index(name)


def _build_routes():
    routes = []
    for a, b, length, color in _ROUTE_SPECS:
        ia, ib = CITY_INDEX[a], CITY_INDEX[b]
        if isinstance(color, tuple):
            r1 = Route(len(routes), ia, ib, length, _color_index(color[0]))
            r2 = Route(len(routes) + 1, ia, ib, length, _color_index(color[1]))
            r1.pair, r2.pair = r2.id, r1.id
            routes += [r1, r2]
        else:
            routes.append(Route(len(routes), ia, ib, length, _color_index(color)))
    return routes


ROUTES = _build_routes()

# (city A, city B, points) - the 30 original USA destination tickets (the standard game).
_TICKET_SPECS = [
    ("Los Angeles", "New York", 21),
    ("Duluth", "Houston", 8),
    ("Sault St. Marie", "Nashville", 8),
    ("New York", "Atlanta", 6),
    ("Portland", "Nashville", 17),
    ("Vancouver", "Montreal", 20),
    ("Duluth", "El Paso", 10),
    ("Toronto", "Miami", 10),
    ("Portland", "Phoenix", 11),
    ("Dallas", "New York", 11),
    ("Calgary", "Salt Lake City", 7),
    ("Calgary", "Phoenix", 13),
    ("Los Angeles", "Miami", 20),
    ("Winnipeg", "Little Rock", 11),
    ("San Francisco", "Atlanta", 17),
    ("Kansas City", "Houston", 5),
    ("Los Angeles", "Chicago", 16),
    ("Denver", "Pittsburgh", 11),
    ("Chicago", "Santa Fe", 9),
    ("Vancouver", "Santa Fe", 13),
    ("Boston", "Miami", 12),
    ("Chicago", "New Orleans", 7),
    ("Montreal", "Atlanta", 9),
    ("Seattle", "New York", 22),
    ("Denver", "El Paso", 4),
    ("Helena", "Los Angeles", 8),
    ("Winnipeg", "Houston", 12),
    ("Montreal", "New Orleans", 13),
    ("Sault St. Marie", "Oklahoma City", 9),
    ("Seattle", "Los Angeles", 9),
]

# The Mega Game deck (the user's list). Some of these share cities with the original tickets
# but carry different points; those values apply only in the Mega Game, so these are
# separate tickets with their own ids (after the original 30).
_MEGA_SPECS = [
    ("Boston", "Miami", 12), ("Boston", "Washington", 4), ("Calgary", "Nashville", 14),
    ("Calgary", "Phoenix", 13), ("Calgary", "Salt Lake City", 7), ("Chicago", "Atlanta", 5),
    ("Chicago", "Boston", 7), ("Chicago", "New Orleans", 7), ("Chicago", "New York", 5),
    ("Chicago", "Santa Fe", 9), ("Dallas", "New York", 11), ("Denver", "El Paso", 4),
    ("Denver", "Pittsburgh", 11), ("Denver", "Saint Louis", 6), ("Duluth", "Dallas", 7),
    ("Duluth", "El Paso", 10), ("Duluth", "Houston", 8), ("Helena", "Los Angeles", 8),
    ("Houston", "Washington", 10), ("Kansas City", "Boston", 11), ("Kansas City", "Houston", 5),
    ("Las Vegas", "Miami", 21), ("Las Vegas", "New York", 19), ("Los Angeles", "Atlanta", 15),
    ("Los Angeles", "Calgary", 12), ("Los Angeles", "Chicago", 16), ("Los Angeles", "Miami", 19),
    ("Los Angeles", "New York", 20), ("Los Angeles", "Oklahoma City", 9), ("Montreal", "Atlanta", 9),
    ("Montreal", "Chicago", 7), ("Montreal", "Dallas", 13), ("Montreal", "New Orleans", 13),
    ("Montreal", "Raleigh", 7), ("Nashville", "New York", 6), ("New York", "Atlanta", 6),
    ("New York", "Miami", 10), ("Omaha", "New Orleans", 8), ("Phoenix", "Boston", 19),
    ("Pittsburgh", "New Orleans", 8), ("Portland", "Houston", 16), ("Portland", "Nashville", 17),
    ("Portland", "Phoenix", 11), ("Portland", "Pittsburgh", 19), ("Saint Louis", "Miami", 8),
    ("Salt Lake City", "Chicago", 11), ("Salt Lake City", "Kansas City", 7),
    ("San Francisco", "Atlanta", 17), ("San Francisco", "Sault St. Marie", 17),
    ("San Francisco", "Washington", 21), ("Sault St. Marie", "Miami", 12),
    ("Sault St. Marie", "Nashville", 8), ("Sault St. Marie", "Oklahoma City", 8),
    ("Seattle", "Las Vegas", 10), ("Seattle", "Los Angeles", 9), ("Seattle", "New York", 20),
    ("Seattle", "Oklahoma City", 14), ("Toronto", "Charleston", 6), ("Toronto", "Miami", 10),
    ("Vancouver", "Denver", 11), ("Vancouver", "Duluth", 13), ("Vancouver", "Montreal", 20),
    ("Vancouver", "Portland", 2), ("Vancouver", "Santa Fe", 13), ("Washington", "Atlanta", 4),
    ("Winnipeg", "Houston", 12), ("Winnipeg", "Little Rock", 11), ("Winnipeg", "Omaha", 6),
    ("Winnipeg", "Santa Fe", 10),
]

# Every ticket either game mode can use; a game only deals from its own mode's deck.
TICKETS = [(CITY_INDEX[a], CITY_INDEX[b], pts) for a, b, pts in _TICKET_SPECS + _MEGA_SPECS]
STANDARD_TICKETS = list(range(len(_TICKET_SPECS)))
MEGA_TICKETS = list(range(len(_TICKET_SPECS), len(TICKETS)))

# Rules that differ by game mode: the ticket deck, how many tickets are dealt at the start
# and drawn during the game (and the fewest you may keep), and the Globetrotter bonus for
# completing the most tickets.
GAME_MODES = {
    "standard": {"tickets": STANDARD_TICKETS, "setup_draw": 3, "setup_keep": 2,
                 "draw": 3, "draw_keep": 1, "globetrotter": 0},
    "mega": {"tickets": MEGA_TICKETS, "setup_draw": 5, "setup_keep": 3,
             "draw": 4, "draw_keep": 1, "globetrotter": 15},
}


def map_json():
    """Everything the UI needs to draw the board."""
    return {
        "cities": [{"name": n, "lat": CITIES[n][0], "lon": CITIES[n][1]} for n in CITY_NAMES],
        "routes": [r.to_dict() for r in ROUTES],
        "tickets": [
            {"id": i, "a": CITY_NAMES[a], "b": CITY_NAMES[b], "points": p}
            for i, (a, b, p) in enumerate(TICKETS)
        ],
        "colors": COLORS,
    }

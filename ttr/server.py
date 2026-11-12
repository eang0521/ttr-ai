"""Local web server: serves the board UI and runs the game and bots.

Run with:  python run.py   (opens http://127.0.0.1:8765 in your browser)
"""

import json
import mimetypes
import os
import pickle
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .bots import bot_levels, make_bot
from .data import BOARDS, map_json
from .engine import CHOOSE_TICKETS, GAME_OVER, SECOND_DRAW, SETUP_TICKETS, Game

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PLAYER_COLORS = ["red", "blue", "green", "yellow", "purple"]


SAVE_PATH = Path(os.environ.get("TTR_SAVE") or
                 Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ttr-ai" / "current_game.pkl")


class Session:
    def __init__(self, seats, seed=None, mode="standard", board="usa"):
        self.seats = seats  # [{"name", "type"}], type "human" or a bot level
        self.seed = seed if seed is not None else random.randrange(10**9)
        self.mode = mode  # "standard" or "mega" (see data.GAME_MODES)
        self.board = board  # a key of data.BOARDS
        self.game = Game(len(seats), seed=self.seed, mode=mode, board=board)
        # Every action taken, as (player, action). The engine is deterministic given the
        # seed, so replaying a prefix of this rebuilds any earlier position for review.
        self.history = []
        self.turn_ends = []  # len(history) at the end of each turn
        self.replayable = True
        self._setup()

    def _setup(self):
        self.bots = {i: make_bot(s["type"], board=self.board)
                     for i, s in enumerate(self.seats) if s["type"] != "human"}
        self.lock = threading.Lock()

    # Saved games keep only the seats and the game; bots are rebuilt on load.
    def __getstate__(self):
        return {"seats": self.seats, "game": self.game, "seed": self.seed, "history": self.history,
                "turn_ends": self.turn_ends, "replayable": self.replayable, "mode": self.mode,
                "board": self.board}

    def __setstate__(self, state):
        self.seats, self.game = state["seats"], state["game"]
        # Games saved before move history existed can't be replayed; record from now on.
        self.seed = state.get("seed")
        self.history = state.get("history", [])
        self.turn_ends = state.get("turn_ends", [])
        self.replayable = state.get("replayable", False)
        self.mode = state.get("mode", "standard")
        self.board = state.get("board", "usa")
        self._setup()

    def save(self):
        """Write the game to disk so a server restart doesn't lose it."""
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SAVE_PATH.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f)
        os.replace(tmp, SAVE_PATH)

    def is_bot(self, i):
        return i in self.bots

    def play(self, action):
        """Apply an action for the current player and record it."""
        g = self.game
        pi = g.current
        g.step(action)
        self.history.append((pi, action))
        if g.phase not in (SECOND_DRAW, CHOOSE_TICKETS):  # the turn is complete
            self.turn_ends.append(len(self.history))

    def bot_step(self):
        g = self.game
        if g.phase == GAME_OVER or not self.is_bot(g.current):
            return False
        self.play(self.bots[g.current].choose(g, g.current))
        return True

    def replay(self, turn):
        """The game as it stood after `turn` complete turns (0 = the opening deal)."""
        g = Game(len(self.seats), seed=self.seed, mode=self.mode, board=self.board)
        for _, action in self.history[:self.turn_ends[turn - 1] if turn else 0]:
            g.step(action, validate=False)
        return g

    def last_claims(self, n_actions):
        """Routes claimed by each player on their most recent turn (within the first
        `n_actions` actions), for highlighting."""
        last = {}
        for pi, action in self.history[:n_actions]:
            last[pi] = action
        return [a[1] for a in last.values() if a[0] == "claim"]

    def view(self, viewer, turn=None):
        """Game state as seen by seat `viewer` (hidden info of others removed). With
        `turn`, the position after that many turns, for reviewing earlier moves."""
        review = turn is not None
        g = self.replay(turn) if review else self.game
        n_actions = (self.turn_ends[turn - 1] if turn else 0) if review else len(self.history)
        over = g.phase == GAME_OVER

        names = g.board.city_names

        def ticket(t, pi):
            a, b, pts = g.board.tickets[t]
            return {"id": t, "a": names[a], "b": names[b], "points": pts,
                    "done": g.ticket_done(pi, t)}

        players = []
        for p in g.players:
            info = {
                "index": p.index, "name": self.seats[p.index]["name"],
                "type": self.seats[p.index]["type"],
                "color": self.seats[p.index].get("color", PLAYER_COLORS[p.index]),
                "trains": p.trains, "hand_size": p.hand_size, "ticket_count": len(p.tickets),
                "route_score": p.route_score, "known_hand": p.known_hand,
            }
            if over:
                info["tickets"] = [ticket(t, p.index) for t in p.tickets]
            players.append(info)

        you = None
        if viewer is not None and 0 <= viewer < g.num_players:
            p = g.players[viewer]
            you = {"index": viewer, "hand": p.hand,
                   "tickets": [ticket(t, viewer) for t in p.tickets]}
            if g.current == viewer and g.phase in (SETUP_TICKETS, CHOOSE_TICKETS):
                you["pending"] = [ticket(t, viewer) for t in p.pending]
                you["min_keep"] = g.min_keep()

        legal = None
        if viewer == g.current and not over and not review:
            acts = g.legal_actions()
            claims = {}
            for a in acts:
                if a[0] == "claim":
                    claims.setdefault(a[1], []).append([a[2], a[3]])
            legal = {
                "draw_deck": ("draw_deck",) in acts,
                "face_up": [a[1] for a in acts if a[0] == "draw_face_up"],
                "draw_tickets": ("draw_tickets",) in acts,
                "claims": claims,
                "pass": ("pass",) in acts,
            }

        return {
            "players": players, "you": you, "legal": legal,
            "current": g.current, "phase": g.phase, "mode": g.mode, "board": g.board.key,
            "current_is_bot": self.is_bot(g.current),
            "face_up": g.face_up, "deck_size": len(g.deck), "discard_size": len(g.discard),
            "ticket_deck_size": len(g.ticket_deck), "route_owner": g.route_owner,
            "final_turns_left": g.final_turns_left, "log": g.log[-60:],
            "result": g.result,
            "review": review, "turn": turn if review else len(self.turn_ends),
            "turns": len(self.turn_ends), "replayable": self.replayable,
            "last_claims": self.last_claims(n_actions),
        }


def load_saved_session():
    try:
        with open(SAVE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:  # no saved game, or one from an incompatible version
        return None


SESSION = None


class Analyst:
    """Runs the Expert network's move analysis (on other boards, the graph network's);
    reloads a network when training saves a new one."""

    def __init__(self):
        self.path = None
        self.net = None
        self.pool = None
        self.mtime = None
        self.lock = threading.Lock()

    def ready(self, board):
        from .ai.model import load
        from .bots import model_for
        path = model_for(board)
        if path is None or not path.exists():
            return False
        mtime = path.stat().st_mtime
        if path != self.path or mtime != self.mtime:
            try:
                net = load(path)
            except Exception:  # file mid-write by the trainer; keep the old one
                return self.net is not None and path == self.path
            self.net, self.path, self.mtime = net, path, mtime
            if self.pool is not None:
                self.pool.terminate()
                self.pool = None
        return True

    def get_pool(self):
        if self.pool is None:
            import multiprocessing
            import os
            from .ai.search import worker_init
            self.pool = multiprocessing.Pool(max(2, (os.cpu_count() or 4) - 2),
                                             initializer=worker_init, initargs=(str(self.path),))
        return self.pool

    def hint(self, game, viewer):
        from .ai.search import describe, instinct
        with self.lock:
            if not self.ready(game.board.key):
                return {"error": "No trained Expert model yet."}
            value, picks = instinct(game, viewer, self.net)
        moves = []
        for idx, prior in picks:
            label, rid = describe(game, viewer, idx)
            moves.append({"label": label, "route": rid, "prior": prior})
        return {"value": value, "moves": moves}

    def analyze(self, game, viewer, sims):
        from .ai.search import analyze
        with self.lock:
            if not self.ready(game.board.key):
                return {"error": "No trained Expert model yet."}
            return analyze(game, viewer, self.net, self.get_pool(), sims=sims)


ANALYST = Analyst()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the console quiet

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/map":
            board = parse_qs(url.query).get("board", ["usa"])[0]
            if board not in BOARDS:
                return self._json({"error": f"unknown board {board}"}, 400)
            boards = [{"key": b.key, "title": b.title, "modes": list(b.modes), "bot_levels": bot_levels(b.key)}
                      for b in BOARDS.values()]
            return self._json({**map_json(board), "bot_levels": bot_levels(board), "boards": boards,
                               "player_colors": PLAYER_COLORS})
        if url.path == "/api/state":
            if SESSION is None:
                return self._json({"no_game": True})
            q = parse_qs(url.query)
            viewer = int(q["viewer"][0]) if "viewer" in q else None
            turn = int(q["turn"][0]) if "turn" in q else None
            with SESSION.lock:
                if turn is not None and not (SESSION.replayable and 0 <= turn < len(SESSION.turn_ends)):
                    turn = None  # nothing to review: show the live game
                return self._json(SESSION.view(viewer, turn))
        if url.path == "/api/hint":
            if SESSION is None:
                return self._json({"error": "no game"}, 400)
            viewer = int(parse_qs(url.query)["viewer"][0])
            with SESSION.lock:
                game = self._own_turn_copy(viewer)
            if game is None:
                return self._json({"error": "Hints are only for your own turn."}, 400)
            return self._json(ANALYST.hint(game, viewer))
        # Static files
        rel = "index.html" if url.path in ("", "/") else url.path.lstrip("/")
        path = (WEB_DIR / rel).resolve()
        if WEB_DIR not in path.parents or not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _own_turn_copy(self, viewer):
        """Snapshot of the game for analysis, only on a human player's own turn."""
        import copy
        g = SESSION.game
        if g.phase == GAME_OVER or g.current != viewer or SESSION.is_bot(viewer):
            return None
        return copy.deepcopy(g)

    def do_POST(self):
        global SESSION
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/analyze":
                if SESSION is None:
                    return self._json({"error": "no game"}, 400)
                viewer = int(body["viewer"])
                with SESSION.lock:
                    game = self._own_turn_copy(viewer)
                if game is None:
                    return self._json({"error": "Analysis is only for your own turn."}, 400)
                sims = max(8, min(int(body.get("sims", 48)), 200))
                return self._json(ANALYST.analyze(game, viewer, sims))
            if url.path == "/api/new":
                seats = body["seats"]
                board = body.get("board", "usa")
                if board not in BOARDS:
                    raise ValueError(f"unknown board {board}")
                if not 2 <= len(seats) <= 5:
                    raise ValueError("2-5 players")
                levels = bot_levels(board)
                for s in seats:
                    if s["type"] != "human" and s["type"] not in levels:
                        raise ValueError(f"unknown seat type {s['type']}")
                # Each seat keeps the color it was shown with, even if the order is shuffled.
                seats = [{**s, "color": PLAYER_COLORS[i]} for i, s in enumerate(seats)]
                if body.get("random_order"):
                    random.shuffle(seats)
                mode = body.get("mode", "standard")
                if mode not in BOARDS[board].modes:
                    raise ValueError(f"unknown game mode {mode} for this board")
                SESSION = Session(seats, body.get("seed"), mode, board)
                SESSION.save()
                return self._json({"ok": True})
            if SESSION is None:
                return self._json({"error": "no game"}, 400)
            with SESSION.lock:
                g = SESSION.game
                if url.path == "/api/action":
                    pi = body["player"]
                    if pi != g.current or SESSION.is_bot(pi):
                        raise ValueError("not your turn")
                    action = body["action"]
                    if action[0] == "keep_tickets":
                        action = ["keep_tickets", tuple(action[1])]
                    SESSION.play(tuple(action))
                    SESSION.save()
                    return self._json({"ok": True})
                if url.path == "/api/bot":
                    moved = SESSION.bot_step()
                    if moved:
                        SESSION.save()
                    return self._json({"ok": moved})
            return self._json({"error": "unknown endpoint"}, 404)
        except (ValueError, KeyError, TypeError, IndexError) as e:
            return self._json({"error": str(e)}, 400)


def serve(port=8765, open_browser=True):
    global SESSION
    SESSION = load_saved_session()
    if SESSION is not None:
        print("Resumed the saved game.")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Ticket to Ride running at {url}  (close this window to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()

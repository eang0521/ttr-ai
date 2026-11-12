"""Boards other than the USA: a small made-up map, plus every board found in ttr/local_boards."""
import copy
import pickle
import random
import unittest
from types import SimpleNamespace

from ttr.bots import bot_levels, make_bot
from ttr.data import BOARDS, USA, board_from_spec
from ttr.engine import GAME_OVER, Game
from ttr.server import Session

# A small ring of six towns with a few shortcuts and one double route.
RING = SimpleNamespace(
    TITLE="Test Ring", SIZE=(1000, 800),
    CITIES={"A": (100, 100), "B": (500, 80), "C": (900, 120), "D": (880, 650), "E": (500, 700), "F": (120, 640)},
    ROUTES=[("A", "B", 4, "red"), ("B", "C", 4, ("blue", "green")), ("C", "D", 5, "gray"),
            ("D", "E", 3, "yellow"), ("E", "F", 3, "black"), ("F", "A", 5, "white"),
            ("B", "E", 6, "gray"), ("A", "E", 6, "orange"), ("C", "E", 6, "purple"), ("A", "D", 6, "gray")],
    TICKETS=[("A", "C", 8), ("A", "D", 6), ("B", "F", 9), ("C", "F", 9), ("B", "D", 9), ("F", "C", 8),
             ("A", "E", 6), ("B", "E", 6)],
    TRAINS=20, COLOR_STYLE={"white": ("Silver", "#c0c0c0")},
)
TEST_BOARD = BOARDS.get("test_ring") or board_from_spec("test_ring", RING)
OTHER_BOARDS = [b for b in BOARDS.values() if b is not USA]


def play(g, bots):
    while g.phase != GAME_OVER:
        g.step(bots[g.current].choose(g, g.current))


class AnyBoardTests(unittest.TestCase):
    def test_double_routes_pair_up(self):
        for board in OTHER_BOARDS:
            for r in board.routes:
                if r.pair is not None:
                    other = board.routes[r.pair]
                    self.assertEqual(other.pair, r.id)
                    self.assertEqual({other.a, other.b}, {r.a, r.b})

    def test_rule_bots_play_full_games(self):
        for board in OTHER_BOARDS:
            for n in (2, 3, 4):
                g = Game(n, seed=n, board=board.key)
                play(g, [make_bot("hard" if i % 2 else "medium", seed=i) for i in range(n)])
                self.assertTrue(all(p.trains >= 0 for p in g.players))

    def test_random_games_finish(self):
        rng = random.Random(5)
        for board in OTHER_BOARDS:
            for seed in range(3):
                g = Game(2 + seed, seed=seed, board=board.key)
                while g.phase != GAME_OVER:
                    g.step(rng.choice(g.legal_actions()))

    def test_board_is_shared_not_copied(self):
        for board in OTHER_BOARDS:
            g = Game(2, seed=1, board=board.key)
            self.assertIs(copy.deepcopy(g).board, board)
            self.assertIs(pickle.loads(pickle.dumps(g)).board, board)

    def test_session_replays_on_its_board(self):
        for board in OTHER_BOARDS:
            s = Session([{"name": "A", "type": "easy"}, {"name": "B", "type": "easy"}], seed=8, board=board.key)
            for _ in range(30):
                s.bot_step()
            self.assertEqual(s.replay(len(s.turn_ends)).route_owner, s.game.route_owner)
            self.assertEqual(s.view(0)["board"], board.key)


class TestBoardTests(unittest.TestCase):
    def test_board_rules(self):
        g = Game(2, seed=3, board="test_ring")
        self.assertEqual(g.players[0].trains, 20)
        self.assertEqual(len(g.players[0].pending), 3)
        self.assertEqual(TEST_BOARD.card_label(1), "silver")
        self.assertEqual(TEST_BOARD.card_label(0), "purple")
        self.assertEqual(len(TEST_BOARD.routes), 11)

    def test_networks_need_their_board_size(self):
        # The trained networks only fit boards shaped like the USA map (36 cities, 100 routes).
        self.assertEqual(bot_levels("test_ring"), ["easy", "medium", "hard"])

    def test_unknown_board_or_mode(self):
        with self.assertRaises(ValueError):
            Game(2, board="mars")
        with self.assertRaises(ValueError):
            Game(2, mode="mega", board="test_ring")


class SavedGameTests(unittest.TestCase):
    def test_old_saves_are_usa_games(self):
        s = Session([{"name": "A", "type": "human"}, {"name": "B", "type": "easy"}], seed=3)
        state = s.__getstate__()
        del state["board"]
        del state["game"].__dict__["board"]  # saved before boards existed
        restored = Session.__new__(Session)
        restored.__setstate__(pickle.loads(pickle.dumps(state)))
        self.assertEqual(restored.board, "usa")
        self.assertIs(restored.game.board, USA)
        self.assertIs(BOARDS["usa"], USA)


if __name__ == "__main__":
    unittest.main()

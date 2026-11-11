import random
import unittest

from ttr.data import CITY_INDEX, CITY_NAMES, LOCO, MEGA_TICKETS, ROUTES, STANDARD_TICKETS, TICKETS
from ttr.engine import (
    CHOOSE_TICKETS, GAME_OVER, MAIN, SECOND_DRAW, SETUP_TICKETS, Game, longest_path,
)


def route_id(a, b, color=None):
    ia, ib = CITY_INDEX[a], CITY_INDEX[b]
    for r in ROUTES:
        if {r.a, r.b} == {ia, ib} and (color is None or r.color == color):
            return r.id
    raise KeyError((a, b, color))


def skip_setup(game):
    while game.phase == SETUP_TICKETS:
        p = game.players[game.current]
        game.step(("keep_tickets", tuple(p.pending)))


def total_cards(game):
    return (len(game.deck) + len(game.discard) + sum(1 for c in game.face_up if c is not None)
            + sum(p.hand_size for p in game.players))


def play_random(game, rng, max_steps=5000):
    for _ in range(max_steps):
        if game.phase == GAME_OVER:
            return
        game.step(rng.choice(game.legal_actions()))
    raise AssertionError("game did not finish")


class DataTests(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(CITY_NAMES), 36)
        self.assertEqual(len(ROUTES), 100)
        self.assertEqual(len(STANDARD_TICKETS), 30)
        self.assertEqual(len(MEGA_TICKETS), 69)
        self.assertEqual(len(TICKETS), 99)

    def test_pairs_symmetric(self):
        for r in ROUTES:
            if r.pair is not None:
                other = ROUTES[r.pair]
                self.assertEqual(other.pair, r.id)
                self.assertEqual({r.a, r.b}, {other.a, other.b})
                self.assertEqual(r.length, other.length)

    def test_board_is_connected(self):
        all_routes = [r.id for r in ROUTES]
        from ttr.engine import connected
        for a, b, _ in TICKETS:
            self.assertTrue(connected(all_routes, a, b))


class SetupTests(unittest.TestCase):
    def test_initial_deal(self):
        g = Game(3, seed=1)
        self.assertEqual(g.phase, SETUP_TICKETS)
        for p in g.players:
            self.assertEqual(p.hand_size, 4)
            self.assertEqual(len(p.pending), 3)
            self.assertEqual(p.trains, 45)
        self.assertEqual(total_cards(g), 110)

    def test_must_keep_two_opening_tickets(self):
        g = Game(2, seed=2)
        sizes = {len(a[1]) for a in g.legal_actions()}
        self.assertEqual(sizes, {2, 3})
        with self.assertRaises(ValueError):
            g.step(("keep_tickets", (g.players[0].pending[0],)))

    def test_returned_tickets_go_to_bottom(self):
        g = Game(2, seed=3)
        returned = g.players[0].pending[2]
        g.step(("keep_tickets", tuple(g.players[0].pending[:2])))
        self.assertEqual(g.ticket_deck[-1], returned)


class DrawTests(unittest.TestCase):
    def setUp(self):
        self.g = Game(2, seed=4)
        skip_setup(self.g)

    def test_two_blind_draws(self):
        g = self.g
        g.step(("draw_deck",))
        self.assertEqual(g.phase, SECOND_DRAW)
        self.assertEqual(g.current, 0)
        g.step(("draw_deck",))
        self.assertEqual(g.current, 1)
        self.assertEqual(g.players[0].hand_size, 6)

    def test_face_up_locomotive_ends_turn(self):
        g = self.g
        g.face_up[0] = LOCO
        g.face_up[1:] = [0, 1, 2, 3]
        g.step(("draw_face_up", 0))
        self.assertEqual(g.current, 1)
        self.assertEqual(g.phase, MAIN)

    def test_second_draw_cannot_take_face_up_locomotive(self):
        g = self.g
        g.face_up = [0, LOCO, 1, 2, 3]
        g.deck.append(4)  # make sure the replacement isn't a locomotive
        g.step(("draw_face_up", 0))
        self.assertEqual(g.phase, SECOND_DRAW)
        self.assertNotIn(("draw_face_up", 1), g.legal_actions())

    def test_three_face_up_locomotives_reset(self):
        g = self.g
        g.deck.extend([0, 1, 2, 3, 4])  # next five drawn are non-locomotive
        g.face_up = [LOCO, LOCO, None, 5, 6]
        g.deck.append(LOCO)  # fills slot 2 -> three locomotives
        g._refill_face_up()
        self.assertLess(sum(1 for c in g.face_up if c == LOCO), 3)
        # Injected 6 deck cards + 4 face-up cards, replacing the original 5 face-up.
        self.assertEqual(total_cards(g), 110 + 6 + 4 - 5)

    def test_known_hand_tracks_face_up_picks(self):
        g = self.g
        g.face_up = [6, 6, 1, 2, 3]
        g.deck.extend([0, 0])
        g.step(("draw_face_up", 0))
        g.step(("draw_face_up", 1))
        self.assertEqual(g.players[0].known_hand[6], 2)


class ClaimTests(unittest.TestCase):
    def setUp(self):
        self.g = Game(2, seed=5)
        skip_setup(self.g)
        self.p = self.g.players[0]
        self.p.hand = [0] * 9

    def test_colored_route_needs_matching_color(self):
        rid = route_id("Montreal", "New York")  # blue 3
        self.p.hand[2] = 2
        self.assertEqual(self.g.claim_options(0, rid), [])
        self.p.hand[LOCO] = 1
        self.assertEqual(self.g.claim_options(0, rid), [(2, 1)])

    def test_gray_route_any_single_color(self):
        rid = route_id("Toronto", "Montreal")  # gray 3
        self.p.hand[0] = 3
        self.p.hand[5] = 3
        opts = self.g.claim_options(0, rid)
        self.assertIn((0, 0), opts)
        self.assertIn((5, 0), opts)
        self.assertNotIn((1, 0), opts)

    def test_claim_scores_and_discards(self):
        rid = route_id("Seattle", "Helena")  # yellow 6
        self.p.hand[3] = 6
        before_discard = len(self.g.discard)
        self.g.step(("claim", rid, 3, 0))
        self.assertEqual(self.p.route_score, 15)
        self.assertEqual(self.p.trains, 39)
        self.assertEqual(self.g.route_owner[rid], 0)
        self.assertEqual(len(self.g.discard), before_discard + 6)

    def test_double_route_closed_in_two_player(self):
        a = route_id("Boston", "New York", 3)  # yellow half
        b = ROUTES[a].pair
        self.p.hand[3] = 2
        self.g.step(("claim", a, 3, 0))
        self.g.players[1].hand[6] = 5
        self.assertEqual(self.g.claim_options(1, b), [])

    def test_double_route_four_player(self):
        g = Game(4, seed=6)
        skip_setup(g)
        a = route_id("Boston", "New York", 3)
        b = ROUTES[a].pair
        g.players[0].hand = [0] * 9
        g.players[0].hand[3] = 2
        g.players[0].hand[6] = 2
        g.step(("claim", a, 3, 0))
        self.assertEqual(g.claim_options(0, b), [])  # same player can't take both
        g.players[1].hand = [0] * 9
        g.players[1].hand[6] = 2
        self.assertEqual(g.claim_options(1, b), [(6, 0)])

    def test_spent_cards_refill_empty_face_up_row(self):
        g = self.g
        # Deck and discard ran out earlier; two face-up slots are empty.
        g.deck, g.discard = [], []
        g.face_up = [0, 1, None, None, 2]
        rid = route_id("Seattle", "Helena")  # yellow 6
        self.p.hand[3] = 6
        g.step(("claim", rid, 3, 0))
        self.assertNotIn(None, g.face_up)
        self.assertEqual(g.face_up.count(3), 2)
        self.assertEqual(len(g.deck) + len(g.discard), 4)

    def test_needs_enough_trains(self):
        rid = route_id("Seattle", "Helena")
        self.p.hand[3] = 6
        self.p.trains = 5
        self.assertEqual(self.g.claim_options(0, rid), [])


class TicketTests(unittest.TestCase):
    def test_mid_game_ticket_draw_keep_one(self):
        g = Game(2, seed=7)
        skip_setup(g)
        g.step(("draw_tickets",))
        self.assertEqual(g.phase, CHOOSE_TICKETS)
        pending = g.players[0].pending
        self.assertEqual(len(pending), 3)
        self.assertEqual(min(len(a[1]) for a in g.legal_actions()), 1)
        g.step(("keep_tickets", (pending[0],)))
        self.assertEqual(g.current, 1)


class ScoringTests(unittest.TestCase):
    def test_longest_path_with_loop(self):
        # Square: Helena-Denver(4), Denver-Omaha(4), Omaha-Helena(5), plus Omaha-Chicago(4)
        ids = [route_id("Helena", "Denver"), route_id("Denver", "Omaha"),
               route_id("Helena", "Omaha"), route_id("Omaha", "Chicago")]
        self.assertEqual(longest_path(ids), 4 + 4 + 5 + 4)

    def test_final_round_and_scoring(self):
        g = Game(2, seed=8)
        skip_setup(g)
        p0 = g.players[0]
        p0.tickets = [TICKETS.index((CITY_INDEX["Kansas City"], CITY_INDEX["Houston"], 5))]
        p0.trains = 3
        p0.hand = [0] * 9
        p0.hand[LOCO] = 1
        rid = route_id("Dallas", "Houston")
        g.step(("claim", rid, LOCO, 1))
        self.assertEqual(g.final_turns_left, 2)
        g.step(("draw_deck",)); g.step(("draw_deck",))  # player 1's last turn
        self.assertEqual(g.current, 0)
        g.step(("draw_deck",)); g.step(("draw_deck",))  # player 0's last turn
        self.assertEqual(g.phase, GAME_OVER)
        row = g.result["scores"][0]
        self.assertEqual(row["route_points"], 1)
        self.assertEqual(row["ticket_points"], -5)  # Kansas City - Houston not connected
        self.assertEqual(row["longest_bonus"], 10)  # player 1 has no routes


class RandomPlayTests(unittest.TestCase):
    def test_random_games_finish_and_conserve_cards(self):
        rng = random.Random(0)
        for n in (2, 3, 4, 5):
            for seed in range(15):
                g = Game(n, seed=seed * 10 + n)
                for _ in range(5000):
                    if g.phase == GAME_OVER:
                        break
                    g.step(rng.choice(g.legal_actions()))
                    self.assertEqual(total_cards(g), 110)
                    for p in g.players:
                        self.assertGreaterEqual(p.trains, 0)
                        self.assertTrue(all(c >= 0 for c in p.hand))
                self.assertEqual(g.phase, GAME_OVER)
                owned = sorted(rid for p in g.players for rid in p.routes)
                self.assertEqual(owned, sorted(i for i, o in enumerate(g.route_owner) if o != -1))


class MegaGameTests(unittest.TestCase):
    def test_deal_and_keep_rules(self):
        g = Game(5, seed=4, mode="mega")
        self.assertTrue(all(t in MEGA_TICKETS for t in g.ticket_deck))
        self.assertTrue(all(len(p.pending) == 5 for p in g.players))
        keeps = g.legal_actions()
        self.assertEqual(len(keeps), 16)  # 10 ways to keep 3, 5 to keep 4, 1 to keep all 5
        self.assertTrue(all(len(a[1]) >= 3 for a in keeps))
        skip_setup(g)
        g.step(("draw_tickets",))
        self.assertEqual(g.phase, CHOOSE_TICKETS)
        self.assertEqual(len(g.players[0].pending), 4)
        self.assertEqual(len(g.legal_actions()), 15)  # any non-empty subset of 4

    def test_mega_points_differ_from_standard(self):
        def pts(deck, a, b):
            return [TICKETS[t][2] for t in deck
                    if {TICKETS[t][0], TICKETS[t][1]} == {CITY_INDEX[a], CITY_INDEX[b]}]
        self.assertEqual(pts(STANDARD_TICKETS, "Los Angeles", "New York"), [21])
        self.assertEqual(pts(MEGA_TICKETS, "Los Angeles", "New York"), [20])

    def test_globetrotter_bonus(self):
        g = Game(2, seed=6, mode="mega")
        skip_setup(g)
        g._finish()
        rows = g.result["scores"]
        most = max(len(r["tickets_completed"]) for r in rows)
        for r in rows:
            expected = 15 if most > 0 and len(r["tickets_completed"]) == most else 0
            self.assertEqual(r["globetrotter_bonus"], expected)
        # A standard game never awards it.
        s = Game(2, seed=6)
        skip_setup(s)
        s._finish()
        self.assertTrue(all(r["globetrotter_bonus"] == 0 for r in s.result["scores"]))

    def test_random_mega_games_finish(self):
        rng = random.Random(3)
        for seed in range(4):
            g = Game(2 + seed % 4, seed=seed, mode="mega")
            play_random(g, rng)
            self.assertEqual(g.phase, GAME_OVER)


if __name__ == "__main__":
    unittest.main()

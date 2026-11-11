import pickle
import unittest

from ttr.server import Session


class ReplayTests(unittest.TestCase):
    def play_out(self, n):
        s = Session([{"name": str(i), "type": "medium"} for i in range(n)], seed=11 + n)
        while s.bot_step():
            pass
        return s

    def test_replay_rebuilds_the_final_position(self):
        for n in (2, 4):
            s = self.play_out(n)
            g, r = s.game, s.replay(len(s.turn_ends))
            self.assertEqual(r.route_owner, g.route_owner)
            self.assertEqual(r.result, g.result)
            self.assertEqual([p.hand for p in r.players], [p.hand for p in g.players])

    def test_review_view_is_read_only(self):
        s = self.play_out(2)
        v = s.view(0, 5)
        self.assertTrue(v["review"])
        self.assertIsNone(v["legal"])
        self.assertEqual(v["turn"], 5)
        self.assertEqual(v["turns"], len(s.turn_ends))

    def test_last_claims_are_each_players_latest_turn(self):
        s = self.play_out(2)
        expected = []
        for pi in range(2):
            last = [a for p, a in s.history if p == pi][-1]
            if last[0] == "claim":
                expected.append(last[1])
        self.assertEqual(sorted(s.view(None)["last_claims"]), sorted(expected))

    def test_history_survives_save_and_old_saves_load(self):
        s = self.play_out(2)
        s2 = pickle.loads(pickle.dumps(s))
        self.assertTrue(s2.replayable)
        self.assertEqual(s2.history, s.history)
        old = Session.__new__(Session)
        old.__setstate__({"seats": s.seats, "game": s.game})
        self.assertFalse(old.replayable)
        self.assertEqual(old.view(0)["last_claims"], [])


if __name__ == "__main__":
    unittest.main()

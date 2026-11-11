import random
import unittest

from ttr.ai.search import determinize
from ttr.bots import make_bot
from ttr.engine import GAME_OVER, MAIN, Game


def total_cards(g):
    return (len(g.deck) + len(g.discard) + sum(1 for c in g.face_up if c is not None)
            + sum(p.hand_size for p in g.players))


class DeterminizeTests(unittest.TestCase):
    def test_redeal_is_consistent_with_what_viewer_sees(self):
        g = Game(3, seed=11)
        bots = [make_bot("medium", seed=i) for i in range(3)]
        for _ in range(60):  # get into the middle of a game
            g.step(bots[g.current].choose(g, g.current))
        while g.phase != MAIN:
            g.step(bots[g.current].choose(g, g.current))
        viewer = g.current
        rng = random.Random(0)
        changed = False
        for _ in range(20):
            d = determinize(g, viewer, rng)
            self.assertEqual(total_cards(d), 110)
            self.assertEqual(sorted(d.ticket_deck + [t for p in d.players for t in p.tickets]),
                             sorted(g.ticket_deck + [t for p in g.players for t in p.tickets]))
            self.assertEqual(d.players[viewer].hand, g.players[viewer].hand)
            self.assertEqual(d.players[viewer].tickets, g.players[viewer].tickets)
            self.assertEqual(d.face_up, g.face_up)
            for p, q in zip(d.players, g.players):
                self.assertEqual(p.hand_size, q.hand_size)
                self.assertEqual(len(p.tickets), len(q.tickets))
                self.assertTrue(all(h >= k for h, k in zip(p.hand, q.known_hand)))
                if p.index != viewer and (p.hand != q.hand or p.tickets != q.tickets):
                    changed = True
            # The imagined game must be playable to the end.
            while d.phase != GAME_OVER:
                d.step(random.choice(d.legal_actions()))
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()

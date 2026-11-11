import random
import unittest

from ttr.ai.encode import ACTION_SIZE, OBS_SIZE, action_index, decode, encode
from ttr.bots import make_bot
from ttr.engine import GAME_OVER, Game


class EncodeTests(unittest.TestCase):
    def test_mask_matches_legal_actions(self):
        rng = random.Random(1)
        for n in (2, 3, 5):
            for seed in range(4):
                g = Game(n, seed=seed)
                while g.phase != GAME_OVER:
                    pi = g.current
                    obs, mask = encode(g, pi)
                    self.assertEqual(obs.shape, (OBS_SIZE,))
                    self.assertEqual(mask.shape, (ACTION_SIZE,))
                    legal = g.legal_actions()
                    # Every masked index decodes to a legal move...
                    decoded = {decode(g, pi, i) for i in mask.nonzero()[0]}
                    self.assertTrue(decoded <= set(legal), decoded - set(legal))
                    # ...and every legal move maps to a masked index.
                    for a in legal:
                        self.assertTrue(mask[action_index(g, pi, a)], a)
                    g.step(rng.choice(legal))

    def test_bot_moves_round_trip(self):
        g = Game(2, seed=3)
        bots = [make_bot("hard", seed=0), make_bot("medium", seed=1)]
        while g.phase != GAME_OVER:
            pi = g.current
            a = bots[pi].choose(g, pi)
            _, mask = encode(g, pi)
            self.assertTrue(mask[action_index(g, pi, a)])
            g.step(a)

    def test_claimable_matches_legal_claims(self):
        rng = random.Random(2)
        g = Game(4, seed=9)
        while g.phase != GAME_OVER:
            if g.phase == "main":
                legal = {(a[1], a[2]) for a in g.legal_actions() if a[0] == "claim"}
                self.assertEqual(set(g.claimable(g.current)), legal)
            g.step(rng.choice(g.legal_actions()))


if __name__ == "__main__":
    unittest.main()

"""Measure a trained network against a rule-based bot or another network.

    python -m ttr.ai.evaluate models/expert.pt --vs hard --games 400
    python -m ttr.ai.evaluate models/expert.pt --vs models/expert_v3.pt
"""

import argparse
import time
from multiprocessing import Pool

from ..bots import make_bot
from ..engine import GAME_OVER, Game

_net = _opp = None


def _init(path, opponent):
    global _net, _opp
    import torch
    torch.set_num_threads(1)
    from .model import load
    _net = load(path)
    _opp = load(opponent) if opponent.endswith(".pt") else None


def _play(args):
    from .model import NNBot
    seed, opponent, players, *rest = args  # (older training runs send fewer fields)
    mode = rest[0] if rest else "standard"
    board = rest[1] if len(rest) > 1 else "usa"
    # Rotate the network's seat so it isn't always first.
    nn_seat = seed % players
    bots = [NNBot(_net) if i == nn_seat else
            NNBot(_opp) if _opp is not None else make_bot(opponent, seed=seed * 11 + i)
            for i in range(players)]
    g = Game(players, seed=seed, mode=mode, board=board)
    while g.phase != GAME_OVER:
        pi = g.current
        g.step(bots[pi].choose(g, pi), validate=False)
    scores = [r["total"] for r in g.result["scores"]]
    winners = g.result["winners"]
    win = 1 / len(winners) if nn_seat in winners else 0.0
    others = [s for i, s in enumerate(scores) if i != nn_seat]
    return win, scores[nn_seat], sum(others) / len(others)


def evaluate(path, opponent="hard", games=400, players=2, workers=14, seed=777_000, mode="standard",
             board="usa"):
    opponent = str(opponent)
    with Pool(workers, initializer=_init, initargs=(str(path), opponent)) as pool:
        results = pool.map(_play, [(seed + i, opponent, players, mode, board) for i in range(games)],
                           chunksize=4)
    wins = sum(r[0] for r in results)
    return {
        "win_rate": wins / games,
        "nn_score": sum(r[1] for r in results) / games,
        "opp_score": sum(r[2] for r in results) / games,
        "games": games,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--vs", default="hard")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--board", default="usa")
    args = ap.parse_args()
    t = time.time()
    r = evaluate(args.model, args.vs, args.games, args.players, board=args.board)
    print(f"vs {args.vs} ({args.players}p): win {100 * r['win_rate']:.1f}%  "
          f"score {r['nn_score']:.1f} vs {r['opp_score']:.1f}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

"""Pit bots against each other: python -m ttr.arena hard medium --games 200"""

import argparse
import time

from .bots import make_bot
from .engine import GAME_OVER, Game


def play(bots, seed):
    game = Game(len(bots), seed=seed)
    while game.phase != GAME_OVER:
        pi = game.current
        game.step(bots[pi].choose(game, pi))
    return game


def match(kinds, games=100, seed=0):
    wins = [0.0] * len(kinds)
    totals = [0] * len(kinds)
    turns = 0
    for g in range(games):
        # Rotate seats so nobody always goes first.
        order = [(i + g) % len(kinds) for i in range(len(kinds))]
        bots = [make_bot(kinds[i], seed=seed + g * 7 + i) for i in order]
        game = play(bots, seed + g)
        turns += game.turn_number
        winners = game.result["winners"]
        for seat, i in enumerate(order):
            totals[i] += game.result["scores"][seat]["total"]
            if seat in winners:
                wins[i] += 1 / len(winners)
    return wins, [t / games for t in totals], turns / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bots", nargs="+")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    start = time.time()
    wins, avg, turns = match(args.bots, args.games, args.seed)
    elapsed = time.time() - start
    for kind, w, s in zip(args.bots, wins, avg):
        print(f"{kind:>8}: win {100 * w / args.games:5.1f}%  avg score {s:6.1f}")
    print(f"{args.games} games, {turns:.0f} turns/game, {args.games / elapsed:.1f} games/sec")


if __name__ == "__main__":
    main()

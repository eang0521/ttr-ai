"""Stage 1: teach the network to copy the Hard bot (behavior cloning).

    python -m ttr.ai.imitate gen --games 6000     # record Hard-bot games
    python -m ttr.ai.imitate train --epochs 10    # fit the network -> models/bc.pt
"""

import argparse
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..bots import make_bot
from ..engine import GAME_OVER, Game
from .encode import ACTION_SIZE, action_index, encode

# Training data is large, so it lives outside the (OneDrive-synced) project folder.
DATA = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ttr-ai" / "data" / "bc.npz"


def outcome(game, seat):
    """Training target for a finished game: +1 win / -1 loss, plus score margin / 100."""
    scores = [r["total"] for r in game.result["scores"]]
    best_other = max(s for i, s in enumerate(scores) if i != seat)
    winners = game.result["winners"]
    win = (1.0 if len(winners) == 1 else 0.0) if seat in winners else -1.0
    return win + (scores[seat] - best_other) / 100


def _gen(args):
    first, count = args
    obs, masks, acts, vals = [], [], [], []
    for s in range(first, first + count):
        rng = random.Random(s)
        kinds = ["hard", "hard" if rng.random() < 0.6 else "medium"]
        rng.shuffle(kinds)
        bots = [make_bot(k, seed=s * 7 + i) for i, k in enumerate(kinds)]
        g = Game(2, seed=s)
        recorded = []
        while g.phase != GAME_OVER:
            pi = g.current
            a = bots[pi].choose(g, pi)
            if kinds[pi] == "hard":
                o, m = encode(g, pi)
                if m.sum() > 1:
                    obs.append(o.astype(np.float16))
                    masks.append(np.packbits(m))
                    acts.append(action_index(g, pi, a))
                    recorded.append(pi)
            g.step(a, validate=False)
        vals.extend(outcome(g, pi) for pi in recorded)
    return (np.array(obs), np.array(masks), np.array(acts, np.int16), np.array(vals, np.float32))


def generate(games, workers=14, chunk=25):
    t = time.time()
    with Pool(workers) as pool:
        parts = pool.map(_gen, [(i, min(chunk, games - i)) for i in range(0, games, chunk)])
    obs, masks, acts, vals = (np.concatenate([p[k] for p in parts]) for k in range(4))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, obs=obs, masks=masks, acts=acts, vals=vals)
    print(f"{games} games -> {len(acts)} decisions in {time.time() - t:.0f}s")


def train(epochs=10, batch=1024, lr=1e-3, out="models/bc.pt"):
    import torch
    import torch.nn.functional as F
    from .model import PolicyNet, save

    torch.set_num_threads(8)
    d = np.load(DATA)
    obs, masks, acts, vals = d["obs"], d["masks"], d["acts"].astype(np.int64), d["vals"]
    n = len(acts)
    idx = np.random.default_rng(0).permutation(n)
    val_idx, tr_idx = idx[: n // 20], idx[n // 20:]

    def batch_tensors(ix):
        o = torch.from_numpy(obs[ix].astype(np.float32))
        m = torch.from_numpy(np.unpackbits(masks[ix], axis=1, count=ACTION_SIZE).astype(bool))
        return o, m, torch.from_numpy(acts[ix]), torch.from_numpy(vals[ix])

    net = PolicyNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(tr_idx) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    rng = np.random.default_rng(1)
    for ep in range(epochs):
        t = time.time()
        net.train()
        perm = rng.permutation(tr_idx)
        for k in range(len(tr_idx) // batch):
            o, m, a, v = batch_tensors(np.sort(perm[k * batch:(k + 1) * batch]))
            logits, value = net(o, m)
            loss = F.cross_entropy(logits, a) + 0.5 * F.mse_loss(value, v)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            correct, vloss, cnt = 0, 0.0, 0
            for k in range(0, len(val_idx), 4096):
                o, m, a, v = batch_tensors(np.sort(val_idx[k:k + 4096]))
                logits, value = net(o, m)
                correct += (logits.argmax(-1) == a).sum().item()
                vloss += F.mse_loss(value, v, reduction="sum").item()
                cnt += len(a)
        print(f"epoch {ep + 1}: move agreement {100 * correct / cnt:.1f}%  "
              f"value error {vloss / cnt:.3f}  ({time.time() - t:.0f}s)", flush=True)
    save(net, out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "train"])
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    if args.cmd == "gen":
        generate(args.games)
    else:
        train(args.epochs)


if __name__ == "__main__":
    main()

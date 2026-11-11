"""Teach the graph network (graph.py) to play like the current Expert.

    python -m ttr.ai.distill --rounds 3 --games 3000

Each round, games are played and every decision is labeled with the Expert's move
preferences and its estimate of the position; the graph network learns to match both.
In the first round the Expert makes the moves. Later rounds let the graph network make
more of them itself (the Expert still supplies the labels), so it also learns what to do
in the positions its own small mistakes lead to, not just in the Expert's games.
Afterwards the graph network is checked head to head against the Expert and against Hard.
"""

import argparse
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..bots import make_bot
from ..engine import GAME_OVER, Game
from .encode import ACTION_SIZE, decode
from .graph import GraphNet
from .model import EXPERT_PATH, MODELS_DIR, load, save

RUN_DIR = Path(os.environ.get("TTR_RUN_DIR") or
               Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ttr-ai" / "graph")
TEACHER = RUN_DIR / "teacher.pt"
STUDENT = RUN_DIR / "student.pt"
TOPK = 8               # teacher preferences kept per decision (the rest are tiny)
OPPONENTS = {"teacher": 0.5, "hard": 0.2, "blocker": 0.2, "medium": 0.1}
EXPLORE_T = 0.25       # chance the playing network samples a move instead of taking its best

_teacher = _student = None


def _init(student_path):
    global _teacher, _student
    torch.set_num_threads(1)
    _teacher = load(TEACHER)
    _student = load(student_path) if student_path else None


@torch.no_grad()
def _games(args):
    seed, n_games, student_share = args
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("obs", "mask", "idx", "prob", "value")}
    kinds, p = list(OPPONENTS), list(OPPONENTS.values())
    for _ in range(n_games):
        kind = kinds[rng.choice(len(kinds), p=p)]
        g = Game(2, seed=int(rng.integers(2**31)))
        labeled = {0, 1} if kind == "teacher" else {int(rng.integers(2))}
        other = None if kind == "teacher" else make_bot(kind, seed=int(rng.integers(2**31)))
        while g.phase != GAME_OVER:
            pi = g.current
            if pi not in labeled:
                g.step(other.choose(g, pi), validate=False)
                continue
            t_obs, mask = _teacher.encode(g, pi)
            if mask.sum() == 1:
                g.step(decode(g, pi, int(mask.argmax())), validate=False)
                continue
            m = torch.from_numpy(mask)[None]
            t_logits, t_value = _teacher(torch.from_numpy(t_obs)[None], m)
            probs = torch.softmax(t_logits[0], -1)
            top = torch.topk(probs, TOPK)
            s_obs, _ = _student.encode(g, pi) if _student is not None else GraphNet.encode(g, pi)
            out["obs"].append(s_obs.astype(np.float16))
            out["mask"].append(np.packbits(mask))
            out["idx"].append(top.indices.numpy().astype(np.int16))
            out["prob"].append(top.values.numpy().astype(np.float16))
            out["value"].append(float(t_value))
            # Who moves: the graph network (part of the time, in later rounds) or the Expert.
            if _student is not None and rng.random() < student_share:
                logits = _student(torch.from_numpy(s_obs)[None], m)[0]
            else:
                logits = t_logits[0]
            if rng.random() < EXPLORE_T:
                a = int(torch.multinomial(torch.softmax(logits, -1), 1))
            else:
                a = int(logits.argmax())
            g.step(decode(g, pi, a), validate=False)
    return {k: np.array(v) for k, v in out.items()}


def collect(games, student_share, workers, seed):
    per = 10
    tasks = [(seed + i, per, student_share) for i in range(games // per)]
    student = str(STUDENT) if STUDENT.exists() and student_share > 0 else None
    with Pool(workers, initializer=_init, initargs=(student,)) as pool:
        parts = pool.map(_games, tasks, chunksize=1)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def fit(net, data, epochs, lr, minibatch=512, vcoef=0.5):
    """Match the teacher's move preferences (cross-entropy on its top moves) and value."""
    n = len(data["value"])
    perm0 = np.random.default_rng(0).permutation(n)
    val, tr = perm0[:max(500, n // 50)], perm0[max(500, n // 50):]
    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(np.unpackbits(data["mask"], axis=1, count=ACTION_SIZE).astype(bool))
    idx = torch.from_numpy(data["idx"].astype(np.int64))
    prob = torch.from_numpy(data["prob"].astype(np.float32))
    prob = prob / prob.sum(1, keepdim=True)
    value = torch.from_numpy(data["value"].astype(np.float32))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(tr) + minibatch - 1) // minibatch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    def batch_loss(ix):
        logits, v = net(obs[ix].float(), mask[ix])
        lp = torch.log_softmax(logits, -1).gather(1, idx[ix])
        pl = -(prob[ix] * lp).sum(-1).mean()
        vl = F.mse_loss(v, value[ix])
        agree = (logits.argmax(-1) == idx[ix, 0]).float().mean()
        return pl, vl, agree

    for ep in range(epochs):
        t = time.time()
        net.train()
        rng = np.random.default_rng(ep + 1)
        order = torch.from_numpy(rng.permutation(tr))
        tot = [0.0, 0.0, 0.0]
        nb = 0
        for k in range(0, len(order), minibatch):
            pl, vl, agree = batch_loss(order[k:k + minibatch])
            loss = pl + vcoef * vl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot = [a + b.item() for a, b in zip(tot, (pl, vl, agree))]
            nb += 1
        net.eval()
        with torch.no_grad():
            vparts = [batch_loss(torch.from_numpy(val[k:k + 2048])) for k in range(0, len(val), 2048)]
        vp, vv, va = (float(np.mean([x[i].item() for x in vparts])) for i in range(3))
        print(f"  epoch {ep + 1}/{epochs}: train policy {tot[0] / nb:.3f} value {tot[1] / nb:.4f} "
              f"agree {tot[2] / nb:.1%} | held-out policy {vp:.3f} value {vv:.4f} "
              f"same top move as Expert {va:.1%}  ({time.time() - t:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--games", type=int, default=3000, help="games per round")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--eval-games", type=int, default=400)
    args = ap.parse_args()
    from .evaluate import evaluate

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if not TEACHER.exists():
        shutil.copy(EXPERT_PATH, TEACHER)
    net = load(STUDENT) if STUDENT.exists() else GraphNet(args.d, args.layers)
    print(f"teacher: {TEACHER}; student: {sum(p.numel() for p in net.parameters()):,} weights", flush=True)
    kept = []
    for rnd in range(args.rounds):
        share = min(0.9, 0.45 * rnd)
        t = time.time()
        data = collect(args.games, share, args.workers, seed=1_000_000 * (rnd + 1))
        print(f"round {rnd + 1}: {len(data['value']):,} positions from {args.games} games, "
              f"graph network moving {share:.0%} of the time ({time.time() - t:.0f}s)", flush=True)
        kept = (kept + [data])[-2:]  # this round's positions plus the previous round's
        both = {k: np.concatenate([d[k] for d in kept]) for k in data}
        fit(net, both, args.epochs, args.lr)
        save(net, STUDENT)
        h2h = evaluate(STUDENT, str(TEACHER), games=args.eval_games, workers=args.workers, seed=8_000_000 + rnd)
        hard = evaluate(STUDENT, "hard", games=args.eval_games, workers=args.workers, seed=8_100_000 + rnd)
        print(f"round {rnd + 1} CHECK: graph network vs Expert {100 * h2h['win_rate']:.1f}%, "
              f"vs Hard {100 * hard['win_rate']:.1f}%", flush=True)
    shutil.copy(STUDENT, MODELS_DIR / "graph_v1.pt")
    print(f"saved {MODELS_DIR / 'graph_v1.pt'}", flush=True)


if __name__ == "__main__":
    main()

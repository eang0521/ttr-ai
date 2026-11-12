"""Stage 2: improve the network by playing (PPO self-play).

    python -m ttr.ai.ppo --iters 2000                # start from models/expert_v4.pt
    python -m ttr.ai.ppo --iters 3000 --resume       # continue an earlier run
    python -m ttr.ai.ppo --init models/graph_v1.pt --players 2:0.4,3:0.2,4:0.2,5:0.2 --publish models/graph.pt
                                                     # a graph network, 2-5 player games

Each iteration, worker processes play games (2 players unless --players mixes in
larger games) with the current network against a mix of opponents (itself, older snapshots of itself, earlier Experts, and rule bots including
a route-blocking one), then the network is updated with PPO. Every few iterations it is
checked against Hard, Blocker, a fixed older Expert, and the current best version; when
it beats the best (see PROMOTE), it becomes the new best and is copied to
models/expert.pt, which the game uses.

The reward is only the final result (win/loss plus score margin / 100), so nothing
nudges the network toward claiming routes early instead of hoarding: whichever plan
actually wins more is what it learns. A KL "anchor" to the starting network keeps
early updates from wrecking what it already knows; it fades to zero so the network
is free to move past its teacher later on.

Two things keep it from getting stuck on a strategy just because it once looked best:
training games include deliberate experiments (see EXPLORE), and ticket decisions are
also learned by looking ahead (see LOOK_TAU), which tries every option every time.
"""

import argparse
import copy
import csv
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..bots import make_bot
from ..engine import CHOOSE_TICKETS, GAME_OVER, MAIN, SETUP_TICKETS, Game
from .encode import A_CLAIM, A_DECK, A_FACE, A_KEEP, A_KEEP5, A_TICKETS, ACTION_SIZE, NC, decode, legal_mask
from .imitate import outcome
from .model import EXPERT_PATH, MODELS_DIR, NNBot, PolicyNet, load, save
from ..planning import ticket_damage

# Snapshots and trainer state churn constantly; keep them out of the OneDrive-synced project.
RUN_DIR = Path(os.environ.get("TTR_RUN_DIR") or
               Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ttr-ai" / "ppo")
LATEST = RUN_DIR / "latest.pt"
BEST = RUN_DIR / "best.pt"
STATE = RUN_DIR / "trainer_state.pt"
LOG = RUN_DIR / "log.csv"

GAMMA, LAMBDA = 1.0, 0.99
# Opponents in training games. "champion" is one of the earlier Experts or the current
# best, so the network has to keep beating strong, varied players rather than just
# learning Hard's habits. "blocker" plays like Hard but sees the network's tickets and
# claims the routes they depend on, so leaving a bottleneck open for too long gets punished.
OPPONENT_MIX = {"self": 0.25, "past": 0.25, "champion": 0.15, "hard": 0.15, "blocker": 0.15,
                "medium": 0.05}
BOTS = ("hard", "blocker", "medium")
CHAMPIONS = [MODELS_DIR / "expert_v2.pt", MODELS_DIR / "expert_v3.pt", MODELS_DIR / "expert_v4.pt"]
CHAMPION_TEMPERATURE = 0.5

# Every check plays four matches on deals never used before: vs Hard, vs Blocker, vs the
# fixed v3 Expert (a yardstick that never changes), and head to head vs the current best.
# A version replaces the best (and becomes the game's Expert) only if it hasn't got
# clearly worse against Hard or Blocker (SLACK), and either wins at least PROMOTE of the
# head-to-head games, or breaks even head to head while clearly improving against
# Blocker (BLOCKER_GAIN). The best never blocks, so better defense alone can't show up
# in the head-to-head games.
EVAL_GAMES = 400
YARDSTICK = MODELS_DIR / "expert_v3.pt"
PROMOTE = 0.55
SLACK = 0.03
BLOCKER_GAIN = 0.05
# Runs with larger games also check 4-player games (one network against three Hard bots,
# and against three copies of the best). A version that does clearly better there
# (MULTI_GAIN over the 1-in-4 share an equal player would win) while breaking even in the
# 2-player games is promoted too.
MULTI_GAIN = 0.05

# Experimenting during training games, so the network doesn't stop trying ideas it once
# rated poorly (e.g. drawing tickets mid-game, or claiming early instead of hoarding):
#   keep  - chance a ticket-keeping decision picks a random legal combination
#   move  - chance a turn picks a random kind of move (blind draw / face-up card /
#           claim / draw tickets), then a move of that kind the network likes
#   style - chance a whole game is played with a lean toward or away from one kind of
#           move (e.g. claim early instead of hoarding), to try out different plans
#   block - chance a turn claims the route the opponent's tickets would miss most, when
#           one is affordable. Picking it peeks at the opponent's real tickets (the
#           network never sees them), so the network gets to find out whether blocking
#           pays and learn to spot good blocks from what is visible on the board.
# Training knows which moves were experiments (see _explore_action), so they can't drag
# the network's normal play around.
EXPLORE = {"keep": 0.15, "move": 0.05, "style": 0.15, "block": 0.1}
BLOCK_MIN = 5  # least damage (see planning.ticket_damage) worth trying a block for
STYLE_LEAN = 2.0  # how far (in logits) a style game leans
CATEGORIES = {
    "deck": [A_DECK],
    "face": list(range(A_FACE, A_FACE + 9)),
    "tickets": [A_TICKETS],
    "claim": list(range(A_CLAIM, A_KEEP5)),
}

# Looking ahead at ticket decisions. Experiments alone almost never land on a good but
# unlikely choice (like keeping all 3 opening tickets) often enough to learn it. So
# whenever the network chooses tickets, every option is also rated by the network's
# value estimate of the position after keeping it, and the network learns toward the
# ones that look best. On some ordinary turns, drawing tickets is checked the same way,
# with a few guessed draws, against the position as it stands.
LOOK_TAU = 0.1        # how sharply the lookahead prefers the best-looking option
LOOK_TICKETS = 0.05   # share of turns where drawing tickets is checked
LOOK_DRAWS = 2        # guessed draws per check
LOOK_SHIFT = 3.0      # most one check can raise or lower drawing tickets (in logits)
LOOK_BLOCK = 0.05     # share of turns where the best available block is checked the same way


def _category_masks():
    masks = {}
    for name, idx in CATEGORIES.items():
        m = torch.zeros(ACTION_SIZE)
        m[idx] = 1.0
        masks[name] = m
    return masks


CATEGORY_MASKS = _category_masks()


def _pick_style(rng, explore):
    """None for a normal game; otherwise a logit offset that leans the whole game."""
    if rng.random() >= explore["style"]:
        return None
    name = list(CATEGORIES)[int(rng.integers(len(CATEGORIES)))]
    # Tickets are the one kind the network almost never tries, so only lean toward them.
    sign = 1.0 if name == "tickets" else float(rng.choice([-1.0, 1.0]))
    return sign * STYLE_LEAN * CATEGORY_MASKS[name]


def _block_dist(g, mask, logits):
    """The network's choice of payment for the claim that would hurt the opponent's tickets
    most (None if no affordable claim does at least BLOCK_MIN damage)."""
    pi = g.current
    damage = {}
    for q in range(g.num_players):
        if q != pi:
            for key, (d, _) in ticket_damage(g, q).items():
                damage[key] = damage.get(key, 0) + d
    best, best_d = None, BLOCK_MIN - 1
    for idx in np.flatnonzero(mask[A_CLAIM:]) + A_CLAIM:
        r = g.board.routes[(idx - A_CLAIM) // NC]
        d = damage.get((min(r.a, r.b), max(r.a, r.b)), 0)
        if d > best_d:
            best, best_d = r.id, d
    if best is None:
        return None
    on = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    on[A_CLAIM + best * NC:A_CLAIM + (best + 1) * NC] = True
    on &= torch.from_numpy(mask)
    return torch.softmax(logits.masked_fill(~on, -1e9), -1)


def _explore_action(g, mask, logits, style, rng, explore):
    """Choose the move to play: usually a sample from the network, sometimes an experiment.

    Also returns the log-probability of the move under the mix of choices actually used
    (network, lean and experiments). Training weighs each move by how likely the network
    itself was to play it compared with that, so a forced experiment that went badly
    doesn't teach the network to avoid a move it plays sensibly, and luck in an experiment
    isn't credited to the moves that led up to it."""
    legal = torch.from_numpy(mask)
    if g.phase == MAIN and explore.get("block") and rng.random() < explore["block"]:
        # The coin is tossed before looking, so on these turns the block is the whole
        # behavior policy (its log-probability is just the payment choice).
        dist = _block_dist(g, mask, logits)
        if dist is not None:
            a = int(torch.multinomial(dist, 1))
            return a, "block", float(torch.log(dist[a]))
    base = torch.softmax(logits if style is None else logits + style, -1)
    mix, a, how = base, None, ("style" if style is not None else None)
    if g.phase in (SETUP_TICKETS, CHOOSE_TICKETS) and explore["keep"]:
        uniform = legal.float() / legal.sum()
        mix = (1 - explore["keep"]) * base + explore["keep"] * uniform
        if rng.random() < explore["keep"]:
            a, how = int(rng.choice(np.flatnonzero(mask))), "keep"
    elif g.phase == MAIN and explore["move"]:
        kinds = [legal & CATEGORY_MASKS[name].bool() for name in CATEGORIES]
        within = [torch.softmax(logits.masked_fill(~k, -1e9), -1) for k in kinds if k.any()]
        mix = (1 - explore["move"]) * base + explore["move"] * sum(within) / len(within)
        if rng.random() < explore["move"]:
            a, how = int(torch.multinomial(within[int(rng.integers(len(within)))], 1)), "move"
    if a is None:
        a = int(torch.multinomial(base, 1))
    return a, how, float(torch.log(mix[a]))


def _after(net, g, pi, idx):
    """Player pi's view of the position after pi plays action idx (or the result, if that ends the game)."""
    g2 = copy.deepcopy(g, {id(g.log): []})  # the move log isn't needed
    g2.step(decode(g2, pi, idx), validate=False)
    return outcome(g2, pi) if g2.phase == GAME_OVER else net.encode(g2, pi)


def _keep_values(net, g, pi, legal):
    """The network's value estimate of the position after each legal ticket choice."""
    after = [_after(net, g, pi, int(i)) for i in legal]
    q = np.array([x if isinstance(x, float) else 0.0 for x in after], np.float32)
    pos = [k for k, x in enumerate(after) if not isinstance(x, float)]
    if pos:
        obs = torch.from_numpy(np.stack([after[k][0] for k in pos]))
        mask = torch.from_numpy(np.stack([after[k][1] for k in pos]))
        with torch.no_grad():
            q[pos] = net(obs, mask)[1].numpy()
    return q


def _soft(q):
    w = np.exp((q - q.max()) / LOOK_TAU)
    return w / w.sum()


def _keep_target(net, g, pi, mask):
    """Lookahead target for a ticket choice: every option, weighted by how good it looks."""
    legal = np.flatnonzero(mask)
    q = _keep_values(net, g, pi, legal)
    target = np.zeros(ACTION_SIZE, np.float32)
    target[legal] = _soft(q)
    return target, int(legal[np.argmax(q)])


def _block_target(net, g, pi, mask, logits, value):
    """Lookahead target for an ordinary turn: the network's own choice, with the best
    available block raised or lowered by how much better the position after it looks."""
    dist = _block_dist(g, mask, logits)
    if dist is None:
        return None, False
    idx = int(dist.argmax())
    q = _after(net, g, pi, idx)
    if not isinstance(q, float):
        with torch.no_grad():
            q = net(torch.from_numpy(q[0])[None], torch.from_numpy(q[1])[None])[1].item()
    gain = (q - value) / LOOK_TAU
    shifted = logits.clone()
    shifted[idx] += float(np.clip(gain, -LOOK_SHIFT, LOOK_SHIFT))
    return torch.softmax(shifted, -1).numpy(), gain > 0


def _draw_target(net, g, pi, logits, value, rng):
    """Lookahead target for an ordinary turn: the network's own choice, with drawing
    tickets raised or lowered by how much better a few guessed draws (each followed by
    the best-looking keep) look than the position as it stands."""
    me = g.players[pi]
    unseen = [t for t in g.rules["tickets"] if t not in me.tickets]
    n = min(g.rules["draw"], len(g.ticket_deck))
    qs = []
    for _ in range(LOOK_DRAWS):
        g2 = copy.deepcopy(g, {id(g.log): []})
        drawn = [int(t) for t in rng.choice(unseen, n, replace=False)]
        g2.ticket_deck = [t for t in g2.ticket_deck if t not in drawn]
        g2.players[pi].pending = drawn
        g2.phase = CHOOSE_TICKETS
        q = _keep_values(net, g2, pi, np.flatnonzero(legal_mask(g2, pi)))
        qs.append(float((_soft(q) * q).sum()))
    gain = (np.mean(qs) - value) / LOOK_TAU
    shifted = logits.clone()
    shifted[A_TICKETS] += float(np.clip(gain, -LOOK_SHIFT, LOOK_SHIFT))
    return torch.softmax(shifted, -1).numpy(), gain > 0


# ---------------------------------------------------------------- workers

_cache = {}


def _init_worker():
    torch.set_num_threads(1)


def _net(path, version):
    key = str(path)
    if key not in _cache or _cache[key][0] != version:
        _cache[key] = (version, load(path))
    return _cache[key][1]


def _opponent(kind, snapshots, champions, rng):
    if kind in BOTS:
        return make_bot(kind, seed=int(rng.integers(2**31)))
    if kind == "champion":
        path, ver = champions[int(rng.integers(len(champions)))]
        return NNBot(_net(path, ver), temperature=CHAMPION_TEMPERATURE, seed=int(rng.integers(2**31)))
    snap = snapshots[int(rng.integers(len(snapshots)))]
    return NNBot(_net(snap, 0), temperature=1.0, seed=int(rng.integers(2**31)))


def _rollout(task):
    version, snapshots, champions, seed, n_games, explore, players, modes = task
    net = _net(LATEST, version)
    rng = np.random.default_rng(seed)
    kinds, probs = list(OPPONENT_MIX), list(OPPONENT_MIX.values())
    # pol: a move the network made (False: an extra value-only sample of the position
    # right after a ticket choice, so the lookahead's value estimates stay accurate).
    buf = {k: [] for k in ("obs", "mask", "act", "logp", "rho", "adv", "ret", "pol")}
    tgt_at, tgt = [], []
    stats = []
    tried = {"keep": 0, "move": 0, "style": 0, "decisions": 0,
             "offers": 0, "look_all": 0, "net_all": 0, "draw_checks": 0, "draw_up": 0,
             "block": 0, "block_checks": 0, "block_up": 0}

    def add(obs, mask, act, logp, rho, adv, ret, pol):
        for k, v in zip(buf, (obs, mask, act, logp, rho, adv, ret, pol)):
            buf[k].append(v)

    sizes, size_p = list(players), list(players.values())
    mode_names, mode_p = list(modes), list(modes.values())
    for _ in range(n_games):
        kind = kinds[rng.choice(len(kinds), p=probs)] if snapshots else rng.choice(["self", "hard"])
        n = int(sizes[rng.choice(len(sizes), p=size_p)])
        mode = mode_names[rng.choice(len(mode_names), p=mode_p)]
        g = Game(n, seed=int(rng.integers(2**31)), mode=mode)
        if kind == "self":
            learners, opps = list(range(n)), {}
        else:
            learners = [int(rng.integers(n))]
            # Every other seat is its own opponent of this kind (in larger games each bot
            # or past network is drawn separately).
            opps = {s: _opponent(kind, snapshots, champions, rng) for s in range(n) if s != learners[0]}
        traj = {s: [] for s in learners}
        styles = {s: _pick_style(rng, explore) for s in learners}
        while g.phase != GAME_OVER:
            pi = g.current
            if pi not in traj:
                g.step(opps[pi].choose(g, pi), validate=False)
                continue
            obs, mask = net.encode(g, pi)
            if mask.sum() == 1:
                g.step(decode(g, pi, int(mask.argmax())), validate=False)
                continue
            with torch.no_grad():
                logits, value = net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
                logits, value = logits[0], value.item()
                a, how, blogp = _explore_action(g, mask, logits, styles[pi], rng, explore)
                logp = torch.log_softmax(logits, -1)[a].item()
            tried["decisions"] += 1
            if how:
                tried[how] += 1
            target = None
            keeping = g.phase in (SETUP_TICKETS, CHOOSE_TICKETS)
            if keeping:
                target, best = _keep_target(net, g, pi, mask)
                full = len(g.players[pi].pending)
                if full == 3:
                    tried["offers"] += 1
                    tried["look_all"] += bin(best - A_KEEP + 1).count("1") == 3
                    tried["net_all"] += bin(int(logits.argmax()) - A_KEEP + 1).count("1") == 3
            elif g.phase == MAIN and mask[A_TICKETS] and rng.random() < LOOK_TICKETS:
                target, up = _draw_target(net, g, pi, logits, value, rng)
                tried["draw_checks"] += 1
                tried["draw_up"] += int(up)
            elif g.phase == MAIN and rng.random() < LOOK_BLOCK:
                target, up = _block_target(net, g, pi, mask, logits, value)
                if target is not None:
                    tried["block_checks"] += 1
                    tried["block_up"] += int(up)
            g.step(decode(g, pi, a), validate=False)
            after = None
            if keeping and g.phase != GAME_OVER:
                o2, m2 = net.encode(g, pi)
                after = (o2, np.packbits(m2))
            traj[pi].append((obs, np.packbits(mask), a, logp, blogp, value, target, after))

        for s, steps in traj.items():
            if not steps:
                continue
            T = len(steps)
            vals = np.array([st[5] for st in steps] + [0.0], np.float32)
            rho = np.minimum(1.0, np.exp([st[3] - st[4] for st in steps])).astype(np.float32)
            rewards = np.zeros(T, np.float32)
            rewards[-1] = outcome(g, s)
            # V-trace: like GAE, but a move the network was unlikely to play itself only
            # passes back as much credit as the network's own odds of playing it.
            vt = np.zeros(T + 1, np.float32)
            acc = 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + GAMMA * vals[t + 1] - vals[t]
                acc = rho[t] * delta + GAMMA * LAMBDA * rho[t] * acc
                vt[t] = vals[t] + acc
            adv = rewards + GAMMA * vt[1:] - vals[:-1]
            for t, st in enumerate(steps):
                add(st[0], st[1], st[2], st[3], rho[t], adv[t], vt[t], True)
                if st[6] is not None:
                    tgt_at.append(len(buf["act"]) - 1)
                    tgt.append(st[6])
                if st[7] is not None:
                    add(st[7][0], st[7][1], 0, 0.0, 0.0, 0.0, rewards[t] + GAMMA * vt[t + 1], False)
        if kind != "self":
            s = learners[0]
            w = g.result["winners"]
            stats.append((kind, n, mode, (1 / len(w)) if s in w else 0.0))
    arrays = {
        "obs": np.array(buf["obs"], np.float32), "mask": np.array(buf["mask"], np.uint8),
        "act": np.array(buf["act"], np.int64), "logp": np.array(buf["logp"], np.float32),
        "rho": np.array(buf["rho"], np.float32), "adv": np.array(buf["adv"], np.float32),
        "ret": np.array(buf["ret"], np.float32), "pol": np.array(buf["pol"], bool),
        "tgt_at": np.array(tgt_at, np.int64),
        "tgt": np.array(tgt, np.float32).reshape(-1, ACTION_SIZE),
    }
    return arrays, stats, tried


def _merge(parts):
    """Concatenate the workers' samples; lookahead targets point at rows, so shift those."""
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "tgt_at"}
    offsets = np.cumsum([0] + [len(p["act"]) for p in parts[:-1]])
    data["tgt_at"] = np.concatenate([p["tgt_at"] + off for p, off in zip(parts, offsets)])
    return data


# ---------------------------------------------------------------- learner

def _masked_logp(logits):
    # Illegal moves have huge negative logits; clamp so 0 * log(0) terms stay finite.
    return torch.log_softmax(logits, -1).clamp(min=-30)


def ppo_update(net, opt, data, ref=None, anchor=0.0, policy=True, epochs=3, minibatch=4096,
               clip=0.2, vcoef=0.5, ent=0.005, lcoef=1.0, target_kl=0.015):
    """One PPO update. `ref` + `anchor` add a KL penalty toward the reference network
    (except where the lookahead is teaching something new); `policy=False` trains only
    the value head (used to warm it up on a new reward)."""
    n = len(data["act"])
    obs = torch.from_numpy(data["obs"])
    mask = torch.from_numpy(np.unpackbits(data["mask"], axis=1, count=ACTION_SIZE).astype(bool))
    act = torch.from_numpy(data["act"])
    old_logp = torch.from_numpy(data["logp"])
    rho = torch.from_numpy(data["rho"])
    pol = torch.from_numpy(data["pol"])
    adv = torch.from_numpy(data["adv"])
    adv = (adv - adv[pol].mean()) / (adv[pol].std() + 1e-8)
    ret = torch.from_numpy(data["ret"])
    tgt_row = torch.full((n,), -1, dtype=torch.long)
    tgt_row[torch.from_numpy(data["tgt_at"])] = torch.arange(len(data["tgt_at"]))
    tgt = torch.from_numpy(data["tgt"])
    # The policy before this update, to measure exactly how far the update moves it.
    with torch.no_grad():
        old_all = torch.cat([_masked_logp(net(obs[k:k + minibatch], mask[k:k + minibatch])[0])
                             for k in range(0, n, minibatch)])
    net.train()
    info = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "kl_ref": 0.0, "look": 0.0}
    count = 0
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_kl, epoch_n = 0.0, 0
        for k in range(0, n, minibatch):
            ix = perm[k:k + minibatch]
            logits, value = net(obs[ix], mask[ix])
            vloss = F.mse_loss(value, ret[ix])
            p = pol[ix]
            pix = ix[p]
            lp = torch.log_softmax(logits[p], -1)
            lpc = lp.clamp(min=-30)
            probs = lpc.exp()
            entropy = -(probs * lpc).sum(-1).mean()
            olp = old_all[pix]
            kl = (olp.exp() * (olp - lpc)).sum(-1).mean()
            has = tgt_row[pix] >= 0
            kl_ref = torch.zeros(())
            if ref is not None:
                # While warming up, hold the whole policy in place; after that, leave the
                # lookahead positions free to move.
                held = ~has if policy else torch.ones_like(has)
                if held.any():
                    with torch.no_grad():
                        ref_lp = _masked_logp(ref(obs[pix[held]], mask[pix[held]])[0])
                    kl_ref = (probs[held] * (lpc[held] - ref_lp)).sum(-1).mean()
            pg = look = torch.zeros(())
            if policy:
                # The played move's log-probability is not clamped, so even moves rated
                # below e^-30 get the right weight; the log-ratio is bounded instead.
                logp = lp.gather(1, act[pix, None])[:, 0]
                ratio = torch.exp((logp - old_logp[pix]).clamp(-20, 20))
                a = adv[pix]
                surr = torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
                # Dual clip: a bad move whose probability has already risen a lot can't
                # dominate the update.
                surr = torch.where(a < 0, torch.max(surr, 3.0 * a), surr)
                pg = -(rho[pix] * surr).mean()
                if has.any():
                    target = tgt[tgt_row[pix][has]]
                    look = -torch.where(target > 0, target * lp[has], torch.zeros(())).sum(-1).mean()
                loss = pg + vcoef * vloss - ent * entropy + anchor * kl_ref + lcoef * look
            else:
                # The value and policy share layers; hold the policy in place while the value learns.
                loss = vcoef * vloss + kl_ref
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                info["pg"] += pg.item()
                info["v"] += vloss.item()
                info["ent"] += entropy.item()
                info["kl"] += kl.item()
                info["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item() if policy else 0.0
                info["kl_ref"] += kl_ref.item()
                info["look"] += look.item()
            epoch_kl += kl.item()
            epoch_n += 1
            count += 1
        # Stop early if this update has already moved the policy far enough.
        if policy and epoch_kl / epoch_n > target_kl:
            break
    net.eval()
    return {k: v / count for k, v in info.items()}


def _keep_snapshots(snapshots, new, recent=5, spaced_every=100, spaced_max=15):
    """A league of opponents: the last few snapshots plus older ones spread over the run,
    so the network keeps beating its past selves instead of chasing only the latest."""
    snapshots = snapshots + [new]

    def it_of(p):
        return int(Path(p).stem.split("_")[1])

    spaced = [p for p in snapshots[:-recent] if it_of(p) % spaced_every == 0][-spaced_max:]
    return spaced + snapshots[-recent:]


def _check_seed(it):
    """Fresh deals for every check, so no version gets kept for luck on a fixed set."""
    return 3_000_000 + 10_000 * it


def train(iters, init, resume, workers, games_per_task, eval_every, lr,
          anchor, anchor_iters, warmup, patience, explore=EXPLORE, players=None, publish=EXPERT_PATH,
          modes=None):
    players = players or {2: 1.0}
    modes = modes or {"standard": 1.0}
    multi = any(n > 2 for n in players)
    mega = "mega" in modes
    from .evaluate import evaluate

    torch.set_num_threads(8)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if resume and STATE.exists():
        state = torch.load(STATE, weights_only=False)
        net = load(LATEST)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        opt.load_state_dict(state["opt"])
        start, snapshots = state["iter"], state["snapshots"]
        best_iter, init = state["best_iter"], state["init"]
        best_multi = state.get("best_multi")
        best_mega = state.get("best_mega")
        if "best_hard" in state:
            best_hard = state["best_hard"]
        else:
            # Older runs judged versions on a fixed set of deals; re-measure on fresh ones.
            best_hard = evaluate(BEST, "hard", games=EVAL_GAMES, workers=workers,
                                 seed=_check_seed(0))["win_rate"]
        if "best_blocker" in state:
            best_blocker = state["best_blocker"]
        else:
            best_blocker = evaluate(BEST, "blocker", games=EVAL_GAMES, workers=workers,
                                    seed=_check_seed(0) + 9000)["win_rate"]
        print(f"resuming at iteration {start}; best is from iteration {best_iter}, "
              f"{100 * best_hard:.1f}% vs hard, {100 * best_blocker:.1f}% vs blocker", flush=True)
    else:
        net = load(init)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        start, snapshots, best_iter = 0, [], 0
        save(net, LATEST)
        shutil.copy(LATEST, BEST)
        if LOG.exists():
            LOG.unlink()
        best_hard = evaluate(BEST, "hard", games=EVAL_GAMES, workers=workers,
                             seed=_check_seed(0))["win_rate"]
        best_blocker = evaluate(BEST, "blocker", games=EVAL_GAMES, workers=workers,
                                seed=_check_seed(0) + 9000)["win_rate"]
        best_multi = best_mega = None
        print(f"starting network: win {100 * best_hard:.1f}% vs hard, "
              f"{100 * best_blocker:.1f}% vs blocker over {EVAL_GAMES} games", flush=True)
    if multi and best_multi is None:
        best_multi = evaluate(BEST, "hard", games=EVAL_GAMES, players=4, workers=workers,
                              seed=_check_seed(0) + 12000)["win_rate"]
        print(f"best so far wins {100 * best_multi:.1f}% of 4-player games vs 3 Hard bots", flush=True)
    if mega and best_mega is None:
        best_mega = evaluate(BEST, "hard", games=EVAL_GAMES, workers=workers, mode="mega",
                             seed=_check_seed(0) + 18000)["win_rate"]
        print(f"best so far wins {100 * best_mega:.1f}% of Mega Games vs Hard", flush=True)
    for gparam in opt.param_groups:
        gparam["lr"] = lr
    ref = load(init)
    ref.eval()

    header = ["iter", "games", "samples", "rollout_win_vs_hard", "rollout_win_vs_blocker",
              "eval_win_vs_hard", "eval_score", "eval_opp_score", "win_vs_blocker",
              "win_vs_v3", "win_vs_best", "promoted",
              "pg", "v", "ent", "kl", "clipfrac", "kl_ref", "look", "anchor",
              "lookahead_keep_all", "net_keep_all", "draw_checks", "draw_better",
              "blocks_tried", "block_checks", "block_better", "secs"]
    if multi:
        header = header[:-1] + ["rollout_win_multi", "win4_vs_hard", "win4_vs_best", "secs"]
    if mega:
        header = header[:-1] + ["mega_win_vs_hard", "secs"]
    if LOG.exists():
        with open(LOG, newline="") as f:
            old_header = next(csv.reader(f), None)
        if old_header != header:
            LOG.rename(RUN_DIR / f"log_until_{start}.csv")
    new_log = not LOG.exists()
    logf = open(LOG, "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(header)

    with Pool(workers, initializer=_init_worker) as pool:
        for it in range(start, iters):
            t = time.time()
            seed = 1_000_003 * (it + 1)
            champions = [(str(p), 0) for p in CHAMPIONS] + [(str(BEST), best_iter)]
            tasks = [(it, list(snapshots), champions, seed + i, games_per_task, explore, players, modes)
                     for i in range(workers)]
            results = pool.map(_rollout, tasks)
            data = _merge([r[0] for r in results])
            stats = [s for r in results for s in r[1]]
            tried = {k: sum(r[2][k] for r in results) for k in results[0][2]}
            vs_hard = [w for kind, n, mode, w in stats if kind == "hard" and n == 2 and mode == "standard"]
            vs_blocker = [w for kind, n, mode, w in stats if kind == "blocker" and n == 2 and mode == "standard"]
            # Larger games: win share times the number of players (1.0 = an equal share).
            vs_multi = [w * n for kind, n, mode, w in stats if n > 2]
            coef = anchor * max(0.0, 1 - it / anchor_iters) if anchor_iters else anchor
            info = ppo_update(net, opt, data, ref=ref, anchor=coef, policy=it >= warmup)
            save(net, LATEST)

            if (it + 1) % 10 == 0:
                snap = RUN_DIR / f"snap_{it + 1:05d}.pt"
                save(net, snap)
                snapshots = _keep_snapshots(snapshots, str(snap))

            ev = {"win_rate": "", "nn_score": "", "opp_score": ""}
            v3 = h2h = bl = m4 = m4b = mg = {"win_rate": ""}
            promoted = ""
            if (it + 1) % eval_every == 0:
                cs = _check_seed(it + 1)
                ev = evaluate(LATEST, "hard", games=EVAL_GAMES, workers=workers, seed=cs)
                bl = evaluate(LATEST, "blocker", games=EVAL_GAMES, workers=workers, seed=cs + 9000)
                v3 = evaluate(LATEST, YARDSTICK, games=EVAL_GAMES, workers=workers, seed=cs + 3000)
                h2h = evaluate(LATEST, BEST, games=EVAL_GAMES, workers=workers, seed=cs + 6000)
                h, b, x = ev["win_rate"], bl["win_rate"], h2h["win_rate"]
                ok4 = better4 = True
                if multi:
                    m4 = evaluate(LATEST, "hard", games=EVAL_GAMES, players=4, workers=workers, seed=cs + 12000)
                    m4b = evaluate(LATEST, BEST, games=EVAL_GAMES, players=4, workers=workers, seed=cs + 15000)
                    ok4 = m4["win_rate"] >= best_multi - SLACK
                    better4 = m4b["win_rate"] >= 0.25 + MULTI_GAIN and x >= 0.5
                ok_mega = True
                if mega:
                    mg = evaluate(LATEST, "hard", games=EVAL_GAMES, workers=workers, mode="mega", seed=cs + 18000)
                    ok_mega = mg["win_rate"] >= best_mega - SLACK
                promoted = (h >= best_hard - SLACK and b >= best_blocker - SLACK and ok4 and ok_mega
                            and (x >= PROMOTE or (x >= 0.5 and b >= best_blocker + BLOCKER_GAIN)
                                 or (multi and better4)))
                if promoted:
                    best_hard, best_blocker, best_iter = h, b, it + 1
                    if multi:
                        best_multi = m4["win_rate"]
                    if mega:
                        best_mega = mg["win_rate"]
                    shutil.copy(LATEST, publish)
                    shutil.copy(LATEST, BEST)

            secs = time.time() - t
            games = workers * games_per_task
            samples = int(data["pol"].sum())
            offers = max(1, tried["offers"])
            look_all, net_all = tried["look_all"] / offers, tried["net_all"] / offers
            log.writerow([it + 1, games, samples,
                          round(float(np.mean(vs_hard)), 3) if vs_hard else "",
                          round(float(np.mean(vs_blocker)), 3) if vs_blocker else "",
                          ev["win_rate"], ev["nn_score"], ev["opp_score"], bl["win_rate"],
                          v3["win_rate"], h2h["win_rate"], promoted,
                          *(round(info[k], 4) for k in ("pg", "v", "ent", "kl", "clipfrac", "kl_ref", "look")),
                          round(coef, 4), round(look_all, 3), round(net_all, 3),
                          tried["draw_checks"], tried["draw_up"],
                          tried["block"], tried["block_checks"], tried["block_up"],
                          *([round(float(np.mean(vs_multi)), 3) if vs_multi else "", m4["win_rate"],
                             m4b["win_rate"]] if multi else []),
                          *([mg["win_rate"]] if mega else []), round(secs, 1)])
            logf.flush()
            phase = " (value warm-up)" if it < warmup else ""
            msg = (f"iter {it + 1}{phase}: {samples} samples, rollout win vs hard "
                   f"{100 * np.mean(vs_hard) if vs_hard else 0:.0f}% / blocker "
                   f"{100 * np.mean(vs_blocker) if vs_blocker else 0:.0f}%, kl {info['kl']:.4f}, "
                   f"drift {info['kl_ref']:.3f}, entropy {info['ent']:.2f}, "
                   f"experiments {tried['keep']}k/{tried['move']}m/{tried['style']}s, "
                   f"keep-all best {100 * look_all:.0f}% / chosen {100 * net_all:.0f}%, "
                   f"draw tickets better {tried['draw_up']}/{tried['draw_checks']}, "
                   f"blocks tried {tried['block']}, block better {tried['block_up']}/{tried['block_checks']}, "
                   + (f"larger games {np.mean(vs_multi) if vs_multi else 0:.2f}x an equal share, " if multi else "")
                   + f"{secs:.0f}s")
            if ev["win_rate"] != "":
                msg += (f"  | CHECK: vs hard {100 * ev['win_rate']:.1f}% "
                        f"(score {ev['nn_score']:.0f} vs {ev['opp_score']:.0f}), "
                        f"vs blocker {100 * bl['win_rate']:.1f}%, "
                        f"vs v3 {100 * v3['win_rate']:.1f}%, vs best {100 * h2h['win_rate']:.1f}%"
                        + (f"; 4 players: vs 3 Hard {100 * m4['win_rate']:.1f}%, "
                           f"vs 3 best {100 * m4b['win_rate']:.1f}% (25% = even)" if multi else "")
                        + (f"; Mega Game vs Hard {100 * mg['win_rate']:.1f}%" if mega else "")
                        + (" -> NEW BEST" if promoted else f" (best: iteration {best_iter})"))
            print(msg, flush=True)
            torch.save({"opt": opt.state_dict(), "iter": it + 1, "best_hard": best_hard,
                        "best_blocker": best_blocker, "best_multi": best_multi, "best_mega": best_mega,
                        "best_iter": best_iter, "snapshots": snapshots, "init": init}, f"{STATE}.tmp")
            os.replace(f"{STATE}.tmp", STATE)  # swap in whole, so a crash mid-save can't corrupt it
            if patience and it + 1 - max(best_iter, warmup) >= patience:
                print(f"no new best for {patience} iterations; stopping. "
                      f"Best is from iteration {best_iter}.", flush=True)
                break
    logf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--init", default=str(MODELS_DIR / "expert_v4.pt"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--games-per-task", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--anchor", type=float, default=0.05, help="starting KL penalty toward --init")
    ap.add_argument("--anchor-iters", type=int, default=400, help="iterations for the anchor to fade to 0")
    ap.add_argument("--warmup", type=int, default=20, help="value-only iterations at the start")
    ap.add_argument("--patience", type=int, default=600, help="stop after this many without a new best")
    ap.add_argument("--explore-keep", type=float, default=EXPLORE["keep"])
    ap.add_argument("--explore-move", type=float, default=EXPLORE["move"])
    ap.add_argument("--explore-style", type=float, default=EXPLORE["style"])
    ap.add_argument("--explore-block", type=float, default=EXPLORE["block"])
    ap.add_argument("--players", default="2:1", help="player counts and their shares, e.g. 2:0.4,3:0.2,4:0.2,5:0.2")
    ap.add_argument("--publish", default=str(EXPERT_PATH), help="where a new best is copied")
    ap.add_argument("--modes", default="standard:1", help="game modes and their shares, e.g. standard:0.5,mega:0.5")
    args = ap.parse_args()
    players = {int(k): float(v) for k, v in (x.split(":") for x in args.players.split(","))}
    total = sum(players.values())
    players = {k: v / total for k, v in players.items()}
    modes = {k: float(v) for k, v in (x.split(":") for x in args.modes.split(","))}
    total = sum(modes.values())
    modes = {k: v / total for k, v in modes.items()}
    train(args.iters, args.init, args.resume, args.workers, args.games_per_task,
          args.eval_every, args.lr, args.anchor, args.anchor_iters, args.warmup, args.patience,
          {"keep": args.explore_keep, "move": args.explore_move, "style": args.explore_style,
           "block": args.explore_block}, players, args.publish, modes)


if __name__ == "__main__":
    main()

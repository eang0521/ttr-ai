"""Policy/value network and the bot that plays with it."""

import os
from pathlib import Path

import numpy as np

import torch
from torch import nn

from .encode import ACTION_SIZE, OBS_SIZE, decode, encode

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
EXPERT_PATH = MODELS_DIR / "expert.pt"


class PolicyNet(nn.Module):
    """One input slot per route and ticket of the USA board (see encode.py)."""

    encode = staticmethod(encode)

    def __init__(self, hidden=512, layers=3):
        super().__init__()
        self.inp = nn.Linear(OBS_SIZE, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.policy = nn.Linear(hidden, ACTION_SIZE)
        self.value = nn.Linear(hidden, 1)
        self.config = {"hidden": hidden, "layers": layers}

    def forward(self, obs, mask):
        h = torch.relu(self.inp(obs))
        for block in self.blocks:
            h = h + block(h)
        h = torch.relu(self.norm(h))
        logits = self.policy(h).masked_fill(~mask, -1e9)
        return logits, self.value(h).squeeze(-1)


def save(net, path, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file and swap it in, so a crash mid-save can't corrupt the model.
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"config": net.config, "state": net.state_dict(), **extra}, tmp)
    os.replace(tmp, path)


def load(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(ckpt["config"])
    if config.pop("kind", None) == "graph":
        from .graph import GraphNet, upgrade_state
        net = GraphNet(**config)
        net.load_state_dict(upgrade_state(net, ckpt["state"]))
        net.eval()
        return net
    net = PolicyNet(**config)
    state = ckpt["state"]
    w = state["inp.weight"]
    if w.shape[1] < OBS_SIZE:
        # Trained before newer inputs were added at the end: they start with zero weight,
        # so the network plays exactly as it did until training teaches it to use them.
        state = {**state, "inp.weight": torch.cat([w, w.new_zeros(w.shape[0], OBS_SIZE - w.shape[1])], 1)}
    w, b = state["policy.weight"], state["policy.bias"]
    if w.shape[0] < ACTION_SIZE:
        # Trained before the Mega Game's keep moves were added at the end of the menu.
        extra = ACTION_SIZE - w.shape[0]
        state = {**state, "policy.weight": torch.cat([w, w.new_zeros(extra, w.shape[1])]),
                 "policy.bias": torch.cat([b, b.new_zeros(extra)])}
    net.load_state_dict(state)
    net.eval()
    return net


class NNBot:
    """Plays the move the network rates best (or samples, if temperature > 0)."""

    def __init__(self, net=None, path=EXPERT_PATH, temperature=0.0, seed=None, name="expert"):
        self.net = net if net is not None else load(path)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name

    @torch.no_grad()
    def choose(self, game, pi):
        obs, mask = self.net.encode(game, pi)
        if mask.sum() == 1:
            return decode(game, pi, int(mask.argmax()))
        logits, _ = self.net(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
        logits = logits[0]
        if self.temperature > 0:
            p = torch.softmax(logits / self.temperature, -1).numpy().astype(np.float64)
            idx = int(self.rng.choice(len(p), p=p / p.sum()))
        else:
            idx = int(logits.argmax())
        return decode(game, pi, idx)

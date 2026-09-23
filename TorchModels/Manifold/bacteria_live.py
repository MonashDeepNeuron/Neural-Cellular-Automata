"""Live bacteria chemotaxis on the NCA manifold -- colour interferences, greedy teacher, camera.

Source of truth for `bacteria_chemotaxis.ipynb` (built by `build_bacteria_notebook.py`
from the `# %%` cells below). Also a CLI:

    python bacteria_live.py --train            ## train, save outputs/bacteria_live.pth
    python bacteria_live.py --verify           ## acceptance tests on held-out worlds
    python bacteria_live.py --figures          ## teacher-vs-NCA GIFs
    python bacteria_live.py --smoke            ## headless fake-camera live loop
    python bacteria_live.py --calibrate 1      ## corner clicks + HSV trackbars on camera 1
    python bacteria_live.py --live 1           ## live on camera index 1 (OBS Virtual Camera)

Design doc: docs/superpowers/specs/2026-09-24-bacteria-live-camera-design.md
"""

# %% [markdown]
# # Bacteria chemotaxis, live -- people in coloured clothing steer the colony
#
# A bird's-eye camera (Sony a6000 -> OBS -> **OBS Virtual Camera**) watches people walking
# over a floor. Each clothing colour is an interference in the bacteria's world:
#
# | Colour | Meaning | Mechanism |
# |---|---|---|
# | RED | food target | attractant source (1.0); colony cells on food bloom |
# | ORANGE | decoy | attractant source (0.5); pulls, but is not food |
# | BLACK | toxin | repellent source; colony cells under black die |
# | BLUE | wall | hard physics: cells inside are exactly 0 every step |
# | GREEN | booster | local speed-up (higher fire rate) |
#
# **No path planning.** The attractant and repellent *diffuse* from their sources (walls
# block diffusion) and the colony climbs the local gradient **greedily**. Like real
# chemotaxis it can be caught at a local maximum -- a U-shaped wall with food behind it
# traps it until somebody moves. The global nutrient `n` sets the colony's **biomass**:
# the manifold maps `n` to the update rule's weights, and `n` decides how long a cell
# lives, so a small `n` is a blob and a large `n` a crawling plume.
#
# Supervision comes from a hand-written, deterministic **teacher** CA (greedy head +
# aging body) run on the same moving worlds; the NCA learns to imitate it from its own
# drifted states, so it stays stable over an unbounded live run.

# %% [markdown]
# ## Setup

# %%
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLOURS = ("red", "orange", "black", "blue", "green")    ## mask channel order
RED, ORANGE, BLACK, BLUE, GREEN = range(5)
print(f"torch {torch.__version__} | device {DEVICE}")

# %% [markdown]
# ## Configuration

# %%
@dataclass
class Config:
    ## cell state
    channels: int = 16
    grid: int = 48
    alive_threshold: float = 0.1
    fire_rate: float = 0.5
    boost_fire: float = 0.4     ## fire rate on green = fire_rate + boost_fire

    ## update rule / manifold
    hidden: int = 128
    leak_init: float = 0.1
    latent_dim: int = 8
    dna_hidden: int = 64
    env_hidden: int = 64

    ## fields: explicit diffusion c <- c + alpha*lap/8 - k*c, k from the length scale
    diff_alpha: float = 0.9
    ell_attr: float = 10.0      ## attractant length scale (cells)
    ell_rep: float = 4.0        ## repellent length scale -- a local push, not a wall
    decoy_level: float = 0.5    ## orange pins the attractant at this level (red at 1)
    field_iters: int = 8        ## relaxation iterations per step (the field LAGS motion)
    init_iters: int = 1500      ## relaxation at episode start

    ## greedy teacher
    rep_weight: float = 1.5     ## utility U = A - rep_weight * R
    move_every: int = 2         ## head speed 0.5 cell/step (1 cell/step on green)
    life_max: int = 48          ## cell lifespan at n = 1 (steps); life(n) = n * life_max
    body_r: int = 2             ## head stamps a disc of this radius
    gain_eps: float = 1e-4      ## the head moves only for a strict gain above this

    ## worlds
    max_people: int = 6
    person_r: tuple = (2, 4)
    person_speed: float = 0.3
    toggle_prob: float = 0.004  ## per person per step: appear / vanish
    colour_probs: tuple = (0.30, 0.20, 0.20, 0.15, 0.15)
    margin: int = 3

    ## training
    epochs: int = 4000
    batch: int = 8
    lr: float = 2e-3
    scalar_lr: float = 1e-4
    n_lo: float = 0.1
    t0_max: int = 400           ## warm-up depth at the end of the curriculum
    window: tuple = (16, 32)    ## gradient window length
    loss_every: int = 8
    seed: int = 0
    log_every: int = 100
    eval_every: int = 50

    ## io
    out_dir: str = "outputs"
    ckpt_name: str = "bacteria_live.pth"

    @property
    def perception(self) -> int:
        return 3 * (self.channels + 3)      ## +3 sensed channels: attractant, repellent, food

    @property
    def k_attr(self) -> float:
        return (3 * self.diff_alpha / 8) / self.ell_attr ** 2

    @property
    def k_rep(self) -> float:
        return (3 * self.diff_alpha / 8) / self.ell_rep ** 2


CFG = Config()
os.makedirs(CFG.out_dir, exist_ok=True)

# %% [markdown]
# ## Fields -- diffusion instead of BFS
#
# Two concentrations, attractant `A` (red pinned at 1, orange at 0.5) and repellent `R`
# (black pinned at 1), relaxed by an explicit diffusion-with-decay scheme. Walls are
# no-flux (a cell only exchanges with free neighbours) and hold zero. The scheme is
# warm-started every step with only a few iterations, so the chemical **lags** behind
# moving people -- the far field of somebody who just walked in is still building up.
# Steady-state length scale `ell = sqrt(D/k)` with `D = 3*alpha/8` for the 8-neighbour
# Laplacian.

# %%
_NBR8 = torch.ones(1, 1, 3, 3)
_NBR8[0, 0, 1, 1] = 0.0


class Fields:
    def __init__(self, cfg, B, H, W, device=DEVICE):
        self.cfg = cfg
        self.c = torch.zeros(B, 2, H, W, device=device)
        self.k = torch.tensor([cfg.k_attr, cfg.k_rep], device=device).view(1, 2, 1, 1)
        self.nbr = _NBR8.to(device).repeat(2, 1, 1, 1)

    def _pins(self, masks):
        """(value, where) for the Dirichlet sources of both channels."""
        red, orange, black = masks[:, RED:RED + 1], masks[:, ORANGE:ORANGE + 1], masks[:, BLACK:BLACK + 1]
        val_a = torch.maximum(red, orange * self.cfg.decoy_level)
        val = torch.cat([val_a, black], 1)
        where = torch.cat([torch.maximum(red, orange), black], 1) > 0
        return val, where

    @torch.no_grad()
    def relax(self, masks, iters):
        ## c stays 0 in walls, so the no-flux Laplacian is conv8(c) - c * n_free. Folding
        ## the wall, decay and Dirichlet pins into per-cell coefficients leaves 3 kernels
        ## per iteration:  c <- pin + coef * c + g * conv8(c)
        free = (1.0 - masks[:, BLUE:BLUE + 1]).expand(-1, 2, -1, -1)
        n_free = F.conv2d(free, self.nbr, padding=1, groups=2)
        val, where = self._pins(masks)
        where &= free > 0                      ## a source under a wall is walled in
        a, keep = self.cfg.diff_alpha, (~where).float() * free
        coef = (1.0 - a * n_free / 8 - self.k) * keep
        g = (a / 8) * keep
        pin = val * where.float()
        c = torch.addcmul(pin, self.c, keep)
        for _ in range(iters):
            c = torch.addcmul(torch.addcmul(pin, coef, c), g,
                              F.conv2d(c, self.nbr, padding=1, groups=2))
        self.c = c
        return c

    def sense(self, masks):
        """(B,3,H,W): attractant, repellent, raw food mask -- what a cell perceives."""
        return torch.cat([self.c, masks[:, RED:RED + 1]], 1)

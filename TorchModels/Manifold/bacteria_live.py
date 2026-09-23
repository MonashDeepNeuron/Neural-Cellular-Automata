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
# only slow diffusion) and the colony climbs the local gradient **greedily**. Like real
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
    wall_perm: float = 0.25     ## walls stop cells, but the chemical seeps through them.
                                ## (fully no-flux walls leave NO local maxima off the
                                ## sources -- greedy ascent would never get trapped)
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
# (black pinned at 1), relaxed by an explicit diffusion-with-decay scheme. Walls stop
# the *bacteria* but only slow the *chemical*: it seeps through them at `wall_perm` of the
# free rate, like a chemical diffusing through the agar under a ridge. That matters --
# with fully no-flux walls the field has no local maxima away from its sources (maximum
# principle), so greedy ascent would always find the food and could never be trapped;
# with seepage, the far side of a U-wall facing the food becomes a real trap. The scheme is
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
    def relax(self, masks, iters, sel=None):
        """`iters` relaxation steps; with `sel` (B,1,1,1) bool only those slots commit."""
        ## Conductance between neighbours i,j: 1 if both free, p = wall_perm if either is a
        ## wall. Then  lap_i = p*conv8(c) + (1-p)*f_i*conv8(f*c) - c_i*G_i  with the total
        ## conductance G_i precomputed. Decay and Dirichlet pins fold into coefficients.
        B, _, H, W = self.c.shape
        p, a = self.cfg.wall_perm, self.cfg.diff_alpha
        f = (1.0 - masks[:, BLUE:BLUE + 1]).expand(-1, 2, -1, -1)
        ones = torch.ones_like(f)
        G = p * F.conv2d(ones, self.nbr, padding=1, groups=2) \
            + (1 - p) * f * F.conv2d(f, self.nbr, padding=1, groups=2)
        val, where = self._pins(masks)
        where &= f > 0                         ## a source standing in a wall is walled in
        keep = (~where).float()
        coef = (1.0 - a * G / 8 - self.k) * keep
        pin = val * where.float()
        g_all, g_free = (a / 8) * p * keep, (a / 8) * (1 - p) * keep * f
        nbr4 = self.nbr.repeat(2, 1, 1, 1)
        c = torch.addcmul(pin, self.c, keep)
        for _ in range(iters):
            s = F.conv2d(torch.cat([c, c * f], 1), nbr4, padding=1, groups=4)
            c = torch.addcmul(torch.addcmul(torch.addcmul(pin, coef, c), g_all, s[:, :2]),
                              g_free, s[:, 2:])
        self.c.copy_(c if sel is None else torch.where(sel, c, self.c))   ## in place: CUDA graphs
        return self.c

    def sense(self, masks):
        """(B,3,H,W): attractant, repellent, raw food mask -- what a cell perceives."""
        return torch.cat([self.c, masks[:, RED:RED + 1]], 1)

# %% [markdown]
# ## Worlds -- walls and walking people
#
# Training worlds are synthetic: 0-3 static wall shapes (blue) and up to six "people",
# discs of radius 2-4 cells with a clothing colour, walking with a slowly turning
# heading, bouncing off the borders, and occasionally stepping in or out of view. The
# camera produces exactly the same `(B,5,H,W)` masks, so the model cannot tell them apart.

# %%
def sample_walls(g, rng):
    """0-3 shapes from {wall with a gap, blob, bar}; the outer 2-cell ring stays free."""
    obs = np.zeros((g, g), bool)
    n_shapes = 0 if rng.random() < 0.25 else int(rng.integers(1, 4))
    for _ in range(n_shapes):
        kind = int(rng.integers(0, 3))
        if kind == 0:                                        ## wall with a gap
            pos, th = int(rng.integers(12, g - 12)), int(rng.integers(2, 4))
            gc, gw = int(rng.integers(8, g - 8)), int(rng.integers(3, 6))
            if rng.random() < 0.5:
                obs[pos:pos + th, :] = True
                obs[pos:pos + th, max(0, gc - gw):gc + gw] = False
            else:
                obs[:, pos:pos + th] = True
                obs[max(0, gc - gw):gc + gw, pos:pos + th] = False
        elif kind == 1:                                      ## blob
            cy, cx = rng.integers(8, g - 8, size=2)
            r = int(rng.integers(3, 7))
            yy, xx = np.mgrid[0:g, 0:g]
            obs |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        else:                                                ## bar
            cy, cx = rng.integers(8, g - 8, size=2)
            L, th = int(rng.integers(8, 16)), int(rng.integers(2, 4))
            if rng.random() < 0.5:
                obs[max(0, cy - L // 2):cy + L // 2, max(0, cx - th // 2):cx + th // 2 + 1] = True
            else:
                obs[max(0, cy - th // 2):cy + th // 2 + 1, max(0, cx - L // 2):cx + L // 2] = True
    obs[:2, :] = False; obs[-2:, :] = False; obs[:, :2] = False; obs[:, -2:] = False
    return obs


class World:
    """Batched walls + people. `pos`, `vel` (B,P,2) float; `r` (B,P); `col` (B,P) colour
    index; `active` (B,P) bool; `walls` (B,H,W) bool; `seed` (B,2) long."""

    def __init__(self, cfg, walls, pos, vel, r, col, active, seed, device,
                 wander=True, gen_seed=0):
        self.cfg, self.device, self.wander = cfg, device, wander
        self.walls = torch.as_tensor(walls, device=device).bool()
        self.pos = torch.as_tensor(pos, dtype=torch.float32, device=device).clone()
        self.vel = torch.as_tensor(vel, dtype=torch.float32, device=device).clone()
        self.r = torch.as_tensor(r, dtype=torch.float32, device=device)
        self.col = torch.as_tensor(col, dtype=torch.long, device=device)
        self.active = torch.as_tensor(active, device=device).bool().clone()
        self.seed = torch.as_tensor(seed, dtype=torch.long, device=device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(gen_seed))
        H, W = self.walls.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                                indexing="ij")
        self._yy, self._xx = yy.float(), xx.float()

    @property
    def B(self):
        return self.walls.shape[0]

    @classmethod
    def random(cls, cfg, B, rng, device=DEVICE):
        g, P = cfg.grid, cfg.max_people
        walls = np.stack([sample_walls(g, rng) for _ in range(B)])
        r = rng.integers(cfg.person_r[0], cfg.person_r[1] + 1, size=(B, P)).astype(np.float32)
        col = rng.choice(5, size=(B, P), p=np.array(cfg.colour_probs))
        col[:, 0] = np.where(rng.random(B) < 0.8, RED, col[:, 0])     ## food is usually there
        pos = rng.uniform(r[..., None], g - 1 - r[..., None], size=(B, P, 2)).astype(np.float32)
        speed = rng.uniform(0, cfg.person_speed, size=(B, P))
        head = rng.uniform(0, 2 * np.pi, size=(B, P))
        vel = np.stack([speed * np.sin(head), speed * np.cos(head)], -1).astype(np.float32)
        n_on = rng.integers(1, P + 1, size=B)
        active = np.arange(P)[None, :] < n_on[:, None]
        w = cls(cfg, walls, pos, vel, r, col, active, np.zeros((B, 2)), device,
                wander=True, gen_seed=int(rng.integers(1 << 31)))
        w.seed = w._pick_seeds(rng)
        return w

    @classmethod
    def scripted(cls, cfg, people, seeds, walls=None, device=DEVICE):
        """people: per batch element, a list of dicts {pos, vel, r, colour}. Straight-line
        motion (bouncing at the borders), nobody appears or vanishes."""
        g, B = cfg.grid, len(people)
        P = max(1, max(len(p) for p in people))
        pos, vel = np.zeros((B, P, 2), np.float32), np.zeros((B, P, 2), np.float32)
        r, col, active = np.full((B, P), 2.0, np.float32), np.zeros((B, P), int), np.zeros((B, P), bool)
        for b, plist in enumerate(people):
            for i, p in enumerate(plist):
                pos[b, i], vel[b, i], r[b, i] = p["pos"], p.get("vel", (0, 0)), p["r"]
                col[b, i], active[b, i] = COLOURS.index(p["colour"]), True
        if walls is None:
            walls = np.zeros((B, g, g), bool)
        return cls(cfg, walls, pos, vel, r, col, active, np.array(seeds), device, wander=False)

    def _pick_seeds(self, rng):
        """A free cell per world: not in a wall, clear of every person by 2 cells."""
        g, m = self.cfg.grid, self.cfg.margin
        d2 = (self._yy - self.pos[..., 0, None, None]) ** 2 + (self._xx - self.pos[..., 1, None, None]) ** 2
        near = ((d2 <= (self.r[..., None, None] + 2) ** 2) & self.active[..., None, None]).any(1)
        bad = (near | self.walls).cpu().numpy()
        bad[:, :m, :] = True; bad[:, -m:, :] = True; bad[:, :, :m] = True; bad[:, :, -m:] = True
        seeds = []
        for b in range(self.B):
            cells = np.argwhere(~bad[b])
            if len(cells) == 0:                     ## crowded: fall back to any free cell
                cells = np.argwhere(~self.walls[b].cpu().numpy())
            seeds.append(cells[rng.integers(len(cells))])
        return torch.as_tensor(np.array(seeds), dtype=torch.long, device=self.device)

    def splice(self, idx, other):
        """Replace slots `idx` with the (len(idx)-slot) world `other`."""
        for k in ("walls", "pos", "vel", "r", "col", "active", "seed"):
            getattr(self, k)[idx] = getattr(other, k)

    def masks(self):
        d2 = (self._yy - self.pos[..., 0, None, None]) ** 2 + (self._xx - self.pos[..., 1, None, None]) ** 2
        inside = ((d2 <= self.r[..., None, None] ** 2) & self.active[..., None, None]).float()
        onehot = (self.col[..., None] == torch.arange(5, device=self.device)).float()
        m = torch.einsum("bphw,bpc->bchw", inside, onehot).clamp_(max=1.0)
        m[:, BLUE] = torch.maximum(m[:, BLUE], self.walls.float())
        return m

    def step(self):
        if self.wander:                             ## slowly turning heading
            dth = torch.randn(self.vel.shape[:2], device=self.device, generator=self.gen) * 0.15
            c, s = torch.cos(dth), torch.sin(dth)
            vy, vx = self.vel[..., 0], self.vel[..., 1]
            self.vel.copy_(torch.stack([c * vy - s * vx, s * vy + c * vx], -1))
            flip = torch.rand(self.active.shape, device=self.device, generator=self.gen) < self.cfg.toggle_prob
            self.active ^= flip
        self.pos += self.vel
        lo = self.r[..., None].expand_as(self.pos)
        hi = self.cfg.grid - 1 - lo
        below, above = self.pos < lo, self.pos > hi
        self.pos.copy_(torch.where(below, 2 * lo - self.pos, torch.where(above, 2 * hi - self.pos, self.pos)))
        self.vel.copy_(torch.where(below | above, -self.vel, self.vel))

# %% [markdown]
# ## The teacher -- a greedy chemotaxis CA (the supervision)
#
# Deterministic and small enough to read in one go:
#
# - a **head** cell moves every `move_every` steps (every step on green) to the best of
#   its 8 neighbours by utility `U = A - rep_weight * R` -- but only for a strict gain.
#   No neighbour better -> it stays: **greedy ascent, trapped at local maxima**. Never
#   into a wall;
# - every step the head stamps a radius-2 disc of **age 0**; every living cell ages by
#   one; a cell dies when `age > life(n) = n * life_max` -- so `n` is the **biomass**, and
#   dropping `n` kills the tail at once;
# - cells under blue (crushed) or black (poisoned) die; if the whole body dies the colony
#   is **extinct** (the live loop reseeds);
# - colour encodes age (bright young front -> dark old tail), cells on red food **bloom**.
#
# Everything the teacher does depends only on local quantities (age, the fields, the
# masks, `n`), which is what makes it learnable by an NCA.

# %%
CORE = np.array([0.30, 0.42, 0.12])      ## old tail
FRONT = np.array([0.90, 0.95, 0.45])     ## young front
BLOOM = np.array([0.98, 0.45, 0.80])     ## feeding on red food

_OFFS = torch.tensor([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
                      [-1, -1], [-1, 1], [1, -1], [1, 1]])      ## self first: ties -> stay


def life(n, cfg):
    """Cell lifespan in steps at nutrient n (float or tensor)."""
    if torch.is_tensor(n):
        return torch.clamp(torch.round(n * cfg.life_max), min=1)
    return max(1, int(round(float(n) * cfg.life_max)))


class Teacher:
    def __init__(self, cfg, seed, masks):
        B, _, H, W = masks.shape
        dev = masks.device
        self.cfg = cfg
        self.t = torch.zeros(B, dtype=torch.long, device=dev)      ## per-slot clock
        self.head = torch.as_tensor(seed, device=dev).long().clone()
        self.age = torch.full((B, H, W), float("inf"), device=dev)
        self.alive = torch.ones(B, dtype=torch.bool, device=dev)
        yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        self._yy, self._xx = yy, xx
        self._offs = _OFFS.to(dev)
        self._bidx = torch.arange(B, device=dev)
        self._stamp()
        self._kill_env(masks)

    def _stamp(self):
        d2 = (self._yy - self.head[:, 0, None, None]) ** 2 + (self._xx - self.head[:, 1, None, None]) ** 2
        disc = (d2 <= self.cfg.body_r ** 2) & self.alive[:, None, None]
        self.age.copy_(torch.where(disc, torch.zeros_like(self.age), self.age))

    def _kill_env(self, masks):
        dead = (masks[:, BLUE] > 0) | (masks[:, BLACK] > 0)
        self.age.copy_(torch.where(dead, torch.full_like(self.age, float("inf")), self.age))
        self.alive &= torch.isfinite(self.age).flatten(1).any(1)

    def _move(self, fields, masks):
        cfg = self.cfg
        H, W = self.age.shape[-2:]
        U = fields.c[:, 0] - cfg.rep_weight * fields.c[:, 1]
        wall = masks[:, BLUE] > 0
        b = self._bidx
        on_green = masks[b, GREEN, self.head[:, 0], self.head[:, 1]] > 0
        due = on_green | (self.t % cfg.move_every == 0)

        cand = self.head[:, None, :] + self._offs[None]                 ## (B,9,2)
        inb = (cand[..., 0] >= 0) & (cand[..., 0] < H) & (cand[..., 1] >= 0) & (cand[..., 1] < W)
        cy, cx = cand[..., 0].clamp(0, H - 1), cand[..., 1].clamp(0, W - 1)
        Uc = U[b[:, None], cy, cx]
        free = inb & ~wall[b[:, None], cy, cx]
        self_ok = free[:, 0]
        ## a move needs a strict gain -- unless the head is standing in a (moving) wall
        better = (Uc > Uc[:, :1] + cfg.gain_eps) | ~self_ok[:, None]
        ok = free & better
        ok[:, 0] = self_ok
        k = torch.where(ok, Uc, torch.full_like(Uc, -float("inf"))).argmax(1)
        go = due & self.alive & ok.any(1)
        self.head.copy_(torch.where(go[:, None], cand[b, k], self.head))

    @torch.no_grad()
    def step(self, fields, masks, n):
        self._move(fields, masks)
        self.age.add_(1)
        self._stamp()
        lf = life(torch.as_tensor(n, dtype=torch.float32, device=self.age.device), self.cfg)
        self.age.copy_(torch.where(self.age > lf.view(-1, 1, 1), torch.full_like(self.age, float("inf")), self.age))
        self._kill_env(masks)
        self.t += 1

    def splice(self, idx, other):
        """Replace slots `idx` with the (len(idx)-slot) teacher `other`."""
        for k in ("t", "head", "age", "alive"):
            getattr(self, k)[idx] = getattr(other, k)

    def rgba(self, masks):
        """(B,4,H,W) target image: colour by age, bloom on food, alpha = body."""
        alive = torch.isfinite(self.age)
        shade = (self.age / self.cfg.life_max).clamp(0, 1).nan_to_num(0.0)
        front = torch.tensor(FRONT, dtype=torch.float32, device=self.age.device).view(1, 3, 1, 1)
        core = torch.tensor(CORE, dtype=torch.float32, device=self.age.device).view(1, 3, 1, 1)
        bloom = torch.tensor(BLOOM, dtype=torch.float32, device=self.age.device).view(1, 3, 1, 1)
        rgb = front + (core - front) * shade[:, None]
        rgb = torch.where((masks[:, RED:RED + 1] > 0), bloom, rgb)
        a = alive[:, None].float()
        return torch.cat([rgb * a, a], 1)

# %% [markdown]
# ## The model
#
# `BacteriaNCA` keeps the manifold of `env_growth.py`: nutrient -> env encoder -> latent
# -> DNA decoder -> the update net's weights, as a residual on a learned base rule
# (predictor zero-init). Three inputs are new:
#
# - **sense** `(A, R, food)` is concatenated to the state before perception, so every cell
#   sees the fields and their Sobel gradients -- it *learns* greedy chemotaxis;
# - **kill** = blue or black: `state *= 1 - kill` after every step, so walls and toxins are
#   exactly empty by construction;
# - **boost** = green: the stochastic fire rate rises from 0.5 to 0.9 -- faster growth.

# %%
class BacteriaNCA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.env_encoder = nn.Sequential(
            nn.Linear(1, cfg.env_hidden), nn.ReLU(),
            nn.Linear(cfg.env_hidden, cfg.latent_dim))

        self.n_generated = cfg.hidden * cfg.perception + cfg.hidden + cfg.channels * cfg.hidden
        self.dna_decoder = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.dna_hidden), nn.ReLU())
        self.predictor = nn.Linear(cfg.dna_hidden, self.n_generated)
        nn.init.zeros_(self.predictor.weight)          ## start as a zero residual: every
        nn.init.zeros_(self.predictor.bias)            ## nutrient decodes to the base rule

        self.base_w1 = nn.Parameter(torch.randn(cfg.hidden, cfg.perception) * 0.001)
        self.base_b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.base_w2 = nn.Parameter(torch.zeros(cfg.channels, cfg.hidden))
        self.log_leak = nn.Parameter(torch.tensor(math.log(cfg.leak_init)))

        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        c3 = cfg.channels + 3
        self.register_buffer("kx", sx.view(1, 1, 3, 3).repeat(c3, 1, 1, 1))
        self.register_buffer("ky", sy.view(1, 1, 3, 3).repeat(c3, 1, 1, 1))

    def weights_for(self, n):
        """(B,) nutrient levels -> the update net's (w1, b1, w2), batched over B."""
        cfg = self.cfg
        z = self.env_encoder(n.view(-1, 1))
        theta = self.predictor(self.dna_decoder(z))
        b, i, k = z.shape[0], 0, cfg.hidden * cfg.perception
        w1 = self.base_w1 + theta[:, i:i + k].view(b, cfg.hidden, cfg.perception)
        i += k
        b1 = self.base_b1 + theta[:, i:i + cfg.hidden]
        i += cfg.hidden
        w2 = self.base_w2 + theta[:, i:].view(b, cfg.channels, cfg.hidden)
        return w1, b1, w2

    def perceive(self, state, sense):
        x = torch.cat([state, sense], dim=1)               ## the fields ARE sensed
        p = F.pad(x, (1, 1, 1, 1), mode="replicate")       ## no-flux, not a torus
        gx = F.conv2d(p, self.kx, groups=self.cfg.channels + 3)
        gy = F.conv2d(p, self.ky, groups=self.cfg.channels + 3)
        return torch.cat([x, gx, gy], dim=1)

    def alive_mask(self, state):
        return F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > self.cfg.alive_threshold

    def forward(self, state, weights, sense, kill, boost):
        """One step. sense (B,3,H,W); kill, boost (B,1,H,W) in {0,1}."""
        w1, b1, w2 = weights
        pre = self.alive_mask(state)
        p = self.perceive(state, sense)
        a = torch.einsum("bop,bpij->boij", w1, p) + b1[:, :, None, None]
        ds = torch.einsum("bcv,bvij->bcij", w2, F.relu(a))
        fire = self.cfg.fire_rate + self.cfg.boost_fire * boost
        m = (torch.rand_like(ds[:, :1]) < fire).to(ds.dtype)
        out = state + self.log_leak.exp() * ds * m
        out = out * (pre & self.alive_mask(out)).to(out.dtype)
        return out * (1.0 - kill)


def seed_state(rgba, cfg):
    """NCA start state: the teacher's first body in RGBA, hidden channels zero."""
    s = torch.zeros(rgba.shape[0], cfg.channels, *rgba.shape[-2:], device=rgba.device)
    s[:, 0:4] = rgba
    return s


class Env:
    """World + fields + teacher, advanced together. The NCA reads `inputs()`.

    On CUDA, `step()` is captured once as a CUDA graph and replayed: the env is ~250 tiny
    kernels per step on a 48x48 grid, so without the graph it is pure launch overhead
    (~12 ms/step on a Windows laptop vs ~0.3 ms of actual GPU work). Every state update
    in World/Fields/Teacher is therefore in place (static addresses)."""

    def __init__(self, cfg, world, use_graph=True):
        self.cfg, self.world = cfg, world
        self.masks = world.masks()
        B, _, H, W = self.masks.shape
        self.fields = Fields(cfg, B, H, W, self.masks.device)
        self.fields.relax(self.masks, cfg.init_iters)
        self.teacher = Teacher(cfg, world.seed, self.masks)
        self.use_graph = use_graph and self.masks.is_cuda
        self._graph, self._init_graph = None, None
        self._n = torch.zeros(B, device=self.masks.device)
        self._sel = torch.zeros(B, 1, 1, 1, dtype=torch.bool, device=self.masks.device)

    def _state_tensors(self):
        w, t = self.world, self.teacher
        return [w.pos, w.vel, w.active, self.masks, self.fields.c, t.age, t.head, t.alive, t.t]

    def _step_impl(self):
        self.world.step()
        self.masks.copy_(self.world.masks())
        self.fields.relax(self.masks, self.cfg.field_iters)
        self.teacher.step(self.fields, self.masks, self._n)

    _INIT_CHUNK = 100

    def _init_impl(self):
        self.fields.relax(self.masks, self._INIT_CHUNK, sel=self._sel)

    def _capture(self, fn):
        saved = [x.clone() for x in self._state_tensors()]
        gen_state = self.world.gen.get_state()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                    ## warm-up, as CUDA graphs require
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        for x, v in zip(self._state_tensors(), saved):   ## undo the warm-up steps
            x.copy_(v)
        self.world.gen.set_state(gen_state)
        g = torch.cuda.CUDAGraph()
        g.register_generator_state(self.world.gen)
        with torch.cuda.graph(g):
            fn()
        return g

    def reset_slots(self, idx, rng):
        """Fresh random worlds (and teachers, fields) in slots idx. The fields of just those
        slots are relaxed from zero, `init_iters` steps, by a replayed CUDA graph."""
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.masks.device)
        self.world.splice(idx, World.random(self.cfg, len(idx), rng, self.masks.device))
        self.masks.copy_(self.world.masks())
        self.fields.c[idx] = 0.0
        self._sel.zero_()
        self._sel[idx] = True
        if self.use_graph:
            if self._init_graph is None:
                self._init_graph = self._capture(self._init_impl)
            for _ in range(max(1, self.cfg.init_iters // self._INIT_CHUNK)):
                self._init_graph.replay()
        else:
            self.fields.relax(self.masks, self.cfg.init_iters, sel=self._sel)
        self.teacher.splice(idx, Teacher(self.cfg, self.world.seed[idx], self.masks[idx]))

    def step(self, n):
        self._n.copy_(torch.as_tensor(n, dtype=torch.float32, device=self._n.device).expand_as(self._n))
        if not self.use_graph:
            return self._step_impl()
        if self._graph is None:
            self._graph = self._capture(self._step_impl)
        self._graph.replay()

    def inputs(self):
        m = self.masks
        kill = torch.maximum(m[:, BLUE:BLUE + 1], m[:, BLACK:BLACK + 1])
        return self.fields.sense(m), kill, m[:, GREEN:GREEN + 1]


def _as_n(v, B, device):
    return torch.as_tensor(v, dtype=torch.float32, device=device).expand(B).contiguous()


@torch.no_grad()
def run_world(model, env, n_fn, steps, state=None, record_every=1, seed=0):
    """Advance env + NCA together under nutrient n_fn(t) (float or (B,) tensor).
    Returns the final NCA state and frames [(teacher_rgba, nca_rgba, masks)] on the CPU,
    starting with the initial frame."""
    torch.manual_seed(seed)
    dev = env.masks.device
    B = env.masks.shape[0]
    if state is None:
        state = seed_state(env.teacher.rgba(env.masks), model.cfg)
    snap = lambda: (env.teacher.rgba(env.masks).cpu(), state[:, :4].clamp(0, 1).cpu(), env.masks.cpu())
    frames, cur, w = [snap()], None, None
    for t in range(steps):
        n = _as_n(n_fn(t), B, dev)
        if cur is None or not torch.equal(n, cur):
            w, cur = model.weights_for(n), n
        env.step(n)
        state = model(state, w, *env.inputs())
        if (t + 1) % record_every == 0:
            frames.append(snap())
    return state, frames

# %% [markdown]
# ## Checkpoint save / load

# %%
def save_checkpoint(model, cfg, path=None):
    path = path or os.path.join(cfg.out_dir, cfg.ckpt_name)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, path)
    print(f"saved {path}")


def load_checkpoint(path=None, device=DEVICE):
    path = path or os.path.join(CFG.out_dir, CFG.ckpt_name)
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["config"])
    m = BacteriaNCA(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, cfg

# %% [markdown]
# ## Training -- imitate the teacher from the NCA's own states
#
# A **persistent pool** of `batch` running worlds. The teacher and the NCA start from the
# same body and run in lockstep on identical fields; each epoch continues every world by
# a short **gradient window** (MSE on RGBA vs the teacher every `loss_every` steps), then
# detaches and carries the state on. Slots are reset to fresh worlds at random -- often
# early in training (short episodes), rarely later (episodes hundreds of steps deep) --
# so the rule is trained to *correct back* toward the teacher from wherever it has
# drifted, which is what keeps an unbounded live run stable. Each slot has its own
# nutrient schedule: constant, a ramp, or a sudden jump, up or down.

# %%
class Schedule:
    """Per-slot nutrient schedules on per-slot clocks: constant / ramp / jump."""

    def __init__(self, B, rng, cfg, device=DEVICE):
        self.cfg, self.device = cfg, device
        self.a, self.b = np.zeros(B), np.zeros(B)
        self.kind, self.t_s, self.dur, self.t = (np.zeros(B, int) for _ in range(4))
        self.reset(np.arange(B), rng)

    def reset(self, idx, rng):
        k = len(idx)
        self.a[idx] = rng.uniform(self.cfg.n_lo, 1.0, k)
        self.b[idx] = rng.uniform(self.cfg.n_lo, 1.0, k)
        self.kind[idx] = rng.integers(0, 3, k)
        self.t_s[idx] = rng.integers(0, 400, k)
        self.dur[idx] = rng.integers(40, 200, k)
        self.t[idx] = 0

    def n(self):
        f = np.clip((self.t - self.t_s) / self.dur, 0.0, 1.0)
        v = np.where(self.kind == 0, self.a,
                     np.where(self.kind == 1, self.a + (self.b - self.a) * f,
                              np.where(self.t >= self.t_s, self.b, self.a)))
        return torch.as_tensor(v, dtype=torch.float32, device=self.device)

    def tick(self):
        self.t += 1


def _fixed_eval_worlds(cfg, device=DEVICE):
    return World.random(cfg, 4, np.random.default_rng(1234), device)


@torch.no_grad()
def _fixed_eval(model, cfg):
    """Deterministic held-out eval for best-checkpoint tracking: MSE vs the teacher at
    t = 100, 200, 300 on four fixed worlds. Saves and restores the global RNG."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        dev = next(model.parameters()).device
        env = Env(cfg, _fixed_eval_worlds(cfg, dev))
        levels = torch.tensor([0.4, 0.7, 1.0, 0.6], device=dev)
        _, frames = run_world(model, env, lambda t: levels, 300, record_every=100, seed=11)
        return float(np.mean([F.mse_loss(n, t).item() for t, n, _ in frames[1:]]))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def reset_prob(epoch, epochs):
    """Per-slot reset probability per epoch: 0.3 early (episodes ~80 steps) decaying to
    0.05 by mid-training (episodes ~500 steps)."""
    f = min(1.0, epoch / (0.5 * epochs))
    return 0.3 + (0.05 - 0.3) * f


def train(model, cfg, rng, epochs=None):
    dev = next(model.parameters()).device
    epochs = epochs or cfg.epochs
    vec = [p for p in model.parameters() if p.numel() > 1]
    sca = [p for p in model.parameters() if p.numel() == 1]
    opt = torch.optim.Adam([{"params": vec, "lr": cfg.lr},
                            {"params": sca, "lr": cfg.scalar_lr}])
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    history, t0 = [], time.time()

    B = cfg.batch
    env = Env(cfg, World.random(cfg, B, rng, dev))
    sched = Schedule(B, rng, cfg, dev)
    state = seed_state(env.teacher.rgba(env.masks), cfg)

    for epoch in range(epochs):
        frac = epoch / max(epochs - 1, 1)
        opt.param_groups[0]["lr"] = cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))

        ## reset some slots: at random, or because the teacher colony went extinct
        idx = np.nonzero((rng.random(B) < reset_prob(epoch, epochs))
                         | ~env.teacher.alive.cpu().numpy())[0]
        if len(idx):
            env.reset_slots(idx, rng)
            sched.reset(idx, rng)
            state[idx] = seed_state(env.teacher.rgba(env.masks)[idx], cfg)

        opt.zero_grad()
        W = int(rng.integers(cfg.window[0], cfg.window[1] + 1))
        loss, k = 0.0, 0
        for i in range(W):
            n = sched.n()
            env.step(n)
            sched.tick()
            state = model(state, model.weights_for(n), *env.inputs())
            if (i + 1) % cfg.loss_every == 0 or i == W - 1:
                loss = loss + F.mse_loss(state[:, 0:4], env.teacher.rgba(env.masks))
                k += 1
        loss = loss / k
        loss.backward()
        for p in vec:                                   ## per-parameter gradient normalisation
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        state = state.detach()
        history.append((epoch, loss.item(), float(sched.t.mean())))

        if epoch % cfg.eval_every == 0 or epoch == epochs - 1:
            ev = _fixed_eval(model, cfg)
            if ev < best:
                best, best_state = ev, copy.deepcopy(model.state_dict())

        if epoch % cfg.log_every == 0 or epoch == epochs - 1:
            recent = np.mean([h[1] for h in history[-cfg.log_every:]])
            print(f"epoch {epoch:5d} | loss {recent:.5f} | mean episode depth {sched.t.mean():5.0f} "
                  f"| best-eval {best:.5f} | {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    print(f"done in {time.time() - t0:.1f}s | best eval {best:.6f} | weights restored to best")
    return history

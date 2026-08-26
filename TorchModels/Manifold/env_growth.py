"""Environment-driven growth on the NCA manifold.

A single seed cell grows into a plant whose developmental stage is decided by two
EXTERNAL signals, not by a latent index:

  1. nutrient n(t)   -- global scalar in [0,1]. An environment encoder Z(n) maps it onto
                        the manifold latent, and kappa = P(D(Z(n))) generates the per-cell
                        update rule's weights. The nutrient therefore selects WHAT grows:
                        seed blob -> sprout -> leafy plant -> flowering plant.
  2. chemical chem(x,y) -- spatial field in [0,1]. Hard multiplicative gate on the state
                        update: state' = state + exp(rho) * ds * fire * chem. No chemical
                        means a cell is EXACTLY frozen, by construction. The chemical is
                        deliberately not in the perception vector: it is physics the rule
                        cannot learn to ignore.

Design doc: docs/superpowers/specs/2026-08-26-env-driven-growth-design.md

Usage:
    python env_growth.py                    # train (~20-30 min on an RTX 4060), verify, render
    python env_growth.py --epochs 200       # quick smoke run
    python env_growth.py --render-only      # skip training, load outputs/env_growth.pth
"""

import argparse
import copy
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## ---------------------------------------------------------------------------
## config
## ---------------------------------------------------------------------------

@dataclass
class Config:
    ## cell state
    channels: int = 16
    grid: int = 40
    alive_threshold: float = 0.1
    fire_rate: float = 0.5

    ## update rule
    hidden: int = 128
    leak_init: float = 0.1

    ## the manifold
    latent_dim: int = 8
    dna_hidden: int = 64
    env_hidden: int = 64

    ## training
    epochs: int = 3000
    lr: float = 2e-3
    scalar_lr: float = 1e-4
    seg1_min: int = 48
    seg1_max: int = 64
    seg2_min: int = 32
    seg2_max: int = 48
    env_jitter: float = 0.15        ## nutrient noise around each stage level. 0.15 makes the
                                    ## four basins [0,.25][.25,.55][.55,.85][.85,1] partition
                                    ## [0,1] EXACTLY -- no untrained gap a slow ramp can visit
    state_noise: float = 0.05       ## perturbation before segment 2 -> rules learn to clean up
    slow_ramp_prob: float = 0.33    ## fraction of epochs trained on true slow-ramp states
    seed: int = 0
    log_every: int = 100
    eval_every: int = 50            ## fixed eval for best-checkpoint tracking

    ## io
    out_dir: str = "outputs"
    ckpt_name: str = "env_growth.pth"

    @property
    def perception(self) -> int:
        return 3 * self.channels


## ---------------------------------------------------------------------------
## targets: four nested developmental stages of a plant, one nutrient level each
## ---------------------------------------------------------------------------

GREEN = (0.20, 0.65, 0.25)
BROWN = (0.50, 0.33, 0.16)
PINK = (0.92, 0.32, 0.45)
YELLOW = (0.96, 0.82, 0.20)


def plant_targets(cfg):
    """(4, 4, grid, grid) RGBA targets + names + nutrient levels.

    Stages are nested (each contains the previous), all anchored at the same ground
    point where the single seed cell is planted, so development is monotone growth.
    """
    g = cfg.grid
    yy, xx = torch.meshgrid(torch.arange(g, dtype=torch.float32),
                            torch.arange(g, dtype=torch.float32), indexing="ij")
    ground, cx = 31.0, g / 2.0

    def ellipse(cy_, cx_, ry, rx):
        return ((yy - cy_) / ry) ** 2 + ((xx - cx_) / rx) ** 2 <= 1.0

    def stem(top):
        return (yy >= top) & (yy <= ground) & ((xx - cx).abs() <= 1.0)

    seed_blob = [(ellipse(ground - 1, cx, 2.2, 2.6), BROWN)]
    mini_leaves = [(ellipse(24.0, cx - 3.2, 1.5, 2.6), GREEN),
                   (ellipse(24.0, cx + 3.2, 1.5, 2.6), GREEN)]
    big_leaves = [(ellipse(20.0, cx - 5.5, 2.0, 4.2), GREEN),
                  (ellipse(20.0, cx + 5.5, 2.0, 4.2), GREEN)]
    flower = [(ellipse(8.0, cx, 4.6, 4.6), PINK),
              (ellipse(8.0, cx, 2.0, 2.0), YELLOW)]

    stages = [
        ("seed", 0.10, seed_blob),
        ("sprout", 0.40, seed_blob + [(stem(24.0), GREEN)] + mini_leaves),
        ("leafy", 0.70, seed_blob + [(stem(16.0), GREEN)] + mini_leaves + big_leaves),
        ("flower", 1.00, seed_blob + [(stem(12.0), GREEN)] + mini_leaves + big_leaves + flower),
    ]

    out = torch.zeros(len(stages), 4, g, g)
    names, envs = [], []
    for i, (name, n, layers) in enumerate(stages):
        for mask, rgb in layers:                        ## later layers paint over earlier
            for c in range(3):
                out[i, c] = torch.where(mask, torch.tensor(rgb[c]), out[i, c])
            out[i, 3] = torch.maximum(out[i, 3], mask.float())
        names.append(name)
        envs.append(n)
    return out, names, torch.tensor(envs)


## ---------------------------------------------------------------------------
## model
## ---------------------------------------------------------------------------

class EnvNCA(nn.Module):
    """A manifold NCA whose latent is COMPUTED FROM THE ENVIRONMENT.

    kappa(n) = P(D(Z(n))): the nutrient level n is encoded to a latent, the latent is
    decoded to the update net's weights (residual on a learned base rule, predictor
    zero-init). forward() additionally takes a chemical field that hard-gates ds.
    """

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
        nn.init.zeros_(self.predictor.bias)            ## environment decodes to the base rule

        self.base_w1 = nn.Parameter(torch.randn(cfg.hidden, cfg.perception) * 0.001)
        self.base_b1 = nn.Parameter(torch.zeros(cfg.hidden))
        self.base_w2 = nn.Parameter(torch.zeros(cfg.channels, cfg.hidden))

        self.log_leak = nn.Parameter(torch.tensor(math.log(cfg.leak_init)))

        sx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        self.register_buffer("kx", sx.view(1, 1, 3, 3).repeat(cfg.channels, 1, 1, 1))
        self.register_buffer("ky", sy.view(1, 1, 3, 3).repeat(cfg.channels, 1, 1, 1))

    def weights_for(self, n):
        """(B,) nutrient levels -> the update net's (w1, b1, w2), batched over B."""
        cfg = self.cfg
        z = self.env_encoder(n.view(-1, 1))
        theta = self.predictor(self.dna_decoder(z))
        b, i = z.shape[0], 0
        k = cfg.hidden * cfg.perception
        w1 = self.base_w1 + theta[:, i:i + k].view(b, cfg.hidden, cfg.perception)
        i += k
        b1 = self.base_b1 + theta[:, i:i + cfg.hidden]
        i += cfg.hidden
        w2 = self.base_w2 + theta[:, i:].view(b, cfg.channels, cfg.hidden)
        return w1, b1, w2

    def perceive(self, state):
        p = F.pad(state, (1, 1, 1, 1), mode="replicate")   ## no-flux, not a torus
        gx = F.conv2d(p, self.kx, groups=self.cfg.channels)
        gy = F.conv2d(p, self.ky, groups=self.cfg.channels)
        return torch.cat([state, gx, gy], dim=1)

    def alive_mask(self, state):
        return F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > self.cfg.alive_threshold

    def forward(self, state, weights, chem=None):
        """One step. chem is (B,1,H,W) in [0,1] or None (== 1 everywhere).

        chem gates the ENTIRE per-cell state transition -- growth AND death -- because
        both are state changes the chemical fuels. The alive-mask kill is not idempotent
        (masking lowers neighbourhood alpha, which can kill further cells), so gating
        only ds would let a chem=0 cell still change on the next step. chem == 0
        therefore freezes a cell exactly, by construction, and the rule cannot ignore it.
        """
        w1, b1, w2 = weights
        pre = self.alive_mask(state)

        p = self.perceive(state)
        a = torch.einsum("bop,bpij->boij", w1, p) + b1[:, :, None, None]
        ds = torch.einsum("bcv,bvij->bcij", w2, F.relu(a))

        m = (torch.rand_like(ds[:, :1]) <= self.cfg.fire_rate).to(ds.dtype)
        out = state + self.log_leak.exp() * ds * m
        new = out * (pre & self.alive_mask(out)).to(out.dtype)

        if chem is None:
            return new
        return state + (new - state) * chem


def new_seed(n, cfg, device=DEVICE):
    """A single living cell where the plant is anchored -- identical for every sample."""
    s = torch.zeros(n, cfg.channels, cfg.grid, cfg.grid, device=device)
    s[:, 3, 30, cfg.grid // 2] = 1.0
    return s


def rollout(model, state, weights, steps, chem=None):
    for _ in range(steps):
        state = model(state, weights, chem)
    return state


def rollout_moving(model, state, n_from, n_to, steps, frac=0.6, chem=None):
    """Rollout under an environment IN MOTION: n glides linearly from n_from to n_to
    over the first `frac` of the segment, then holds at n_to. Weights are regenerated
    every step, exactly like a nutrient ramp at eval -- this is what makes slow ramps
    in-distribution instead of a hope."""
    ramp_steps = max(1, int(steps * frac))
    for i in range(steps):
        t = min(1.0, (i + 1) / ramp_steps)
        state = model(state, model.weights_for(n_from + (n_to - n_from) * t), chem)
    return state


@torch.no_grad()
def simulate(model, env_fn, steps, chem_fn=None, state=None, record_every=1, seed=0):
    """Eval rollout under a TIME-VARYING environment.

    env_fn(t) -> nutrient level; weights are regenerated whenever it changes.
    chem_fn(t) -> chemical field (1,1,H,W) or None. Reseeds the global RNG.
    """
    cfg = model.cfg
    torch.manual_seed(seed)
    if state is None:
        state = new_seed(1, cfg, next(model.parameters()).device)
    frames, env_trace = [state.clone()], [float(env_fn(0))]
    cur_n, weights = None, None
    for t in range(steps):
        n = float(env_fn(t))
        if n != cur_n:
            weights = model.weights_for(torch.tensor([n], device=state.device))
            cur_n = n
        chem = chem_fn(t) if chem_fn is not None else None
        state = model(state, weights, chem)
        if (t + 1) % record_every == 0:
            frames.append(state.clone())
            env_trace.append(n)
    return state, frames, env_trace


@torch.no_grad()
def grow(model, n_values, steps, chem=None, state=None, seed=0):
    """Fixed-environment eval rollout, one row per nutrient level."""
    torch.manual_seed(seed)
    if state is None:
        state = new_seed(n_values.shape[0], model.cfg, n_values.device)
    return rollout(model, state, model.weights_for(n_values), steps, chem)


## ---------------------------------------------------------------------------
## sanity checks (before any training time is spent)
## ---------------------------------------------------------------------------

def sanity_check(model, targets, envs):
    cfg = model.cfg
    dev = next(model.parameters()).device
    k = envs.shape[0]

    w1, b1, w2 = model.weights_for(envs)
    assert w1.shape == (k, cfg.hidden, cfg.perception), w1.shape
    assert b1.shape == (k, cfg.hidden), b1.shape
    assert w2.shape == (k, cfg.channels, cfg.hidden), w2.shape

    state = new_seed(k, cfg, dev)
    out = model(state, (w1, b1, w2))
    assert out.shape == state.shape
    assert torch.isfinite(out).all(), "non-finite values after a single step"

    ## chem gate: zero chemical must freeze the state EXACTLY
    chem0 = torch.zeros(k, 1, cfg.grid, cfg.grid, device=dev)
    frozen = model(state, (w1, b1, w2), chem0)
    assert (frozen - state).abs().max().item() == 0.0, "chem=0 did not freeze the state"

    ## gradient must reach the predictor at init (the thing that moves it off zero)
    final = rollout(model, new_seed(k, cfg, dev), (w1, b1, w2), 8)
    loss = F.mse_loss(final[:, 0:4], targets)
    g_pred = torch.autograd.grad(loss, model.predictor.weight)[0].norm().item()
    assert math.isfinite(g_pred) and g_pred > 0, f"grad->predictor {g_pred}"

    ## the env-encoder path is probed on a copy with a small non-zero predictor:
    ## at init theta = P(D(Z(n))) = 0 for every n, so its gradient is legitimately zero.
    probe = copy.deepcopy(model)
    nn.init.normal_(probe.predictor.weight, std=1e-3)
    p_final = rollout(probe, new_seed(k, cfg, dev), probe.weights_for(envs), 8)
    g_env = torch.autograd.grad(F.mse_loss(p_final[:, 0:4], targets),
                                probe.env_encoder[0].weight)[0].norm().item()
    assert math.isfinite(g_env) and g_env > 0, f"grad->env_encoder {g_env}"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"sanity: {n_params:,} params | grad->predictor {g_pred:.2e} | "
          f"grad->env_encoder (probe) {g_env:.2e} | chem=0 freeze exact | all checks passed")


## ---------------------------------------------------------------------------
## training: two-segment rollouts so environment CHANGE is explicitly trained
## ---------------------------------------------------------------------------

def _slow_ramp_episode(model, targets, envs, cfg):
    """Truncated BPTT on a slow nutrient ramp -- the eval distribution itself.

    Error on slow ramps accrues PER STEP spent under in-motion rules, so short training
    glides never expose the states a 480-step ramp reaches. Here the actual slow ramp is
    rolled from seed under no_grad to a random depth (zero memory cost), and only the
    continuation is trained: a few more ramp steps, then a hold at the nearest stage
    level, with loss against that stage's target. This teaches every rule to finish the
    transition and clean up from any genuine mid-ramp state.
    """
    dev = targets.device
    k = envs.shape[0]
    b = 2 * k
    levels = envs.tolist()

    tv = int(torch.randint(300, 701, (1,)).item())      ## virtual ramp duration
    ramp_len = max(1, int(0.85 * tv))
    T = int(torch.randint(0, tv - 60, (1,)).item())     ## truncation depth
    descending = torch.rand(1).item() < 0.5             ## both directions, or downward
                                                        ## transitions keep stable ghosts

    def rampn(t):
        if descending:
            return max(0.10, 1.00 - 0.90 * t / ramp_len)
        return min(1.0, 0.10 + 0.90 * t / ramp_len)

    state = new_seed(b, cfg, dev)
    with torch.no_grad():
        for t in range(T):
            n = torch.full((b,), rampn(t), device=dev)
            state = model(state, model.weights_for(n))

    g = int(torch.randint(8, 17, (1,)).item())          ## trained ramp continuation
    for t in range(T, T + g):
        n = torch.full((b,), rampn(t), device=dev)
        state = model(state, model.weights_for(n))

    n_end = rampn(T + g)
    stage = min(range(k), key=lambda i: abs(levels[i] - n_end))
    h = int(torch.randint(24, 41, (1,)).item())         ## hold at the snapped stage
    n_hold = envs[stage:stage + 1].expand(b)
    state = rollout(model, state, model.weights_for(n_hold), h)
    return F.mse_loss(state[:, 0:4], targets[stage].unsqueeze(0).expand(b, -1, -1, -1))


@torch.no_grad()
def _fixed_eval(model, targets, envs, cfg):
    """Deterministic eval for best-checkpoint tracking: stage growth + a 240-step ramp.
    Saves and restores the global RNG so training randomness is untouched."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        final = grow(model, envs, cfg.seg1_max, seed=2)
        mse_stages = F.mse_loss(final[:, 0:4], targets).item()
        ramp = lambda t: min(1.0, 0.10 + 0.90 * t / (240 * 0.85))
        end, _, _ = simulate(model, ramp, 240, seed=5)
        mse_ramp = F.mse_loss(end[:, 0:4], targets[-1:]).item()
        ## grow the flower, then ramp the nutrient back down: must end a clean sprout
        def down(t):
            if t < 100:
                return 1.00
            return max(0.40, round(1.00 - 0.60 * (t - 100) / 60, 3))
        end, _, _ = simulate(model, down, 260, seed=8)
        mse_ramp += F.mse_loss(end[:, 0:4], targets[1:2]).item()
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return mse_stages + mse_ramp


def train(model, targets, envs, cfg):
    """Each epoch runs a batch of 2K two-segment episodes.

    Segment 1 (from seed, env a): identity samples use their own stage; transition
    samples use a random stage a. Loss vs target a.
    Segment 2 (env b, from the DETACHED segment-1 state): identity keeps b = a
    (persistence); transitions pick random b != a, both directions (development AND
    regression). Loss vs target b. Detaching keeps memory at one-segment cost and is
    exactly pool-style training.

    Two robustness sources, both needed for TIME-VARYING environments at eval:
    - nutrient levels are jittered by +-env_jitter, so every stage owns a basin of
      environments instead of a single trained point;
    - the segment-2 start state is perturbed with noise half the time, so every rule
      learns to clean up residue left behind while the environment was in motion.
    """
    dev = targets.device
    k = envs.shape[0]

    vec = [p for p in model.parameters() if p.numel() > 1]
    sca = [p for p in model.parameters() if p.numel() == 1]
    opt = torch.optim.Adam([{"params": vec, "lr": cfg.lr},
                            {"params": sca, "lr": cfg.scalar_lr}])

    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    history = []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        opt.param_groups[0]["lr"] = cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))

        ident = torch.arange(k, device=dev)
        a_tr = torch.randint(0, k, (k,), device=dev)
        shift = torch.randint(1, k, (k,), device=dev)
        b_tr = (a_tr + shift) % k
        a_idx = torch.cat([ident, a_tr])               ## (2K,) segment-1 stages
        b_idx = torch.cat([ident, b_tr])               ## (2K,) segment-2 stages

        def jitter(n):
            j = (torch.rand_like(n) - 0.5) * 2 * cfg.env_jitter
            return (n + j).clamp(0.0, 1.0)

        opt.zero_grad()

        if torch.rand(1).item() < cfg.slow_ramp_prob:
            ## slow-ramp episode: train from true mid-ramp states (truncated BPTT)
            loss_r = _slow_ramp_episode(model, targets, envs, cfg)
            loss_r.backward()
            l1, l2 = float("nan"), loss_r.item()
        else:
            ## segment 1: seed -> target a
            s1 = int(torch.randint(cfg.seg1_min, cfg.seg1_max + 1, (1,)).item())
            state = new_seed(2 * k, cfg, dev)
            state = rollout(model, state, model.weights_for(jitter(envs[a_idx])), s1)
            loss1 = F.mse_loss(state[:, 0:4], targets[a_idx])
            loss1.backward()

            ## segment 2: grown state -> env changes to b -> target b. Half the episodes
            ## switch instantly; half GLIDE there (rollout_moving), because at eval the
            ## nutrient is a ramp that visits every value in between.
            s2 = int(torch.randint(cfg.seg2_min, cfg.seg2_max + 1, (1,)).item())
            state = state.detach()
            if torch.rand(1).item() < 0.5:             ## half the time: dirty start state
                alive = model.alive_mask(state).to(state.dtype)
                state = state + torch.randn_like(state) * cfg.state_noise * alive
            if torch.rand(1).item() < 0.5:
                state = rollout_moving(model, state, jitter(envs[a_idx]), jitter(envs[b_idx]), s2)
            else:
                state = rollout(model, state, model.weights_for(jitter(envs[b_idx])), s2)
            loss2 = F.mse_loss(state[:, 0:4], targets[b_idx])
            loss2.backward()
            l1, l2 = loss1.item(), loss2.item()

        ## per-parameter gradient normalisation, scalars excluded
        for p in vec:
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()

        history.append((epoch, l1, l2))

        ## best-checkpoint tracking on a FIXED eval (stage growth + a 240-step ramp):
        ## episode types have incomparable training losses, so those cannot be used
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            ev = _fixed_eval(model, targets, envs, cfg)
            if ev < best:
                best, best_state = ev, copy.deepcopy(model.state_dict())

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:5d} | grow {l1:.5f} | switch {l2:.5f} "
                  f"| best-eval {best:.5f} | {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    print(f"done in {time.time() - t0:.1f}s | best {best:.6f} | weights restored to best")
    return history


## ---------------------------------------------------------------------------
## verification: the acceptance tests
## ---------------------------------------------------------------------------

def verify(model, targets, envs, names, cfg):
    dev = targets.device
    k = envs.shape[0]
    results = {}

    ## 1. frozen without chemical: EXACTLY zero change over 50 steps
    grown = grow(model, envs[-1:], cfg.seg1_max, seed=1)
    chem0 = torch.zeros(1, 1, cfg.grid, cfg.grid, device=dev)
    frozen = grown
    for _ in range(50):
        frozen = model(frozen, model.weights_for(envs[-1:]), chem0)
    d = (frozen - grown).abs().max().item()
    results["freeze"] = (d == 0.0, f"max|delta| over 50 chem=0 steps = {d}")

    ## 2. per-stage growth from seed
    final = grow(model, envs, cfg.seg1_max, seed=2)
    per = (final[:, 0:4] - targets).pow(2).mean(dim=(1, 2, 3))
    worst = per.max().item()
    detail = ", ".join(f"{n} {v:.5f}" for n, v in zip(names, per.tolist()))
    results["stages"] = (worst < 0.005, f"per-stage MSE [{detail}], worst {worst:.5f} (< 0.005)")

    ## 3. env-shuffle ablation: the wrong nutrient must be much worse
    rolled = envs.roll(1)
    shuf = grow(model, rolled, cfg.seg1_max, seed=2)
    mse_ok = F.mse_loss(final[:, 0:4], targets).item()
    mse_sh = F.mse_loss(shuf[:, 0:4], targets).item()
    ratio = mse_sh / max(mse_ok, 1e-12)
    results["ablation"] = (ratio >= 10, f"correct {mse_ok:.6f} -> shuffled {mse_sh:.6f} "
                                        f"({ratio:.0f}x worse, need >= 10x)")

    ## 4. environment change: flower regresses to sprout when nutrient drops
    flower = grow(model, envs[-1:], cfg.seg1_max, seed=3)
    back = grow(model, envs[1:2], cfg.seg2_max + 16, state=flower, seed=4)
    mse_back = F.mse_loss(back[:, 0:4], targets[1:2]).item()
    results["regression"] = (mse_back < 0.01, f"flower -> n={envs[1]:.2f} -> sprout "
                                              f"MSE {mse_back:.5f} (< 0.01)")

    ## 5. continuous ramp simulation: nutrient 0.1 -> 1.0 changing EVERY step must still
    ##    end on a clean flower (this is the simulation the whole component exists for)
    total = 480
    ramp = lambda t: min(1.0, 0.10 + 0.90 * t / (total * 0.85))
    end, _, _ = simulate(model, ramp, total, seed=5)
    mse_ramp = F.mse_loss(end[:, 0:4], targets[-1:]).item()
    results["ramp"] = (mse_ramp < 0.01, f"continuous 480-step ramp -> flower "
                                        f"MSE {mse_ramp:.5f} (< 0.01)")

    ## 6. rise-then-fall simulation: grow the flower, then ramp nutrient down to sprout
    def updown(t):
        if t < 160:
            return 1.00
        if t < 200:
            return round(1.0 - 0.6 * (t - 160) / 40, 3)
        return 0.40
    end, _, _ = simulate(model, updown, 420, seed=8)
    mse_ud = F.mse_loss(end[:, 0:4], targets[1:2]).item()
    ## the sprout target has no alpha above row 18: any alpha there is a ghost of the
    ## dead flower that the low-nutrient rule failed to clean up
    ghost = end[0, 3].clamp(0, 1)[:18].sum().item()
    results["rise_fall"] = (mse_ud < 0.01 and ghost < 8.0,
                            f"flower then graded drop to n=0.40 -> sprout "
                            f"MSE {mse_ud:.5f} (< 0.01), ghost alpha {ghost:.1f} (< 8)")

    print("\n=== verification ===")
    ok = True
    for name, (passed, msg) in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<11} {msg}")
        ok &= passed
    print(f"=== {'ALL PASSED' if ok else 'FAILURES PRESENT'} ===\n")
    return ok, results


## ---------------------------------------------------------------------------
## rendering
## ---------------------------------------------------------------------------

def _checkerboard(h, w, tile=2):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // tile) + (xx // tile)) % 2 == 0, 0.74, 0.82)


def composite(state_chw, chem=None):
    """(C,H,W) -> (H,W,3). RGBA over a checkerboard; chem (if given) tints the
    background pale yellow where the chemical is present, so the field is visible."""
    rgba = state_chw[0:4].detach().clamp(0, 1).cpu().numpy()
    rgb, a = rgba[0:3].transpose(1, 2, 0), rgba[3][..., None]
    bg = _checkerboard(*rgb.shape[:2])[..., None] * np.ones(3)
    if chem is not None:
        c = chem.detach().clamp(0, 1).cpu().numpy().reshape(*rgb.shape[:2], 1)
        tint = np.array([1.00, 0.96, 0.72])
        bg = bg * (1 - 0.5 * c) + tint * 0.5 * c * bg.mean(axis=2, keepdims=True) * 1.25
        bg = np.clip(bg, 0, 1)
    return np.clip(rgb * a + bg * (1 - a), 0, 1)


def upscale(a, s=5):
    return np.kron(a, np.ones((s, s, 1)))


def frame_with_bar(img, level, label, bar_h=10):
    """Append a nutrient-level bar (and text-free tick marks) below an (H,W,3) image."""
    h, w, _ = img.shape
    bar = np.full((bar_h + 4, w, 3), 0.92)
    fill = int(round(level * (w - 4)))
    bar[2:-2, 2:2 + fill] = np.array([0.30, 0.62, 0.32]) if label == "nutrient" \
        else np.array([0.85, 0.75, 0.25])
    return np.concatenate([img, bar], axis=0)


def save_gif(frames_rgb, path, duration=70):
    imgs = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames_rgb]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print(f"saved {path} ({len(imgs)} frames)")


def save_grid_png(images, ncols, path, labels=None, scale=5, pad=4):
    """Tile (H,W,3) arrays row-major into one PNG with a header row of pixel labels."""
    tiles = [upscale(im, scale) for im in images]
    th, tw, _ = tiles[0].shape
    nrows = math.ceil(len(tiles) / ncols)
    canvas = np.ones((nrows * (th + pad) + pad, ncols * (tw + pad) + pad, 3))
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncols)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        canvas[y:y + th, x:x + tw] = t
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)
    print(f"saved {path}" + (f"  [{' | '.join(labels)}]" if labels else ""))


def render_all(model, targets, envs, names, cfg):
    out = cfg.out_dir
    dev = targets.device
    g = cfg.grid

    ## --- stages: targets vs grown vs env-shuffled -------------------------
    final = grow(model, envs, cfg.seg1_max, seed=2)
    shuf = grow(model, envs.roll(1), cfg.seg1_max, seed=2)
    images = [composite(targets[i]) for i in range(4)] \
        + [composite(final[i]) for i in range(4)] \
        + [composite(shuf[i]) for i in range(4)]
    save_grid_png(images, 4, os.path.join(out, "env_stages.png"),
                  labels=["rows: targets / grown at correct n / grown at shuffled n; "
                          f"cols: {', '.join(f'{nm} n={e:.2f}' for nm, e in zip(names, envs.tolist()))}"])

    ## --- ramp: nutrient 0 -> 1 over the rollout, development through stages
    total = 480
    ramp = lambda t: min(1.0, 0.10 + 0.90 * t / (total * 0.85))
    _, frames, trace = simulate(model, ramp, total, record_every=4, seed=5)
    gif = [frame_with_bar(upscale(composite(f[0]), 5), n, "nutrient")
           for f, n in zip(frames, trace)]
    save_gif(gif, os.path.join(out, "env_ramp.gif"))

    strip_idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    save_grid_png([frame_with_bar(composite(frames[i][0]), trace[i], "nutrient")
                   for i in strip_idx], 8, os.path.join(out, "env_ramp_strip.png"),
                  labels=[f"nutrient ramp, steps 0..{total}"])

    ## --- chem gate in time: no chemical until step 120, then full ---------
    onset = 120
    chem_t = lambda t: torch.full((1, 1, g, g), 0.0 if t < onset else 1.0, device=dev)
    _, frames, trace = simulate(model, lambda t: 1.0, 360, chem_fn=chem_t,
                                record_every=4, seed=6)
    gif = [frame_with_bar(upscale(composite(f[0], chem_t(i * 4)[0, 0]), 5),
                          0.0 if i * 4 < onset else 1.0, "chem")
           for i, f in enumerate(frames)]
    save_gif(gif, os.path.join(out, "env_chem_gate.gif"))

    ## --- chem gate in space: chemical only on the left half ---------------
    half = torch.zeros(1, 1, g, g, device=dev)
    half[..., : g // 2 + 1] = 1.0
    grown_half = grow(model, envs[-1:], cfg.seg1_max * 2, chem=half, seed=7)
    grown_full = grow(model, envs[-1:], cfg.seg1_max * 2, seed=7)
    save_grid_png([composite(grown_full[0]), composite(grown_half[0], half[0, 0])],
                  2, os.path.join(out, "env_chem_half.png"),
                  labels=["left: chem everywhere; right: chem only on left half (tinted)"])

    ## --- environment drop: flower regresses when nutrient falls -----------
    total = 420
    def updown(t):
        if t < 160:
            return 1.00
        if t < 200:
            return round(1.0 - 0.6 * (t - 160) / 40, 3)
        return 0.40
    _, frames, trace = simulate(model, updown, total, record_every=4, seed=8)
    gif = [frame_with_bar(upscale(composite(f[0]), 5), n, "nutrient")
           for f, n in zip(frames, trace)]
    save_gif(gif, os.path.join(out, "env_regress.gif"))


## ---------------------------------------------------------------------------
## checkpoint
## ---------------------------------------------------------------------------

def save_checkpoint(model, names, envs, cfg, path):
    torch.save({"state_dict": model.state_dict(), "names": names,
                "envs": envs.cpu(), "config": asdict(cfg)}, path)
    print(f"saved {path}")


def load_checkpoint(path, device=DEVICE):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["config"])
    m = EnvNCA(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["names"], ck["envs"].to(device), cfg


## ---------------------------------------------------------------------------
## main
## ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt = args.ckpt or os.path.join(cfg.out_dir, cfg.ckpt_name)

    if args.render_only:
        model, names, envs, cfg = load_checkpoint(ckpt)
        targets, _, _ = plant_targets(cfg)
        targets = targets.to(DEVICE)
    else:
        torch.manual_seed(cfg.seed)
        targets, names, envs = plant_targets(cfg)
        targets, envs = targets.to(DEVICE), envs.to(DEVICE)
        model = EnvNCA(cfg).to(DEVICE)
        print(f"torch {torch.__version__} | device {DEVICE} | stages "
              f"{dict(zip(names, [round(e, 2) for e in envs.tolist()]))}")
        sanity_check(model, targets, envs)
        train(model, targets, envs, cfg)
        save_checkpoint(model, names, envs, cfg, ckpt)

    ok, _ = verify(model, targets, envs, names, cfg)
    render_all(model, targets, envs, names, cfg)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

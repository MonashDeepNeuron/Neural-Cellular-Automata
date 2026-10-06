import math
import os
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLOURS = ("red", "orange", "black", "blue", "green")    ## mask channel order
RED, ORANGE, BLACK, BLUE, GREEN = range(5)


## ---------------------------------------------------------------------------
## config
## ---------------------------------------------------------------------------

@dataclass
class Config:
    ## cell state
    channels: int = 16
    grid: int = 48
    alive_threshold: float = 0.1
    fire_rate: float = 1.0      ## deterministic updates: the teacher is deterministic, so a
                                ## stochastic rule can only blur toward it (it did: a grey smear)

    ## update rule / manifold
    hidden: int = 128
    leak_init: float = 0.1
    latent_dim: int = 8
    dna_hidden: int = 64
    env_hidden: int = 64

    ## fields: explicit diffusion c <- c + alpha*lap/8 - k*c, k from the length scale
    diff_alpha: float = 0.9
    ell_attr: float = 10.0      ## attractant length scale (cells)
    ell_rep: float = 3.0        ## repellent length scale -- a local push, not a wall
    decoy_level: float = 0.5    ## orange pins the attractant at this level (red at 1)
    attr_floor: float = 1e-4    ## log-sensing floor: la = 1 + log(A + floor) / log(1/floor)
    wall_perm: float = 0.25     ## walls stop cells, but the chemical seeps through them.
                                ## (fully no-flux walls leave NO local maxima off the
                                ## sources -- greedy ascent would never get trapped)
    field_iters: int = 8        ## relaxation iterations per step (the field LAGS motion)
    init_iters: int = 1500      ## relaxation at episode start

    ## greedy teacher
    rep_weight: float = 0.4     ## utility U = log_attr - rep_weight * R: the colony keeps
                                ## ~6 cells from a toxin and still reaches food behind it
    life_max: int = 48          ## cell lifespan at n = 1 (steps); life(n) = n * life_max
    body_r: int = 2             ## radius of the seed colony
    gain_eps: float = 1e-4      ## a cell divides toward a neighbour only for a gain above this

    ## worlds (unused here, kept for config/checkpoint compatibility)
    max_people: int = 6
    person_r: tuple = (2, 4)
    person_speed: float = 0.3
    toggle_prob: float = 0.004
    colour_probs: tuple = (0.30, 0.20, 0.20, 0.15, 0.15)
    margin: int = 3

    ## training (unused here, kept for config/checkpoint compatibility)
    epochs: int = 4000
    batch: int = 16
    lr: float = 2e-3
    scalar_lr: float = 1e-4
    n_lo: float = 0.1
    dagger_prob: float = 0.75
    window: tuple = (16, 32)
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


## ---------------------------------------------------------------------------
## diffusion fields (attractant / repellent)
## ---------------------------------------------------------------------------

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

    def log_attr(self):
        """(B,1,H,W) attractant as bacteria sense it: LOG concentration (Weber's law --
        E. coli responds to relative change), normalised to [0,1]; 0 below the floor."""
        e = self.cfg.attr_floor
        return (1.0 + torch.log(self.c[:, 0:1] + e) / math.log(1.0 / e)).clamp(0.0, 1.0)

    def sense(self, masks):
        """(B,3,H,W): log-attractant, repellent, raw food mask -- what a cell perceives."""
        return torch.cat([self.log_attr(), self.c[:, 1:2], masks[:, RED:RED + 1]], 1)


## ---------------------------------------------------------------------------
## teacher (used here only for Teacher(...).rgba() -- the initial seed shape)
## ---------------------------------------------------------------------------

CORE = np.array([0.30, 0.42, 0.12])      ## old tail
FRONT = np.array([0.90, 0.95, 0.45])     ## young front
BLOOM = np.array([0.98, 0.45, 0.80])     ## feeding on red food

_OFFS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))   ## tie order: axes first


def life(n, cfg):
    """Cell lifespan in steps at nutrient n (float or tensor)."""
    if torch.is_tensor(n):
        return torch.clamp(torch.round(n * cfg.life_max), min=1)
    return max(1, int(round(float(n) * cfg.life_max)))


def _nbr(x, dy, dx, fill):
    """out[..., y, x] = x[..., y + dy, x + dx], `fill` outside the grid."""
    H, W = x.shape[-2:]
    p = F.pad(x, (1, 1, 1, 1), value=fill)
    return p[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]


class Teacher:
    """The state is just the age grid `age` (B,H,W), inf = dead; `alive` (B,) flags colonies
    that still exist; `t` is a per-slot step counter."""

    def __init__(self, cfg, seed, masks):
        B, _, H, W = masks.shape
        dev = masks.device
        self.cfg = cfg
        self.t = torch.zeros(B, dtype=torch.long, device=dev)
        self.age = torch.full((B, H, W), float("inf"), device=dev)
        self.alive = torch.ones(B, dtype=torch.bool, device=dev)
        yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        seed = torch.as_tensor(seed, device=dev).long()
        d2 = (yy - seed[:, 0, None, None]) ** 2 + (xx - seed[:, 1, None, None]) ** 2
        self.age.copy_(torch.where(d2 <= cfg.body_r ** 2, torch.zeros_like(self.age), self.age))
        self._kill_env(masks)

    def _kill_env(self, masks):
        dead = (masks[:, BLUE] > 0) | (masks[:, BLACK] > 0)
        self.age.copy_(torch.where(dead, torch.full_like(self.age, float("inf")), self.age))
        self.alive &= torch.isfinite(self.age).flatten(1).any(1)

    def choice(self, fields, masks):
        """(B,H,W) long: 0 = stay (a local maximum), k = divide toward _OFFS[k-1]."""
        cfg, inf = self.cfg, float("inf")
        wall = masks[:, BLUE] > 0
        U = fields.log_attr()[:, 0] - cfg.rep_weight * fields.c[:, 1]
        U = torch.where(wall, torch.full_like(U, -inf), U)
        best_u, best_k = U + cfg.gain_eps, torch.zeros_like(U, dtype=torch.long)
        for k, (dy, dx) in enumerate(_OFFS, start=1):
            un = _nbr(U, dy, dx, -inf)
            better = un > best_u
            best_u = torch.where(better, un, best_u)
            best_k = torch.where(better, torch.full_like(best_k, k), best_k)
        return best_k

    @torch.no_grad()
    def step(self, fields, masks, n):
        wall, black = masks[:, BLUE] > 0, masks[:, BLACK] > 0
        best_k = self.choice(fields, masks)
        self.age.add_(1.0 + masks[:, GREEN])        ## local time runs twice as fast on green
        alive = torch.isfinite(self.age)
        mature = alive & (self.age >= 2)
        born = torch.zeros_like(alive)
        for k, (dy, dx) in enumerate(_OFFS, start=1):   ## a mature cell divides into its choice
            born |= _nbr((mature & (best_k == k)).float(), -dy, -dx, 0.0) > 0
        tips = alive & (best_k == 0)                    ## local maxima keep dividing in place
        born |= F.max_pool2d(tips[:, None].float(), 3, stride=1, padding=1)[:, 0] > 0
        born &= ~alive & ~wall & ~black
        self.age.copy_(torch.where(born | tips, torch.zeros_like(self.age), self.age))
        ## feeding: the colony spreads over red food one cell per step, and feeding cells
        ## don't age -- it engulfs the food person and trails after them when they walk off
        alive = torch.isfinite(self.age)[:, None].float()
        feed = (F.max_pool2d(alive, 3, stride=1, padding=1)[:, 0] > 0) & (masks[:, RED] > 0)
        self.age.copy_(torch.where(feed, torch.zeros_like(self.age), self.age))
        lf = life(torch.as_tensor(n, dtype=torch.float32, device=self.age.device), self.cfg)
        self.age.copy_(torch.where(self.age > lf.view(-1, 1, 1), torch.full_like(self.age, float("inf")), self.age))
        self._kill_env(masks)
        self.t += 1

    def splice(self, idx, other):
        """Replace slots `idx` with the (len(idx)-slot) teacher `other`."""
        for k in ("t", "age", "alive"):
            getattr(self, k)[idx] = getattr(other, k)

    @torch.no_grad()
    def anchor(self, rgba, masks, sel):
        """DAgger: re-read the slots in `sel` ((B,) bool) from a colony IMAGE -- alive where
        alpha > 0.5 (and not in a wall/toxin), age from the colour along FRONT -> CORE (bloom
        cells count as young). The teacher then continues from the learner's colony."""
        dev = self.age.device
        front = torch.tensor(FRONT, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        d = torch.tensor(CORE - FRONT, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        shade = ((rgba[:, :3] - front) * d).sum(1) / (d * d).sum()
        age = torch.round(shade.clamp(0, 1) * self.cfg.life_max)
        age = torch.where(masks[:, RED] > 0, torch.zeros_like(age), age)
        kill = (masks[:, BLUE] > 0) | (masks[:, BLACK] > 0)
        ## the image is drawn THICK (see rgba): erode it back to the thin state, treating
        ## walls/toxins and the border as "inside" so cells beside them are not eroded away
        thick = (rgba[:, 3] > 0.5) | kill
        pad = F.pad(thick[:, None].float(), (1, 1, 1, 1), value=1.0)
        alive = (-F.max_pool2d(-pad, 3, stride=1)[:, 0] > 0) & ~kill & (rgba[:, 3] > 0.5)
        age = torch.where(alive, age, torch.full_like(age, float("inf")))
        ok = sel & alive.flatten(1).any(1)
        self.age.copy_(torch.where(ok[:, None, None], age, self.age))
        self.alive.copy_(torch.where(ok, torch.ones_like(self.alive), self.alive))

    def rgba(self, masks):
        """(B,4,H,W) target image: the colony drawn THICK -- dilated by one cell (a 1-cell
        trail is too thin a target for an NCA to hold), each drawn cell coloured by the
        youngest colony cell next to it, bloom on food, nothing on walls/toxins."""
        big = torch.full_like(self.age, 1e6)
        age = -F.max_pool2d(-torch.where(torch.isfinite(self.age), self.age, big)[:, None], 3,
                            stride=1, padding=1)[:, 0]
        kill = (masks[:, BLUE] > 0) | (masks[:, BLACK] > 0)
        alive = (age < 1e6) & ~kill
        shade = (age / self.cfg.life_max).clamp(0, 1)
        front = torch.tensor(FRONT, dtype=torch.float32, device=self.age.device).view(1, 3, 1, 1)
        core = torch.tensor(CORE, dtype=torch.float32, device=self.age.device).view(1, 3, 1, 1)
        bloom = torch.tensor(BLOOM, dtype=torch.float32, device=self.age.device).view(1, 3, 1, 1)
        rgb = front + (core - front) * shade[:, None]
        rgb = torch.where((masks[:, RED:RED + 1] > 0), bloom, rgb)
        a = alive[:, None].float()
        return torch.cat([rgb * a, a], 1)


## ---------------------------------------------------------------------------
## model
## ---------------------------------------------------------------------------

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

    def _update(self, state, weights, sense):
        w1, b1, w2 = weights
        pre = self.alive_mask(state)
        p = self.perceive(state, sense)
        a = torch.einsum("bop,bpij->boij", w1, p) + b1[:, :, None, None]
        ds = torch.einsum("bcv,bvij->bcij", w2, F.relu(a))
        if self.cfg.fire_rate < 1.0:                       ## optional stochastic updates
            ds = ds * (torch.rand_like(ds[:, :1]) < self.cfg.fire_rate).to(ds.dtype)
        out = state + self.log_leak.exp() * ds
        return out * (pre & self.alive_mask(out)).to(out.dtype)

    def forward(self, state, weights, sense, kill, boost):
        """One step. sense (B,3,H,W); kill, boost (B,1,H,W) in {0,1}. Green (boost) cells take
        a SECOND update in the same step -- local time runs twice as fast there."""
        out = self._update(state, weights, sense)
        out = out + (self._update(out, weights, sense) - out) * boost
        return out * (1.0 - kill)


def seed_state(rgba, cfg):
    """NCA start state: the teacher's first body in RGBA, hidden channels zero."""
    s = torch.zeros(rgba.shape[0], cfg.channels, *rgba.shape[-2:], device=rgba.device)
    s[:, 0:4] = rgba
    return s


## ---------------------------------------------------------------------------
## rendering
## ---------------------------------------------------------------------------

PERSON_RGB = np.array([[0.86, 0.16, 0.16],     ## red: food
                       [0.96, 0.58, 0.12],     ## orange: decoy
                       [0.10, 0.10, 0.12],     ## black: toxin
                       [0.20, 0.34, 0.86],     ## blue: wall
                       [0.22, 0.70, 0.30]])    ## green: booster


def _checkerboard(h, w, tile=2):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // tile) + (xx // tile)) % 2 == 0, 0.74, 0.82)


def composite(rgba, masks=None, bg=None):
    """(4,H,W) colony RGBA (+ optional (5,H,W) masks, (H,W,3) bg) -> (H,W,3) image in [0,1]."""
    rgba = torch.as_tensor(rgba).detach().float().clamp(0, 1).cpu().numpy()
    rgb, a = rgba[0:3].transpose(1, 2, 0), rgba[3][..., None]
    h, w = rgb.shape[:2]
    if bg is None:
        bg = _checkerboard(h, w)[..., None] * np.ones(3)
    elif bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)   ## cv2.resize wants (w, h)
    if masks is not None:
        m = torch.as_tensor(masks).cpu().numpy() > 0
        for c in (GREEN, ORANGE, RED, BLACK):          ## people: tinted, colony shows through
            bg[m[c]] = 0.35 * bg[m[c]] + 0.65 * PERSON_RGB[c]
        bg[m[BLUE]] = PERSON_RGB[BLUE]
    return np.clip(rgb * a + bg * (1 - a), 0, 1)


## ---------------------------------------------------------------------------
## checkpoint
## ---------------------------------------------------------------------------

def save_checkpoint(model, cfg, path=None):
    path = path or os.path.join(cfg.out_dir, cfg.ckpt_name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, path)
    print(f"saved {path}")


def load_checkpoint(path=None, device=DEVICE):
    path = path or os.path.join(CFG.out_dir, CFG.ckpt_name)
    ck = torch.load(path, map_location=device, weights_only=False)
    known = Config.__dataclass_fields__             ## tolerate configs from older runs
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
    m = BacteriaNCA(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, cfg
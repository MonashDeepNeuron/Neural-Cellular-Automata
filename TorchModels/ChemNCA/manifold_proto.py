## Minimal NCA-Manifold prototype (Hernandez Ruiz et al., CVPR 2021, arXiv 2006.12155).
##
## One NCA. K latents. The latent is decoded into the WEIGHTS of the per-cell update
## rule -- kappa(e) = P(D(e)) -- so each latent IS a different cellular automaton.
## Every organism grows from the identical seed, so the latent is the only thing that
## can possibly explain the difference between them. That is the whole point.
##
## Deviations from the paper, both deliberate (see plan):
##   - weights are generated as a RESIDUAL on a learned base rule, predictor zero-init,
##     so training starts from one well-conditioned shared rule instead of noise.
##   - no InstanceNorm (it is non-local, which is not an NCA); per-cell channel RMS
##     norm is available behind a flag instead.
import argparse
import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as f

CHANNELS = 16
HIDDEN = 128
PERCEPTION = 3 * CHANNELS  ## identity + sobel_x + sobel_y
GRID = 40
LATENT_DIM = 8
DNA_HIDDEN = 64
ALIVE_THRESHOLD = 0.1
FIRE_RATE = 0.5
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## ---------------------------------------------------------------- targets
## Four synthetic RGBA shapes. Deliberately simple and maximally distinct so that
## latent interpolation is unmistakable rather than something you have to squint at.
def make_targets(grid=GRID):
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    cy = cx = (grid - 1) / 2.0
    dy, dx = yy - cy, xx - cx
    r = 9.0

    circle = (dy**2 + dx**2).sqrt() <= r
    square = (dy.abs() <= r * 0.85) & (dx.abs() <= r * 0.85)
    ## upward triangle
    tri = (dy <= r * 0.9) & (dy >= -r * 0.9) & (dx.abs() <= (dy + r * 0.9) * 0.55)
    plus = ((dy.abs() <= r * 0.32) & (dx.abs() <= r)) | ((dx.abs() <= r * 0.32) & (dy.abs() <= r))

    specs = [
        (circle, (0.90, 0.20, 0.20), "red circle"),
        (square, (0.20, 0.35, 0.90), "blue square"),
        (tri, (0.20, 0.75, 0.30), "green triangle"),
        (plus, (0.95, 0.80, 0.15), "yellow plus"),
    ]
    out = torch.zeros(len(specs), 4, grid, grid)
    names = []
    for i, (mask, rgb, name) in enumerate(specs):
        m = mask.float()
        for c in range(3):
            out[i, c] = rgb[c] * m
        out[i, 3] = m
        names.append(name)
    return out, names


## ---------------------------------------------------------------- model
class ManifoldGCA(nn.Module):
    SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    SOBEL_Y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    def __init__(self, n_latents, latent_dim=LATENT_DIM, dna_hidden=DNA_HIDDEN, rms_norm=False):
        super().__init__()
        self.n_latents = n_latents
        self.rms_norm = rms_norm

        ## The genome: one free embedding per regime, optimised jointly with the weights
        ## (auto-decoded, DeepSDF style -- no encoder needed for a fixed target set).
        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.1)

        ## DNA-decoder D, then Parameter Predictor P. P emits every weight of the update net.
        self.n_generated = HIDDEN * PERCEPTION + HIDDEN + CHANNELS * HIDDEN
        self.dna_decoder = nn.Sequential(nn.Linear(latent_dim, dna_hidden), nn.ReLU())
        self.predictor = nn.Linear(dna_hidden, self.n_generated)
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

        ## Base rule the generated weights are a residual on.
        self.base_w1 = nn.Parameter(torch.randn(HIDDEN, PERCEPTION) * 0.001)
        self.base_b1 = nn.Parameter(torch.zeros(HIDDEN))
        self.base_w2 = nn.Parameter(torch.zeros(CHANNELS, HIDDEN))

        ## Paper's leak factor: a learnable Euler timestep, parameterised in log space.
        self.log_leak = nn.Parameter(torch.tensor(math.log(0.1)))

    def to(self, device):
        self.SOBEL_X = self.SOBEL_X.to(device)
        self.SOBEL_Y = self.SOBEL_Y.to(device)
        return super().to(device)

    def generate_weights(self, z):
        ## kappa(e) = P(D(e)) -- computed ONCE per rollout, not per step and not per cell.
        theta = self.predictor(self.dna_decoder(z))
        b = z.size(0)
        i = 0
        n = HIDDEN * PERCEPTION
        w1 = self.base_w1 + theta[:, i : i + n].view(b, HIDDEN, PERCEPTION)
        i += n
        b1 = self.base_b1 + theta[:, i : i + HIDDEN]
        i += HIDDEN
        w2 = self.base_w2 + theta[:, i:].view(b, CHANNELS, HIDDEN)
        return w1, b1, w2

    def perceive(self, state):
        ## replicate (no-flux) padding, NOT circular: a torus makes a boundary signal
        ## wrap around, and it is why the 40x40 lizard degrades on larger grids.
        padded = f.pad(state, (1, 1, 1, 1), mode="replicate")
        kx = self.SOBEL_X.unsqueeze(0).unsqueeze(0).repeat(CHANNELS, 1, 1, 1)
        ky = self.SOBEL_Y.unsqueeze(0).unsqueeze(0).repeat(CHANNELS, 1, 1, 1)
        gx = f.conv2d(padded, kx, groups=CHANNELS)
        gy = f.conv2d(padded, ky, groups=CHANNELS)
        return torch.cat([state, gx, gy], dim=1)

    def alive_mask(self, state):
        return f.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > ALIVE_THRESHOLD

    def forward(self, state, weights):
        w1, b1, w2 = weights
        pre = self.alive_mask(state)

        p = self.perceive(state)
        a = torch.einsum("bop,bpij->boij", w1, p) + b1[:, :, None, None]
        if self.rms_norm:
            a = a / (a.pow(2).mean(dim=1, keepdim=True) + 1e-6).sqrt()
        h = f.relu(a)
        ds = torch.einsum("bcv,bvij->bcij", w2, h)

        ## per-cell fire mask, shared across channels, active in train AND eval
        m = (torch.rand_like(ds[:, :1]) <= FIRE_RATE).to(ds.dtype)
        out = state + self.log_leak.exp() * ds * m

        post = self.alive_mask(out)
        return out * (pre & post).to(out.dtype)


def new_seed(n, grid=GRID):
    s = torch.zeros(n, CHANNELS, grid, grid)
    s[:, 3, grid // 2, grid // 2] = 1.0
    return s


def rollout(model, state, weights, steps):
    for _ in range(steps):
        state = model(state, weights)
    return state


## ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--steps-min", type=int, default=48)
    ap.add_argument("--steps-max", type=int, default=64)
    ap.add_argument("--rms-norm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    targets, names = make_targets()
    targets = targets.to(DEVICE)
    k = targets.size(0)
    print(f"device={DEVICE}  latents={k}  targets={names}")

    model = ManifoldGCA(k, rms_norm=args.rms_norm).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"params: {total:,}  (generates {model.n_generated:,} update-net weights per latent)")

    ## Scalars (the leak factor) must NOT go through per-parameter gradient
    ## normalisation -- for a scalar that yields a gradient of exactly +-1 every step.
    vec = [p for p in model.parameters() if p.numel() > 1]
    sca = [p for p in model.parameters() if p.numel() == 1]
    opt = torch.optim.Adam([{"params": vec, "lr": args.lr}, {"params": sca, "lr": 1e-4}])

    z_idx = torch.arange(k, device=DEVICE)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for epoch in range(args.epochs):
        frac = epoch / max(args.epochs - 1, 1)
        lr = args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * frac)))
        opt.param_groups[0]["lr"] = lr

        opt.zero_grad()
        ## One sample per latent, identical seeds: z is the ONLY differentiator,
        ## which is what stops the model collapsing onto an average rule.
        state = new_seed(k).to(DEVICE)
        weights = model.generate_weights(model.latents[z_idx])
        steps = int(torch.randint(args.steps_min, args.steps_max + 1, (1,)).item())
        state = rollout(model, state, weights, steps)

        loss = f.mse_loss(state[:, 0:4], targets)
        loss.backward()
        for p in vec:
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()

        if epoch % 200 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                per = (state[:, 0:4] - targets).pow(2).mean(dim=(1, 2, 3))
            if loss.item() < best:
                best = loss.item()
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"epoch {epoch:5d}  loss {loss.item():.5f}  leak {model.log_leak.exp().item():.3f}"
                f"  per-latent {[round(v, 4) for v in per.tolist()]}"
            )

    model.load_state_dict(best_state)
    print(f"done in {time.time() - t0:.1f}s  best loss {best:.5f}")

    ## shuffle-z ablation: if the latent is being ignored, permuting it costs nothing.
    with torch.no_grad():
        state = new_seed(k).to(DEVICE)
        state = rollout(model, state, model.generate_weights(model.latents[z_idx]), 64)
        honest = f.mse_loss(state[:, 0:4], targets).item()
        perm = torch.tensor([1, 2, 3, 0], device=DEVICE)
        state = new_seed(k).to(DEVICE)
        state = rollout(model, state, model.generate_weights(model.latents[perm]), 64)
        shuffled = f.mse_loss(state[:, 0:4], targets).item()
    print(f"shuffle-z ablation: correct z {honest:.5f} -> shuffled z {shuffled:.5f}"
          f"  ({shuffled / max(honest, 1e-9):.1f}x worse)")

    torch.save({"state_dict": model.state_dict(), "names": names}, os.path.join(OUT_DIR, "manifold.pth"))
    print("saved manifold.pth")


if __name__ == "__main__":
    main()

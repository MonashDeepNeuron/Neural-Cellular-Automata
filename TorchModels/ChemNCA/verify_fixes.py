## Numerically verify the claims behind the uncommitted PersistingGCA fixes.
## Read-only: loads lizard.pth, writes nothing.
import sys

import torch
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image

sys.path.insert(0, r"C:\Users\krrish\documents\Neural-Cellular-Automata\TorchModels\PersistingGCA")
from persistingmodel import GCA

CH = 16
GRID = 40
WEIGHTS = r"C:\Users\krrish\documents\Neural-Cellular-Automata\TorchModels\PersistingGCA\lizard.pth"
TARGET = r"C:\Users\krrish\documents\Neural-Cellular-Automata\TorchModels\PersistingGCA\lizard.png"
device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_grid(n=1, grid=GRID):
    s = torch.zeros(n, CH, grid, grid)
    s[:, 3, grid // 2, grid // 2] = 1.0
    return s.to(device)


def roll(model, steps, grid=GRID, seed=0):
    torch.manual_seed(seed)
    s = seed_grid(1, grid)
    with torch.no_grad():
        for _ in range(steps):
            s = model(s)
    return s


model = GCA().to(device)
model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location=device))

print("=" * 62)
print("1. train()/eval() equivalence  (dropout fix)")
model.train(True)
a = roll(model, 200)
model.train(False)
b = roll(model, 200)
delta = (a - b).abs().max().item()
print(f"   max|train - eval| over 200 steps = {delta:.3e}   -> {'PASS' if delta == 0.0 else 'FAIL'}")
print("   (identical means model.train()/eval() are behavioural no-ops, as intended)")

print("\n2. Determinism under a fixed seed")
d = (roll(model, 96, seed=0) - roll(model, 96, seed=0)).abs().max().item()
n = (roll(model, 96, seed=0) - roll(model, 96, seed=1)).abs().max().item()
print(f"   same seed  -> max|delta| = {d:.3e}   {'PASS' if d == 0.0 else 'FAIL'}")
print(f"   diff seed  -> max|delta| = {n:.3e}   {'PASS (stochastic)' if n > 0 else 'FAIL'}")

print("\n3. load_image Pad bug  (Pad's 2nd positional arg is `fill`, not padding)")
img = read_image(TARGET, ImageReadMode.RGB_ALPHA)
img = T.Resize((28, 28))(img)
buggy = (T.Pad((GRID - 28) // 2, (GRID - 28) // 2)(img) / 255.0)
fixed = (T.Pad((GRID - 28) // 2)(img) / 255.0)
print(f"   as written : border RGBA = {[round(v, 6) for v in buggy[:, 0, 0].tolist()]}")
print(f"   fixed      : border RGBA = {[round(v, 6) for v in fixed[:, 0, 0].tolist()]}")
print(f"   alive_mask threshold is 0.1, so border alpha {buggy[3,0,0]:.4f} is UNREACHABLE")
floor = (buggy - fixed).pow(2).mean().item()
print(f"   irreducible MSE contributed by the bad border = {floor:.3e}")

print("\n4. Persistence: RGBA loss vs rollout length (fixed seed)")
tgt = fixed.to(device)
for steps in (48, 96, 200, 400, 600, 1000):
    s = roll(model, steps)
    loss = (s[0, 0:4] - tgt).pow(2).mean().item()
    alive = (s[0, 3] > 0.1).sum().item()
    print(f"   {steps:5d} steps  loss = {loss:.5f}   alive = {alive:4d}")

print("\n5. Grid-size generalisation (trained at 40x40)")
for g in (40, 48, 60, 80):
    s = roll(model, 200, grid=g)
    alive = (s[0, 3] > 0.1).sum().item()
    print(f"   {g}x{g:<3d} after 200 steps  alive = {alive:4d}")
print("   note: web/app/simulator/growing/client.tsx runs this model at SIZE = 48")

print("\n6. Scalar-parameter hazard in the grad-norm loop")
print("   train.py:217  p.grad /= (p.grad.norm() + 1e-8)")
sc = torch.tensor(1.5, requires_grad=True)
(sc * 3).backward()
sc.grad /= sc.grad.norm() + 1e-8
print(f"   a scalar param's normalised grad = {sc.grad.item():.6f}  -> always +-1, full-size step forever")
print("=" * 62)

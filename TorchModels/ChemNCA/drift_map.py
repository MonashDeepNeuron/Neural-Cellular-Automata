## Where exactly does the lizard degrade over a long rollout?
## Total alive count RISES (432 -> 482), which hides the fact that some regions
## lose alpha while a diffuse halo gains it. This localises the drift.
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image

REPO = r"C:\Users\krrish\documents\Neural-Cellular-Automata"
PGCA = os.path.join(REPO, "TorchModels", "PersistingGCA")
sys.path.insert(0, PGCA)
from persistingmodel import GCA

HERE = os.path.dirname(os.path.abspath(__file__))
CH, GRID = 16, 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKS = [96, 250, 1000]

model = GCA().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(PGCA, "lizard.pth"), weights_only=True, map_location=DEVICE))

img = read_image(os.path.join(PGCA, "lizard.png"), ImageReadMode.RGB_ALPHA)
target = (T.Pad((GRID - 28) // 2)(T.Resize((28, 28))(img)) / 255.0).to(DEVICE)
tgt_alive = target[3] > 0.1

torch.manual_seed(0)
s = torch.zeros(1, CH, GRID, GRID)
s[:, 3, GRID // 2, GRID // 2] = 1.0
s = s.to(DEVICE)

snaps = {}
with torch.no_grad():
    for i in range(1, max(CHECKS) + 1):
        s = model(s)
        if i in CHECKS:
            snaps[i] = s[0, 3].clamp(0, 1).cpu().numpy()

ref = snaps[CHECKS[0]]
fig, ax = plt.subplots(1, len(CHECKS) + 1, figsize=(4.0 * (len(CHECKS) + 1), 4.0))
ax[0].imshow(tgt_alive.cpu().numpy(), cmap="gray")
ax[0].set_title("target support (alpha > 0.1)")
for j, st in enumerate(CHECKS):
    a = ax[j + 1]
    if j == 0:
        im = a.imshow(ref, cmap="viridis", vmin=0, vmax=1)
        a.set_title(f"alpha @ step {st} (reference)")
    else:
        d = snaps[st] - ref
        im = a.imshow(d, cmap="coolwarm", vmin=-0.6, vmax=0.6)
        a.set_title(f"alpha change, step {CHECKS[0]} -> {st}")
    plt.colorbar(im, ax=a, fraction=0.046)
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle("Long-rollout drift is LOCALISED: blue = losing alpha, red = gaining", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "drift_map.png"), dpi=105)
print("saved drift_map.png")

## quantify: inside the target support vs outside it
print(f"{'step':>6} {'mean a inside':>14} {'mean a outside':>15} {'alive in':>9} {'alive out':>10}")
for st in CHECKS:
    a = torch.from_numpy(snaps[st]).to(DEVICE)
    inside, outside = a[tgt_alive], a[~tgt_alive]
    print(
        f"{st:>6} {inside.mean().item():>14.4f} {outside.mean().item():>15.4f}"
        f" {(inside > 0.1).sum().item():>9} {(outside > 0.1).sum().item():>10}"
    )
print(f"\ntarget support = {tgt_alive.sum().item()} of {GRID * GRID} px")

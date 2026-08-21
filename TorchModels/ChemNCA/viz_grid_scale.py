## Does the 40x40-trained rule survive on the 96-128 grids the ChemNCA plan needs?
## Read-only apart from the PNG it writes into the scratchpad.
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, r"C:\Users\krrish\documents\Neural-Cellular-Automata\TorchModels\PersistingGCA")
from persistingmodel import GCA

CH = 16
WEIGHTS = r"C:\Users\krrish\documents\Neural-Cellular-Automata\TorchModels\PersistingGCA\lizard.pth"
OUT = r"C:\Users\krrish\AppData\Local\Temp\claude\C--Users-krrish-documents-Neural-Cellular-Automata\3d38b136-2e07-4af2-a6b8-8443bf7441ee\scratchpad\grid_scale.png"
device = "cuda" if torch.cuda.is_available() else "cpu"

GRIDS = (40, 64, 96, 128)
STEPS = (96, 200, 600)

model = GCA().to(device)
model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location=device))

fig, axes = plt.subplots(len(GRIDS), len(STEPS), figsize=(3 * len(STEPS), 3 * len(GRIDS)))
for r, g in enumerate(GRIDS):
    torch.manual_seed(0)
    s = torch.zeros(1, CH, g, g)
    s[:, 3, g // 2, g // 2] = 1.0
    s = s.to(device)
    nxt = 0
    with torch.no_grad():
        for i in range(1, max(STEPS) + 1):
            s = model(s)
            if i == STEPS[nxt]:
                rgb = s[0, 0:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                alive = (s[0, 3] > 0.1).sum().item()
                ax = axes[r, nxt]
                ax.imshow(rgb)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.02, 0.03, f"alive={alive}", color="w", fontsize=8, transform=ax.transAxes)
                if r == 0:
                    ax.set_title(f"{STEPS[nxt]} steps")
                if nxt == 0:
                    ax.set_ylabel(f"{g}x{g}", fontsize=11)
                nxt += 1
                if nxt == len(STEPS):
                    break
    print(f"grid {g}x{g}: done")

fig.suptitle("40x40-trained rule evaluated on larger grids (same seed)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print("saved", OUT)

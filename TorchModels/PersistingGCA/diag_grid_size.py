"""Diagnostic: same saved model, same seed, rolled out on the 40x40 grid it was
trained on vs the 60x60 grid the original script evaluated on. Isolates the
grid-size effect now that the dropout mismatch is fixed."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from persistingmodel import GCA

CH = 16
STEPS = [48, 96, 200, 400, 600]
GRIDS = [40, 60]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load():
    m = GCA().to(device)
    m.load_state_dict(torch.load("lizard.pth", weights_only=True, map_location=device))
    m.eval()
    return m


def seed(grid):
    s = torch.zeros(1, CH, grid, grid, device=device)
    s[:, 3, grid // 2, grid // 2] = 1
    return s


fig, axes = plt.subplots(len(GRIDS), len(STEPS), figsize=(3 * len(STEPS), 3.2 * len(GRIDS)))
for row, grid in enumerate(GRIDS):
    torch.manual_seed(0)
    model = load()
    with torch.no_grad():
        state = seed(grid)
        done = 0
        for col, n in enumerate(STEPS):
            for _ in range(n - done):
                state = model(state)
            done = n
            alive = (state[0, 3] > 0.1).sum().item()
            rgb = state[0, 0:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"{n} steps")
            ax.text(1, grid - 2, f"alive={alive}", color="w", fontsize=8, va="bottom")
            if col == 0:
                ax.axis("on")
                ax.set_ylabel(f"{grid}x{grid}", fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("diag_grid_size.png", bbox_inches="tight", dpi=110)
print("saved diag_grid_size.png")

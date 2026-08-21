"""Quick diagnostic: render the trained lizard at several step counts to see
whether the shape forms at the training horizon and when it diverges."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from persistingmodel import GCA

GRID = 40
CH = 16
STEPS = [24, 48, 72, 96, 200, 400]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCA().to(device)
model.load_state_dict(torch.load("lizard.pth", weights_only=True, map_location=device))
model.eval()

def seed():
    s = torch.zeros(1, CH, GRID, GRID, device=device)
    s[:, 3, GRID // 2, GRID // 2] = 1
    return s

fig, axes = plt.subplots(1, len(STEPS), figsize=(3 * len(STEPS), 3))
with torch.no_grad():
    state = seed()
    done = 0
    for ax, n in zip(axes, STEPS):
        for _ in range(n - done):
            state = model(state)
        done = n
        rgb = state[0, 0:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        ax.imshow(rgb)
        ax.set_title(f"{n} steps")
        ax.axis("off")
plt.tight_layout()
plt.savefig("eval_steps.png", bbox_inches="tight", dpi=110)
print("saved eval_steps.png")

"""Diagnostic for H1: does the train/eval dropout mismatch cause divergence?

Rolls the SAVED model from the same seed in both model.train() (dropout ON, the
dynamics it was optimised under) and model.eval() (dropout OFF, the dynamics used
for the final gif). If train-mode stays coherent while eval-mode breaks up, the
dropout-as-stochastic-mask mismatch is the persistence bug.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from persistingmodel import GCA

GRID = 40
CH = 16
STEPS = [48, 96, 200, 400, 600]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load():
    m = GCA().to(device)
    m.load_state_dict(torch.load("lizard.pth", weights_only=True, map_location=device))
    return m


def seed():
    s = torch.zeros(1, CH, GRID, GRID, device=device)
    s[:, 3, GRID // 2, GRID // 2] = 1
    return s


def rollout(model, training):
    torch.manual_seed(0)  # same dropout draws across runs for fairness
    model.train(training)
    frames = {}
    with torch.no_grad():
        state = seed()
        done = 0
        for n in STEPS:
            for _ in range(n - done):
                state = model(state)
            done = n
            frames[n] = state[0, 0:3].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return frames


modes = [("train() dropout ON", True), ("eval() dropout OFF", False)]
fig, axes = plt.subplots(2, len(STEPS), figsize=(3 * len(STEPS), 6.4))
for row, (label, training) in enumerate(modes):
    frames = rollout(load(), training)
    for col, n in enumerate(STEPS):
        ax = axes[row, col]
        ax.imshow(frames[n])
        ax.axis("off")
        if row == 0:
            ax.set_title(f"{n} steps")
        if col == 0:
            ax.set_ylabel(label, fontsize=11)
            ax.axis("on")
            ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("diag_train_vs_eval.png", bbox_inches="tight", dpi=110)
print("saved diag_train_vs_eval.png")

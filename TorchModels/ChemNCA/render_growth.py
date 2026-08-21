## Legible growth rendering for the existing PersistingGCA lizard.
##
## Fixes what makes eval_steps.png hard to read:
##   - samples the EARLY steps densely (all the growth happens before step ~40)
##   - nearest-neighbour upscale, so individual cells are visible as cells
##   - composites straight-alpha RGB over a checkerboard, so the organism's actual
##     extent is legible instead of dissolving into a black background
##   - separates the alpha/aliveness field into its own strip
##   - adds the quantitative view: alive count and RGBA loss against step
##   - writes an animation, because a filmstrip cannot show a process
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.io import ImageReadMode, read_image

REPO = r"C:\Users\krrish\documents\Neural-Cellular-Automata"
PGCA = os.path.join(REPO, "TorchModels", "PersistingGCA")
sys.path.insert(0, PGCA)
from persistingmodel import GCA

OUT = os.path.dirname(os.path.abspath(__file__))
CH, GRID = 16, 40
SCALE = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

## The alive-count curve is a sigmoid: flat until ~20, explosive 20-70, saturated after ~80.
## Sample the explosion, not the dead time before it.
EARLY = [8, 20, 28, 34, 40, 48, 58, 72]
LATE = [96, 150, 250, 400, 600, 1000]
MAX_STEP = max(LATE)
GIF_UNTIL = 240


def upscale(a, s=SCALE):
    return np.kron(a, np.ones((s, s, 1))) if a.ndim == 3 else np.kron(a, np.ones((s, s)))


def checkerboard(h, w, tile=2):
    ## subtle: enough to read transparency, not enough to compete with the organism
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // tile) + (xx // tile)) % 2 == 0, 0.74, 0.82)


def composite(state):
    ## straight (non-premultiplied) alpha over a grey checkerboard
    rgba = state[0, 0:4].clamp(0, 1).cpu().numpy()
    rgb, a = rgba[0:3].transpose(1, 2, 0), rgba[3][..., None]
    bg = checkerboard(*rgb.shape[:2])[..., None]
    return np.clip(rgb * a + bg * (1 - a), 0, 1)


def load_target():
    img = read_image(os.path.join(PGCA, "lizard.png"), ImageReadMode.RGB_ALPHA)
    img = T.Resize((28, 28))(img)
    return (T.Pad((GRID - 28) // 2)(img) / 255.0).to(DEVICE)  ## Pad bug fixed


def main():
    model = GCA().to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(PGCA, "lizard.pth"), weights_only=True, map_location=DEVICE)
    )
    target = load_target()

    torch.manual_seed(0)
    state = torch.zeros(1, CH, GRID, GRID)
    state[:, 3, GRID // 2, GRID // 2] = 1.0
    state = state.to(DEVICE)

    frames, alive_curve, loss_curve, gif = {}, [], [], []
    want = set(EARLY + LATE)
    with torch.no_grad():
        for step in range(1, MAX_STEP + 1):
            state = model(state)
            alive_curve.append((state[0, 3] > 0.1).sum().item())
            loss_curve.append((state[0, 0:4] - target).pow(2).mean().item())
            if step in want:
                frames[step] = (composite(state), state[0, 3].clamp(0, 1).cpu().numpy())
            if step <= GIF_UNTIL:
                gif.append(Image.fromarray((upscale(composite(state), 6) * 255).astype(np.uint8)))

    ## ---- filmstrip: growth on top, persistence below
    for tag, steps in (("growth", EARLY), ("persistence", LATE)):
        fig, ax = plt.subplots(2, len(steps), figsize=(2.1 * len(steps), 4.6))
        for c, s in enumerate(steps):
            rgb, alpha = frames[s]
            ax[0, c].imshow(upscale(rgb), interpolation="nearest")
            ax[0, c].set_title(f"step {s}", fontsize=10)
            ax[1, c].imshow(upscale(alpha), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
            ax[1, c].set_xlabel(f"alive {alive_curve[s - 1]}", fontsize=8)
            for r in (0, 1):
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])
        ax[0, 0].set_ylabel("RGB over\ncheckerboard", fontsize=9)
        ax[1, 0].set_ylabel("alpha\n(aliveness)", fontsize=9)
        fig.suptitle(
            f"PersistingGCA lizard -- {tag}"
            + ("  (growth is essentially over by step ~36)" if tag == "growth" else "  (stable, texture drifts)"),
            fontsize=12,
        )
        fig.tight_layout()
        p = os.path.join(OUT, f"lizard_{tag}.png")
        fig.savefig(p, dpi=105)
        plt.close(fig)
        print("saved", p)

    ## ---- quantitative view
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(1, MAX_STEP + 1)
    ax[0].plot(x, alive_curve, lw=1.2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("step (log)")
    ax[0].set_ylabel("alive cells (alpha > 0.1)")
    ax[0].set_title("growth saturates, then holds")
    ax[0].grid(alpha=0.3)
    ax[1].plot(x, loss_curve, lw=1.2, color="crimson")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("step (log)")
    ax[1].set_ylabel("RGBA MSE vs target")
    ax[1].set_title("no collapse out to 1000 steps")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, "lizard_curves.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print("saved", p)

    p = os.path.join(OUT, "lizard_growth.gif")
    gif[0].save(p, save_all=True, append_images=gif[1:], duration=60, loop=0)
    print(f"saved {p}  ({len(gif)} frames)")
    print(f"alive: step1={alive_curve[0]}  step36={alive_curve[35]}  step1000={alive_curve[-1]}")


if __name__ == "__main__":
    main()

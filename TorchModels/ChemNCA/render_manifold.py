## Make the manifold VISIBLE.
##
## Three figures, each answering a different question:
##   1. manifold_growth.png  -- does the latent alone decide what grows?
##      (4 latents, identical seed, identical weights-except-for-z)
##   2. manifold_interp.png  -- is the latent space CONTINUOUS, i.e. actually a manifold?
##      (a 2D bilinear sweep with the 4 trained latents at the corners; every
##       interior tile is a rule that was never trained on anything)
##   3. manifold_ablation.png -- is the latent load-bearing, or decoration?
##      (correct z vs shuffled z, same seeds)
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from manifold_proto import GRID, ManifoldGCA, make_targets, new_seed, rollout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 5
STEPS = [8, 20, 30, 40, 52, 64]
FINAL = 64
NGRID = 7


def upscale(a, s=SCALE):
    return np.kron(a, np.ones((s, s, 1))) if a.ndim == 3 else np.kron(a, np.ones((s, s)))


def checkerboard(h, w, tile=2):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // tile) + (xx // tile)) % 2 == 0, 0.74, 0.82)


def composite(state_1chw):
    rgba = state_1chw[0:4].clamp(0, 1).cpu().numpy()
    rgb, a = rgba[0:3].transpose(1, 2, 0), rgba[3][..., None]
    bg = checkerboard(*rgb.shape[:2])[..., None]
    return np.clip(rgb * a + bg * (1 - a), 0, 1)


def load():
    ck = torch.load(os.path.join(HERE, "manifold.pth"), weights_only=False, map_location=DEVICE)
    names = ck["names"]
    model = ManifoldGCA(len(names)).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, names


def run(model, z, steps, snap=None):
    ## z: (B, latent_dim). Identical seed for every sample -- z is the only variable.
    torch.manual_seed(0)
    state = new_seed(z.size(0)).to(DEVICE)
    w = model.generate_weights(z)
    out = {}
    with torch.no_grad():
        for i in range(1, steps + 1):
            state = model(state, w)
            if snap and i in snap:
                out[i] = state.clone()
    return state, out


def main():
    model, names = load()
    targets, _ = make_targets()
    z_train = model.latents.detach()
    k = z_train.size(0)

    ## ---- 1. per-latent growth from an identical seed
    _, snaps = run(model, z_train, FINAL, snap=set(STEPS))
    fig, ax = plt.subplots(k, len(STEPS) + 1, figsize=(2.0 * (len(STEPS) + 1), 2.05 * k))
    for r in range(k):
        for c, s in enumerate(STEPS):
            ax[r, c].imshow(upscale(composite(snaps[s][r])), interpolation="nearest")
            if r == 0:
                ax[r, c].set_title(f"step {s}", fontsize=10)
        tgt = np.concatenate(
            [targets[r, 0:3].permute(1, 2, 0).numpy(), targets[r, 3:4].permute(1, 2, 0).numpy()], axis=2
        )
        rgb, a = tgt[..., 0:3], tgt[..., 3:4]
        bgc = checkerboard(GRID, GRID)[..., None]
        ax[r, -1].imshow(upscale(np.clip(rgb * a + bgc * (1 - a), 0, 1)), interpolation="nearest")
        if r == 0:
            ax[r, -1].set_title("target", fontsize=10)
        ax[r, 0].set_ylabel(f"z{r}\n{names[r]}", fontsize=9)
        for c in range(len(STEPS) + 1):
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
    fig.suptitle(
        "One NCA, one seed, 4 latents. The latent is decoded into the update rule's WEIGHTS,\n"
        "so each row is a different cellular automaton -- nothing else differs.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "manifold_growth.png"), dpi=105)
    plt.close(fig)
    print("saved manifold_growth.png")

    ## ---- 2. the manifold itself: bilinear sweep, corners = trained latents
    us = torch.linspace(0, 1, NGRID, device=DEVICE)
    zs = []
    for v in us:
        for u in us:
            top = (1 - u) * z_train[0] + u * z_train[1]
            bot = (1 - u) * z_train[2] + u * z_train[3]
            zs.append((1 - v) * top + v * bot)
    zs = torch.stack(zs)
    final, _ = run(model, zs, FINAL)

    fig, ax = plt.subplots(NGRID, NGRID, figsize=(1.6 * NGRID, 1.6 * NGRID))
    for i in range(NGRID * NGRID):
        r, c = divmod(i, NGRID)
        ax[r, c].imshow(upscale(composite(final[i]), 4), interpolation="nearest")
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        corner = {(0, 0): names[0], (0, NGRID - 1): names[1],
                  (NGRID - 1, 0): names[2], (NGRID - 1, NGRID - 1): names[3]}.get((r, c))
        if corner:
            ax[r, c].set_title(corner, fontsize=8, color="crimson")
            for sp in ax[r, c].spines.values():
                sp.set_edgecolor("crimson")
                sp.set_linewidth(2)
    fig.suptitle(
        "A 2D slice of the NCA manifold. Corners are the 4 TRAINED latents (red).\n"
        "Every interior tile is a rule generated from a latent that was never trained --\n"
        "if the space is a manifold, these interpolate smoothly rather than snapping.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "manifold_interp.png"), dpi=105)
    plt.close(fig)
    print("saved manifold_interp.png")

    ## ---- 3. shuffle-z ablation: is z load-bearing?
    perm = torch.tensor([1, 2, 3, 0], device=DEVICE)
    ok, _ = run(model, z_train, FINAL)
    sh, _ = run(model, z_train[perm], FINAL)
    tg = targets.to(DEVICE)
    l_ok = (ok[:, 0:4] - tg).pow(2).mean().item()
    l_sh = (sh[:, 0:4] - tg).pow(2).mean().item()
    fig, ax = plt.subplots(2, k, figsize=(2.1 * k, 4.5))
    for r, (batch, lab) in enumerate(((ok, "correct z"), (sh, "shuffled z"))):
        for c in range(k):
            ax[r, c].imshow(upscale(composite(batch[c])), interpolation="nearest")
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(f"target: {names[c]}", fontsize=9)
        ax[r, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle(
        f"Shuffle-z ablation. MSE {l_ok:.4f} -> {l_sh:.4f} ({l_sh / max(l_ok, 1e-9):.1f}x worse).\n"
        "If the bottom row still matched its column, the latent would be decoration.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "manifold_ablation.png"), dpi=105)
    plt.close(fig)
    print(f"saved manifold_ablation.png   correct {l_ok:.5f} -> shuffled {l_sh:.5f}")

    ## ---- 4. animated walk around the manifold loop
    loop, frames = [0, 1, 3, 2, 0], []
    for a, b in zip(loop[:-1], loop[1:]):
        for t in torch.linspace(0, 1, 14, device=DEVICE):
            frames.append((1 - t) * z_train[a] + t * z_train[b])
    zloop = torch.stack(frames)
    fin, _ = run(model, zloop, FINAL)
    imgs = [Image.fromarray((upscale(composite(fin[i]), 6) * 255).astype(np.uint8)) for i in range(zloop.size(0))]
    p = os.path.join(HERE, "manifold_walk.gif")
    imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=90, loop=0)
    print(f"saved manifold_walk.gif  ({len(imgs)} frames)")

    ## per-latent quality
    with torch.no_grad():
        per = (ok[:, 0:4] - tg).pow(2).mean(dim=(1, 2, 3))
    for i, n in enumerate(names):
        print(f"   z{i} {n:16s} MSE {per[i].item():.5f}")


if __name__ == "__main__":
    main()

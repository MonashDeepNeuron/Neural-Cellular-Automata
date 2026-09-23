"""Unit tests for bacteria_live.py. No pytest in this env: `python test_bacteria_live.py`
runs every `test_*` function (optionally filtered: `python test_bacteria_live.py teacher`)."""
import sys
import time

import numpy as np
import torch

import bacteria_live as bl

CFG = bl.Config()
DEV = bl.DEVICE
G = CFG.grid


def disc(y, x, r, g=G):
    yy, xx = np.mgrid[0:g, 0:g]
    return (yy - y) ** 2 + (xx - x) ** 2 <= r * r


def masks_from(red=None, orange=None, black=None, blue=None, green=None, g=G):
    """(1,5,g,g) mask tensor from optional (g,g) bool arrays."""
    m = np.zeros((1, 5, g, g), np.float32)
    for i, a in enumerate((red, orange, black, blue, green)):
        if a is not None:
            m[0, i] = a
    return torch.from_numpy(m).to(DEV)


def relaxed(masks, iters=3000):
    f = bl.Fields(CFG, 1, G, G, DEV)
    f.relax(masks, iters)
    return f


## ---------------------------------------------------------------- fields

def test_fields_open_space():
    red = disc(24, 24, 3)
    f = relaxed(masks_from(red=red))
    A = f.c[0, 0].cpu().numpy()
    assert np.allclose(A[red], 1.0), "attractant not pinned to 1 on the red source"
    row = A[24, 27:46]                          ## walking away from the disc edge
    assert np.all(np.diff(row) < 0), f"A not monotone with distance: {row}"
    assert 0.15 <= A[24, 37] <= 0.6, f"A ten cells from the source edge = {A[24, 37]:.3f}"


def test_fields_orange_weaker_and_walls_shade():
    orange = disc(24, 10, 3)
    wall = np.zeros((G, G), bool)
    wall[:, 20:23] = True
    A = relaxed(masks_from(orange=orange, blue=wall)).c[0, 0].cpu().numpy()
    A_open = relaxed(masks_from(orange=orange)).c[0, 0].cpu().numpy()
    assert np.allclose(A[orange], 0.5), "decoy must pin at 0.5"
    assert 0 < A[24, 26] < 0.6 * A_open[24, 26], \
        f"a solid wall must shade (but not block) the chemical: {A[24, 26]:.4f} vs {A_open[24, 26]:.4f}"


def u_cup():
    """Cup opening LEFT toward x=0, back wall at x=30..32, food behind it at x=40."""
    wall = np.zeros((G, G), bool)
    wall[12:37, 30:33] = True                   ## back
    wall[12:15, 18:33] = True                   ## top arm
    wall[34:37, 18:33] = True                   ## bottom arm
    return wall


def test_fields_u_cup_is_a_trap():
    """The point of wall seepage: inside the cup, right at the back wall, the utility is
    a LOCAL MAXIMUM -- greedy ascent from the open side gets stuck there."""
    wall = u_cup()
    f = relaxed(masks_from(red=disc(24, 40, 3), blue=wall))
    A = f.c[0, 0].cpu().numpy()
    inner = A[16:33, 19:30]                     ## the cup's free interior
    y, x = np.unravel_index(inner.argmax(), inner.shape)
    y, x = y + 16, x + 19
    nb = A[y - 1:y + 2, x - 1:x + 2].copy()
    nb[wall[y - 1:y + 2, x - 1:x + 2]] = -1
    assert A[y, x] >= nb.max() - 1e-9, "no local maximum in the cup"
    assert x == 29, f"the trap should sit against the back wall, got x={x}"
    assert A[24, 5] < A[y, x], "the cup must be uphill from the open side"


def test_fields_slit_is_weaker_than_open():
    red = disc(24, 10, 3)
    wall = np.zeros((G, G), bool)
    wall[:, 20:23] = True
    wall[22:27, 20:23] = False                  ## a 5-cell slit
    A_slit = relaxed(masks_from(red=red, blue=wall)).c[0, 0].cpu().numpy()
    A_open = relaxed(masks_from(red=red)).c[0, 0].cpu().numpy()
    behind = A_slit[10, 30] / A_open[10, 30]
    through = A_slit[24, 30] / A_open[24, 30]
    assert behind < 0.8, f"the wall does not shade the field ({behind:.2f})"
    assert through > behind, f"the slit should pass more than the wall ({through:.2f} vs {behind:.2f})"


def test_fields_repellent_shorter_range():
    black = disc(24, 24, 3)
    f = relaxed(masks_from(red=disc(24, 24, 3), black=black))
    A, R = f.c[0, 0].cpu().numpy(), f.c[0, 1].cpu().numpy()
    assert np.allclose(R[black], 1.0)
    assert R[24, 35] < A[24, 35], "repellent should decay faster than attractant"


def test_fields_lag_moving_source():
    """Warm-started relaxation: a moved source is followed, but not instantly far away."""
    f = relaxed(masks_from(red=disc(24, 10, 3)))
    far_before = f.c[0, 0, 24, 44].item()
    m2 = masks_from(red=disc(24, 30, 3))
    f.relax(m2, CFG.field_iters)
    lagged = f.c[0, 0, 24, 44].item()
    f.relax(m2, 3000)
    settled = f.c[0, 0, 24, 44].item()
    assert far_before < lagged < settled, (far_before, lagged, settled)


def test_fields_sense():
    red = disc(24, 24, 3)
    m = masks_from(red=red)
    f = relaxed(m)
    s = f.sense(m)
    assert s.shape == (1, 3, G, G)
    assert torch.equal(s[:, 2], m[:, 0]), "third sense channel must be the raw food mask"


## ---------------------------------------------------------------- world

def test_world_masks_binary_and_discs():
    w = bl.World.scripted(CFG, [[dict(pos=(20, 20), vel=(0, 0), r=3, colour="orange"),
                                 dict(pos=(30, 35), vel=(0, 0), r=2, colour="green")]],
                          seeds=[(5, 5)], device=DEV)
    m = w.masks()
    assert m.shape == (1, 5, G, G)
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}
    assert np.array_equal(m[0, bl.ORANGE].cpu().numpy() > 0, disc(20, 20, 3))
    assert np.array_equal(m[0, bl.GREEN].cpu().numpy() > 0, disc(30, 35, 2))
    assert m[0, bl.RED].sum() == 0 and m[0, bl.BLUE].sum() == 0


def test_world_scripted_motion_and_bounce():
    w = bl.World.scripted(CFG, [[dict(pos=(24, 24), vel=(0.0, 0.25), r=2, colour="red")]],
                          seeds=[(5, 5)], device=DEV)
    for _ in range(8):
        w.step()
    assert torch.allclose(w.pos[0, 0], torch.tensor([24.0, 26.0], device=DEV)), w.pos[0, 0]
    for _ in range(400):                        ## long enough to hit a border many times
        w.step()
        x = w.pos[0, 0, 1].item()
        assert 2 - 1e-5 <= x <= G - 3 + 1e-5, f"person left the grid: x={x}"


def test_world_static_walls_are_blue():
    walls = np.zeros((1, G, G), bool)
    walls[0, 10:12, :] = True
    w = bl.World.scripted(CFG, [[]], seeds=[(30, 30)], walls=walls, device=DEV)
    assert np.array_equal(w.masks()[0, bl.BLUE].cpu().numpy() > 0, walls[0])


def test_world_random_seed_valid_and_deterministic():
    w1 = bl.World.random(CFG, 16, np.random.default_rng(3), DEV)
    w2 = bl.World.random(CFG, 16, np.random.default_rng(3), DEV)
    m = w1.masks()
    for b in range(16):
        y, x = w1.seed[b].tolist()
        assert m[b, bl.BLUE, y, x] == 0 and m[b, bl.BLACK, y, x] == 0, "seed on blue/black"
    for _ in range(50):
        w1.step(); w2.step()
    assert torch.equal(w1.masks(), w2.masks()), "same rng must give the same world"
    assert (m[:, bl.RED].flatten(1).sum(1) > 0).float().mean() >= 0.5, "red too rare"


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and pat in k]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Exception as e:                  ## report every failure, keep going
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

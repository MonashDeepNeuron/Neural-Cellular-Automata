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


def test_fields_orange_weaker_and_walls():
    orange = disc(24, 10, 3)
    wall = np.zeros((G, G), bool)
    wall[:, 30:33] = True
    f = relaxed(masks_from(orange=orange, blue=wall))
    A = f.c[0, 0].cpu().numpy()
    assert np.allclose(A[orange], 0.5), "decoy must pin at 0.5"
    assert A[wall].max() == 0.0, "attractant inside a wall"
    assert A[24, 36] < 1e-3, "attractant leaked through a solid wall"


def test_fields_slit_is_weaker_than_open():
    red = disc(24, 10, 3)
    wall = np.zeros((G, G), bool)
    wall[:, 20:23] = True
    wall[22:27, 20:23] = False                  ## a 5-cell slit
    A_slit = relaxed(masks_from(red=red, blue=wall)).c[0, 0].cpu().numpy()
    A_open = relaxed(masks_from(red=red)).c[0, 0].cpu().numpy()
    assert A_slit[10, 30] < 0.5 * A_open[10, 30], "the wall does not shadow the field"
    assert A_slit[24, 30] > 0, "no field through the slit"


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

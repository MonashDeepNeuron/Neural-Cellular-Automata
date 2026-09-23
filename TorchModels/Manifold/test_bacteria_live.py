"""Unit tests for bacteria_live.py. No pytest in this env: `python test_bacteria_live.py`
runs every `test_*` function (optionally filtered: `python test_bacteria_live.py teacher`)."""
import math
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


## ---------------------------------------------------------------- teacher

def run_teacher(masks, seed, n, steps, masks_fn=None):
    """Static (or scripted via masks_fn(t)) world; returns the teacher and its head track."""
    f = relaxed(masks)
    T = bl.Teacher(CFG, torch.tensor([seed], device=DEV), masks)
    track = [tuple(T.head[0].tolist())]
    for t in range(steps):
        m = masks if masks_fn is None else masks_fn(t)
        f.relax(m, CFG.field_iters)
        T.step(f, m, torch.full((1,), float(n), device=DEV))
        track.append(tuple(T.head[0].tolist()))
    return T, track


def test_teacher_climbs_to_food():
    T, track = run_teacher(masks_from(red=disc(24, 40, 3)), (24, 8), 0.5, 120)
    hy, hx = track[-1]
    assert (hy - 24) ** 2 + (hx - 40) ** 2 <= 9, f"head ended at {track[-1]}, not on the food"
    assert track[-1] == track[-10], "head should rest once on the food plateau"


def test_teacher_trapped_in_u_cup():
    T, track = run_teacher(masks_from(red=disc(24, 40, 3), blue=u_cup()), (24, 8), 1.0, 250)
    hy, hx = track[-1]
    assert hx == 29 and 15 <= hy <= 33, f"greedy head should be stuck at the cup's back, got {track[-1]}"
    assert track[-1] == track[-30], "a trapped head must not move"


def test_teacher_biomass_and_retreat():
    m = masks_from(red=disc(24, 44, 2))
    counts = {}
    for n in (0.25, 1.0):
        T, _ = run_teacher(m, (24, 4), n, 100)
        alive = torch.isfinite(T.age)
        counts[n] = alive.sum().item()
        assert T.age[alive].max().item() <= bl.life(n, CFG), "a cell outlived life(n)"
    assert counts[1.0] > 1.5 * counts[0.25], counts
    ## drop the nutrient: the tail dies on the very next step
    f = relaxed(m)
    T.step(f, m, torch.full((1,), 0.25, device=DEV))
    assert T.age[torch.isfinite(T.age)].max().item() <= bl.life(0.25, CFG)


def test_teacher_kills_under_black_and_blue():
    m = masks_from(red=disc(24, 44, 2))
    T, _ = run_teacher(m, (24, 4), 1.0, 60)
    body = torch.isfinite(T.age[0]).cpu().numpy()
    ys, xs = np.nonzero(body)
    y0, x0 = int(ys.mean()), int(xs.min()) + 2     ## the old tail
    tox = disc(y0, x0, 2)
    wall = np.zeros((G, G), bool)
    wall[:, 30:32] = True
    m2 = masks_from(red=disc(24, 44, 2), black=tox, blue=wall)
    f = relaxed(m2, 50)
    T.step(f, m2, torch.full((1,), 1.0, device=DEV))
    alive = torch.isfinite(T.age[0]).cpu().numpy()
    assert not alive[tox].any(), "cells survived under black"
    assert not alive[wall].any(), "cells survived under blue"
    assert alive.any()


def test_teacher_green_doubles_speed():
    red = disc(24, 46, 1)
    green = np.zeros((G, G), bool)
    green[18:31, :] = True                       ## a green carpet along the path
    _, slow = run_teacher(masks_from(red=red), (24, 4), 0.5, 20)
    _, fast = run_teacher(masks_from(red=red, green=green), (24, 4), 0.5, 20)
    d_slow, d_fast = slow[-1][1] - 4, fast[-1][1] - 4
    assert d_slow == 10 and d_fast == 20, f"head displacement slow {d_slow}, fast {d_fast}"


def test_teacher_extinct_when_crushed():
    m = masks_from(red=disc(24, 44, 2))
    T, _ = run_teacher(m, (24, 4), 0.5, 10)
    everything = np.ones((G, G), bool)
    m2 = masks_from(blue=everything)
    f = relaxed(m2, 10)
    T.step(f, m2, torch.full((1,), 0.5, device=DEV))
    assert not T.alive[0].item(), "teacher should be extinct"
    assert (T.rgba(m2)[0, 3] == 0).all()


def test_teacher_rgba_colours():
    m = masks_from(red=disc(24, 20, 3))
    T, _ = run_teacher(m, (24, 8), 1.0, 60)
    rgba = T.rgba(m)[0].cpu().numpy()
    alive = torch.isfinite(T.age[0]).cpu().numpy()
    assert np.array_equal(rgba[3] > 0, alive), "alpha must equal the body"
    on_food = alive & disc(24, 20, 3)
    assert on_food.any() and np.allclose(rgba[0:3, on_food].T, bl.BLOOM), "no bloom on the food"


## ---------------------------------------------------------------- model

def probe_model():
    """A model with a small non-zero predictor (at init every n decodes to the base rule,
    and the base rule outputs zero -- so sensitivities are legitimately 0 there)."""
    m = bl.BacteriaNCA(CFG).to(DEV)
    torch.nn.init.normal_(m.predictor.weight, std=1e-3)
    return m


def random_env(B=4, seed=0):
    return bl.Env(CFG, bl.World.random(CFG, B, np.random.default_rng(seed), DEV))


def test_model_shapes_and_kill_exact():
    m, env = probe_model(), random_env()
    n = torch.full((4,), 0.7, device=DEV)
    w1, b1, w2 = m.weights_for(n)
    assert w1.shape == (4, CFG.hidden, CFG.perception)
    assert b1.shape == (4, CFG.hidden) and w2.shape == (4, CFG.channels, CFG.hidden)
    state = bl.seed_state(env.teacher.rgba(env.masks), CFG)
    for _ in range(20):
        env.step(n)
        state = m(state, (w1, b1, w2), *env.inputs())
    kill = env.inputs()[1][:, 0] > 0
    assert state.abs().amax(1)[kill].max().item() == 0.0, "state non-zero under blue/black"


def test_model_senses_fields():
    m, env = probe_model(), random_env()
    n = torch.full((4,), 0.7, device=DEV)
    w = m.weights_for(n)
    state = bl.seed_state(env.teacher.rgba(env.masks), CFG)
    sense, kill, boost = env.inputs()
    with torch.no_grad():
        torch.manual_seed(1); a = m(state, w, sense, kill, boost)
        torch.manual_seed(1); b = m(state, w, sense.roll(5, -1), kill, boost)
    assert (a - b).abs().max().item() > 0, "rolling the sensed fields changed nothing"


def test_model_grad_reaches_predictor_at_init():
    m, env = bl.BacteriaNCA(CFG).to(DEV), random_env()
    n = torch.full((4,), 0.5, device=DEV)
    state = bl.seed_state(env.teacher.rgba(env.masks), CFG)
    for _ in range(8):
        env.step(n)
        state = m(state, m.weights_for(n), *env.inputs())
    loss = torch.nn.functional.mse_loss(state[:, :4], env.teacher.rgba(env.masks))
    g = torch.autograd.grad(loss, m.predictor.weight)[0].norm().item()
    assert math.isfinite(g) and g > 0, g


def test_model_boost_raises_fire_rate():
    m = probe_model()
    with torch.no_grad():
        m.base_w2.normal_(std=0.1)              ## make every cell's update non-zero
    s = torch.zeros(1, CFG.channels, G, G, device=DEV)
    s[:, 3] = 1.0
    sense = torch.zeros(1, 3, G, G, device=DEV)
    zero = torch.zeros(1, 1, G, G, device=DEV)
    w = m.weights_for(torch.tensor([0.5], device=DEV))
    with torch.no_grad():
        f0 = ((m(s, w, sense, zero, zero) - s).abs().amax(1) > 0).float().mean().item()
        f1 = ((m(s, w, sense, zero, torch.ones_like(zero)) - s).abs().amax(1) > 0).float().mean().item()
    assert abs(f0 - 0.5) < 0.05 and abs(f1 - 0.9) < 0.05, (f0, f1)


def test_run_world_records_frames():
    m, env = probe_model(), random_env(B=2)
    final, frames = bl.run_world(m, env, lambda t: 0.6, 12, record_every=4)
    assert final.shape == (2, CFG.channels, G, G) and len(frames) == 4
    t_rgba, n_rgba, masks = frames[-1]
    assert t_rgba.shape == n_rgba.shape == (2, 4, G, G) and masks.shape == (2, 5, G, G)


def test_env_cuda_graph_matches_eager():
    if not torch.cuda.is_available():
        return
    envs = [bl.Env(CFG, bl.World.random(CFG, 4, np.random.default_rng(5), DEV), use_graph=g)
            for g in (False, True)]
    n = torch.tensor([0.3, 0.6, 0.9, 1.0], device=DEV)
    for t in range(40):
        for e in envs:
            e.step(n)
        if t == 20:                             ## a slot reset mid-run must work with the graph
            for e in envs:
                e.reset_slots(np.array([1, 2]), np.random.default_rng(9))
    a, b = envs
    assert torch.equal(a.masks, b.masks), "masks diverged"
    assert torch.allclose(a.fields.c, b.fields.c, atol=1e-6), "fields diverged"
    assert torch.equal(a.teacher.age, b.teacher.age) and torch.equal(a.teacher.head, b.teacher.head)


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

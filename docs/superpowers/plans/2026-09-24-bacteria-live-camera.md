# Bacteria live-camera + greedy teacher — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the bacteria NCA on moving colour-coded worlds with a diffusion field and a greedy-ascent teacher, and drive it live from an OBS virtual camera.

**Architecture:** One source module `bacteria_live.py` (percent-format `# %%` cells: config, fields, world, teacher, model, training, verification, camera, live loop, CLI). A builder turns its cells into the self-contained notebook `bacteria_chemotaxis.ipynb` and appends notebook driver cells. Tests are plain-assert functions in `test_bacteria_live.py` (no pytest in this env), mirroring the repo's in-script check style.

**Tech Stack:** Python 3.14, torch 2.11 (CUDA, RTX 4060), numpy, OpenCV 5.0, PIL, nbformat/nbconvert.

Spec: `docs/superpowers/specs/2026-09-24-bacteria-live-camera-design.md`.

---

## File structure

- Create `TorchModels/Manifold/bacteria_live.py` — all library code + CLI (`--train`, `--verify`, `--figures`, `--live SRC`, `--smoke`).
- Create `TorchModels/Manifold/test_bacteria_live.py` — unit tests for fields, world, teacher, model, camera; `python test_bacteria_live.py` runs all.
- Create `TorchModels/Manifold/build_bacteria_notebook.py` — cells of `bacteria_live.py` (minus `# %% [cli]`) + driver cells → `bacteria_chemotaxis.ipynb`.
- Replace `TorchModels/Manifold/bacteria_chemotaxis.ipynb` (generated, then executed).
- Outputs: `TorchModels/Manifold/outputs/bacteria_live.pth`, `bact_live_*.gif/png`, `live_calib.json` (untracked dir).

## Shared interfaces (all tasks use exactly these)

- Colour channel order: `COLOURS = ("red", "orange", "black", "blue", "green")`; masks tensor `(B,5,H,W)` float {0,1}.
- `Fields(cfg, B, H, W, device)`: `.relax(masks, iters)` updates `.c` `(B,2,H,W)` (ch0 attractant A, ch1 repellent R); `.sense(masks)` → `(B,3,H,W)` = `[A, R, red]`.
- `World`: `World.random(cfg, B, rng, device)`, `World.scripted(cfg, walls, people, device)` where `people` = list per batch of dicts `{pos:(y,x), vel:(vy,vx), r, colour}`; `.masks()` → `(B,5,H,W)`; `.step()` advances people; `.seed` `(B,2)` long.
- `Teacher(cfg, seed, masks)`: `.age` `(B,H,W)` float, `inf` = dead; `.head` `(B,2)` long; `.alive` `(B,)` bool; `.step(fields, masks, n)`; `.rgba(masks)` → `(B,4,H,W)`.
- `BacteriaNCA(cfg)`: `.weights_for(n)`; `.forward(state, weights, sense, kill, boost)` where `kill`=`max(blue, black)` `(B,1,H,W)`, `boost`=green `(B,1,H,W)`.
- `Episode` helper `run_world(model, world, n_fn, steps, record_every)` → frames of `(teacher_rgba, nca_state, masks)`.
- Camera: `Calib` dataclass (json io), `segment(frame_bgr, calib, grid)` → `(5,g,g)` float; `FakeCamera(world)` with `.read()`; `LiveSession(model, cfg, calib, grid)` with `.step(frame)` → BGR panel image; `run_live(source, ...)`.

## Tasks

### Task 1: Config + diffusion fields
- [ ] Test: open space, single red disc → A pinned 1.0 on disc, monotonically decreasing with distance, A(10 cells) in [0.15, 0.6]; wall (blue) cells have A == 0; a cell behind a solid wall with a slit has lower A than the same distance in open space; R from a black disc decays faster than A (ℓ_R < ℓ_A).
- [ ] Implement `Config`, `Fields` (explicit scheme `c ← c + α·lap/8 − k·c`, `lap = conv8(c·free) − c·conv8(free)`, Dirichlet pins red=1/orange=0.5 on A and black=1 on R, zero in walls, `k = (3α/8)/ℓ²`).
- [ ] Run tests, commit.

### Task 2: World generator
- [ ] Test: masks binary with 5 channels; disc radius respected; people move by `vel` per step and bounce inside the grid; seed never on blue/black; scripted world reproduces given positions exactly; `random` with the same rng seed is deterministic.
- [ ] Implement `sample_walls` (from the old `sample_obstacles`), `World`.
- [ ] Run tests, commit.

### Task 3: Greedy teacher
- [ ] Tests: (a) open space → head moves toward red and reaches its disc edge; (b) U-cup wall with food behind → head stuck inside the cup (greedy, not BFS); (c) body cell count grows with n and `age <= life(n)` everywhere; lowering n kills old cells the next step; (d) cells under black/blue are dead; (e) green under the head → twice the head speed; (f) extinct when the whole body is killed.
- [ ] Implement `Teacher` (age grid CA: age+=1, stamp radius-2 disc of age 0 at the head, kill `age > life(n)`, kill under blue/black, head moves every `move_every` steps (1 on green) to argmax of `U = A − w_R·R` over self+8 nbrs with strict gain > eps, self first). RGBA colour = lerp(FRONT, CORE, age/life_max), BLOOM on red.
- [ ] Run tests, commit.

### Task 4: Model + episode runner
- [ ] Tests: weights shapes; kill=1 cells exactly 0 after a step; sense actually changes the output (probe with non-zero predictor); grad reaches predictor at init; boost raises the fraction of fired cells.
- [ ] Implement `BacteriaNCA`, `seed_state(teacher_rgba, cfg)`, `run_world`.
- [ ] Run tests, commit.

### Task 5: Training
- [ ] Smoke: `python bacteria_live.py --train --epochs 30` completes, loss finite, checkpoint loads.
- [ ] Implement `train()`: per epoch fresh `World.random`, relax fields, n schedule (const / ramp / jump), no-grad warm-up `T0 ~ U[0, T0max(epoch)]`, grad window 16–32 steps with MSE vs teacher every 8 steps, per-param grad norm, cosine LR, fixed held-out eval → best checkpoint.
- [ ] Commit.

### Task 6: Verification suite
- [ ] Implement `scenes(cfg)` (chase, decoy, decoy_only, toxin, trap, booster±) and `verify(model)` with the 8 acceptance tests from the spec, printing PASS/FAIL lines.
- [ ] Run on the smoke checkpoint (expect FAILs, no crashes). Commit.

### Task 7: Camera + live loop
- [ ] Tests: `segment(FakeCamera frame)` recovers each colour's mask with IoU ≥ 0.8; black heads on non-black people produce no black mask; homography calib from 4 corners round-trips; `Calib` json round-trips; `LiveSession.step` returns an image of the expected shape and the colony evolves; headless 300-frame smoke reports fps.
- [ ] Implement `Calib`, `segment`, `FakeCamera`, `calibrate()` (corner clicks + HSV trackbars), `LiveSession`, `run_live()` (keys q/space/click/r/+/-/a/c, auto-reseed).
- [ ] Run tests, commit.

### Task 8: Notebook
- [ ] Implement builder; generate notebook (markdown concept cells + code cells + driver cells: data figure, sanity, TRAIN flag, verify, figures, fake-camera smoke, LIVE flag).
- [ ] Commit.

### Task 9: Full training + verification + figures
- [ ] Full train (`CFG.epochs` ≈ 4000) in the background; monitor loss.
- [ ] Execute notebook with `TRAIN=False` via nbconvert so outputs (verification, figures, smoke) are embedded.
- [ ] Iterate on training if acceptance tests fail (record lessons in the spec).
- [ ] Commit notebook + code; update memory.

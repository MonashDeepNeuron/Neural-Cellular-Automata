# Bacteria chemotaxis — live camera, colour interferences, greedy teacher — design

**Date:** 2026-09-24
**Supersedes (in place):** `2026-08-30-bacteria-chemotaxis-design.md` — the notebook
`TorchModels/Manifold/bacteria_chemotaxis.ipynb` is rewritten; the old version lives in git.
**Checkpoint:** `TorchModels/Manifold/outputs/bacteria_live.pth` (the old `bacteria.pth` is kept).

## Goal

A bird's-eye camera (Sony a6000 → OBS → OBS Virtual Camera) watches people in coloured
clothing walking freely over a floor. Each colour is an interference in the bacteria's
world. A manifold NCA colony lives on a grid mapped onto that floor and reacts in real
time. The BFS shortest-path field is replaced by **real diffusion + greedy local ascent**:
the colony climbs the local gradient and can be trapped at local maxima, like real
chemotaxis.

## Colours

| Colour | Meaning | Mechanism |
|---|---|---|
| RED | food target | attractant source, strength 1.0; colony cells on food render a bloom colour |
| ORANGE | decoy | attractant source, strength 0.5; no feeding |
| BLACK | toxin | source of a separate repellent field `R`; colony cells under black die |
| BLUE | wall | hard: `state *= (1 − wall)` after every step (exactly 0, moving walls crush) |
| GREEN | booster | NCA fire rate `0.5 + 0.4·green`; teacher head moves every step instead of every 2 |

Other colours are ignored. Static random walls (`sample_obstacles()`) are also blue.

## Fields (replace BFS)

Attractant `A` and repellent `R`: diffusion with decay, relaxed by warm-started Jacobi
iterations `c ← c + α·(Σ_free-nbrs (c_n − c))/8 − k·c + s`, walls no-flux (a wall
neighbour contributes the cell's own value) and `c = 0` inside walls. Length scale
≈ 10 cells. Normalised to [0,1] by a fixed saturating map. A few iterations per NCA step
so the field **lags** moving sources; a long relaxation at episode start. `food` = the
raw red mask. The NCA senses `A`, `R`, `food` (identity + Sobel x/y).

## Teacher (the greedy search — supervision)

Deterministic, runs on the same fields as the NCA:

- **head**: one cell. Every `move_every` steps (2; 1 when on green) it moves to the best
  of its 8 neighbours by `U = A − 1.5·R`, never into a wall; if no neighbour beats the
  current cell it stays (**trapped at a local maximum**). Fixed tie-breaking order.
- **body**: the last `L(n) = max(1, round(n · L_max))` head positions (`L_max = 24`), each
  stamped as a radius-2 disc. `n` = biomass. Lowering `n` drops the oldest positions.
- removed: cells under walls or black.
- RGBA: alpha 1 on the body; colour shades dark (old) → bright (young); body cells on
  red food take a bloom colour.
- if the head is crushed by a wall/toxin it is kept at its last free position.

## World generator (training)

48×48. 0–3 static wall shapes; 1–6 people (discs radius 2–4), colour drawn with red
likely-present; each random-walks with speed 0–0.3 cells/step, bounces off borders, and
may appear/vanish mid-episode. Random seed cell (free, not under a person). Everything
is batched torch on the GPU.

## Model

`BacteriaNCA`, same manifold as before (`n` → env encoder → latent → DNA decoder →
generated weights as residual on a learned base rule, zero-init predictor, learned leak,
alive masking, replicate padding). Changes: perception over `cat[state, A, R, food]`
(`3(C+3)`), per-cell fire-rate map, hard wall zeroing after each step.

## Training

Truncated BPTT against the teacher, B = 8 fresh worlds per epoch:

- NCA and teacher start from the same seed and run in lockstep on identical fields.
- warm-up of `T0` steps under `no_grad` with the NCA on its own (curriculum: `T0` max
  grows 0 → ~400 over training), then a 16–32-step gradient window; MSE on RGBA vs
  the teacher at the window end.
- `n` schedule per episode: constant, slow ramp (up/down), or sudden jumps.
- per-parameter gradient normalisation, cosine LR, best checkpoint on a fixed held-out
  world set. ~4000 epochs; estimate 1–2 h on an RTX 4060 (report actual).

## Verification (held-out worlds + hand-built scenes)

1. **wall** — state inside blue exactly 0 at every step, including moving walls.
2. **imitation** — MSE vs teacher at t ∈ {100, 200, 400} on unseen moving worlds below
   threshold.
3. **chase** — a red person walks a path; the colony front ends within ~4 cells of them.
4. **food vs decoy** — equidistant red and orange: colony goes to red; orange alone:
   colony goes to orange.
5. **toxin** — a black person on the path: colony detours; alpha under black ≈ 0.
6. **trap** — U-wall with food behind it: colony stuck in the cup, like the teacher
   (greedy, not BFS).
7. **biomass** — colony mass increases with `n`; dropping `n` retreats with no ghost.
8. **booster** — the front covers more ground in a fixed number of steps on green.

## Camera adapter and live loop

- **Source**: `cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)` (OBS Virtual Camera), or a
  video file path, or `FakeCamera` (renders the synthetic world as an overhead image —
  used for automated tests).
- **Calibration** (saved to `outputs/live_calib.json`): click 4 floor corners →
  homography; per-colour HSV ranges (defaults + trackbars; red wraps hue); minimum blob
  area.
- **Segmentation**: warp → blur → HSV inRange → morphological open/close → drop
  components below a torso-sized area → area-fraction pool to the grid → mask where
  fraction > 0.3.
- **Gotchas documented**: black hair from overhead (area filter; toxin people wear large
  black tops), skin/wood/warm light reading orange/red (saturation threshold), shadows
  reading black (value threshold + area). a6000: manual white balance and exposure.
- **Loop**: frame → masks → diffusion update → 2 NCA steps → render. OpenCV window with
  warped camera | detected fields | colony, plus nutrient bar. Keys: `q` quit, `space`
  pause, click = place seed, `r` reseed, `+`/`-` nutrient, `a` auto-ramp, `c` recalibrate.
  Auto-reseed if the colony dies out.
- **Notebook**: `LIVE = False` flag cell; a headless fake-camera smoke test running the
  full path for ~300 frames and reporting fps.

## Changes made during implementation (2026-09-24)

- **Code location:** everything lives in `bacteria_chemotaxis.ipynb` (user preference — no
  companion `.py` module); unit tests are a sanity-check cell section (`run_tests()`).
- **Walls seep chemical** (`wall_perm = 0.25`) instead of being no-flux. With no-flux walls the
  diffusion field has no local maxima away from its sources (maximum principle + Hopf lemma on
  the Neumann boundary), so greedy ascent could never be trapped — it would be BFS in disguise.
  With seepage the far side of a U-wall facing the food is a genuine local maximum (tested).
- **Log-sensing:** the colony climbs `log A` (Weber's law — E. coli responds to relative
  change), normalised `la = 1 + log(A + 1e-4)/log(1e4)`; the NCA senses `la` too. With linear
  `A` the far field is ~0.02, so even a weak repellent tail dominated and the colony fled to the
  border. Repellent retuned to `rep_weight = 0.4`, `ell_rep = 3` — the colony keeps ~6 cells
  from a toxin and still reaches food behind it.
- **Training speed:** the env is ~250 tiny kernels per step; on this Windows laptop each launch
  costs ~45 µs, so `Env.step` is captured once as a **CUDA graph** (all state updates in place;
  the world RNG registered with the graph): 12.7 ms → 0.37 ms per step. Slot resets relax only
  the reset slots through a second graph (0.7 s → 0.03 s).
- **Persistent pool** instead of warm-up + window: `batch` running worlds carried across epochs,
  reset at random (often early, rarely late), so episodes reach hundreds of steps for the cost
  of one gradient window per epoch.
- **DAgger:** the first run plateaued at ~0.6× the empty-grid MSE — compounding divergence
  (once the NCA's colony is a few cells off, the independently running teacher can't be
  matched by any local rule). Before each window most slots re-anchor the teacher on the NCA's
  own colony (age read back from colour, head = youngest cell), so the loss asks "from where
  you are, what would the teacher do next".

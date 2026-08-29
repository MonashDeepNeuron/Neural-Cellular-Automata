# Bacteria chemotaxis on the NCA manifold — design

**Date:** 2026-08-30
**Deliverable:** `TorchModels/Manifold/bacteria_chemotaxis.ipynb` (self-contained, same style as `env_growth.ipynb`), checkpoint at `TorchModels/Manifold/outputs/bacteria.pth`.

## Goal

Emulate a bacteria colony growing from a seed toward a food target on a grid with
obstacles it must avoid — everything driven by external signals, in the manifold
framework built by `env_growth.py`:

- **Nutrient `n` ∈ [0,1]** — global scalar. The env-encoder maps it onto the manifold
  latent and the DNA decoder generates the update rule's weights, exactly as in
  `env_growth.py`. The nutrient controls **progress**: what fraction of the geodesic
  distance from seed to target the colony has covered. `n≈0` → small blob at the seed;
  `n=1` → plume reaching the target. Ramping `n` at eval animates the journey.
- **Attractant `A(x,y)` ∈ [0,1]** — spatial field, `A = exp(−D_t/τ)` where `D_t` is
  the **geodesic** (around obstacles, 8-connected BFS) distance to the target; `A = 0`
  inside obstacles. **Sensed**: concatenated to the state before perception, so the
  rule receives `A` and its Sobel gradients and *learns* chemotaxis. Because the field
  is geodesic, its gradient already points around obstacles.
- **Obstacles** — hard physics, via the chem-gate mechanism from `env_growth.py`:
  `gate = 1 − obstacle_mask` multiplies the whole per-cell state transition, so cells
  inside obstacles are exactly frozen, by construction. The rule cannot ignore them.

The rule must **generalize to unseen layouts**: every training episode samples a fresh
random layout; verification runs on held-out layouts never seen in training.

## Environment generation

- Grid 48×48, obstacles sampled per episode from {wall with a gap (H/V), circular
  blobs, bars}, 1–3 shapes, kept off a 2-cell border.
- Target: uniform over free cells (3-cell margin from border). Seed: free cell with
  geodesic distance to target in [14, 26] steps (guarantees a journey and
  reachability; resample layout if no valid seed exists).
- `D_t`, `D_s`: 8-connected unit-cost BFS over free cells from target / seed.
- Attractant `A = exp(−D_t/τ)`, τ = 10, zero on obstacles.

## Procedural colony targets (supervision)

Ideal colony at nutrient `n` = a plume along near-shortest paths:

- corridor: free cells with `D_s + D_t ≤ D_st + slack` (slack = 6, `D_st` = geodesic
  seed→target distance) — naturally widens in open space, hugs the detour around
  obstacles;
- progress: corridor ∩ `{D_t ≥ (1−n)·K}` with the fixed constant `K = d_max`, plus a
  home blob `{D_s ≤ r0}` (r0 = 2) that is always present. The front contour is the
  iso-attractant line `A*(n) = exp(−(1−n)K/τ)` — a pure function of the nutrient, so a
  cell can decide **locally** whether to keep advancing. (A fraction-of-`D_st`
  parameterization would make the stopping contour depend on the journey length, which
  a cell cannot sense — inconsistent supervision.)

RGBA target: colony cells get a yellow-green bacteria color, alpha 1. MSE loss on
RGBA channels, as in `env_growth.py`.

## Model

`BacteriaNCA` — same manifold architecture as `EnvNCA` (env-encoder → latent →
DNA-decoder → zero-init predictor as a residual on a learned base rule; learned
log-leak step scale; alive masking; fire-rate stochastic update; replicate padding).
One change: perception is computed over `cat([state, A], dim=1)`, widening the
perception vector from `3C` to `3(C+1)` (identity + Sobel x/y of the attractant too).
The obstacle gate enters exactly like `chem` in `EnvNCA.forward`.

## Training

Reuses the `env_growth.py` machinery, adapted to per-episode layouts:

- batch = B random layouts (B ≈ 8), each with its own attractant and gate;
- nutrient sampled **continuously**, `n ~ U[0.05, 1]` (targets exist for any `n`, no
  discrete stages needed);
- two-segment episodes: seg1 from seed at nutrient `a`, loss vs target(`a`); seg2
  from the detached state at nutrient `b` (both directions — advance AND retreat),
  loss vs target(`b`); state-noise perturbation half the time;
- slow-ramp episodes (truncated BPTT on a genuine nutrient ramp) at ~1/3 probability;
- rollout lengths scale with the journey: seg1 ≈ `2·n·D_st + [20, 40]` steps
  (front speed is < 1 cell/step under fire-rate 0.5);
- per-parameter gradient normalisation, cosine LR, best-checkpoint tracking on a
  **fixed held-out eval set** of layouts (grow at several `n` + a ramp).
- budget: ~3000 epochs, ballpark 30–60 min on an RTX 4060.

## Verification (acceptance tests, on held-out layouts)

1. **freeze** — 50 steps with the colony grown: max |Δstate| inside obstacles is
   exactly 0.
2. **stages** — grow at n ∈ {0.25, 0.5, 0.75, 1.0} on unseen layouts; per-level MSE
   below threshold.
3. **ablation** — swap in the attractant of a *different* target: MSE vs the true
   target ≥ 10× worse, i.e. the colony provably follows the sensed field.
4. **reached** — at `n=1.0`, mean colony alpha in a 2-cell disc around the target
   above threshold on every held-out layout.
5. **ramp** — continuous nutrient ramp 0→1 changing every step still ends with the
   target reached.
6. **retreat** — after reaching, drop the nutrient; colony regresses to a clean blob
   near the seed (no ghost alpha along the abandoned plume).

## Figures

- sample layouts + procedural targets grid (before training — sanity of the data);
- ramp GIFs on 3 unseen demo layouts (wall-with-slit, two pillars, S-corridor):
  obstacles dark, attractant tinted, nutrient bar below;
- targets-vs-grown stage grid; ablation grid; retreat GIF.

## Notebook layout

Mirrors `env_growth.ipynb`: title/concept markdown → setup → config → rendering
helpers → environment generation (+ data figure) → model → sanity checks →
checkpoint io → training (+ `TRAIN` flag cell) → verification → figures → recipe
markdown for adapting it.

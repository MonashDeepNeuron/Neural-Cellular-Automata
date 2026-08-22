# NCA-Manifold notebook — design

**Date:** 2026-08-23
**Branch:** NCA-Manifold
**Deliverable:** `TorchModels/Manifold/nca_manifold.ipynb`

## Purpose

Turn the working NCA-Manifold prototype (`TorchModels/ChemNCA/manifold_proto.py` +
`render_manifold.py`) into a reusable, debuggable notebook that can grow a manifold over
*any* set of targets, and that is fit to merge into `main` as a first-class component.

Two problems with the current prototype motivate this:

1. It is a pair of CLI scripts. You cannot step into the model, inspect an intermediate
   state, or re-run one figure without re-running everything.
2. Its target set is four hard-coded synthetic shapes and `K = 4` is baked into the
   interpolation figure. Training it on anything else means editing source.

## Scope

In scope: one self-contained notebook, three target loaders, `K`-agnostic model and
figures, a sanity-check cell, live training feedback, checkpoint save/load.

Out of scope: sample-pool / persistence training, the chemical field, division mechanics,
WebGPU export. The notebook grows static shapes, same as the prototype.

## Location

`TorchModels/Manifold/` — a new sibling of `PersistingGCA`, `Texture`, `RebuildingGCA`.

`ChemNCA/README.md` declares itself "untracked scratch work … treat everything here as
disposable". A component intended to merge into `main` should not inherit that framing.
`ChemNCA` keeps its prototype; this is the reusable version.

## Architecture

Fully self-contained: model, training loop and every visualisation live in notebook
cells. This duplicates logic from `manifold_proto.py`, which is accepted — the point is
that every internal is editable in place without a kernel restart, and the prototype it
duplicates is explicitly disposable.

### Mechanism (unchanged from the prototype)

A latent `e` is decoded into the **weights** of the per-cell update rule,
`κ(e) = P(D(e))`, computed once per rollout. Each latent is therefore a different
cellular automaton, not a different input. All organisms grow from an identical seed, so
the latent is the only possible explanation for the difference between them.

Two deliberate deviations from Hernandez Ruiz et al. (CVPR 2021, arXiv 2006.12155), both
carried over:

- Weights are generated as a **residual on a learned base rule**, predictor zero-init, so
  training starts from one well-conditioned shared rule instead of noise.
- **No InstanceNorm.** It reduces over all of `H*W`, which destroys the locality that
  makes an NCA an NCA, and cannot be expressed in the WebGPU shader. Per-cell channel RMS
  norm is available behind a config flag instead.

### Generalisation over `K`

Nothing assumes four latents. `K` is inferred from the target tensor's leading dimension.

- Shuffle-z ablation uses a cyclic roll, valid for any `K >= 2`.
- The interpolation sweep takes four caller-chosen latent indices as corners, and falls
  back to a 1-D interpolation strip between two latents when `K < 4`.
- The growth filmstrip and walk GIF iterate over whatever `K` is.

### Config

One `CFG` cell holds every constant that is currently a module-level global in
`manifold_proto.py`: `CHANNELS`, `HIDDEN`, `GRID`, `LATENT_DIM`, `DNA_HIDDEN`,
`ALIVE_THRESHOLD`, `FIRE_RATE`, `EPOCHS`, `LR`, `STEPS_MIN`/`STEPS_MAX`, `SEED`,
`RMS_NORM`, `LEARN_LEAK`.

`LEARN_LEAK` exists because `ChemNCA/README.md` records that the learnable leak factor
`exp(ρ)` never moved off its 0.100 initialisation and is degenerate with `‖W2‖`. The
parameter is kept — removing it changes the mechanism — but can be frozen to a constant.

## Target loaders

Three functions, one return contract: an `(K, 4, GRID, GRID)` float RGBA tensor in
`[0, 1]`, plus a `list[str]` of names.

| loader | source | notes |
|---|---|---|
| `targets_from_shapes()` | four synthetic shapes | default; trains in minutes, keeps debugging cheap |
| `targets_from_folder(path)` | any directory of images | one latent per image, resized to `GRID` |
| `targets_from_emoji([...])` | PIL-rendered glyphs | quick experiments with no files to find |

All three **alpha-premultiply** RGB, matching the synthetic convention where colour is
`rgb * mask`. Loaders that read files must fail with a clear message naming the path when
the directory is empty or a font is missing, not with a `FileNotFoundError` from three
frames down.

A preview cell renders the loaded targets before any training happens.

## Cell order

1. Markdown intro — the mechanism, the two deviations, what the notebook produces
2. Imports and device
3. `CFG`
4. Target loaders
5. Target selection + preview figure
6. Model (`ManifoldNCA`), seed, rollout
7. **Sanity checks**
8. Training loop
9. Verification — shuffle-z ablation
10. Visualisation — growth, interpolation, ablation, walk GIF
11. Save / load + "use this for your own targets" recipe

## The sanity-check cell

This is the cell that does not exist in the current scripts and is the main reason a
notebook is worth building. Before any training time is spent it asserts:

- one forward step preserves state shape `(K, CHANNELS, GRID, GRID)` and produces finite values
- generated weight tensors have the expected shapes and batch over `K`
- total and generated parameter counts, printed
- **gradient reaches `model.latents`** with non-zero norm

The last check catches the failure mode where the latent silently becomes decoration —
the exact thing the shuffle-z ablation exists to detect after the fact, caught in seconds
instead of after a full training run.

## Training

Carried over from the prototype: one sample per latent from identical seeds, random
rollout length in `[STEPS_MIN, STEPS_MAX]`, MSE against RGBA targets, cosine LR schedule,
best-checkpoint tracking.

Two details that are load-bearing and must survive the port:

- **Per-parameter gradient normalisation excludes scalars.** For a scalar parameter,
  dividing by its own gradient norm yields exactly `±1` every step. Scalars go in a
  separate optimiser group at a fixed low LR.
- **The stochastic fire mask is active in eval as well as train**, and is shared across
  channels within a cell.

Added: a live loss plot and a per-latent loss readout that update during training, so a
run that is going nowhere is visible immediately.

## Verification

The shuffle-z ablation is the acceptance test. Rolling the latents by one and re-running
should make the loss dramatically worse; the prototype measured 2663×. If shuffling is
cheap, the latent is not load-bearing and the manifold is not real.

## Success criteria

- Notebook runs top to bottom on the default synthetic targets and reproduces the
  prototype's result: near-zero final loss and a shuffle-z ablation that is orders of
  magnitude worse.
- Pointing `targets_from_folder` at a directory of images trains without editing any cell
  other than the target-selection cell.
- All four figures render for `K != 4`.

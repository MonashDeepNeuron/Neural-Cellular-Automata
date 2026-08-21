# Plan: Isolate simulator from training + transparent overlay rendering

Status: DRAFT — awaiting review. Nothing implemented yet.

## Context

The persisting NCA now trains and holds its lizard shape (dropout mismatch and
training bugs fixed earlier — see `WHY_I_CHANGED_THIS.md`). But everything besides
the model lives in one monolithic `train.py` — seed creation, rollout loop,
matplotlib rendering, training — and the three diagnostic scripts each hand-roll
their own copies of seed + rollout. Goals:

1. **Isolate grid + model** into reusable modules — a `Simulator` you can seed,
   reset, step, and record anywhere (training, diagnostics, standalone rendering).
2. **Overlay rendering with true transparency**: RGBA outputs where dead cells are
   fully transparent, so the growing pattern can be layered over anything.
   Python side only — do **not** touch `web/`.
3. A **standalone runner** that loads `lizard.pth` and emits overlay
   frames/animation with no training code involved.
4. `train.py` keeps working with training behaviour unchanged, consuming the new
   modules.

Environment verified: Pillow 12.1.1 (APNG write support), torch 2.11.0+cu126.

Key facts that shape the design:

- State channels: 0-2 RGB, 3 alpha. The loss trains channels 0:4 directly against
  a **straight-alpha** RGBA target (`load_image` uses `ImageReadMode.RGB_ALPHA`),
  so state RGB is straight (not premultiplied) and alpha clamps to [0,1] for display.
- Alive threshold is `alpha > 0.1` (`GCA.alive_mask`, `persistingmodel.py:123-130`).
- The stochastic update mask fires in eval too, so rollouts need
  `torch.manual_seed` for reproducibility.
- **Globals gotcha:** `update_pass` reads `LOSS_FN`; `standard_train`/`pool_train`
  read `UPDATES_RANGE`, `MODEL`, `EPOCHS`, `BATCH_SIZE`, `POOL_SIZE` as train.py
  module globals assigned in `__main__`. These functions must stay in `train.py`;
  only pure functions move out.

## Step 1 — New `simulator.py`

Move from `train.py`, behaviour-identical:

- `GRID_SIZE = 40`, `CHANNELS = 16` module constants (train.py imports them).
- `new_seed(batch=1, grid_size=GRID_SIZE, channels=CHANNELS)` — keep both branches:
  int → fresh zeros with centre alpha=1; Tensor → in-place reset (used by
  `standard_train`), deriving the centre from `batch.size(2)//2, batch.size(3)//2`.
- `forward_pass(model, state, updates, record=False)` — identical loop; allocate
  the record buffer as `torch.empty(updates, *state.shape[1:])` instead of
  hardcoding module constants (same CPU-buffer <- CUDA-state assignment pattern
  as today).
- New `Simulator` class:
  - `__init__(self, model=None, grid_size=GRID_SIZE, channels=CHANNELS, device=None)`
    — device defaults to cuda-if-available, model defaults to `GCA()`, calls
    `model.eval()`; `self.state = None` until reset.
  - `from_weights(cls, path, **kwargs)` classmethod —
    `torch.load(weights_only=True, map_location=device)`.
  - `reset(self, batch=1)` — `self.state = new_seed(batch, ...).to(device)`.
  - `step(self, n=1)` — under `torch.no_grad()`, returns `self.state`.
  - `run(self, updates, record=False, record_every=1)` — wraps `forward_pass`;
    returns `(frames, C, H, W)` CPU tensor when recording, else final state.

Match existing code style (plain functions, module constants, `##` comments).

## Step 2 — New `render.py`

- `ALPHA_THRESHOLD = 0.1` (matches the alive threshold).
- `state_to_rgba(state, alpha_threshold=ALPHA_THRESHOLD) -> np.ndarray` — accepts
  `(C,H,W)` or `(1,C,H,W)`; channels 0:4 clamped to [0,1]; RGB used as-is
  (straight alpha), A scaled to 0-255; force `[0,0,0,0]` wherever
  alpha <= threshold (dead cells fully transparent, RGB zeroed). Returns
  `(H,W,4)` uint8. Above threshold keep the continuous trained alpha — don't
  binarise; the model was optimised to reproduce the target's alpha channel.
- `save_png(rgba, path, scale=1)` — PIL RGBA, optional `Image.NEAREST` upscale
  (40x40 is tiny; nearest keeps the cell look).
- `save_apng(frames, path, fps=15, scale=1)` —
  `imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000/fps), loop=0)`.
  Full 8-bit alpha per frame.
- `save_gif(frames, path, fps=15, scale=1)` — binary-transparency fallback:
  per-frame adaptive palette (255 colours), reserved index 255 where A <= 128,
  `transparency=255, disposal=2, loop=0`. GIF alpha is 1-bit — this is the
  compatibility option, not the default.
- `composite_over_checkerboard(rgba, tile=8) -> np.ndarray` — straight-alpha blend
  over a grey checkerboard, returns RGB uint8, used for verification output.
- `visualise(...)` — moved **verbatim** from `train.py` (with the
  `matplotlib.use("Agg")` backend selection), so training/diagnostic visuals are
  unchanged.

Default outputs: `overlay.png` (final frame, full alpha) + `overlay_anim.png`
(APNG — browsers auto-detect it; degrades to showing frame 1 elsewhere). GIF only
behind a flag.

## Step 3 — New `run_overlay.py` (standalone runner)

Imports only `simulator` and `render` (not `train`). CLI:

```
python run_overlay.py [--weights lizard.pth] [--steps 600] [--every 4]
                      [--out overlay] [--fps 15] [--scale 8]
                      [--gif] [--check] [--seed N]
```

Flow: optional `torch.manual_seed` → `Simulator.from_weights(args.weights)` →
`reset(1)` → `run(steps, record=True, record_every=every)` → `state_to_rgba` per
frame → `save_png` (last frame), `save_apng` (all frames), optional `--gif`,
`--check` writes `overlay_check.png` (checkerboard composite). Prints output paths.

## Step 4 — Modify `train.py` (minimal churn, training behaviour unchanged)

- Delete local `visualise`, `new_seed`, `forward_pass`, and the
  `GRID_SIZE`/`CHANNELS` constants.
- Add `from simulator import GRID_SIZE, CHANNELS, new_seed, forward_pass` and
  `from render import visualise`.
- Delete the `GRID_SIZE = 40` reassignment near line 511 (it reassigns the
  identical value; keep its comment).
- Everything else (`apply_lr_decay`, `update_pass`, `standard_train`,
  `pool_train`, `initialiseGPU`, `load_image`, `__main__`) stays untouched so the
  globals mechanism keeps working.

## Step 5 (optional, do last or skip) — Migrate diagnostics

`diag_train_vs_eval.py`, `eval_preview.py`, `diag_grid_size.py` → use
`Simulator.from_weights` instead of hand-rolled seed/rollout (`diag_grid_size`
passes `grid_size=g`; `diag_train_vs_eval` toggles `sim.model.train(flag)` between
rollouts). They work as-is, so this is a nice-to-have.

## Verification

1. Import smoke: `python -c "import simulator, render, run_overlay"`
2. `python run_overlay.py --weights lizard.pth --steps 600 --every 4 --scale 8 --check --seed 0`
3. Transparency checks:
   - `python -c "from PIL import Image; im=Image.open('overlay.png'); print(im.mode, im.getpixel((0,0)))"`
     → expect `RGBA (0, 0, 0, 0)`
   - `python -c "from PIL import Image; im=Image.open('overlay_anim.png'); print(im.is_animated, im.n_frames)"`
     → expect `True 150`
4. Visual: view `overlay_check.png` (lizard over checkerboard) and
   `overlay_anim.png` in a browser.
5. Training smoke: **back up weights first** (`Copy-Item lizard.pth lizard.pth.bak`)
   because train.py saves over `lizard.pth`; then `python train.py --epochs 8`;
   confirm both phases run and `pool.gif`/`train.gif`/`train.png`/`loss.png`/
   `weights.bin` regenerate; restore `lizard.pth.bak` after.
6. `python eval_preview.py` still works.

## Files

- New: `simulator.py`, `render.py`, `run_overlay.py`
- Modified: `train.py` (imports swap + deletions only); optionally the three
  diagnostic scripts
- Untouched: `persistingmodel.py`, `model.py`, `learning_rate_adjuster.py`,
  everything under `web/`

## Risks

- The `lizard.pth` overwrite during the training smoke run is the only destructive
  hazard — the backup/restore in verification step 5 covers it.
- Preserve the exact CPU-buffer <- CUDA-state recording pattern in `forward_pass`
  to avoid memory-behaviour changes on 600-frame rollouts.

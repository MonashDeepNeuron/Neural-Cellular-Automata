# ChemNCA — scratch prototypes and verification output

**Status: untracked scratch work, nothing committed.** This directory holds the visual
verification of the uncommitted `PersistingGCA` fixes plus a working prototype of the
NCA-Manifold mechanism. It is the planned home for the real ChemNCA work; treat everything
here as disposable until that lands.

Everything is runnable in place. Requires the repo's conda env (torch 2.11.0+cu126 verified).

## The manifold prototype

`manifold_proto.py` — a minimal but paper-faithful implementation of *Neural Cellular Automata
Manifold* (Hernandez Ruiz, Vilalta & Moreno-Noguer, CVPR 2021, arXiv 2006.12155). A latent `e`
is decoded into the **weights** of the per-cell update rule, `kappa(e) = P(D(e))`, so each
latent is a different cellular automaton rather than a different input.

```powershell
python manifold_proto.py --epochs 4000     # ~10 min on an RTX 4060, writes manifold.pth
python render_manifold.py                  # writes the three manifold_*.png + the walk gif
```

> The reusable version of this lives in `TorchModels/Manifold/nca_manifold.ipynb` — same
> mechanism, but `K`-agnostic, with pluggable target loaders so it can be trained on any
> set of images rather than the four shapes hard-coded here. Use that one for new work;
> these two scripts stay as the reference the notebook was verified against.

Config: 4 latents (dim 8) -> `Linear(8,64)+ReLU` -> `Linear(64,8320)`, zero-init, emitting all
8,320 update-net weights as a residual on a learned base rule. 549,729 params total.

Two deliberate deviations from the paper, both documented in the source: weights are generated
as a **residual** on a learned base rule (so training starts from one well-conditioned rule
instead of noise), and there is **no InstanceNorm** (it reduces over all of H*W, which destroys
the locality that makes an NCA an NCA, and it cannot be implemented in the WebGPU shader).

### Results

| | |
|---|---|
| final loss, all 4 targets | 0.06589 -> **0.00000** (per-latent max 0.00006) |
| shuffle-z ablation | 0.00002 -> 0.04622, **2663x worse** |
| latent interpolation | smooth in shape and colour, no snapping |
| leak factor `exp(rho)` | stayed at 0.100 — **inert**, drop it (degenerate with `\|\|W2\|\|`) |

## What each output shows

| file | what it answers |
|---|---|
| `outputs/manifold_growth.png` | Does the latent alone decide what grows? 4 latents, identical seed, identical weights except `z`. |
| `outputs/manifold_interp.png` | **Is it actually a manifold?** 7x7 bilinear sweep, trained latents at the corners; every interior tile is a rule generated from a latent never trained on anything. |
| `outputs/manifold_ablation.png` | Is `z` load-bearing or decoration? Under shuffled `z` the model grows the *shuffled* latent's shape. |
| `outputs/manifold_walk.gif` | Animated loop z0 -> z1 -> z3 -> z2 -> z0 around the manifold. |
| `outputs/lizard_growth.png` | The existing lizard growing, sampled on the window where growth actually happens (steps 8-72, not 1-36). |
| `outputs/lizard_persistence.png` | Steps 96-1000. Form and alive count hold. |
| `outputs/lizard_curves.png` | Why the old render was unreadable: alive count is flat until ~step 20, explodes 20-70, saturates. |
| `outputs/lizard_growth.gif` | 240 frames. A filmstrip cannot show a process. |
| `outputs/drift_map.png` | Where long rollouts degrade. Localised, not global. |
| `outputs/grid_scale.png` | The 40x40-trained rule at 40/64/96/128. Survives everywhere, shape quality falls off. |

## Verification of the uncommitted PersistingGCA fixes

```powershell
python verify_fixes.py
```

| claim | measured |
|---|---|
| dropout fix: `train()` == `eval()` | `max\|delta\| = 0.0` over 200 steps |
| determinism | same seed `0.0`, different seed `2.77` |
| `Pad` bug (2nd positional arg is `fill`) | border alpha `0.0235` vs `0.0`, MSE floor `2.82e-4` |
| scalar grad-norm hazard | a scalar param's normalised grad is exactly `1.000000` |

Persistence is **better** than `WHY_I_CHANGED_THIS.md` claims: RGBA MSE is 0.0149 @200,
0.0160 @600, 0.0161 @1000 — no collapse.

But `drift_map.py` shows the drift is a **growing halo, not decay**. Split by the target's
support (361 of 1600 px):

| step | mean alpha inside | mean alpha outside | alive inside | alive outside |
|---|---|---|---|---|
| 96 | 0.7378 | 0.0187 | 350 | 82 |
| 1000 | 0.7305 | 0.0341 | 348 | 134 |

The organism is frozen (350 -> 348 cells over 900 steps). *All* of the 432 -> 482 alive growth
is outside the shape. Relevant to ChemNCA because an unbounded halo corrupts `N = sum nu` as a
cell count.

## Not demonstrated yet

This prototype grows **static shapes**. The chemical field, the division mechanics and the
process objective are all unbuilt — that is still the hard part. See the plan for staging.

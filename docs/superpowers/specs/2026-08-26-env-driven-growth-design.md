# Environment-driven growth on the NCA manifold — design

**Date:** 2026-08-26
**Branch:** NCA-Manifold
**Deliverable:** `TorchModels/Manifold/env_growth.py` (+ rendered outputs)

## Purpose

The manifold notebook grows static shapes selected by free latents. This component makes
growth a *process driven by external signals*: a single seed cell develops into a whole
organism, and both **what** it grows into and **whether** it can grow at all are decided
by the environment, not by an index.

Concrete example: a plant with four developmental stages — seed blob, sprout, leafy
plant, flowering plant — driven by a nutrient level, and fuelled by a chemical field.

## The two external signals

| signal | shape | mechanism | what it controls |
|---|---|---|---|
| nutrient `n(t)` | global scalar in `[0,1]`, time-varying | environment encoder `Z(n)` replaces the free latents: `kappa = P(D(Z(n)))` | *what* grows — the developmental stage (morphology) |
| chemical `chem(x,y)` | spatial field in `[0,1]` | hard multiplicative gate on the WHOLE per-cell transition: `state' = state + (nca_step(state) - state) * chem` | *whether/where* growth happens at all |

The nutrient reaches every channel through the generated weights (the whole update rule
is a function of it). The chemical gates the entire state transition of every channel
identically — growth *and* death, because the alive-mask kill is not idempotent and is
also a state change the chemical fuels (gating only `ds` lets a chem=0 cell still die on
the next step; found and fixed during verification). `chem = 0` means a cell is exactly
frozen, by construction, with no training required. The chemical is deliberately **not**
in the perception vector: it is physics, not an input the rule can learn to ignore.

## Model

`EnvNCA` = `ManifoldNCA` from the notebook with two changes:

1. `latents` (free embeddings) are replaced by `env_encoder`:
   `Linear(1, 64) -> ReLU -> Linear(64, latent_dim)`. Everything downstream
   (`dna_decoder`, zero-init `predictor`, residual-on-base-rule weights, replicate
   padding, shared-across-channels fire mask active in eval) is carried over unchanged.
2. `forward` takes a `chem` field `(B,1,H,W)` and multiplies `ds` by it.

Weights are regenerated whenever the environment changes (cheap: two small MLPs).

## Targets — plant stages

Four synthetic RGBA targets on the 40x40 grid, drawn programmatically, **nested** so
that development is monotone growth:

| stage | nutrient `n` | form |
|---|---|---|
| 0 | 0.10 | small brown seed blob at the ground line |
| 1 | 0.40 | short green stem + tiny leaves |
| 2 | 0.70 | taller stem + two side leaves |
| 3 | 1.00 | full stem + leaves + flower head |

All share the same anchor point (where the seed cell is planted), so every stage grows
from the identical single-cell seed.

## Training — environment change is trained explicitly, not hoped for

Two episode types, mixed per epoch (batch of 8 = 4 identity + 4 transition samples):

**Two-segment episodes (2/3 of epochs).** Segment 1: seed -> rollout under jittered
`n_a` for `[48, 64]` steps -> MSE vs target `a`. Segment 2, from the *detached*
segment-1 state (pool-style, keeps memory at one-segment cost): env changes to jittered
`n_b` (identity samples keep `b = a`; transitions pick random `b != a`, both directions)
for `[32, 48]` steps -> MSE vs target `b`. Half the time the switch is instant; half the
time the env *glides* `a -> b` with weights regenerated every step. Half the time the
segment-2 start state is perturbed with noise so rules learn to clean up.

**Slow-ramp episodes (1/3 of epochs, truncated BPTT).** Found necessary during
verification: end-state error on a slow nutrient ramp scales with ramp *duration*
(0.002/step ramps sit for many steps under in-motion interpolated rules that short
training glides never visit). The actual slow ramp — ascending or descending, random
virtual duration 300–700 steps — is rolled from seed under `no_grad` to a random depth
(zero memory), then only the continuation is trained: a few more ramp steps, a hold at
the nearest stage level, MSE vs that stage's target.

**Nutrient jitter is ±0.15**, which makes the four basins `[0,.25][.25,.55][.55,.85]
[.85,1]` partition `[0,1]` exactly — there is no untrained nutrient value for a ramp to
visit. (±0.12 left gaps; ramps crossing them sprouted duplicate flower heads.)

`chem = 1` everywhere during training: the gate is exact physics and needs no training;
partial chemical simply time-dilates the same dynamics (it composes with the fire rate).

Carried over from the notebook: per-parameter gradient normalisation with scalars in a
separate low-LR group, cosine LR schedule, random rollout lengths. Best-checkpoint
tracking uses a **fixed periodic eval** (stage growth + an ascending and a descending
240-step ramp, RNG state saved/restored) because the two episode types have
incomparable training losses.

A three-segment variant with an extra persistence hold was tried and **rejected**: it
taught the rules to preserve whatever exists, including mid-ramp ghost residue.

## Verification (acceptance tests, all programmatic)

1. **Frozen without chemical** — with `chem = 0`, 50 steps change the state by exactly
   0.0 (max abs diff).
2. **Env-shuffle ablation** — growing each stage under the *wrong* nutrient must be
   >= 10x worse in MSE than under the right one (the manifold's shuffle-z test, applied
   to the environment).
3. **Per-stage growth** — each stage grown from seed reaches its target under MSE 0.005.
4. **Environment change** — grow the flower at `n = 1.0`, then drop to `n = 0.40`: the
   state must move to the sprout target (MSE < 0.01), demonstrating regression.
5. **Continuous slow ramp** — nutrient 0.1 -> 1.0 changing every step over 480 steps
   must end on a clean flower (MSE < 0.01).
6. **Rise then fall** — flower, then a graded drop to `n = 0.40`, must end a clean
   sprout: MSE < 0.01 *and* ghost alpha above the sprout's extent < 8 (a stable wisp of
   dead flower passed the MSE bar once; the alpha bound encodes that failure).

## Rendered outputs

- `env_ramp.gif` + filmstrip PNG — nutrient ramps 0 -> 1 over the rollout with a level
  bar; the organism develops seed -> sprout -> leafy -> flower through *untrained
  intermediate* nutrient values (the manifold earning its keep).
- `env_chem_gate.gif` — chemical absent until step ~80 (nothing happens), then present
  (growth starts): the chemical is necessary, not decorative.
- `env_chem_half.png` — chemical only on the left half: growth is spatially confined.
- `env_stages.png` — the four stages grown side by side vs targets, with the shuffle
  ablation row.

## Out of scope

Chemical consumption/diffusion dynamics, division mechanics, WebGPU export. The chemical
field here is exogenous (set by the simulation, not produced or consumed by cells).

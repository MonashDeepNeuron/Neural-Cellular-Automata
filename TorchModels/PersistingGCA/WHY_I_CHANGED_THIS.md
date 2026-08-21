## The actual bug: dropout that only fired during training

The original update step ended with `nn.Dropout(p=0.5)`. That looked fine, and there
was even a TODO admitting the problem:

```python
nn.Dropout(p=0.5) # Apply stochastic mask,
# TODO: update such that this dropout remains in, even when training mode is off
```

`nn.Dropout` is a no-op in `model.eval()`. So the model learned its dynamics with a
stochastic update mask switched on, and then the final gif was rendered with that
mask switched off. The training-time physics and the inference-time physics were two
different systems. Of course it diverged.

I pulled the dropout out of the network and apply the mask by hand in `forward()`
instead:

```python
update_mask = (torch.rand_like(ds_grid[:, :1, :, :]) <= 0.5).to(ds_grid.dtype)
output_raw_grid = input_grid + ds_grid * update_mask
```

Two things matter here. The mask is per cell, shared across all 16 channels of that
cell, which is how the canonical NCA does it (a whole cell either updates this step or
it doesn't, you don't update channel 4 but skip channel 5). And it runs in both train
and eval, so the dynamics finally match.

I left two diagnostic scripts in the repo that I used to confirm this was the real
cause. `diag_train_vs_eval.py` rolls the saved model from the same seed in `train()`
and `eval()` side by side. If train-mode stayed coherent while eval-mode broke up, the
dropout mismatch was the culprit. `eval_preview.py` just renders the lizard at a range
of step counts so I could see exactly when it stopped looking like a lizard.

## Alive masking before and after the update

The old `apply_alive_mask` only checked life after the update. I renamed it to
`alive_mask` (it now returns the boolean mask instead of the masked grid) and a cell
survives only if it was alive both before and after:

```python
life_mask = (pre_life_mask & post_life_mask).to(output_raw_grid.dtype)
return output_raw_grid * life_mask
```

This stops cells from spontaneously flickering into existence in dead regions, which
is the behaviour you want if the pattern is meant to hold its shape rather than spread.

## Weight init: start from a do-nothing model

The original init code was duplicated (two identical blocks, probably a bad merge) and
gave every layer small random weights. I changed the last layer to zero init and zeroed
the biases:

```python
torch.nn.init.normal_(self.update_network[0].weight, mean=0.0, std=0.001)
torch.nn.init.zeros_(self.update_network[0].bias)
torch.nn.init.zeros_(self.update_network[2].weight)
```

A zero last layer means the model starts by predicting "no change anywhere." That's the
standard NCA trick. The model has to earn every update from zero, which makes the first
few hundred epochs far less chaotic.

## The loss was only looking at one image in the batch

This one annoyed me. The training loop built a batch of 12, ran it forward, and then
computed loss on `batch[0]` only:

```python
batch_losses = LOSS_FN(batch[0, 0:4], target)
```

Eleven out of twelve samples contributed nothing to the gradient. No wonder it stalled.
I broadcast the target across the batch and average over all of it:

```python
target_b = target.unsqueeze(0).expand(batch.size(0), -1, -1, -1)
batch_loss = LOSS_FN(batch[:, 0:4], target_b)
```

## Gradient normalisation, and the learning-rate story that comes with it

I added per-parameter gradient normalisation before the optimiser step:

```python
for p in model.parameters():
    if p.grad is not None:
        p.grad /= (p.grad.norm() + 1e-8)
```

This is the other canonical NCA trick. It rescales each parameter's gradient to unit
norm so no single layer dominates the update. Without it these models tend to plateau at
high loss and never sharpen up.

But it changes what the learning rate means. Once every gradient is unit norm, the
learning rate is literally the step size, so the old adaptive learning-rate adjuster
(tuned for raw, un-normalised gradients) was now doing the wrong thing and driving the
rate far too low. I disabled it with a flag (`USE_LR_ADJUSTER = False`) rather than
deleting it, set the base rate to the canonical-ish `2e-3`, and added optional cosine
decay (`USE_LR_DECAY`).

The decay matters more than it looks. With a constant step size forever, the model
keeps taking fixed-size steps even after it's converged, and it can eventually
random-walk into the absorbing "all cells dead" state and never come back. Decaying the
rate lets it settle.

## Pool training: re-seed the worst samples, train on a longer horizon

Two changes in `pool_train`, both aimed at persistence rather than just growth.

First, instead of re-seeding random samples in the pool each step, I re-seed the worst
ones (highest loss):

```python
per_sample_loss = (batch[:, 0:4] - target).pow(2).mean(dim=(1, 2, 3))
worst = torch.argsort(per_sample_loss, descending=True)[:seedrate]
batch[worst] = new_seed(seedrate).to(device)
```

The idea is to evict the most-diverged states before they pollute the pool, while still
forcing the model to grow from a fresh seed every step.

Second, I rolled the pool states out for longer (`UPDATES_RANGE` from `(10, 50)` up to
`(64, 96)`) and changed model selection to reward staying on target. At save intervals
the pool phase now evaluates a 250-step rollout (`PERSIST_STEPS`) and keeps the weights
that survive it, not just the ones that reach the shape quickly. There's a separate
`TEST_STEPS = 96` horizon for the per-epoch loss curve so it stays comparable to the
old runs.

## Evaluating on the grid size the model trained on

The original evaluated on a 60x60 grid after training on 40x40:

```python
GRID_SIZE = 60
```

A model that never saw a 60x60 grid has no reason to behave on one, and it destabilised.
I set evaluation back to 40 so I'm testing the thing I actually trained.

## Smaller fixes that were quietly breaking things

- `best_model = model.state_dict()` keeps a live reference, so "best" kept mutating as
  training continued and you never actually saved the best weights. Switched to
  `copy.deepcopy`. Same reason `best_loss` now starts at `float("inf")` instead of the
  loss of an untrained seed.
- Set the matplotlib backend to `Agg` and passed `show=False` through the eval calls so
  a training run doesn't block on a `plt.show()` window. This thing should run headless.
- Added `.cpu()` in `visualise` before going to numpy, so plotting a CUDA tensor doesn't
  crash.
- Added a `--epochs` CLI arg so I can do short smoke-test runs without editing the
  constant every time.

## The cosmetic part: cat to lizard

`MODEL_PATH`/`SAVE_PATH` moved from `abc_4.pth` to `lizard.pth`, and the target loads
`./lizard.png` instead of `./cat.png`. This is the only change that was purely about
what I wanted to grow. Everything else was about getting it to actually stay grown.

## Honest status

The persistence is much better than the original, but I won't pretend it's perfect.
Long rollouts can still drift, which is why the lr decay and the worst-sample re-seeding
are in there as defences rather than guarantees. The two diagnostic scripts are checked
in on purpose so the next person (probably me in a month) can re-run the train-vs-eval
comparison instead of rediscovering the dropout bug from scratch.

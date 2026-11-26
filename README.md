# PeakCount

A neural network that counts the peaks of a function.

Given a function `f: [0, 1] → ℝ`, sampled at `num_samples` points, the network
predicts whether it has **0, 1 or 2 peaks**. It takes the samples as input and
outputs three logits, one per class.

## Model

`PeakCNN` in [`model.py`](model.py) is a small 1D convolutional network. Conv
layers and max pooling alternate until a single position with 32 channels is
left:

| Layer | Output shape |
|---|---|
| Input (unsqueezed to 1 channel) | `(batch, 1, 284)` |
| Conv1d(1 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 280)` |
| MaxPool1d(2) | `(batch, 32, 140)` |
| Conv1d(32 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 136)` |
| MaxPool1d(2) | `(batch, 32, 68)` |
| Conv1d(32 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 64)` |
| MaxPool1d(2) | `(batch, 32, 32)` |
| Conv1d(32 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 28)` |
| MaxPool1d(2) | `(batch, 32, 14)` |
| Conv1d(32 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 10)` |
| MaxPool1d(2) | `(batch, 32, 5)` |
| Conv1d(32 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 1)` |
| Flatten | `(batch, 32)` |
| Linear(32 → 3) | `(batch, 3)` |

About 26k parameters. With 284 samples, every step divides evenly, and the last
conv layer's receptive field covers the whole input. The number of channels per
conv layer is a constructor argument, `PeakCNN(num_samples, channels=32)`. With
16 channels, the model has about 6.6k parameters.

- **Input:** a tensor of shape `(batch, 284)` holding the function values at 284
  sample points. The input is not normalized inside the model.
- **Output:** raw logits of shape `(batch, 3)` for 0, 1 or 2 peaks. Train with
  `nn.CrossEntropyLoss`.
- The input length must shrink to exactly one position. `PeakCNN` raises an
  error for lengths that don't, such as 256.

The two runs from 2026-11-22 (`runs/2026-11-22_16-49-44/` and
`runs/2026-11-22_17-40-36/`) used an earlier architecture: 256 samples, three
conv layers without pooling, then flatten → Linear(7808 → 3). Their `model.pt`
files can only be loaded with the `model.py` from before this change (see the
git history).

## Training data

[`data.py`](data.py) generates random functions with exactly 0, 1 or 2 peaks.
Every class is equally likely, and fresh functions are generated for every epoch.

A **peak** is an interior local maximum. The edges `x = 0` and `x = 1` never
count, and valleys (local minima) don't count either.

Each function is built from **knots**:

1. Pick a pattern of interior extrema for the class. `P` stands for a peak and
   `V` for a valley, and each class's patterns are equally likely:

   | Class | Patterns |
   |---|---|
   | 0 peaks | (none, monotone), `V` |
   | 1 peak | `P`, `VP`, `PV`, `VPV` |
   | 2 peaks | `PVP`, `VPVP`, `PVPV`, `VPVPV` |

2. Place the interior knots at sorted uniform random positions in `[0, 1]` and
   add the edges `x = 0` and `x = 1`.
3. Give each knot a value from its band: peaks `U[2/3, 1]`, edges `U[1/3, 2/3]`,
   valleys `U[0, 1/3]`. This keeps every peak higher and every valley lower than
   its neighbours.

   Since 2026-11-25, neighbouring knots (edges included) keep a horizontal and a
   vertical distance of at least `MIN_KNOT_DISTANCE = 0.05` before warping. The
   knots are drawn from the distribution above conditioned on these distances,
   which is exactly what rejection sampling would give, but without rejection:
   - Horizontally, `k` sorted uniform points in `[0, 1 − (k+1)·0.05]` are
     shifted by `0.05, 0.10, …, k·0.05`.
   - Vertically, peak and valley bands are already `1/3` apart, so only pairs
     with an edge can be too close. An edge next to a peak is drawn from the
     band rectangle minus its forbidden corner (a rectangle plus a trapezoid
     with a quadratic inverse CDF). For the patterns `P` and `V`, where both
     edges border the same knot, that knot comes from its exact marginal (a
     cubic inverse CDF) and the edges uniformly from their allowed interval.
     The two edges of a monotone function are drawn by sort-and-shift in random
     order. Valleys use the peak samplers mirrored at `y = 1/2`.

   `MIN_KNOT_DISTANCE = 0` reproduces the earlier functions exactly.
4. Connect neighbouring knots with a smoothstep curve
   `y_i + (y_{i+1} - y_i)·(3t² − 2t³)`. It is strictly monotone between knots
   and has zero slope at each knot, so the function has exactly the intended
   extrema.
5. Warp the function: `post(f(pre(x)))`, where `pre` and `post` are random,
   strictly increasing maps of `[0, 1]` onto itself, drawn independently for
   each function. Precomposing only shifts the x-positions, and postcomposing
   only distorts the values in `[0, 1]`. All peaks and valleys are kept, so the
   label stays correct. `WARP` in [`data.py`](data.py) selects the family:
   - `"cubic"` (default since 2026-11-25): the cubic `w` with `w(0) = 0`,
     `w(1) = 1`, `w′(0) = α`, `w′(1) = β`, i.e.
     `w(x) = (α+β−2)x³ + (3−2α−β)x² + αx`. It is monotone exactly when
     `α, β ≥ 0` and `α + β − √(αβ) ≤ 3`, a region of area `6 + 4π/√3 ≈ 13.26`
     (an ellipse in `(√α, √β)`). `(α, β)` is drawn uniformly from it without
     rejection: the region splits into the kite `(0,0), (3,0), (2,2), (0,3)`
     and the sector of the ellipse `(α−2)² + (α−2)(β−2) + (β−2)² ≤ 3` around
     `(2, 2)` beyond the kite. Each sample uses three uniform numbers: one picks
     a piece by area, two give a point in it.
   - `"quadratic"` (all runs before 2026-11-25): `w(x) = x·(c·(x − 1) + 1)`
     with `c ~ U[-1, 1]`, the special case `α = 1 − c`, `β = 1 + c`. With this
     setting the generator reproduces those runs' data exactly.
   - `"shavgulidze"`: a sample of the Shavgulidze measure on the increasing
     diffeomorphisms of `[0, 1]`, the image of the Wiener measure under
     `ξ ↦ w(x) = ∫₀ˣ e^ξ / ∫₀¹ e^ξ` for a Brownian motion `ξ` with variance
     `σ²t` (`SHAVGULIDZE_SIGMA = 1`). `ξ` is drawn from its Karhunen–Loève
     expansion `ξ(t) = Σ c_k √2 sin((k−½)πt)/((k−½)π)`, `c_k = σ·Z_k`, truncated
     to 64 terms (99.7% of the variance at `t = 1`); `w` is integrated with the
     trapezoidal rule on 2,048 intervals and interpolated linearly. With `σ = 1`
     the largest slope of a warp is typically 3–5.5 times its smallest (median
     3.9). Compared with an exact Brownian motion of 8,192 steps, `w(0.1)`,
     `w(0.5)` and `w(0.9)` have the same distribution (KS `D ≤ 0.013`). The
     loss-guided and adversarial samplers don't support this warp yet. Drawing
     the functions takes about 3 times longer than with the cubic warp
     (20 ms instead of 7 ms per batch of 64).
6. Sample at 284 evenly spaced points (256 in the two runs from 2026-11-22).

Knots very close together, or a warping that squeezes a narrow peak further,
can make a peak disappear between two sample points. The label is still the
intended class. A second, harmless case looks similar in a strict count: where
a warp is very flat near a peak's value, neighbouring samples can be exactly
equal in float32, so the peak shows as a flat top of two or more equal samples
instead of one sample above both neighbours. On the 51,200 test functions
(seed 12345), the strict count (a sample above both neighbours) differs from
the label for this many functions, split into peaks that are really not
visible and flat tops:

| | No minimum distance | Minimum distance 0.05 |
|---|---|---|
| Quadratic warp | 11 (4 not visible, 7 flat tops) | 4 (0 not visible, 4 flat tops) |
| Cubic warp | 37 (15 not visible, 22 flat tops) | 21 (0 not visible, 21 flat tops) |
| Shavgulidze warp | – | 6 (0 not visible, 6 flat tops) |

So with the minimum distance, no test function has a peak that can't be seen in
the samples. The models count flat tops mostly correctly; the 600-epoch
Schedule-Free model with cubic warps, for example, gets 20 of the 21 right. The
failure analyses below use the strict count, so their category "label doesn't
match the samples" can contain flat tops; where the numbers are given, it is
said which kind they are.

**All results below used the quadratic warp and no minimum knot distance**,
except the runs from 2026-11-25 onward, which say so (the two runs from
2026-11-22 used no warp at all).

## Training

[`train.py`](train.py) trains `PeakCNN` on the GPU via DirectML. The settings are
constants at the top of the file:

| Setting | Value |
|---|---|
| Epochs | 50 |
| Functions per epoch | 12,800 (freshly generated) |
| Batch size | 64 (256 in runs before the batch size comparison) |
| Optimizer | Adam from [`optim.py`](optim.py), learning rate 1e-3 |
| Optimizer | Adam (`OPTIMIZER = "adam"`). Optionally Adam with AMSGrad (`"adam_amsgrad"`), Schedule-Free AdamW (`"schedule_free_adamw"`, Defazio et al. 2024, linear warmup over `SF_WARMUP_FRACTION` of all steps; the saved `model.pt` holds its averaged weights x, while the logged training accuracy is measured at its gradient point y) or SGD with momentum (`"sgd"`, `SGD_MOMENTUM`). All run fully on DirectML |
| Channels per conv layer | 32 (`CHANNELS`) |
| Sampling of training functions | Uniform (`SAMPLER = "uniform"`). Optionally loss-guided (`SAMPLER = "loss_guided"`, see [`sampler.py`](sampler.py)): three small networks (one per class, parameters → 32 → ReLU → 1) learn the CNN's per-function log-loss from the function parameters. Each batch keeps a uniform class mix, and within each class it is sampled without replacement from a pool of `POOL_FACTOR` candidates per slot, with probability proportional to the predicted loss. Or adversarial (`SAMPLER = "adversarial"`, see [`adversary.py`](adversary.py)): `ADVERSARY_COPIES` parameter tuples per extrema pattern (default 1), each drawn like a normal training function, form the batch (needs `BATCH_SIZE = 10 * ADVERSARY_COPIES`). After each backward pass, the network takes its descent step and Adam (`ADVERSARY_LR`) takes one ascent step on the tuples, which increases their loss. The tuples are reparametrized (knot gaps via softmax, values via sigmoid into their bands; with a minimum knot distance, gaps are `0.05 + (1 − (k+1)·0.05)·softmax` and each edge value is mapped into the part of its band that keeps the distance to its neighbour; quadratic warps via tanh, cubic warps via a direction and a radius in `(√α, √β)` inside the ellipse, without sin/cos because DirectML computes those only to about 6·10⁻⁵), so they always stay valid. The loss-guided predictors get the warp parameters as inputs (`c` per warp, or `α, β` per warp) |
| Online hard example mining | Off (`OHEM_POOL = None`). If set, each step scores `OHEM_POOL` fresh candidates without gradients and trains on the `BATCH_SIZE` with the highest loss |
| Loss | Cross-entropy (`LOSS = "cross_entropy"`). Optionally focal loss (`LOSS = "focal"`, focusing parameter `FOCAL_GAMMA`, see [`losses.py`](losses.py)) |
| Seed | 0 |

Each run gets its own folder, named after its start time:

```
runs/<date>_<time>/
  config.json training settings (epochs, functions per epoch, batch size,
              learning rate, optimizer, momentum, sf_warmup_steps, seed, warp,
              min_knot_distance, shavgulidze_sigma, shavgulidze_terms, num_samples,
              channels, loss, focal_gamma, ohem_pool, sampler,
              pool_factor, predictor_lr, adversary_lr, eval_functions,
              fixed_eval_functions, fixed_eval_seed, batch_scores)
  model.pt    weights (state_dict), overwritten after every epoch
  log.csv     one row per epoch
  chart.png   drawn after the last epoch
  tuples.csv  adversarial sampler only: the tuples after every epoch
              (epoch 0 = start), with each tuple's loss; warp parameters in
              the columns pre and post (g and h in runs before the cubic warp)
  tuples.png  adversarial sampler only: the final ten functions with their samples
```

After every epoch, `train.py` prints the mean training loss and accuracy, saves
the weights, and adds a row to the log, so an interrupted run keeps its
progress. The log has these columns:

| Column | Meaning |
|---|---|
| `epoch` | Epoch number, starting at 1 |
| `loss` | Mean cross-entropy loss over the epoch, even when training with focal loss, so runs stay comparable |
| `objective` | Mean of the loss actually optimized (equal to `loss` for cross-entropy runs; missing in runs before focal loss was added) |
| `accuracy` | Fraction of correctly classified functions (0 to 1) |
| `seconds` | Duration of the epoch |
| `functions_seen` | Total number of training functions so far |
| `timestamp` | Local time when the epoch ended (ISO 8601) |
| `predictor_corr` | Loss-guided sampler only: correlation between predicted and actual log-loss (see below) |
| `fixed_loss`, `fixed_accuracy` | Only with `FIXED_EVAL_FUNCTIONS`: loss and accuracy of the model's weights on a fixed set (see below). For Schedule-Free AdamW instead `fixed_loss_x`, `fixed_accuracy_x`, and the same for `y` and `z` |
| `train_loss`, `train_accuracy` | Only with `BATCH_SCORES`: loss and accuracy on the epoch's training batches, each scored just before its update. For Schedule-Free AdamW instead `train_loss_x`, `train_accuracy_x`, and the same for `y` (equal to `loss` and `accuracy`) and `z` |

With OHEM, `loss` and `accuracy` are measured over all scored candidates, so
they stay comparable with runs without OHEM. `objective` refers to the selected
(hard) functions, and `functions_seen` counts the functions trained on.

With the loss-guided sampler, the training batches are deliberately hard. So
after each epoch, `EVAL_FUNCTIONS` (12,800) fresh uniform functions are scored
without gradients, and `loss` and `accuracy` come from these. An extra column
`predictor_corr` holds the correlation between the predicted and the actual
log-loss on them; it is empty in other runs.

The adversarial sampler also logs `loss` and `accuracy` on `EVAL_FUNCTIONS`
fresh uniform functions after each epoch. There, `objective` is the mean loss
on the ten tuples during the epoch.

With `FIXED_EVAL_FUNCTIONS` set (default `None`), the same uniform functions,
drawn once with `FIXED_EVAL_SEED` (2026), are scored after every epoch. Unlike
the fresh functions of the training log, these curves contain no sampling
noise between epochs, so every change comes from the weights. For
Schedule-Free AdamW, the averaged weights x, the gradient point y and the
plain iterate z are scored separately. This evaluation runs after the epoch's
time is taken, so `seconds` stays comparable with other runs.

With `BATCH_SCORES = True` (uniform sampler only, default `False`), every
training batch is also scored just before its update; for Schedule-Free
AdamW with x, y and z, which needs two extra forward passes per step. Together
with the fixed set, that gives 12 values per epoch (x, y, z × loss, accuracy ×
fixed set, training data). The extra passes cost about 8 s per 200-step epoch
on the UHD 620 and are included in `seconds`, so those runs' times are not
comparable with others. Timing them separately would need a sync per step,
which slowed training by about a quarter in a test.

After the last epoch, [`plot.py`](plot.py) draws the log as `chart.png`. The
chart shows loss (left
axis) and error rate `1 − accuracy` (right axis, in %) per epoch in a single
chart. Both axes are logarithmic, so the curves are `ln(loss)` and
`ln(1 − accuracy)`, while the ticks show the original values. Both axes span
the same number of powers of ten, so their ticks line up with the same
gridlines. An epoch with 100% accuracy would have `ln(0) = −∞`. Its error rate
is therefore shown as one mistake (`1/n` for `n` functions in that epoch), and
the subtitle notes it.
`plot.py` is only called from `train.py` and does not run on its own.

The logs and charts are committed to the repository. The model files are
git-ignored.

### Optimizer

`torch.optim.Adam` updates its first moment with `lerp_`. `torch-directml` does
not implement `lerp`, so that step falls back to the CPU. [`optim.py`](optim.py)
provides an `Adam` with the same update, computed with `mul_` + `add_` so it
runs entirely on the GPU. Like `torch.optim.Adam`, it supports `weight_decay`
(L2 weight decay added to the gradient) and `amsgrad`.

In float64 it matches `torch.optim.Adam` to within about 1e-9 after 200
training steps. In float32, small rounding differences grow during training,
so the weights end up slightly different. On the UHD 620 the optimizer step
takes about 9 ms instead of 23 ms. There is no separate evaluation set.
Because every function is new, the training accuracy already shows how well
the model handles unseen functions.

```powershell
python train.py
```

## Setup

The target machine has an Intel UHD 620, so the model runs via
[DirectML](https://github.com/microsoft/DirectML) (`torch-directml`) and not CUDA.
`torch-directml` works on any GPU that supports DirectX 12 on Windows.

```powershell
pip install -r requirements.txt
```

`torch-directml` requires one specific PyTorch version (currently 2.4.1). It
replaces any other PyTorch version installed in the same Python environment.

## Usage

```python
# Generated by Claude (Claude Code).
import torch
import torch_directml

from model import PeakCNN

device = torch_directml.device()

model = PeakCNN().to(device)
x = torch.randn(64, 284).to(device)   # 64 functions, 284 samples each
logits = model(x)                     # (64, 3)
num_peaks = logits.argmax(dim=1)      # predicted peak count per function
```

## Status

- [x] Model (`model.py`)
- [x] Training data generation and peak definition (`data.py`)
- [x] Training script (`train.py`)
- [x] Training run
- [ ] Evaluation

The first full training run (50 epochs on the UHD 620, about 9.5 minutes)
reached a training accuracy of 98.20% in its last epoch, and 99.42% at best.
See [`runs/2026-11-22_16-49-44/`](runs/2026-11-22_16-49-44/). That run used
the data generation without the warping step (5), which was added afterwards.

The second run, with warped functions and the same settings (about 10
minutes), reached 96.65% in its last epoch and 97.04% at best. Its accuracy was
still rising at the end. See
[`runs/2026-11-22_17-40-36/`](runs/2026-11-22_17-40-36/).

The third run was the first with the pooled architecture (284 samples, conv +
max pooling down to one position) and used the same warped data and settings
(about 8.5 minutes). It reached 98.49% in its last epoch and 98.53% at best,
with no drop of more than one percentage point between epochs. See
[`runs/2026-11-22_18-17-39/`](runs/2026-11-22_18-17-39/).

### Batch size comparison

Six runs with batch sizes 256 down to 8. Everything else matched the third
run: 50 epochs × 12,800 warped functions, learning rate 1e-3, seed 0. The
batch-256 run reproduced the third run exactly (same loss and accuracy in
every epoch).

| Batch size | Steps | Time | ms per step | Last epoch | Best epoch | ≥ 98% after | ≥ 99% after | After 10 min |
|---|---|---|---|---|---|---|---|---|
| 256 | 2,500 | 8.4 min | 201 | 98.49% | 98.53% | 7.2 min | not reached | 98.49% (done) |
| 128 | 5,000 | 8.8 min | 106 | 98.77% | 99.00% | 5.7 min | 8.3 min | 98.77% (done) |
| 64 | 10,000 | 9.7 min | 58 | 99.10% | 99.34% | 4.2 min | 6.2 min | 99.10% (done) |
| 32 | 20,000 | 12.6 min | 38 | 99.52% | 99.53% | 3.4 min | 6.3 min | 99.18% |
| 16 | 40,000 | 16.8 min | 25 | 99.48% | 99.55% | 2.7 min | 5.9 min | 99.28% |
| 8 | 80,000 | 26.6 min | 20 | 99.61% | 99.61% | 3.8 min | 9.0 min | 99.01% |

Smaller batches give more updates from the same functions, and per epoch they
learn faster all the way down to batch size 16. Below about 32, the time per
step hardly shrinks any more (38 → 25 → 20 ms), so the runs get much longer.
Per minute of training, batch size 16 is ahead. Batch size 8 learns about as
much per epoch as 16, but is slower per minute. Chart:
[`comparisons/batch_size.png`](comparisons/batch_size.png). Runs:
[`runs/2026-11-22_18-46-09/`](runs/2026-11-22_18-46-09/) (256),
[`runs/2026-11-22_18-54-34/`](runs/2026-11-22_18-54-34/) (128),
[`runs/2026-11-22_19-03-23/`](runs/2026-11-22_19-03-23/) (64),
[`runs/2026-11-22_19-25-33/`](runs/2026-11-22_19-25-33/) (32),
[`runs/2026-11-22_19-38-11/`](runs/2026-11-22_19-38-11/) (16),
[`runs/2026-11-22_19-55-02/`](runs/2026-11-22_19-55-02/) (8).

### Focal loss

A run with focal loss (γ = 2) and otherwise the same settings as the batch-64
cross-entropy run. Both models were also evaluated on the same 51,200 fresh
warped functions (seed 12345).

| | Cross-entropy | Focal loss (γ = 2) |
|---|---|---|
| Training time | 9.7 min | 9.8 min |
| Training accuracy, last epoch | 99.10% | 99.14% |
| Training accuracy, mean of last 10 epochs | 99.19% | 99.16% |
| ≥ 99% after | 6.2 min | 6.9 min |
| Accuracy on 51,200 fresh functions | 99.46% (274 wrong) | 99.45% (281 wrong) |
| Narrow peaks missed (< 4 samples) | 190 | 170 |

The difference is not significant. 194 functions are wrong for both models,
80 only for cross-entropy and 87 only for focal loss (McNemar z = −0.54).
Focal loss misses slightly fewer narrow peaks but makes more other errors.
Cross-entropy stays the default. Chart:
[`comparisons/focal_loss.png`](comparisons/focal_loss.png). Run:
[`runs/2026-11-22_21-18-41/`](runs/2026-11-22_21-18-41/).

### Channels per conv layer: 16, 32, 48, 64

Runs with 16, 48 and 64 channels per conv layer, with otherwise the same
settings as the batch-64 run with 32 channels (50 epochs each). All models were
also evaluated with their final weights on the same 51,200 fresh warped
functions (seed 12345).

| | 16 channels | 32 channels | 48 channels | 64 channels |
|---|---|---|---|---|
| Parameters | 6,627 | 26,051 | 58,275 | 103,299 |
| Training time | 8.7 min | 9.7 min | 12.1 min | 13.8 min |
| Training accuracy, last epoch | 98.24% | 99.10% | 99.44% | 99.41% |
| Training accuracy, mean of last 10 epochs | 98.15% | 99.19% | 99.37% | 99.40% |
| ≥ 98% after | epoch 43 / 7.6 min | epoch 21 / 4.2 min | epoch 14 / 4.3 min | epoch 14 / 4.1 min |
| ≥ 99% after | not reached | epoch 31 / 6.2 min | epoch 23 / 6.2 min | epoch 25 / 7.3 min |
| After 10 min (mean of last 5 epochs) | 98.23% (done) | 99.21% (done) | 99.19% (epoch 40) | 99.16% (epoch 34) |
| Accuracy on 51,200 fresh functions | 98.85% | 99.46% | 99.47% | 99.36% |
| McNemar vs 32 channels | z = −14.58 | — | z = +0.36 | z = −3.60 |

- **Per epoch**, 48 and 64 channels learn clearly faster than 32: they reach
  98% after 14 instead of 21 epochs and end about 0.2 points higher.
- **Per minute** they cost more time per step, so after 10 minutes all three
  are about equal (99.16–99.21%). 16 channels are clearly worse throughout.
- **The fresh-data evaluation of the final weights** differs from the training
  averages. 48 channels equal 32 (n.s.), and 64 channels are even
  significantly worse. With a constant learning rate, accuracy fluctuates from
  epoch to epoch (64 channels: 99.0–99.67% in the last 10 epochs), so the
  final weights are a random snapshot. The McNemar tests compare these
  snapshots, not the channel counts in general.

32 channels stay the default. Chart:
[`comparisons/channels.png`](comparisons/channels.png). Runs:
[`runs/2026-11-22_21-35-28/`](runs/2026-11-22_21-35-28/) (16),
[`runs/2026-11-23_20-19-06/`](runs/2026-11-23_20-19-06/) (48),
[`runs/2026-11-23_20-31-14/`](runs/2026-11-23_20-31-14/) (64).

### Adam vs SGD with momentum

A run with SGD (momentum 0.9) and otherwise the same settings as the batch-64
Adam run with 32 channels. A short sweep with 2 epochs each picked the
learning rate: 0.01 was best. At 0.003 SGD barely learned in 2 epochs, at 0.03
it was unstable, and from 0.1 on it didn't learn at all. Both final models
were also evaluated on the same 51,200 fresh warped functions (seed 12345).

| | Adam (lr 1e-3) | SGD, momentum 0.9 (lr 0.01) |
|---|---|---|
| Training time | 9.7 min | 7.9 min |
| Accuracy after epoch 2 | 82.5% | 43.4% |
| ≥ 95% after | epoch 10 / 2.1 min | epoch 14 / 2.1 min |
| ≥ 98% after | epoch 21 / 4.2 min | epoch 26 / 3.9 min |
| ≥ 99% after | epoch 31 / 6.2 min | epoch 40 / 6.2 min |
| Training accuracy, last epoch | 99.10% | 98.88% |
| Training accuracy, mean of last 10 epochs | 99.19% | 99.02% |
| Accuracy on 51,200 fresh functions | 99.46% (274 wrong) | 99.25% (384 wrong) |
| Extra peaks at a high edge | 10 | 80 |

SGD starts much more slowly but catches up. Its steps are cheaper (the
optimizer in `torch.optim` is faster than our `optim.Adam`), so per minute it
is about as fast as Adam. Per epoch it stays behind. On fresh data its final
model is significantly worse: 197 functions are wrong for both models, 77
only with Adam and 187 only with SGD (McNemar z = −6.77). The training
averages point the same way. Adam stays the default. Chart:
[`comparisons/optimizer.png`](comparisons/optimizer.png). Run:
[`runs/2026-11-23_21-00-36/`](runs/2026-11-23_21-00-36/).

### Online hard example mining (OHEM)

A run with OHEM: each step scores 128 fresh candidates without gradients and
trains on the 64 with the highest loss. Otherwise the settings match the
batch-64 Adam run. Both final models were also evaluated on the same 51,200
fresh warped functions (seed 12345).

| | Plain training | OHEM, pool 128 |
|---|---|---|
| Training time | 9.7 min | 19.0 min |
| ≥ 95% / 98% / 99% after (epochs) | 10 / 21 / 31 | 9 / 20 / 30 |
| ≥ 99% after (minutes) | 6.2 min | 11.5 min |
| Training accuracy, last epoch | 99.10% | 99.48% |
| Training accuracy, mean of last 10 epochs | 99.19% | 99.34% |
| After 10 min (mean of last 5 epochs) | 99.21% | 98.40% (epoch 25) |
| Accuracy on 51,200 fresh functions | 99.46% (274 wrong) | 99.47% (271 wrong) |
| Missed peaks / extra peaks | 264 / 10 | 180 / 91 |
| Narrow peaks missed (< 4 samples) | 190 | 125 |
| Extra peaks at a high edge | 10 | 91 |

Per epoch, OHEM learns slightly faster and ends higher in training. Each step
costs about twice as much, so it is clearly behind per minute. On fresh data
its final model is as good as the baseline's: 154 functions are wrong for both
models, 120 only for the baseline and 117 only with OHEM (McNemar z = +0.19).
OHEM shifts the error types. It misses about a third fewer narrow peaks but
often counts a high edge as a peak. OHEM stays off by default. Chart:
[`comparisons/ohem.png`](comparisons/ohem.png). Run:
[`runs/2026-11-24_11-16-36/`](runs/2026-11-24_11-16-36/).

### Loss-guided sampling

A run with the loss-guided sampler (pool of 8 candidates per batch slot,
predictor learning rate 1e-3) and otherwise the same settings as the batch-64
run with uniform sampling. Both final models were also evaluated on the same
51,200 fresh warped functions (seed 12345).

| | Uniform sampling | Loss-guided sampling |
|---|---|---|
| Training time | 9.7 min | 29.9 min |
| ≥ 95% / 98% / 99% after (epochs) | 10 / 21 / 31 | 9 / 18 / 25 |
| ≥ 99% after (minutes) | 6.2 min | 16.8 min |
| Accuracy, last epoch | 99.10% | 99.71% |
| Accuracy, mean of last 10 epochs | 99.19% | 99.47% |
| Accuracy on 51,200 fresh functions | 99.46% (274 wrong) | 99.66% (174 wrong) |
| Missed peaks / extra peaks | 264 / 10 | 113 / 61 |
| Narrow peaks missed (< 4 samples) | 190 | 55 |
| Predictor correlation (epoch 1 → 50) | — | 0.57 → 0.83 |

The loss-guided model makes about a third fewer errors on fresh data, a
significant difference: 86 functions are wrong for both models, 188 only with
uniform sampling and 88 only with loss-guided sampling (McNemar z = +6.02). It
misses less than a third as many narrow peaks, but counts a high edge as a peak
more often (56 instead of 10). Drawing and scoring the candidate pool in Python
makes each step about three times as expensive, so per minute it is clearly
behind.

The two training curves are measured differently. The loss-guided run scores
fresh uniform functions after each epoch, while the uniform run averages over
the epoch's training batches as the weights change. The loss-guided curve
therefore looks better in the first epochs and is noisier. Uniform sampling
stays the default. Chart:
[`comparisons/loss_guided.png`](comparisons/loss_guided.png). Run:
[`runs/2026-11-24_12-09-32/`](runs/2026-11-24_12-09-32/).

### 300 epochs with the defaults

A run with the current defaults (batch 64, 32 channels, Adam with lr 1e-3,
cross-entropy, uniform sampling, seed 0) for 300 instead of 50 epochs, taking
52.3 minutes. Its first 50 epochs are identical to the 50-epoch run.

| Epochs | Minutes (end) | Mean accuracy | Range |
|---|---|---|---|
| 1–50 | 9.2 | 96.31% | 64.12–99.34% |
| 51–100 | 18.1 | 99.52% | 99.17–99.75% |
| 101–150 | 26.6 | 99.69% | 99.50–99.81% |
| 151–200 | 35.2 | 99.79% | 99.55–99.91% |
| 201–250 | 43.6 | 99.82% | 99.69–99.95% |
| 251–300 | 52.3 | 99.84% | 99.63–99.92% |

On the same 51,200 fresh warped functions (seed 12345), the final model reaches
**99.86% (73 wrong)**, compared with 99.46% (274 wrong) after 50 epochs
(McNemar z = +11.70) and 99.66% for the 50-epoch loss-guided run (McNemar
z = +7.23). After 29.9 minutes, the duration of the loss-guided run, it was
already at 99.81% (mean of 5 epochs). Narrow peaks are now rarely missed (15
instead of 190). Most remaining errors are high edges counted as
peaks (47 of 73). The accuracy is still rising slowly. Run:
[`runs/2026-11-24_12-41-02/`](runs/2026-11-24_12-41-02/).

### Adam vs AMSGrad vs Schedule-Free AdamW

Two runs with the Adam variants, with otherwise the same settings as the
batch-64 Adam run (50 epochs each). AMSGrad used lr 1e-3. Schedule-Free AdamW
used lr 2.5e-3 with a linear warmup over the first 5% of steps, and its saved
model is the averaged weights x. All final models were also evaluated on the
same 51,200 fresh warped functions (seed 12345).

| | Adam | AMSGrad | Schedule-Free AdamW |
|---|---|---|---|
| Training time | 9.7 min | 8.1 min | 9.1 min |
| ≥ 95% / 98% / 99% after (epochs) | 10 / 21 / 31 | 8 / 25 / 40 | 10 / 22 / 31 |
| Training accuracy, last epoch | 99.10% | 99.00% | 99.46% |
| Training accuracy, mean of last 10 epochs | 99.19% | 99.10% | 99.36% |
| Accuracy on 51,200 fresh functions | 99.46% (274 wrong) | 99.19% (416 wrong) | 99.54% (237 wrong) |
| McNemar vs Adam | — | z = −9.28 | z = +2.78 |
| Narrow peaks missed (< 4 samples) | 190 | 280 | 162 |

Schedule-Free AdamW makes about 13% fewer errors than Adam on fresh data in the
same time, a significant difference (167 functions wrong for both, 107 only
with Adam, 70 only with Schedule-Free). AMSGrad starts faster but then falls
behind. Its effective step sizes can only shrink, and it ends significantly
worse, mostly by missing narrow peaks. The training accuracies of
Schedule-Free AdamW are measured at its gradient point y, not at the saved
averaged weights x. Adam stays the default. Chart:
[`comparisons/adam_variants.png`](comparisons/adam_variants.png). Runs:
[`runs/2026-11-24_14-39-28/`](runs/2026-11-24_14-39-28/) (AMSGrad),
[`runs/2026-11-24_14-27-12/`](runs/2026-11-24_14-27-12/) (Schedule-Free AdamW).

### Adversarial tuples

One run with the adversarial sampler: the batch is always the same ten
functions, one parameter tuple per extrema pattern (2 / 4 / 4 functions with
0 / 1 / 2 peaks). After each backward pass, the network takes an Adam descent
step (lr 1e-3) and the tuples an Adam ascent step (lr 1e-2), both from the same
gradients. The tuples started as a random draw (seed 0). There was no guard
against degenerate tuples. The run had 50 epochs of 320 steps (16,000 steps,
160,000 functions) and took 11.2 min.

| | Batch 64, uniform (baseline) | Adversarial tuples |
|---|---|---|
| Training steps / functions | 10,000 / 640,000 | 16,000 / 160,000 |
| Training time | 9.7 min | 11.2 min |
| Mean loss on the ten tuples, epoch 1 → 4 → 50 | — | 0.27 → 0.00005 → < 0.000001 |
| Accuracy on 12,800 uniform functions, epoch 1 / 12 / 50 | — | 48.6% / 50.8% / 46.4% |
| Accuracy on 51,200 fresh functions | 99.46% | 46.50% |

The run failed. The network won the game against the tuples:

- **Memorization instead of learning.** The network has 26,051 parameters and
  always sees the same ten functions. It learns them by heart within the first
  epoch; their loss drops below 0.001 and later to 0. It never learns to count
  peaks in general. On uniform data it stays near 47% accuracy and mostly
  predicts 1 or 2 peaks.
- **The ascent stalls.** The gradient of the cross-entropy is proportional to
  1 − p(correct class). Once a tuple is classified with certainty, it gets
  almost no gradient, so the tuples barely moved between epochs 2 and 12.
- **Breakdown at epoch 13.** The loss on all tuples suddenly jumped back to
  about 1, and uniform accuracy fell to 33%, which is chance level.
  The tuples moved strongly while the network was disturbed, and then the
  network memorized the new tuples again.
- **The tuples became degenerate.** In the end, several of them hide their
  peaks between neighbouring samples or squeeze them against the edge (e.g.
  PVPV: smallest knot gap 0.05 samples, so the samples show no peak at all).
  The network still labels them correctly, which only works by memorization.

Uniform sampling stays the default. Run:
[`runs/2026-11-24_16-19-53/`](runs/2026-11-24_16-19-53/) (see `tuples.png`
there for the final ten functions).

### Adversarial tuples, 8 per pattern

A second run with the adversarial sampler, now with 8 tuples per extrema pattern
(`ADVERSARY_COPIES = 8`, batch of 80, each tuple drawn with `sample_params`),
otherwise with the same settings as the first run: network Adam lr 1e-3,
tuple ascent Adam lr 1e-2, same backward pass, no guard. 25 epochs of 320 steps
(8,000 steps, 640,000 functions), 11.1 min.

| | 10 tuples (first run) | 80 tuples |
|---|---|---|
| Training steps / functions | 16,000 / 160,000 | 8,000 / 640,000 |
| Mean loss on the tuples, first → last epoch | 0.27 → < 0.000001 | 0.96 → 0.79 |
| Tuples with a knot gap < 1 sample, start → end | 0 of 10 → 3 of 10 | 5 of 80 → 58 of 80 |
| Tuples whose samples show a different peak count than the label, end | — | 16 of 80 |
| Accuracy on 12,800 uniform functions, last epoch (range) | 46.4% | 43.9% (27.5–51.9%) |
| Accuracy on 51,200 fresh functions | 46.50% | 43.65% |

This run also failed, but the other way round. With 10 tuples, the network
memorized them and the ascent stalled. With 80 tuples, the network never gets
their loss down: it stays near chance (ln 3 ≈ 1.10) for the whole run, also
for the tuples whose samples still show the right peak count (median 0.85 in
the last epoch). The tuples keep moving and push their features into corners:
in the end, 58 of 80 have two knots closer than one sample, and 57 of 80 have
|g| or |h| above 0.95. 16 of them no longer show their label in the samples at
all, mostly 2-peak tuples. On fresh data the model mostly predicts 1 or 2 peaks,
even for functions with 0 peaks (65% of them predicted as 1 peak).

A likely reason why the balance flipped: Adam normalizes each tuple's ascent
step, so every tuple keeps its full speed however many tuples there are, while
the network's progress on each single tuple shrinks as its gradient is shared
across 80 functions. Run:
[`runs/2026-11-24_23-34-33/`](runs/2026-11-24_23-34-33/) (`tuples.png` shows all
80 final tuples, one column per pattern).

### 300 epochs with Schedule-Free AdamW

A 300-epoch run with Schedule-Free AdamW (lr 2.5e-3, linear warmup over the
first 5% of steps = 15 epochs), otherwise with the defaults, compared with the
300-epoch Adam run. The saved model is the averaged weights x. Both final
models were evaluated on the same 51,200 fresh warped functions (seed 12345).

| | Adam, 300 epochs | Schedule-Free AdamW, 300 epochs |
|---|---|---|
| Training time | 52.3 min | 60.5 min |
| ≥ 99% / 99.5% / 99.8% / 99.9% after (epochs) | 31 / 59 / 136 / 189 | 39 / 56 / 95 / 144 |
| Training accuracy, mean of epochs 251–300 | 99.84% | 99.89% |
| Accuracy on 51,200 fresh functions | 99.86% (73 wrong) | **99.95% (28 wrong)** |
| McNemar vs Adam, 300 epochs | — | z = +5.86 |
| Missed peaks / extra peaks | 25 / 48 | 27 / 1 |
| Narrow peaks missed (< 4 samples) | 15 | 18 |
| Extra peaks at a high edge | 47 | 1 |
| Errors on peaks not visible in the samples | 3 | 3 |

Schedule-Free AdamW makes 62% fewer errors than Adam on fresh data, a
significant difference: 21 functions are wrong for both, 52 only with Adam and
7 only with Schedule-Free. Compared with its own 50-epoch run (237 wrong), the
longer training gives z = +14.32. The improvement comes almost entirely from
the high edges: Adam counts 47 of them as peaks, Schedule-Free only 1. Missed
narrow peaks stay about the same (15 vs 18). Of the 28 remaining errors, 3 are
functions with a peak that isn't visible in the samples at all (knots closer
than one sample), 18 are missed narrow peaks and
7 missed low peaks (rise < 0.03).

The training accuracies of Schedule-Free AdamW are measured at the gradient
point y. There it is only slightly ahead of Adam, while its averaged weights x
are clearly better on fresh data (0.05% vs 0.14% error). The epochs took 12.1 s
on average, compared with 10.9 s in the 50-epoch Schedule-Free run and 10.5 s
in the 300-epoch Adam run, so the run took 60.5 instead of the expected 55
minutes. Per minute, Schedule-Free is still ahead: after 52.3 minutes (the
length of the Adam run) its training accuracy averaged 99.89% over the last 5
epochs, vs 99.86% for Adam. Chart:
[`comparisons/schedule_free_300.png`](comparisons/schedule_free_300.png). Run:
[`runs/2026-11-24_16-48-22/`](runs/2026-11-24_16-48-22/).

### 600 epochs: quadratic vs cubic warp

Two 600-epoch runs with the current defaults (batch 64, Adam lr 1e-3, 32
channels, minimum knot distance 0.05), one with the quadratic and one with the
cubic warp. After every epoch, each run also scored a fixed set of 12,800
functions from its own generator (`FIXED_EVAL_FUNCTIONS`), so those curves
contain no sampling noise. The first 50 epochs of the cubic run are identical
to the 50-epoch run with the new defaults
([`runs/2026-11-25_00-45-09/`](runs/2026-11-25_00-45-09/): 98.34% in the last
epoch). Both final models were evaluated on two test sets of 51,200 functions
(seed 12345, minimum distance 0.05), one per warp.

| | Quadratic-warp model | Cubic-warp model |
|---|---|---|
| Training time (without the fixed-set scoring) | 104.0 min | 104.9 min |
| Fixed-set accuracy, mean of epochs 501–600 | 99.986% | 99.749% |
| **Quadratic test set** | 99.998% (1 wrong) | 99.994% (3 wrong) |
| **Cubic test set** | 96.186% (1,953 wrong) | 99.748% (129 wrong) |

- **With the minimum distance, the quadratic task is practically solved.**
  Both models make 1 and 3 errors on the quadratic test set (McNemar z = −1.00,
  not significant). For comparison, the 300-epoch Adam run without the
  minimum distance made 73 errors on the old quadratic test set, mostly missed
  narrow peaks and high edges.
- **The cubic task stays harder.** After 600 epochs the cubic model still
  makes 129 errors on its own test set. 63 are missed low peaks (rise < 0.03,
  i.e. peaks flattened by the post-warp), 48 other missed peaks and 10 missed
  peaks near an edge. Its fixed-set accuracy still rises slowly (99.69% in
  epochs 301–400, 99.75% in epochs 501–600).
- **Cubic training transfers, quadratic training doesn't.** The cubic model is
  as good as the quadratic one on quadratic data, but the quadratic model
  reaches only 96.19% on cubic data (McNemar z = +41.24). 1,109 of its errors
  are high edges counted as peaks.
- **The fixed-set curves still fluctuate from epoch to epoch** (cubic: between
  about 0.1% and 1% error after epoch 200). Since the set is fixed, these
  fluctuations come from the weights alone, the jitter of a constant learning
  rate. Epochs with 0 errors are drawn as one error (1/12,800).

Charts: [`comparisons/warp_600.png`](comparisons/warp_600.png) (training
logs) and [`comparisons/warp_600_fixed.png`](comparisons/warp_600_fixed.png)
(fixed sets). Runs:
[`runs/2026-11-25_01-03-41/`](runs/2026-11-25_01-03-41/) (quadratic),
[`runs/2026-11-25_03-00-28/`](runs/2026-11-25_03-00-28/) (cubic).

### 600 epochs with Schedule-Free AdamW: quadratic vs cubic warp

The same comparison as above with Schedule-Free AdamW (lr 2.5e-3, warmup over
5% of the steps = 6,000 steps), otherwise the current defaults (minimum knot
distance 0.05). Each run scored x, y and z after every epoch on a fixed set of
12,800 functions and, with `BATCH_SCORES`, on every training batch just before
its update: 12 values per epoch. All final models (x for Schedule-Free) were
evaluated on the two test sets of 51,200 functions used above.

Loss (mean cross-entropy) and error (with the number of misclassified functions)
by training warp and test set:

| Training warp | Test set | SF x loss | SF x error | Adam loss | Adam error |
|---|---|---|---|---|---|
| Quadratic | Quadratic | 0.000003 | 0.0000% (0) | 0.000051 | 0.0020% (1) |
| Quadratic | Cubic | 0.2266 | 2.5840% (1,323) | 0.2342 | 3.8145% (1,953) |
| Cubic | Quadratic | 0.000026 | 0.0000% (0) | 0.000307 | 0.0059% (3) |
| Cubic | Cubic | 0.001581 | 0.0469% (24) | 0.008884 | 0.2520% (129) |

SF x is the saved averaged weights of Schedule-Free AdamW; y and z were not
saved at the end of the runs, so they can't be scored on the test sets.

| Mean of epochs 501–600 | SF, quadratic | SF, cubic |
|---|---|---|
| Fixed set: error of x / y / z | 0.0000% / 0.016% / 58.6% | 0.040% / 0.102% / 65.8% |
| Fixed set: loss of z | 342 | 1,064 |
| Training batches: error of x / y / z | 0.0009% / 0.0087% / 59.0% | 0.048% / 0.105% / 66.1% |

- **Schedule-Free beats Adam clearly on the cubic task.** On the cubic test set
  the cubic models make 24 instead of 129 errors (McNemar z = +9.17: 11 wrong
  for both, 118 only with Adam, 13 only with Schedule-Free). The remaining 24
  are mostly missed low peaks (10) and high edges counted as peaks (5). The
  quadratic models also transfer better (1,323 instead of 1,953 errors on cubic
  data, z = +16.10). On the quadratic test set all four models make at most 3
  errors, too few for a significant difference.
- **The quadratic task is solved.** Both Schedule-Free models make no error on
  the quadratic test set, and x of the quadratic run made no error on its
  fixed set in any epoch after about epoch 200. On the cubic task, x still
  improved slowly at the end (fixed-set error 0.055% in epochs 301–400, 0.040%
  in epochs 501–600).
- **x is the best of the three, y close behind, z unusable on its own.** z,
  the plain iterate with a constant learning rate, has a loss of 300–1,000 and
  misclassifies about 60% of the functions, although its average x is nearly
  perfect. That is how the method is meant to work (only x is the model), but
  the gap is extreme.
- **Training batches and fixed set agree.** Each batch is scored before the
  model trains on it, so for the model it is new data too; the per-batch
  scores are a second, independent measurement rather than a sign of
  memorization.
- The training times (246 and 215 min) include the per-batch scoring and are
  not comparable with other runs. The quadratic run stopped after its last
  epoch on a chart bug (fixed in `plot.py`); its chart was drawn afterwards.

Chart: [`comparisons/schedule_free_600_xyz.png`](comparisons/schedule_free_600_xyz.png), one
column per run and source (fixed set, training batches), loss on top and error
rate below, x drawn on top of y and z, and the 600-epoch Adam run with the
same warp in yellow on top (its fixed set and its training log, which also
scores each batch before its update). On the fixed set, x falls smoothly and
in steps of single errors, while y and Adam keep jumping from epoch to epoch:
the averaging in x is directly visible. Adam behaves much like y; on the cubic
task both stay clearly above x. Runs:
[`runs/2026-11-25_22-08-32/`](runs/2026-11-25_22-08-32/) (quadratic),
[`runs/2026-11-26_03-17-17/`](runs/2026-11-26_03-17-17/) (cubic).

## Feature visualizations

Projections of the 32 features of 600-epoch cubic-warp models: the output of
the conv stack, i.e. the input of the final linear layer, for the 51,200 cubic
test functions (seed 12345, minimum knot distance 0.05). Images of the Adam
model ([`runs/2026-11-25_03-00-28/`](runs/2026-11-25_03-00-28/), 99.748% on
this test set) are in [`features/adam/`](features/adam/), those of the
Schedule-Free AdamW model
([`runs/2026-11-26_03-17-17/`](runs/2026-11-26_03-17-17/), 99.953%) in
[`features/schedule_free/`](features/schedule_free/) under the same names. Colours: blue / green / red for 0 / 1 / 2
true peaks, shades for the patterns within a class. The scripts are in
[`analysis/`](analysis/) and are run from the project folder, e.g.
`python analysis/pca_features.py . runs/2026-11-25_03-00-28 features/adam/pca_panels.png`.
5 (Adam) and 4 (Schedule-Free) of the 32 channels are 0 for every test
function (dead ReLU channels).

### PCA

[`features/adam/pca.png`](features/adam/pca.png) (one scatter) and
[`features/adam/pca_panels.png`](features/adam/pca_panels.png) (explained variance and
one panel per pattern), from `analysis/pca_features.py` and
`analysis/pca_single.py` (raw features, centered).

Adam model:

- PC1 explains 66.8% of the variance, PC2 10.9%; 5 components cover 90%.
- PC1 measures roughly how much peak signal a function contains: the class
  means are −18.7 / −1.6 / 20.6 for 0 / 1 / 2 peaks, and within the one-peak
  functions PC1 correlates with the peak's rise (+0.62).
- All functions without a peak collapse onto almost one point.
- PC2 encodes where a single peak is (correlation +0.86 with its position
  after the pre-warp), which splits the one-peak functions into two lobes.
- One and two peaks overlap in this view, although the model separates them
  at 99.75%.

Schedule-Free model ([`features/schedule_free/pca.png`](features/schedule_free/pca.png),
[`features/schedule_free/pca_panels.png`](features/schedule_free/pca_panels.png)):

- The same structure, but even more concentrated on PC1: 78.2% of the
  variance (PC2 5.5%); 4 components cover 90%.
- The classes lie further apart along PC1 (means −44.8 / −7.3 / 53.1), on a
  scale about 2.5 times larger than Adam's; PC1 again correlates with the rise
  of a single peak (+0.60).
- PC2 still encodes the position of a single peak, but more weakly
  (correlation +0.64 instead of +0.86).

### LDA

[`features/adam/lda.png`](features/adam/lda.png), from `analysis/lda_features.py`. The
two discriminant directions were fitted on 51,200 separate functions (seed
54321) and applied to the test set; axes in units of the pooled within-class
standard deviation, without the 5 dead channels.

Adam model:

- Three clearly separated clusters at (5.5, 1.6), (−0.4, −3.5) and (−5.1, 2.0)
  for 0 / 1 / 2 peaks. LD1 (75% of the separation) orders them by peak count,
  LD2 (25%) separates one peak from both extremes.
- Assigning each test function to the nearest class centre in this plane gives
  99.40%, close to the model's 99.75%.
- The 0- and 2-peak clusters never touch, matching the model's errors (2 → 1
  and 1 → 0 only, no 0 ↔ 2 confusion).

Schedule-Free model ([`features/schedule_free/lda.png`](features/schedule_free/lda.png),
28 live channels):

- The same V-shaped arrangement, with the clusters at (4.7, 1.4), (0.3, −2.6)
  and (−5.1, 1.2); LD1 carries more of the separation (83% instead of 75%).
  Within the no-peak cluster, monotone and V functions are more clearly apart
  than in the Adam model.
- Nearest class centre in this plane: 99.29%, slightly below the Adam model's
  99.40%, although the Schedule-Free model itself is much better (99.953%
  instead of 99.748%). So its extra accuracy is not visible in a 2D linear
  view with equal class covariances; presumably the model's own 32-D linear
  layer uses directions or cluster shapes that LDA's assumptions miss.

### UMAP

[`features/adam/umap.png`](features/adam/umap.png), from `analysis/umap_features.py`
(all 51,200 test functions, n_neighbors 15, min_dist 0.1, seed 0). UMAP and
t-SNE need `umap-learn` and `scikit-learn` (`pip install umap-learn`, which
brings both); they are only used for these analyses and are not in
`requirements.txt`.

Adam model:

- Three completely separate islands, one per peak count.
- Within a class the patterns are partly sorted. Among the 15 nearest
  neighbours in the 32-D features, the share with the same pattern is 93.6% /
  56.5% / 37.1% for 0 / 1 / 2 peaks (chance: 50% / 25% / 25%); the UMAP layout
  keeps 88.5% / 44.1% / 33.8%. So the features clearly tell monotone functions
  from V functions, although the task doesn't require it.
- 92 of the model's 129 errors lie among functions of the class it predicted:
  their features look like the other class.
- Distances between the islands, their sizes and shapes carry no meaning.

Schedule-Free model ([`features/schedule_free/umap.png`](features/schedule_free/umap.png)):

- Again three separate islands, one per peak count.
- The patterns within the one- and two-peak classes are more sorted than in
  the Adam model: same pattern among the 15 nearest neighbours 91.8% / 61.7% /
  42.9% in the 32-D features (Adam: 93.6% / 56.5% / 37.1%), 86.1% / 48.8% /
  36.8% in the layout.
- 20 of its 24 errors lie among functions of the class it predicted.

### t-SNE

[`features/adam/tsne_10k.png`](features/adam/tsne_10k.png), from
`analysis/tsne_features.py` (a random 10,000 of the test functions, seed 0;
perplexity 30, PCA initialization, seed 0; 1.4 min).

Adam model:

- Again three separate clusters, ordered 0 – 1 – 2 peaks; that order comes
  from the PCA initialization.
- The no-peak cluster splits into monotone and V functions, as in UMAP. Among
  the 15 nearest neighbours, the same pattern appears for 89.9% / 50.8% /
  35.0% in the 32-D features of this subset and 89.0% / 47.8% / 34.3% in the
  layout, slightly more faithful than UMAP.
- 16 of the 24 errors in the subset lie among functions of the predicted
  class.

## License

[MIT](LICENSE)

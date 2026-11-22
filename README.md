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
conv layer's receptive field covers the whole input.

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

2. Place the interior knots at sorted uniform random positions in `[0, 1]`,
   with no minimum distance between them. Add the edges `x = 0` and `x = 1`.
3. Give each knot a value from its band: peaks `U[2/3, 1]`, edges `U[1/3, 2/3]`,
   valleys `U[0, 1/3]`. This keeps every peak higher and every valley lower than
   its neighbours.
4. Connect neighbouring knots with a smoothstep curve
   `y_i + (y_{i+1} - y_i)·(3t² − 2t³)`. It is strictly monotone between knots
   and has zero slope at each knot, so the function has exactly the intended
   extrema.
5. Warp the function. Draw `g, h ~ U[-1, 1]` independently for each function
   and use `warp_h(f(warp_g(x)))` with `warp_c(x) = x·(c·(x − 1) + 1)`. For
   `c ∈ [-1, 1]`, `warp_c` is strictly increasing on `[0, 1]` and keeps `0` and
   `1` fixed. Precomposing therefore only shifts the x-positions, and
   postcomposing only distorts the values in `[0, 1]`. All peaks and valleys
   are kept, so the label stays correct.
6. Sample at 284 evenly spaced points (256 in the two runs from 2026-11-22).

Knots very close together, or a warping that squeezes a narrow peak further,
can make a peak disappear between two sample points. The label is still the
intended class. In a check of 12,800 functions, 99.98% of labels matched the
number of peaks counted in the samples.

## Training

[`train.py`](train.py) trains `PeakCNN` on the GPU via DirectML. The settings are
constants at the top of the file:

| Setting | Value |
|---|---|
| Epochs | 50 |
| Functions per epoch | 12,800 (freshly generated) |
| Batch size | 256 |
| Optimizer | Adam from [`optim.py`](optim.py), learning rate 1e-3 |
| Loss | Cross-entropy |
| Seed | 0 |

Each run gets its own folder, named after its start time:

```
runs/<date>_<time>/
  config.json training settings (epochs, functions per epoch, batch size,
              learning rate, seed, num_samples)
  model.pt    weights (state_dict), overwritten after every epoch
  log.csv     one row per epoch
  chart.png   drawn after the last epoch
```

After every epoch, `train.py` prints the mean training loss and accuracy, saves
the weights, and adds a row to the log, so an interrupted run keeps its
progress. The log has these columns:

| Column | Meaning |
|---|---|
| `epoch` | Epoch number, starting at 1 |
| `loss` | Mean cross-entropy loss over the epoch |
| `accuracy` | Fraction of correctly classified functions (0 to 1) |
| `seconds` | Duration of the epoch |
| `functions_seen` | Total number of training functions so far |
| `timestamp` | Local time when the epoch ended (ISO 8601) |

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

Three runs with batch sizes 256, 128 and 64. Everything else matched the third
run: 50 epochs × 12,800 warped functions, learning rate 1e-3, seed 0. The
batch-256 run reproduced the third run exactly (same loss and accuracy in
every epoch).

| Batch size | Steps | Time | ms per step | Last epoch | Best epoch | ≥ 98% after | ≥ 99% after |
|---|---|---|---|---|---|---|---|
| 256 | 2,500 | 8.4 min | 201 | 98.49% | 98.53% | 7.2 min | not reached |
| 128 | 5,000 | 8.8 min | 106 | 98.77% | 99.00% | 5.7 min | 8.3 min |
| 64 | 10,000 | 9.7 min | 58 | 99.10% | 99.34% | 4.2 min | 6.2 min |

Smaller batches give more updates from the same functions. On the UHD 620, a
step costs much less with a small batch, so the runs take only a little longer.
Batch size 64 is ahead both per epoch and per minute of training. Chart:
[`comparisons/batch_size.png`](comparisons/batch_size.png). Runs:
[`runs/2026-11-22_18-46-09/`](runs/2026-11-22_18-46-09/) (256),
[`runs/2026-11-22_18-54-34/`](runs/2026-11-22_18-54-34/) (128),
[`runs/2026-11-22_19-03-23/`](runs/2026-11-22_19-03-23/) (64).

## License

[MIT](LICENSE)

# PeakCount

A neural network that counts the peaks of a function.

Given a function `f: [0, 1] → ℝ`, sampled at `num_samples` points, the network
predicts whether it has **0, 1 or 2 peaks**. It takes the samples as input and
outputs three logits, one per class.

## Model

`PeakCNN` in [`model.py`](model.py) is a small 1D convolutional network:

| Layer | Output shape |
|---|---|
| Input (unsqueezed to 1 channel) | `(batch, 1, 256)` |
| Conv1d(1 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 252)` |
| Conv1d(32 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 248)` |
| Conv1d(32 → 32, kernel 5, no padding) + ReLU | `(batch, 32, 244)` |
| Flatten | `(batch, 7808)` |
| Linear(7808 → 3) | `(batch, 3)` |

About 34k parameters, most of them in the final linear layer.

- **Input:** a tensor of shape `(batch, 256)` holding the function values at 256
  sample points. The input is not normalized inside the model.
- **Output:** raw logits of shape `(batch, 3)` for 0, 1 or 2 peaks. Train with
  `nn.CrossEntropyLoss`.
- The input length is fixed, because the flattened features feed into a linear
  layer. `PeakCNN(num_samples=...)` builds a model for a different length, which
  must then be trained separately.

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
5. Sample at 256 evenly spaced points.

Knots very close together can make a peak disappear between two sample points.
The label is still the intended class. In a check of 12,800 functions, 99.99%
of labels matched the number of peaks counted in the samples.

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

After every epoch it prints the mean training loss and accuracy and saves the
weights (`state_dict`) to `peakcnn.pt`.

Each run also writes a training log to `logs/training_log_<date>_<time>.csv`,
adding one row after every epoch so an interrupted run keeps its log:

| Column | Meaning |
|---|---|
| `epoch` | Epoch number, starting at 1 |
| `loss` | Mean cross-entropy loss over the epoch |
| `accuracy` | Fraction of correctly classified functions (0 to 1) |
| `seconds` | Duration of the epoch |
| `functions_seen` | Total number of training functions so far |
| `timestamp` | Local time when the epoch ended (ISO 8601) |

The logs are committed to the repository.

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
x = torch.randn(64, 256).to(device)   # 64 functions, 256 samples each
logits = model(x)                     # (64, 3)
num_peaks = logits.argmax(dim=1)      # predicted peak count per function
```

## Status

- [x] Model (`model.py`)
- [x] Training data generation and peak definition (`data.py`)
- [x] Training script (`train.py`)
- [ ] Training run
- [ ] Evaluation

A forward and backward pass runs on the UHD 620 via DirectML. The model has not
been trained yet.

## License

[MIT](LICENSE)

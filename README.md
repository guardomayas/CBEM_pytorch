# CBEM_pytorch

PyTorch implementation and utilities for conductance-based encoding model (CBEM) workflows.

Reference: _Kenneth W Latimer, Fred Rieke, Jonathan W Pillow_ (2019) **Inferring synaptic inputs from spikes with a conductance-based neural encoding model** eLife 8:e47012.


## Quickstart

```python
import torch
from utils import CBEM, convolveStimulusWithBasis_torch, makeRaisedCosBasis

dt = 1e-4
model = CBEM(binsize_s=dt)

# Build temporal basis and design matrix
t, B_orth, _ = makeRaisedCosBasis(nb=8, dt=dt, endpoints=[0.02, 0.2], b=0.01)
stimulus = torch.randn(5000)
basis_t = torch.as_tensor(B_orth, dtype=torch.float32)
X = convolveStimulusWithBasis_torch(stimulus, basis_t, add_ones=True)

# Forward pass
rate_hz, aux = model(X)
print(rate_hz.shape, aux["V"].shape, aux["gs"].shape)
```

## Public API

Main imports exposed from `utils`:

- `CBEM`
- `makeRaisedCosBasis`
- `convolveStimulusWithBasis_torch`
- `logOnePlusExpX_torch`
- `get_voltage_exp_recurrence`
- `firingRateNonlinearity`
- `load_mat_v73`
- `flatten_cell`

## Repository Layout

- `utils/CBEM.py`: CBEM model definition and spike-train simulation.
- `utils/cbem_utils.py`: core numerical utilities (stimulus convolution, nonlinearities, voltage recurrence).
- `utils/_raised_cosine_basis.py`: raised cosine basis generation.
- `utils/load_matlab.py`: MATLAB v7.3 loading helpers.
- `utils/stim_utils.py`: optional JAX-based stimulus generation and plotting helpers.
- `Data/`: sample data files used by notebooks.
- `tests/`: unit tests for core math and model behavior.

## Reproducibility Notes

- Use `simulateSpikeTrains(..., seed=...)` for deterministic sampling.
- `CBEM.voltage` computes recurrence in `float64` then casts back for better numerical stability.


## TODO:
- Initialize filters using linear model for k_e and k_i = -k_e
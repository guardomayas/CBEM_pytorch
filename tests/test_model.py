import torch

from utils.CBEM import CBEM


def test_cbem_forward_output_shapes():
    T = 32
    D = 5
    X = torch.randn(T, D, dtype=torch.float32)
    model = CBEM(binsize_s=1e-3)

    rate, aux = model(X)

    assert rate.shape == (T,)
    assert torch.all(rate >= 0.0)
    assert aux["V"].shape == (T,)
    assert aux["gs"].shape == (T, 2)


def test_simulate_spike_trains_is_seed_reproducible():
    T = 40
    D = 4
    X = torch.randn(T, D, dtype=torch.float32)
    Y_init = torch.zeros((5, 1), dtype=torch.float32)
    model = CBEM(binsize_s=1e-3)

    sps1 = model.simulateSpikeTrains(X_cond=X, Y_init=Y_init, N=3, seed=123)
    sps2 = model.simulateSpikeTrains(X_cond=X, Y_init=Y_init, N=3, seed=123)

    assert sps1.shape == (T, 3)
    assert torch.equal(sps1, sps2)

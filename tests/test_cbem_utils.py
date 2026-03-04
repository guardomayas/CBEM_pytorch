import math

import torch

from utils.cbem_utils import get_voltage_exp_recurrence, logOnePlusExpX_torch


def test_log_one_plus_exp_piecewise_behavior():
    x = torch.tensor([-40.0, 0.0, 100.0], dtype=torch.float32)
    max_g = torch.tensor([80.0], dtype=torch.float32)

    out = logOnePlusExpX_torch(x, max_g)

    assert torch.isclose(out[0], torch.tensor(1e-15, dtype=torch.float32))
    assert torch.isclose(out[1], torch.tensor(math.log(2.0), dtype=torch.float32), atol=1e-6)
    assert torch.isclose(out[2], torch.tensor(100.0, dtype=torch.float32))


def test_voltage_recurrence_stays_at_leak_equilibrium_when_no_synaptic_conductance():
    T = 10
    gs = torch.zeros((T, 2), dtype=torch.float64)
    E_s = torch.tensor([0.0, -80.0], dtype=torch.float64)
    g_l = torch.tensor(200.0, dtype=torch.float64)
    E_l = torch.tensor(-60.0, dtype=torch.float64)
    V0 = torch.tensor(-60.0, dtype=torch.float64)

    V = get_voltage_exp_recurrence(gs=gs, E_s=E_s, g_l=g_l, E_l=E_l, V0=V0, dt_s=1e-3)

    assert V.shape == (T,)
    assert torch.allclose(V, torch.full((T,), -60.0, dtype=torch.float64), atol=1e-10)

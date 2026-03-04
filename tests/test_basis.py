import numpy as np

from utils._raised_cosine_basis import makeRaisedCosBasis


def test_make_raised_cos_basis_basic_shapes_and_bounds():
    iht, ihbas, ihbasis = makeRaisedCosBasis(nb=6, dt=1e-3, endpoints=[0.02, 0.2], b=0.01)

    assert iht.ndim == 1
    assert ihbasis.shape[0] == len(iht)
    assert ihbasis.shape[1] == 6
    assert ihbas.shape[0] == len(iht)
    assert ihbas.shape[1] <= 6
    assert np.all(ihbasis >= 0.0)
    assert np.all(ihbasis <= 1.0)

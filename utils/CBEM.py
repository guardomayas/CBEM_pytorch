import torch
import torch.nn as nn
from .cbem_utils import (
    firingRateNonlinearity,
    get_voltage_exp_recurrence,
    logOnePlusExpX_torch,
)


# # ----------------------------
# Model
# ----------------------------
class CBEM(nn.Module):
    def __init__(self, binsize_s: float,
                 gbar_exc: float = 80.0, 
                 gbar_inh: float = 80.0,
                 E_l: float = -60.0, 
                 g_l: float = 200.0, 
                 V0: float | None = None,
                 alpha: float = 90.0, 
                 beta: float = 1.67, 
                 mu: float = -53.0,
                 add_ones: bool = True):
        super().__init__()
        self.binsize_s = float(binsize_s)
        self.add_ones = bool(add_ones)

        self.register_buffer("E_s", torch.tensor([0.0, -80.0], dtype=torch.float32))
        self.register_buffer("E_l", torch.tensor(E_l, dtype=torch.float32))
        self.register_buffer("g_l", torch.tensor(g_l, dtype=torch.float32))
        self.register_buffer("V0", torch.tensor(E_l if V0 is None else V0, dtype=torch.float32))
        self.register_buffer("gbar", torch.tensor([gbar_exc, gbar_inh], dtype=torch.float32))

        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta",  torch.tensor(beta,  dtype=torch.float32))
        self.register_buffer("mu",    torch.tensor(mu,    dtype=torch.float32))

        self.B_cond = None  # lazy nn.Parameter

    def _maybe_init_B_cond_from_D(self, D: int, device, dtype):
        if self.B_cond is None:
            self.B_cond = nn.Parameter(torch.zeros(D, 2, device=device, dtype=dtype))

    def conductances_from_X(self, X_cond: torch.Tensor) -> torch.Tensor:
        x = X_cond @ self.B_cond                          # [T,2]
        return logOnePlusExpX_torch(x, self.gbar.to(x.device, x.dtype))

    def voltage(self, gs: torch.Tensor) -> torch.Tensor:
        # run voltage in float64 for stability, then cast back
        gs64 = gs.to(torch.float64)

        E_s64 = self.E_s.to(device=gs.device, dtype=torch.float64)
        E_l64 = self.E_l.to(device=gs.device, dtype=torch.float64)
        g_l64 = self.g_l.to(device=gs.device, dtype=torch.float64)
        V064  = self.V0.to(device=gs.device, dtype=torch.float64)

        V64 = get_voltage_exp_recurrence(
            gs64,
            E_s64,
            g_l=g_l64,
            E_l=E_l64,
            V0=V064,
            dt_s=float(self.binsize_s),
        )
        return V64.to(gs.dtype)
    
    def forward(self, X_cond: torch.Tensor, window=None):
        # X_cond is already the design matrix [T, D]
        if window is not None:
            window = torch.as_tensor(window, device=X_cond.device, dtype=torch.long)
            X_cond = X_cond[window]

        self._maybe_init_B_cond_from_D(X_cond.shape[1], device=X_cond.device, dtype=X_cond.dtype)

        gs = self.conductances_from_X(X_cond)             # [T,2]
        V  = self.voltage(gs)                             # [T]
        rate = 1e-4 + firingRateNonlinearity(V, self.alpha, self.mu, self.beta)  # [T]
        return rate, {"V": V, "gs": gs}
    
    @torch.no_grad()
    def simulateSpikeTrains(
        self,
        X_cond: torch.Tensor,
        Y_init: torch.Tensor,
        N: int | None = None,
        seed: int = 0,
        eps: float = 0.0,
    ) -> torch.Tensor:
        """
        Simulates spike trains using Bernoulli-per-bin approximation to Poisson:
            p_t = 1 - exp(-rate_t * binsize_s)

        Args:
          X_cond: [T, D] design matrix for conductances (already built / cached)
          Y_init: [T0, N] or [T0, 1] initial spikes (0/1). Used only to seed first T0 bins.
          N:      number of simulations. If Y_init has 1 column, it will be broadcast to N.
          seed:   RNG seed
          eps:    optional clamp for probabilities (0 means no clamp)

        Returns:
          sps: [T, N] spikes in {0,1} (float tensor)
        """
        device = X_cond.device
        dtype  = X_cond.dtype

        Y_init = torch.as_tensor(Y_init, device=device, dtype=dtype)
        if Y_init.ndim == 1:
            Y_init = Y_init[:, None]  # [T0,1]
        T0, N0 = Y_init.shape

        if N is None:
            N = N0
        if not (N == N0 or N0 == 1):
            raise ValueError("Invalid N: must match Y_init.shape[1] or Y_init must have 1 column")

        if N0 == 1 and N > 1:
            Y_init = Y_init.expand(T0, N).contiguous()

        T = X_cond.shape[0]
        if T0 >= T:
            raise ValueError("Y_init must be shorter than total T (T0 < T).")

        # Get stimulus-driven rate once: [T]
        rate, aux = self.forward(X_cond)          # rate in Hz, shape [T]
        lam = rate * float(self.binsize_s)        # expected spikes per bin, [T]
        p = (-torch.expm1(-lam))                  # stable 1-exp(-lam), [T]
        if eps > 0:
            p = p.clamp(min=eps, max=1.0 - eps)
        else:
            p = p.clamp(min=0.0, max=1.0)

        # Sample N spike trains
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        rs = torch.rand((T, N), generator=gen, device=device, dtype=dtype)

        sps = (rs < p[:, None]).to(dtype)         # [T,N]
        sps[:T0, :] = Y_init                      # overwrite seed segment
        return sps
    

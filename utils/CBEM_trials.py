import torch
import torch.nn as nn
from .cbem_utils import (
    firingRateNonlinearity,
    get_voltage_exp_recurrence_batched_loop,
    logOnePlusExpX_torch,
)


# # ----------------------------
# Model
# ----------------------------
class CBEM_trials(nn.Module):
    def __init__(self, binsize_s: float,
                 gbar_exc: float = 80.0,
                 gbar_inh: float = 80.0,
                 E_l: float = -60.0,
                 g_l: float = 200.0,
                 V0: float | None = None,
                 alpha: float = 90.0,
                 beta: float = 1.67,
                 mu: float = -53.0):
        super().__init__()
        self.binsize_s = float(binsize_s)

        self.register_buffer("E_s", torch.tensor([0.0, -80.0], dtype=torch.float32))
        self.register_buffer("E_l", torch.tensor(E_l, dtype=torch.float32))
        self.register_buffer("g_l", torch.tensor(g_l, dtype=torch.float32))
        self.register_buffer("V0",  torch.tensor(E_l if V0 is None else V0, dtype=torch.float32))
        self.register_buffer("gbar", torch.tensor([gbar_exc, gbar_inh], dtype=torch.float32))

        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.register_buffer("beta",  torch.tensor(beta,  dtype=torch.float32))
        self.register_buffer("mu",    torch.tensor(mu,    dtype=torch.float32))

        self.B_cond = None  # lazy nn.Parameter

    def _maybe_init_B_cond_from_D(self, D: int, device, dtype):
        if self.B_cond is None:
            self.B_cond = nn.Parameter(torch.zeros(D, 2, device=device, dtype=dtype))

    def conductances_from_X(self, X_btd: torch.Tensor) -> torch.Tensor:
        # [B,T,D] @ [D,2] -> [B,T,2]
        x = torch.einsum("btd,dk->btk", X_btd, self.B_cond)
        return logOnePlusExpX_torch(x, self.gbar.to(x.device, x.dtype))

    def voltage(self, gs_bt2: torch.Tensor) -> torch.Tensor:
        return get_voltage_exp_recurrence_batched_loop(
            gs_bt2,
            E_s=self.E_s,
            g_l=self.g_l,
            E_l=self.E_l,
            V0=self.V0,
            dt_s=float(self.binsize_s),
        )  # [B,T]

    def forward(self, X_btd: torch.Tensor):
        if X_btd.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {X_btd.shape}")

        self._maybe_init_B_cond_from_D(X_btd.shape[-1], X_btd.device, X_btd.dtype)

        gs = self.conductances_from_X(X_btd)                          # [B,T,2]
        V  = self.voltage(gs)                                         # [B,T]
        rate = 1e-4 + firingRateNonlinearity(V, self.alpha, self.mu, self.beta)  # [B,T]
        return rate, {"V": V, "gs": gs}
    
    @torch.no_grad()
    def simulateSpikeTrains_trials(
        self,
        X_btd: torch.Tensor,        # [B,T,D]
        Y_init: torch.Tensor | None = None,  # [B,T0] or [B,T0,1] or None
        seed: int = 0,
        eps: float = 0.0,
        ) -> torch.Tensor:
        """
        Returns:
        sps: [B,T] in {0,1} (float)
        """
        if X_btd.ndim != 3:
            raise ValueError(f"X must be [B,T,D], got {X_btd.shape}")

        device, dtype = X_btd.device, X_btd.dtype
        B, T, _ = X_btd.shape

        rate_bt, _ = self.forward(X_btd)                 # [B,T] Hz
        lam_bt = rate_bt * float(self.binsize_s)         # [B,T]
        p_bt = (-torch.expm1(-lam_bt))                   # [B,T]  stable 1-exp(-lam)

        if eps > 0:
            p_bt = p_bt.clamp(min=eps, max=1.0 - eps)
        else:
            p_bt = p_bt.clamp(min=0.0, max=1.0)

        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        rs = torch.rand((B, T), generator=gen, device=device, dtype=dtype)

        sps = (rs < p_bt).to(dtype)                      # [B,T]

        if Y_init is not None:
            Y_init = torch.as_tensor(Y_init, device=device, dtype=dtype)
            if Y_init.ndim == 2:                         # [B,T0]
                Y_init = Y_init
            elif Y_init.ndim == 3 and Y_init.shape[-1] == 1:
                Y_init = Y_init[..., 0]                  # [B,T0]
            else:
                raise ValueError("Y_init must be [B,T0] or [B,T0,1]")

            T0 = Y_init.shape[1]
            if T0 >= T:
                raise ValueError("Y_init must have T0 < T")
            sps[:, :T0] = Y_init

        return sps
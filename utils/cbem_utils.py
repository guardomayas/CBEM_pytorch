# from ._raised_cosine_basis import makeRaisedCosBasis
import torch
import torch.nn.functional as F
# ----------------------------
# Utilities
# ----------------------------
def convolveStimulusWithBasis_torch(stimulus: torch.Tensor, basis: torch.Tensor, add_ones: bool = True):
    """
    stimulus: [T] or [T,Npix]
    basis:    [L,P]
    returns X: [T, Npix*P + (1 if add_ones else 0)]
    """
    if stimulus.ndim == 1:
        stimulus = stimulus[:, None]  # [T, Npix]
    T, Npix = stimulus.shape
    L, P = basis.shape

    x = stimulus.transpose(0, 1).unsqueeze(1)          # [Npix, 1, T]
    w = basis.transpose(0, 1).flip(1).unsqueeze(1)     # [P, 1, L]

    y = F.conv1d(F.pad(x, (L - 1, 0)), w)              # [Npix, P, T]
    X = y.permute(2, 0, 1).reshape(T, Npix * P)        # [T, Npix*P]

    if add_ones:
        ones = torch.ones(T, 1, device=X.device, dtype=X.dtype)
        X = torch.cat([X, ones], dim=1)
    return X

# ----------------------------
# Conductance Nonlinearit
# ----------------------------
def logOnePlusExpX_torch(x, maxG):
    """
    MATLAB logOnePlusExpX.m:
      if x <= -30: f = 1e-15
      elif x >= maxG: f = x
      else: f = log(1+exp(x))
    """
    x = torch.as_tensor(x)
    maxG = torch.as_tensor(maxG, device=x.device, dtype=x.dtype)
    while maxG.ndim < x.ndim:
        maxG = maxG.unsqueeze(0)

    f = x.clone()
    lessT = x <= -30.0
    greaterT = x >= maxG
    toFit = (~lessT) & (~greaterT)

    f[toFit] = F.softplus(x[toFit])
    f[lessT] = 1e-15
    return f


# ----------------------------
# Voltage recurrence
# ----------------------------
# def get_voltage_exp_recurrence(
#     gs: torch.Tensor,
#     E_s: torch.Tensor,
#     g_l: torch.Tensor,
#     E_l: torch.Tensor,
#     V0: torch.Tensor,
#     dt_s: float,
#     eps: float = 1e-12,
# ) -> torch.Tensor:
#     """
#     Exponential-Euler voltage recurrence for conductance-based membrane dynamics.

#     gs: [T, 2] conductances (excitatory, inhibitory)
#     E_s: [2] reversal potentials
#     returns: V [T]
#     """
#     gs = torch.as_tensor(gs)
#     E_s = torch.as_tensor(E_s, device=gs.device, dtype=gs.dtype)
#     g_l = torch.as_tensor(g_l, device=gs.device, dtype=gs.dtype)
#     E_l = torch.as_tensor(E_l, device=gs.device, dtype=gs.dtype)
#     V0 = torch.as_tensor(V0, device=gs.device, dtype=gs.dtype)

#     g_tot = gs.sum(dim=-1) + g_l                      # [T]
#     I_tot = (gs * E_s).sum(dim=-1) + g_l * E_l        # [T]
#     V_inf = I_tot / (g_tot + eps)                     # [T]
#     a = torch.exp(-dt_s * g_tot)                      # [T]

#     V = torch.empty(gs.shape[0], device=gs.device, dtype=gs.dtype)
#     v_prev = V0
#     for t in range(gs.shape[0]):
#         v_prev = a[t] * v_prev + (1.0 - a[t]) * V_inf[t]
#         V[t] = v_prev
#     return V

def get_voltage_exp_recurrence(gs, 
                               E_s, 
                               g_l,
                               E_l, 
                               V0, 
                               dt_s, 
                               eps=1e-12, chunk=256):
    """
    Exact same recurrence as the for-loop:
        v <- a[t]*v + (1-a[t])*Vinf[t]
    but computed in chunks to avoid Python-per-timestep overhead
    AND avoid global cumprod underflow.
    
    """
    g_tot = g_l + gs.sum(dim=1)                    # [T]
    I_tot = E_l * g_l + gs @ E_s                   # [T]
    V_inf = I_tot / (g_tot + eps)                  # [T]
    a = torch.exp(-dt_s * g_tot)                   # [T]
    b = (1.0 - a) * V_inf                          # [T]
    #does b/a so may be unstable. 
    T = a.shape[0]
    V = torch.empty_like(a)

    v_prev = torch.as_tensor(V0, device=gs.device, dtype=gs.dtype)

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        a_c = a[start:end]                         # [C]
        b_c = b[start:end]                         # [C]

        # Within-chunk closed form (safe because chunk is small):
        A = torch.cumprod(a_c, dim=0)              # [C], won't underflow for small C
        s = torch.cumsum(b_c / (A + eps), dim=0)   # [C]
        V_c = A * (v_prev + s)                     # [C]

        V[start:end] = V_c
        v_prev = V_c[-1]

    return V

def get_voltage_exp_recurrence_batched_loop(
    gs_bt2: torch.Tensor,  # [B,T,2]
    E_s: torch.Tensor,     # [2]
    g_l: torch.Tensor,     # scalar
    E_l: torch.Tensor,     # scalar
    V0: torch.Tensor,      # scalar
    dt_s: float,
    eps: float = 1e-12,
    chunk: int = 256,
) -> torch.Tensor:
    """
    Batched solver:
      - vectorized over trials (B)
      - processes time in chunks (optional)
    Returns V: [B,T]
    """
    B, T, _ = gs_bt2.shape

    gs = gs_bt2.to(torch.float64)

    E_s64 = E_s.to(gs.device, torch.float64)
    E_l64 = E_l.to(gs.device, torch.float64)
    g_l64 = g_l.to(gs.device, torch.float64)
    V064  = V0.to(gs.device, torch.float64)

    g_tot = g_l64 + gs.sum(dim=-1)                        # [B,T]
    I_tot = E_l64 * g_l64 + (gs * E_s64).sum(dim=-1)      # [B,T]
    E_inf = I_tot / (g_tot + eps)                         # [B,T]

    a = torch.exp(-dt_s * g_tot)                          # [B,T]
    b = (1.0 - a) * E_inf                                 # [B,T]

    V = torch.empty((B, T), device=gs.device, dtype=torch.float64)
    v_prev = V064.expand(B).clone()                       # [B]

    for start in range(0, T, chunk):
        end = min(start + chunk, T)

        a_c = a[:, start:end]                             # [B,C]
        b_c = b[:, start:end]                             # [B,C]

        # Within-chunk closed form (over time dim=1)
        P = torch.cumprod(a_c, dim=1)                     # [B,C]
        S = torch.cumsum(b_c / (P + eps), dim=1)          # [B,C]
        V_c = P * (v_prev[:, None] + S)                   # [B,C]

        V[:, start:end] = V_c
        v_prev = V_c[:, -1]                               # [B]

    return V.to(gs_bt2.dtype)

def firingRateNonlinearity(V_t, alpha, mu, beta):
    return alpha * F.softplus((V_t - mu) / beta)
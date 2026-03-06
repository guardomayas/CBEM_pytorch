import torch 

def poisson_nll_truncated_from_spike_bins(rate: torch.Tensor,
                                         spkTimes_bins: torch.Tensor,
                                         binsize_s: float,
                                         eps= 1e-8) -> torch.Tensor:
    """
    Truncated-Poisson / Bernoulli-per-bin likelihood.
     Based on Citi et. al 2014: product of T bernoulli trials. Works if max 1 spk/bin
    logp
    \log_p(y_{1:T}\mid \textbf{x}_{1:T}, \Theta ) = 
    \sum_{t=1}^T y_t \log(1- \exp(-\lambda_t \Delta ))-(1- y_t)\lambda_t\Delta
    
    Args:
    rate: [T] firing rate in spikes/second (Hz)
    binsize_s: seconds per bin (self.binsize_s)
    spkTimes_bins: indices (0..T-1) of bins that contain a spike (or >=1 spike)

    Log-likelihood per bin:
      ll_t = -lam_t                                  if y_t = 0
      ll_t = log(1 - exp(-lam_t))                    if y_t = 1
    where lam_t = rate_t * binsize_s  (expected count in bin)

    Returns:
      nll: scalar negative log-likelihood (sum over bins)
    """
    lam = rate * binsize_s                      # [T]
    ll = -lam                                   # default no-spike

    spk_idx = torch.as_tensor(spkTimes_bins, device=rate.device, dtype=torch.long)
    spk_idx = spk_idx[(spk_idx >= 0) & (spk_idx < lam.shape[0])]
    if spk_idx.numel() > 0:
        spk_idx = torch.unique(spk_idx)

        # log(1 - exp(-lam)) = log(-expm1(-lam)) stable for small lam
        p_spike = (-torch.expm1(-lam[spk_idx])).clamp_min(eps)  # in (0,1]
        ll_spike = torch.log(p_spike)

        ll = ll.clone()
        ll[spk_idx] = ll_spike

    return -ll.sum()
  
def cbem_penalized_nll_trials(
    model,
    X_btd: torch.Tensor,                 # [B,T,D]
    spk_bins_list: list[torch.Tensor],   # length B, each [Nspk_b]
    *,
    window: torch.Tensor | None = None,  # [W] indices into time
    conductance_penalty=(0.01, 0.001),
    eps_rate=1e-12
    ):
    """
    Penalized NLL across trials. Uses one forward pass.
    """
    B, T, D = X_btd.shape
    device = X_btd.device

    if window is not None:
        X_use = X_btd[:, window, :]           # [B,W,D]
        W = window.numel()
        # shift spikes into local window indices
        spk_use = []
        start = int(window[0].item())
        end   = int(window[-1].item()) + 1
        for b in range(B):
            idx = spk_bins_list[b]
            idxw = idx[(idx >= start) & (idx < end)] - start
            spk_use.append(idxw)
    else:
        X_use = X_btd
        W = T
        spk_use = spk_bins_list

    rate_bt, aux = model(X_use)               # rate [B,W], aux["gs"] [B,W,2]
    gs = aux["gs"]

    # Poisson NLL up to additive constants:
    # sum_t lam - sum_{spikes} log(rate)
    lam = rate_bt * float(model.binsize_s)    # [B,W]
    nll = lam.sum()

        
    for b in range(B):
        idx = spk_use[b]
        if idx.numel() > 0:
            p_spk = (-torch.expm1(-lam[b, idx])).clamp_min(eps_rate)  # 1-exp(-lam)
            nll = nll - torch.log(p_spk).sum()
    if conductance_penalty is not None:
      lam_e, lam_i = conductance_penalty
      pen = lam_e * gs[..., 0].mean() + lam_i * gs[..., 1].mean()
    else: 
      pen = 0
    return (nll / B) + pen

def cbem_penalized_nll(model,
                       stimulus: torch.Tensor,
                       spkTimes_bins: torch.Tensor,
                       window=None,
                       conductance_penalty=(1.0, 0.2)) -> torch.Tensor:
    """
    Penalized NLL:
      nll = truncated_poisson_nll(rate)
      + sum_c pen[c] * ||B_cond[:-1, c]||^2

    Excludes last row of B_cond (baseline term), matching JAX.
    """
    rate, _ = model(stimulus, window=window)  # calls forward
    nll = poisson_nll_truncated_from_spike_bins(rate, spkTimes_bins, model.binsize_s)

    B = model.B_cond
    if B is None:
        raise RuntimeError("model.B_cond is None. Run model once to initialize B_cond.")

    W = B[:-1, :]  # exclude baseline row
    pen = torch.as_tensor(conductance_penalty, device=W.device, dtype=W.dtype)  # [2]

    # generic form (works even if you later change #conductances)
    if pen.numel() != W.shape[1]:
        raise ValueError(f"conductance_penalty has {pen.numel()} elements but B_cond has {W.shape[1]} columns.")

    nll = nll + (pen * (W**2).sum(dim=0)).sum()
    return nll
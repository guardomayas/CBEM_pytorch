import torch
from .loss import cbem_penalized_nll_trials
def train_cbem_trials(
    model,
    X_cond: torch.Tensor,                 # [B,T,D]
    spkTimes_bins: list[torch.Tensor],    # length B, each 1D long (spike bins)
    *,
    lr=1e-2,
    weight_decay=0.0,
    conductance_penalty=(0.01, 0.001),
    n_steps=2000,
    print_every=100,
    clip_grad_norm=1.0,
    window_size=None,
    seed=0,
):
    torch.manual_seed(seed)
    device = X_cond.device

    if X_cond.ndim != 3:
        raise ValueError(f"X_cond must be [B,T,D], got {X_cond.shape}")
    B, T, D = X_cond.shape
    if len(spkTimes_bins) != B:
        raise ValueError(f"len(spkTimes_bins) must be {B}, got {len(spkTimes_bins)}")

    # Ensure parameters exist (B_cond is lazy)
    with torch.no_grad():
        _ = model(X_cond[:, :10, :])
    print("model device:", next(model.parameters()).device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_vals, history_steps = [], []

    for step in range(1, n_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        if window_size is None:
            window = None
        else:
            W = min(int(window_size), T)
            start = torch.randint(low=0, high=max(1, T - W + 1), size=(1,), device=device).item()
            window = torch.arange(start, start + W, device=device, dtype=torch.long)

        loss = cbem_penalized_nll_trials(
            model=model,
            X_btd=X_cond,
            spk_bins_list=spkTimes_bins,
            window=window,
            conductance_penalty=conductance_penalty,
        )

        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))

        opt.step()

        if step == 1 or (step % print_every) == 0:
            model.eval()
            with torch.no_grad():
                rate_bt, _ = model(X_cond)  # [B,T]
                lam_bt = rate_bt * float(model.binsize_s)
                mean_rate_hz = float(rate_bt.mean())
                mean_p = float((-torch.expm1(-lam_bt)).mean())
            loss_vals.append(float(loss.detach().cpu()))
            history_steps.append(step)
            print(f"step {step:5d} | loss {float(loss):.3f} | mean rate {mean_rate_hz:.3f} Hz | mean p {mean_p:.6f}")

    return model, (history_steps, loss_vals)
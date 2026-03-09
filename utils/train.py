import torch
from .loss import cbem_penalized_nll_trials


def train_cbem_trials(
    model,
    X_cond: torch.Tensor,                 # [B,T,D]
    basis_matrix: torch.Tensor,           # [L,P]
    spkTimes_bins: list[torch.Tensor],    # length B, each 1D long (spike bins)
    *,
    lr=1e-2,
    weight_decay=0.0,
    conductance_penalty=None,
    n_steps=2000,
    print_every=100,
    clip_grad_norm=1.0,
    window_size=None,
    seed=0):
    
    torch.manual_seed(seed)
    device = X_cond.device

    if X_cond.ndim != 3:
        raise ValueError(f"X_cond must be [B,T,D], got {X_cond.shape}")
    B, T, D = X_cond.shape
    if len(spkTimes_bins) != B:
        raise ValueError(f"len(spkTimes_bins) must be {B}, got {len(spkTimes_bins)}")

    # ---- Lazy init guard for B_cond ----
    bcond = getattr(model, "B_cond", None)

    if bcond is None:
        # Not initialized yet -> trigger it with a tiny forward pass
        with torch.no_grad():
            _ = model(X_cond[:, :1, :])
        bcond = getattr(model, "B_cond", None)
        if bcond is None:
            raise RuntimeError(
                "Model did not create B_cond during forward(). "
                "Expected lazy init in forward/_maybe_init_B_cond_from_D."
            )
        print(f"[train] Initialized B_cond with zeros with shape {tuple(bcond.shape)}")
    else:
        # Already initialized: sanity check dimensions
        if bcond.ndim != 2:
            raise ValueError(f"B_cond should be rank-2 [D,K], got {bcond.shape}")
        if bcond.shape[0] != D:
            raise ValueError(
                f"B_cond already initialized with D={bcond.shape[0]}, "
                f"but X_cond has D={D}. (Did you change the design matrix?)"
            )
        print(f"[train] Using existing B_cond with shape {tuple(bcond.shape)}")

    # Device sanity print (handles models with only buffers + B_cond)
    first_param = next(model.parameters(), None)
    if first_param is None:
        raise RuntimeError("Model has no parameters (did B_cond become a Parameter?)")
    print("model device:", first_param.device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_vals, history_steps = [], []

    for step in range(1, n_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        if window_size is None:
            window = None
        else:
            W = min(int(window_size), T)
            start = torch.randint(
                low=0, high=max(1, T - W + 1), size=(1,), device=device
            ).item()
            window = torch.arange(start, start + W, device=device, dtype=torch.long)

        loss = cbem_penalized_nll_trials(
            model=model,
            X_btd=X_cond,
            basis_matrix=basis_matrix,
            spk_bins_list=spkTimes_bins,
            window=window,
            conductance_penalty=conductance_penalty,
        )

        loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(clip_grad_norm)
            )

        opt.step()

        if step == 1 or (step % print_every) == 0:
            model.eval()
            with torch.no_grad():
                rate_bt, _ = model(X_cond)  # [B,T]
                lam_bt = rate_bt * float(model.binsize_s)
                mean_rate_hz = float(rate_bt.mean())
                mean_p = float((-torch.expm1(-lam_bt)).mean())
                if conductance_penalty is not None:
                    lam_e, lam_i, lam_se, lam_si = conductance_penalty
                    ke = model.B_cond[:-1, 0]
                    ki = model.B_cond[:-1, 1]

                    f_e = basis_matrix @ ke
                    f_i = basis_matrix @ ki

                    d2_fe = f_e[2:] - 2*f_e[1:-1] + f_e[:-2]
                    d2_fi = f_i[2:] - 2*f_i[1:-1] + f_i[:-2]

                    pen_w = 0.5 * (lam_e * torch.sum(ke**2) + lam_i * torch.sum(ki**2))
                    pen_sm = 0.5 * lam_se * torch.sum(d2_fe**2) + 0.5 * lam_si * torch.sum(d2_fi**2)

                else: 
                    pen_w = 0.0
                    pen_sm = 0.0
                    
                data_term = loss - pen_w - pen_sm
                            
                
            loss_vals.append(float(loss.detach().cpu()))
            history_steps.append(step)
            print(
                    f"step {step:5d} | "
                    f"loss {loss:.3f} | "
                    f"data {data_term:.3f} | "
                    f"pen_w {pen_w:.3f} | "
                    f"pen_sm {pen_sm:.3f} | "
                    f"mean rate {mean_rate_hz:.3f} Hz"
                )

    return model, (history_steps, loss_vals)
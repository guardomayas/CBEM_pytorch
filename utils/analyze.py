from .cbem_utils import convolveStimulusWithBasis_torch
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(trained, cbem_lin, 
                   stimulus_test, 
                   B_orth_t, dt, 
                   psth, plot=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained.eval()
    ##Design Matrix test stim
    stim_test_t = torch.as_tensor(stimulus_test, dtype=torch.float64, device=device)
    # stimulus_t: [B, T]
    # basis_t: [L, P]
    X_test_tr = [
        convolveStimulusWithBasis_torch(stim_test_t[b], B_orth_t, add_ones=True)  # -> [T, P+1]
        for b in range(stimulus_test.shape[0])
    ]
    X_test = torch.stack(X_test_tr, dim=0)  # [B,T,D]
    print('X_full:', X_test.shape)

    with torch.no_grad(): ## LIN MODEL STATS
        psth_lin, aux_lin = cbem_lin(X_test)  # [B,T]
        
    with torch.no_grad(): ## FULL MODEL
        rate_test, aux = trained(X_test)  # [B,T]
    B = X_test.shape[0]
    T0 = 500
    Y_init = torch.zeros(B, T0, device=X_test.device, dtype=X_test.dtype)  # [B,T0]
    sps_t = trained.simulateSpikeTrains_trials(X_test, Y_init, seed=0).detach().cpu().numpy()
    
    rate_hat = gaussian_filter1d(rate_test[0].detach().cpu(), sigma=max(1, int(0.02 / dt)))  # 20 ms
    rate_lin = gaussian_filter1d(psth_lin[0].detach().cpu(), sigma=max(1, int(0.02 / dt)))  # 20 ms
    
    r2_full = r2_score(psth, rate_hat)
    r2_lin  = r2_score(psth, rate_lin)
    
    
    ## Filtes 
    B_cond_learned = trained.B_cond.detach().cpu()
    # Learned conductance filters in time domain
    B = trained.B_cond.detach().cpu()   # [D, 2]
    np.savez("B_trained", B)
    W_orth = B[:-1, :]                  # drop bias row

    # Borth = torch.as_tensor(B_orth, dtype=W_orth.dtype)
    filters = B_orth_t @ W_orth    

    results = {
        "r2_cbem_full": r2_full,
        "r2_cbem_lin" : r2_lin, 
        "K_exc"       : filters[:, 0],
        "K_inh"       : filters[:, 1],
        "rate_full"   : rate_hat,
        "rate_lin"    : rate_lin
        
    }
    
    if plot: 
        t_ms = np.arange(filters.shape[0]) * dt * 1e3
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        axes[0].plot(t_ms, filters[:, 0].numpy(), lw=2.0)
        axes[0].set_title('Excitatory Filter')
        axes[0].set_xlabel('Lag (ms)')

        axes[1].plot(t_ms, filters[:, 1].numpy(), lw=2.0)
        axes[1].set_title('Inhibitory Filter')
        axes[1].set_xlabel('Lag (ms)')

        for ax in axes:
            ax.axhline(0.0, color='k', linestyle= '--', alpha=0.5)

        plt.tight_layout()
        plt.show()
    return results
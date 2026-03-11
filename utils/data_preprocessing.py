import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100
def bin_one_trial(spike_times_s, edges_s):
    spikes = np.asarray(spike_times_s, dtype=np.float64)
    counts, _ = np.histogram(spikes, bins=edges_s)
    return counts

def upsample_hold_edges(I_frame, dt_stim_s, dt_cell_s, T_cell):
    t_cell = np.arange(T_cell) * dt_cell_s
    idx = np.floor(t_cell / dt_stim_s).astype(np.int32)
    idx = np.clip(idx, 0, I_frame.shape[-1] - 1)
    return I_frame[..., idx]

def preprocess_split(stim_frames, spk_times_obj,
                     presamples, postsamples, 
                     T_cell_full,
                     dt_stim, dt, 
                     bin_edges_s, 
                     n_trials=None):
    if n_trials is None:
        n_trials = len(spk_times_obj)

    spk_full = np.stack(
        [bin_one_trial(spk_times_obj[trial], bin_edges_s) for trial in range(n_trials)],
        axis=0,
    )
    stim_full = upsample_hold_edges(
        stim_frames[:n_trials], dt_stim_s=dt_stim, dt_cell_s=dt, T_cell=T_cell_full
    )
    if presamples is not None: 
        t0 = presamples
        t1 = T_cell_full - postsamples
        stim_full = stim_full[:, t0:t1]
        spk_full = spk_full[:, t0:t1]
        
    return stim_full, spk_full


def plot_responses(spk_t_s_train, 
                   spk_t_s_test, 
                   psth_hz, 
                   psth_train,
                   cell_type, 
                   dt, B, B_tr, T
                   ):

    t_sec = np.arange(T) * dt
    tmin, tmax = 0.0, min(5.0, t_sec[-1])  
    fig, axes= plt.subplots(
        2, 2, figsize=(11, 5),
        sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    fig.suptitle("     RGC responses to gaussian white noise light", y=0.96)
    fig.text(0.5, 0.82, f"Cell type: {cell_type}", ha="center", fontsize=20)

    raster = axes[0,:]
    psths  = axes[1, :]

    ## NON REPEATED SEED
    raster[0].eventplot(
        spk_t_s_train,
        lineoffsets=np.arange(1,B_tr+1),
        linelengths=0.8,
        colors="k",
        alpha=0.8,
    )

    raster[0].set_ylim(0, B_tr+1)
    raster[0].set_yticks([1,B_tr])

    psths[0].plot(t_sec, psth_train, label= 'PSTH')

    # REPEATED SEED
    raster[1].eventplot(
        spk_t_s_test,
        lineoffsets=np.arange(1,B+1),
        linelengths=0.8,
        colors="k",
        alpha=0.8,
    )

    raster[1].set_ylim(0, B+1)
    raster[1].set_yticks([1,B])

    # PSTH
    psths[1].plot(t_sec, psth_hz, label="PSTH")

    for ax in raster:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    for ax in psths:
        ax.set_ylim(-1,psth_hz.max()+10)

    raster[0].set_ylabel("Trial")

    psths[0].set_ylabel("Firing rate \n (Hz)")
    fig.text(0.5, -0.025, "Time (s)")
    psths[0].set_xticks([0,5,10])
    psths[0].legend(fontsize=12) 
    
    fig.tight_layout()
    # fig.savefig(fname='Off_tr_merf_psth', bbox_inches='tight', dpi=300)
    plt.show()
    
    return fig
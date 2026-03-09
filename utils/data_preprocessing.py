import numpy as np

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

    t0 = presamples
    t1 = T_cell_full - postsamples
    return stim_full[:, t0:t1], spk_full[:, t0:t1]
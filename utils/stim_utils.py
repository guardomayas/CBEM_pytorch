import seaborn as sns
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

def generate_1_f_noise(beta, num_samples, frame_rate, frame_dwell,
                       mean_intensity, desired_contrast, noise_seed=1, eps=1e-12):

    key = jax.random.PRNGKey(noise_seed)

    # Frequency bins for real FFT
    freqs = jnp.fft.rfftfreq(num_samples, d=1.0 / frame_rate)

    # 1/f^(beta/2) amplitude scaling (skip DC to avoid div-by-zero)
    amp = jnp.where(freqs > 0, freqs ** (-beta / 2.0), 0.0)

    # Random phases (uniform on [0, 2pi))
    key, subkey = jax.random.split(key)
    phases = jax.random.uniform(subkey, shape=freqs.shape, minval=0.0, maxval=2.0 * jnp.pi)
    spectrum = amp * jnp.exp(1j * phases)

    # Back to time domain
    raw_noise = jnp.fft.irfft(spectrum, n=num_samples)

    # Normalize to unit std (avoid NaNs if something degenerates)
    raw_noise = raw_noise / (jnp.std(raw_noise) + eps)

    # Apply contrast around mean intensity
    noise_intensity_adj = mean_intensity * (1.0 + desired_contrast * raw_noise)
    noise_intensity_adj = jnp.clip(noise_intensity_adj, 0.0, 1.0)

    contrast_values = (noise_intensity_adj - mean_intensity) / mean_intensity

    # Frame dwell
    noise_intensity = jnp.repeat(noise_intensity_adj, frame_dwell)
    contrast_over_time = jnp.repeat(contrast_values, frame_dwell)

    return noise_intensity, contrast_over_time

def plot_stim(noise_t, contrast_mat, beta, 
              mean_intensity, desired_contrast, 
              tau=10, frame_rate=60,figsize=(20, 8)):
    time = jnp.arange(len(contrast_mat[0,:])) / frame_rate


    valid_indices = jnp.arange(len(noise_t) - tau)
    intensity_pairs = jnp.vstack((noise_t[valid_indices], noise_t[valid_indices + tau])).T

    hist, xedges, yedges = jnp.histogram2d(intensity_pairs[:, 0], intensity_pairs[:, 1], bins=50, density=True)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(rf"$\frac{{1}}{{f^\beta}}$ noise. $\beta={beta}$, Mean Intensity = {mean_intensity}, Contrast = {desired_contrast}")
    # plt.subplots_adjust(top=1.1, bottom=0.1, left=0.1, right=0.9, hspace=0.6, wspace=0.4)

    # **Plot Contrast Over Time**
    axes[0].plot(time, contrast_mat[0,:], linewidth=1, c=sns.color_palette("dark")[4])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Contrast")
    axes[0].set_title("Contrast Over Time")
    # axes[0].set_xlim([0, 15])
    axes[0].set_ylim([-1.1,1.1])
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

    # **Plot Two-Point Intensity Distribution**
    img = axes[1].imshow(hist.T, origin="lower", extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                        aspect='auto', cmap="inferno")  # Keep probability density between 0 and 1
    cbar = fig.colorbar(img, ax=axes[1])
    cbar.set_label("Probability Density")
    # axes[1].set_xlim(0, 1)
    # axes[1].set_ylim(0, 1)

    axes[1].set_xlabel("Intensity at t")
    axes[1].set_ylabel(f"Intensity at t+{tau} frames \n ({tau/frame_rate*1000:.2f} ms)")
    axes[1].set_title("Two-Point Intensity Distribution\n" 
                    r"$1/f^\beta$ Noise")
    ## intensity distribution
    bins = jnp.linspace(-0.2, 1.2, 200)
    axes[2].hist(noise_t, bins=bins, alpha=0.6, label='Raw (before clipping)', density=True)
    axes[2].axvline(0, color='red', linestyle='--')
    axes[2].axvline(1, color='red', linestyle='--')
    axes[2].set_xlabel("Intensity")
    axes[2].set_ylabel("Probabiltiy")
    sns.set_context('poster')
    for ax in axes:
        ax.grid(False)
    plt.tight_layout()
    plt.show()
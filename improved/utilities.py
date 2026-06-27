"""
utilities.py — Data loading, preprocessing, and visualization utilities
for the GW anomaly detection pipeline.

Q-transform version: replaces STFT spectrogram with gwpy Q-transform.

Changes from spectrogram version:
- compute_spectrogram  → compute_qtransform
- preprocess_waveforms_to_spectrograms → preprocess_waveforms_to_qtransforms
- load_and_inject_spectrograms → load_and_inject_qtransforms
- Noise downsampled from 8192 Hz → 4096 Hz before processing
- Q-transform parameters: frange=(10,2048), qrange=(4,64), output log-scaled mag
- All other logic (whitening, normalization, train/test split) unchanged
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import welch, resample
from scipy.ndimage import zoom
from multiprocessing import Pool


# --------------------------
# Whitening
# --------------------------
def whiten(x, fs, nperseg=1024, noise_psd=None):
    '''
    Whiten a waveform by dividing by the square root of the PSD.
    If noise_psd is provided as (f, psd), use that instead of estimating from x.
    '''
    if noise_psd is not None:
        f, psd = noise_psd
    else:
        f, psd = welch(x, fs=fs, nperseg=nperseg)
    X          = np.fft.rfft(x)
    freqs      = np.fft.rfftfreq(len(x), 1.0 / fs)
    psd_interp = np.interp(freqs, f, psd)
    X_white    = X / np.sqrt(psd_interp + 1e-40)
    x_white    = np.fft.irfft(X_white, n=len(x))
    return x_white


# --------------------------
# Normalization
# --------------------------
def min_max_norm(x):
    '''
    Per-sample min-max normalization to [0, 1].
    x: numpy array of shape (N, H, W)
    '''
    x_min = x.min(axis=(1, 2), keepdims=True)
    x_max = x.max(axis=(1, 2), keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-12)


def noise_referenced_norm(x, noise_mean, noise_std):
    '''
    Normalize by noise population statistics, preserving relative amplitude.

    Unlike per-sample min-max, this keeps the amplitude of each Q-transform
    relative to the noise floor: a faint injection at SNR=10 stays near 0
    while a loud one at SNR=100 produces pixel values well above 0.  This
    gives the CAE an amplitude-based discriminant in addition to shape.

    x          : numpy array, shape (N, H, W)
    noise_mean : scalar — mean of Q-transform pixels across the noise pool
    noise_std  : scalar — std  of Q-transform pixels across the noise pool
    Returns    : (x - noise_mean) / noise_std, same shape as x
    '''
    return (x - noise_mean) / (noise_std + 1e-12)


# --------------------------
# Q-transform computation
# --------------------------
def compute_qtransform(waveform, sample_rate=4096, frange=(10, 2048),
                       qrange=(4, 64), target_shape=(256, 128)):
    '''
    Compute Q-transform of a 2s waveform segment using gwpy.

    Returns 2D numpy array resized to target_shape (freq × time),
    log-scaled magnitude, normalized to [0, 1].

    Parameters
    ----------
    waveform    : 1D numpy array, length sample_rate*2 (2 seconds)
    sample_rate : int, Hz (default 4096)
    frange      : (f_low, f_high) Hz
    qrange      : (q_min, q_max)
    target_shape: (n_freq, n_time) output shape after resize
    '''
    from gwpy.timeseries import TimeSeries

    ts  = TimeSeries(waveform, sample_rate=sample_rate)
    qgram = ts.q_transform(
        qrange=qrange,
        frange=frange,
        tres=0.002,
        fres=0.5,
        norm='median',
        whiten=False,   # whitening is done upstream
    )

    # qgram is a Spectrogram with shape (n_time, n_freq); transpose to (n_freq, n_time)
    mag = np.array(qgram).T  # shape: (n_freq, n_time)

    # log scale — add 1 to avoid log(0)
    mag = np.log1p(np.abs(mag))

    # resize to target_shape using bilinear-equivalent (zoom)
    zoom_f = target_shape[0] / mag.shape[0]
    zoom_t = target_shape[1] / mag.shape[1]
    mag    = zoom(mag, (zoom_f, zoom_t), order=1)

    return mag.astype(np.float32)


def _qtransform_worker(args):
    """Module-level worker for multiprocessing.Pool — must be picklable."""
    waveform, sample_rate, frange, qrange, target_shape = args
    return compute_qtransform(waveform, sample_rate=sample_rate,
                              frange=frange, qrange=qrange,
                              target_shape=target_shape)


def preprocess_waveforms_to_qtransforms(waveforms, sample_rate=4096,
                                         frange=(10, 2048), qrange=(4, 64),
                                         target_shape=(256, 128), n_workers=32):
    '''
    Compute Q-transforms for a batch of waveforms using a multiprocessing Pool.

    Parameters
    ----------
    waveforms : (N, T) array, already at sample_rate Hz
    n_workers : number of parallel workers (default 32)
    Returns   : (N, 256, 128) float32 array
    '''
    args = [(w, sample_rate, frange, qrange, target_shape) for w in waveforms]
    with Pool(processes=n_workers) as pool:
        results = pool.map(_qtransform_worker, args)
    return np.array(results)


# --------------------------
# Amplitude matching
# --------------------------
def match_rms_batch(x, target_rms):
    '''
    Scale each waveform in x so its RMS matches target_rms.
    x: (N, T), target_rms: (N,)
    '''
    current_rms = np.sqrt(np.mean(x**2, axis=1))
    scale       = target_rms / (current_rms + 1e-12)
    return x * scale[:, None]


# --------------------------
# Reference PSD estimation
# --------------------------
def estimate_reference_psd(noise_segments, fs, nperseg):
    '''
    Estimate a reference PSD by averaging per-segment Welch PSDs.
    noise_segments: (N, T) array
    Returns: (f, psd_ref) tuple, ready to pass as noise_psd to whiten()
    '''
    psds = []
    for seg in noise_segments:
        f, psd = welch(seg, fs=fs, nperseg=nperseg)
        psds.append(psd)
    return f, np.mean(psds, axis=0)


# --------------------------
# Downsampling helper
# --------------------------
def downsample_waveforms(waveforms, fs_in, fs_out):
    '''
    Downsample (N, T) array from fs_in to fs_out using scipy.signal.resample.
    '''
    n_out = int(waveforms.shape[1] * fs_out / fs_in)
    return resample(waveforms, n_out, axis=1)


# --------------------------
# Main data loading function
# --------------------------
def load_and_inject_qtransforms(samples_dict, n_files=50, sample_freq_noise=8192,
                                 sample_freq=4096, frange=(10, 2048), qrange=(4, 64),
                                 random_state=42, apply_whiten=True, noise_psd=None):
    '''
    Load waveforms, inject signals into noise, optionally whiten,
    compute Q-transforms.

    samples_dict format:
        {label: [directory, signal_bool, glitch_bool]}
        signal_bool: if True, compute RMS for amplitude matching
        glitch_bool: if True, scale amplitude to match signal RMS

    apply_whiten: if False, skip whitening entirely.

    Noise is downsampled from sample_freq_noise to sample_freq before processing.
    Signals are assumed to already be at sample_freq.

    Returns:
        qtransform_dict: {label: {'train': tensor, 'test': tensor}}
    '''
    rng = np.random.default_rng(random_state)

    # --- Pass 1: load raw waveforms ---
    signal_rms = None
    raw = {}
    for sample_key in samples_dict.keys():
        files = sorted(os.listdir(samples_dict[sample_key][0]))[:n_files]
        waveforms = np.concatenate([
            np.load(os.path.join(samples_dict[sample_key][0], f))
            for f in files
        ])

        if sample_key == 'Noise' and sample_freq_noise != sample_freq:
            print(f"Downsampling noise from {sample_freq_noise} Hz to {sample_freq} Hz...")
            waveforms = downsample_waveforms(waveforms, sample_freq_noise, sample_freq)

        if samples_dict[sample_key][1]:
            signal_rms = np.sqrt(np.mean(waveforms**2, axis=1))

        raw[sample_key] = waveforms

    # Align lengths: noise and signals must have same T
    noise_len = raw['Noise'].shape[1]
    for k in raw:
        if k != 'Noise':
            sig_len = raw[k].shape[1]
            if sig_len > noise_len:
                raw[k] = raw[k][:, :noise_len]
            elif sig_len < noise_len:
                raw[k] = np.pad(raw[k], ((0, 0), (0, noise_len - sig_len)))

    # --- Split raw waveforms into train/test BEFORE injection ---
    raw_train, raw_test = {}, {}
    for sample_key, waveforms in raw.items():
        n_total   = len(waveforms)
        n_test    = int(n_total * 0.2)
        idx       = rng.permutation(n_total)
        test_idx  = idx[:n_test]
        train_idx = idx[n_test:]
        raw_train[sample_key] = waveforms[train_idx]
        raw_test[sample_key]  = waveforms[test_idx]

    # --- Reference PSD for whitening ---
    nperseg_psd = 1024
    if not apply_whiten:
        print("Skipping whitening (apply_whiten=False)...")
        noise_psd = None
    elif noise_psd is not None:
        print("Using supplied reference PSD for whitening.")
    else:
        print("Estimating reference noise PSD...")
        noise_psd = (lambda f, p: (f, p))(
            *estimate_reference_psd(raw['Noise'], fs=sample_freq, nperseg=nperseg_psd)
        )

    def compute_qts_raw(waveforms):
        '''Q-transform without normalization — returns raw float32 numpy array.'''
        if apply_whiten:
            processed = [whiten(w, fs=sample_freq, nperseg=nperseg_psd,
                                noise_psd=noise_psd) for w in waveforms]
        else:
            processed = list(waveforms)
        return preprocess_waveforms_to_qtransforms(
            np.array(processed), sample_rate=sample_freq,
            frange=frange, qrange=qrange)

    def inject_and_compute_qts_raw(signal_waveforms, noise_pool):
        '''Inject signals into noise, then compute Q-transform without normalization.'''
        n        = len(signal_waveforms)
        idx      = rng.choice(len(noise_pool), size=n, replace=(n > len(noise_pool)))
        injected = signal_waveforms + noise_pool[idx]
        return compute_qts_raw(injected)

    # --- Build train/test splits — raw Q-transforms (no normalisation yet) ---
    raw_qts = {}

    for sample_key in samples_dict.keys():
        print(f"Processing: {sample_key}")

        if sample_key == 'Noise':
            raw_qts[sample_key] = {
                'train': compute_qts_raw(raw_train['Noise']),
                'test':  compute_qts_raw(raw_test['Noise']),
            }
        else:
            sig_train = raw_train[sample_key]
            sig_test  = raw_test[sample_key]

            if samples_dict[sample_key][2] and signal_rms is not None:
                sig_train = match_rms_batch(sig_train, signal_rms[:len(sig_train)])
                sig_test  = match_rms_batch(sig_test,  signal_rms[:len(sig_test)])

            raw_qts[sample_key] = {
                'train': inject_and_compute_qts_raw(sig_train, raw_train['Noise']),
                'test':  inject_and_compute_qts_raw(sig_test,  raw_test['Noise']),
            }

    # --- Compute normalisation stats from noise pool only ---
    noise_all  = np.concatenate([raw_qts['Noise']['train'],
                                  raw_qts['Noise']['test']], axis=0)
    noise_mean = float(noise_all.mean())
    noise_std  = float(noise_all.std())
    print(f"Normalization: min-max per sample (noise pool stats for reference: mean={noise_mean:.4f}  std={noise_std:.4f})")

    # --- Apply normalisation and convert to tensors ---
    qtransform_dict = {'_noise_stats': {'mean': noise_mean, 'std': noise_std}}

    for sample_key in raw_qts:
        qtransform_dict[sample_key] = {
            split: torch.tensor(
                min_max_norm(raw_qts[sample_key][split]).astype(np.float32),
                dtype=torch.float32,
            )
            for split in ('train', 'test')
        }

    return qtransform_dict


# --------------------------
# DataLoader helper
# --------------------------
def get_train_test_loader(qtransform_dict, batch_size=64):
    '''
    Build train/test DataLoaders from qtransform_dict.
    '''
    X_train = torch.cat([qtransform_dict[k]['train'] for k in qtransform_dict])
    X_test  = torch.cat([qtransform_dict[k]['test']  for k in qtransform_dict])

    y_train = torch.cat([torch.full((len(qtransform_dict[k]['train']),), i, dtype=torch.long)
                         for i, k in enumerate(qtransform_dict)])
    y_test  = torch.cat([torch.full((len(qtransform_dict[k]['test']),),  i, dtype=torch.long)
                         for i, k in enumerate(qtransform_dict)])

    if X_train.ndim == 3:
        X_train = X_train.unsqueeze(1)
        X_test  = X_test.unsqueeze(1)

    perm    = torch.randperm(len(X_train))
    X_train = X_train[perm]; y_train = y_train[perm]
    perm    = torch.randperm(len(X_test))
    X_test  = X_test[perm];  y_test  = y_test[perm]

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# --------------------------
# Visualization
# --------------------------
def visualize_sample_qtransforms(qtransforms, labels, class_names, n_samples=3):
    '''
    Visualize sample Q-transforms from each class.
    qtransforms: Tensor of shape [N, 1, H, W]
    labels:      Tensor of shape [N]
    '''
    n_classes = len(class_names)
    labels    = labels.cpu()

    fig, axes = plt.subplots(n_classes, n_samples,
                             figsize=(3 * n_samples, 3 * n_classes))

    if n_classes == 1:
        axes = np.expand_dims(axes, 0)
    if n_samples == 1:
        axes = np.expand_dims(axes, 1)

    for class_idx in range(n_classes):
        class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
        if len(class_indices) == 0:
            continue
        for sample_idx in range(min(n_samples, len(class_indices))):
            spec_idx = class_indices[sample_idx].item()
            qt       = qtransforms[spec_idx, 0].cpu().numpy()
            ax       = axes[class_idx, sample_idx]
            im       = ax.imshow(qt, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'{class_names[class_idx]} #{sample_idx+1}', fontsize=10)
            ax.set_xlabel('Time', fontsize=8)
            if sample_idx == 0:
                ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    return fig

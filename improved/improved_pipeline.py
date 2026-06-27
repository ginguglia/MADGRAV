"""
Best baseline pipeline for GW anomaly detection from this workspace.

This script trains a baseline CAE jointly for reconstruction and classification:

- Baseline CAE with ReLU nonlinearities
- Q-transform input only
- Joint noise reconstruction + signal/noise classification from epoch 1
- MDC evaluation with thresholds derived from validation noise

It is intended to be run from /scratch/ginguglia/GW/codex-agent.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from itertools import accumulate
from multiprocessing import Pool
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from scipy.ndimage import zoom
from scipy.signal import filtfilt, iirnotch
from torch import optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGENT_GW_DIR = os.path.join(MADGRAV_ROOT, "improved")   # utilities.py is vendored here
if AGENT_GW_DIR not in sys.path:
    sys.path.insert(0, AGENT_GW_DIR)

from utilities import whiten, downsample_waveforms, min_max_norm
from prepare_o1_data import load_frame


SEED = 42
FS_NOISE = 8192
FS = 4096
SPEC_SIZE = (256, 128)
QTRANSFORM_FRANGE = (10, 1291)
QTRANSFORM_QRANGE = (4, 64)
QTRANSFORM_N_WORKERS = 16
QTRANSFORM_CHUNK_SIZE = 64
O1_CONTEXT_SECONDS = 2.0
O1_CENTER_CROP_SECONDS = 1.0
O1_WHITEN_FDURATION = 2.0
MASS_BIN_EDGES = (10.0, 60.0, 100.0, 240.0)
DISTANCE_BIN_EDGES = (100.0, 1000.0, 5000.0)
MASS_BIN_LABELS = ("10-60", "60-100", "100-240")
DISTANCE_BIN_LABELS = ("100-1000", "1000-5000")
POWERLINE_BASE_HZ = 60.0
POWERLINE_HARMONICS = tuple(POWERLINE_BASE_HZ * n for n in range(1, 9))
PROJECTED_TARGET_NETWORK_SNR_RANGE = (8.0, 25.0)
O1_CAL_LINES = {
    "H1": (35.9, 36.7, 37.3, 331.9),
    "L1": (33.7, 34.7, 35.3, 331.3),
}
EARLY_O1_L1_DITHER_LINES = (600.1, 625.1, 650.1, 675.1)
O3A_CAL_LINES = {
    "H1": (15.1, 15.6, 16.4, 16.7, 17.1, 17.6, 35.9, 36.7, 331.9, 410.3, 1001.3, 1083.7, 1153.1, 1501.3),
    "L1": (15.1, 15.7, 16.3, 16.9, 30.8, 31.4, 32.0, 32.6, 33.2, 33.8, 434.9, 451.2, 451.8, 1083.1, 1153.1, 1503.1, 1653.1),
}

# Training-only data dirs (NOT used by the search/inference path). Env-overridable so the
# package ships no absolute host paths; set these only if you run the training utilities.
NOISE_DIR = os.environ.get("MADGRAV_NOISE_DIR", "")
SIGNAL_DIR = os.environ.get("MADGRAV_SIGNAL_DIR", "")
O1_SIGNAL_BANK_DIR = os.path.join(os.getcwd(), "data", "o1_signal_bank")
MDC_DIR = os.environ.get("MADGRAV_MDC_DIR", "")
PSD_FILE = os.environ.get("MADGRAV_PSD_FILE", "")
O1_PREPARED_DIR = os.path.join(os.getcwd(), "data", "o1_prepared")
WEAKSUP_CRITERION = nn.BCEWithLogitsLoss()
LABEL_SMOOTH_NOISE = 0.05
LABEL_SMOOTH_SIGNAL = 0.95
CHECKPOINT_GATE_ACC_MIN = 0.3
NOISE_ACC_EARLY_STOP_EPOCHS = 2


@dataclass
class Config:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.20
    recon_weight: float = 0.1
    clf_weight: float = 1.0
    epochs: int = 20
    weaksup_es_patience: int = 10
    n_files: int = 50
    bottleneck: int = 128
    num_workers: int = 0
    output_dir: str = os.path.join(os.getcwd(), "outputs_best_baseline")
    dataset_mode: str = "synthetic"
    o1_data_dir: str = O1_PREPARED_DIR
    o1_signal_bank_dir: str = O1_SIGNAL_BANK_DIR
    o1_inj_eval_count: int = 100
    o1_notch_lines: bool = True
    detector_mode: str = "both"
    device: str = ""
    training_mode: str = "joint"
    unsup_epochs: int = 10
    weaksup_epochs: int = 10
    checkpoint_gate_acc_min: float = CHECKPOINT_GATE_ACC_MIN
    noise_acc_early_stop_epochs: int = NOISE_ACC_EARLY_STOP_EPOCHS
    margin: float = 3.0
    lambda_anom: float = 2.0
    artifact_prefix: str = "o1"
    summary_label: str = "O1"
    qt_cache_dir: str = ""
    prepare_only: bool = False
    use_prepared_qt: bool = False
    qt_progress_interval: int = 1000
    precomputed_noise_qt_dir: str = ""
    precomputed_noise_raw_dir: str = ""


class BaselineCAE(nn.Module):
    def __init__(self, dropout=0.20):
        super().__init__()
        d = dropout
        self.latent_dim = 128 * 32 * 16
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(d)
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(d)
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(d)
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (128, 32, 16))
        self.classifier = nn.Linear(self.latent_dim, 1)

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout2d(d)

        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.drop5 = nn.Dropout2d(d)

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)

    def encode(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        size1 = x.size()
        x, i1 = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        size2 = x.size()
        x, i2 = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        size3 = x.size()
        x, i3 = self.pool3(x)
        z = self.flatten(x)
        return z, (i1, i2, i3), (size1, size2, size3)

    def decode(self, z, indices, sizes):
        i1, i2, i3 = indices
        size1, size2, size3 = sizes
        x = self.unflatten(z)

        x = self.unpool1(x, i3, output_size=size3)
        x = F.relu(self.bn4(self.deconv1(x)))
        x = self.drop4(x)

        x = self.unpool2(x, i2, output_size=size2)
        x = F.relu(self.bn5(self.deconv2(x)))
        x = self.drop5(x)

        x = self.unpool3(x, i1, output_size=size1)
        x = self.deconv3(x)
        return x

    def get_latent(self, x):
        z, _, _ = self.encode(x)
        return z

    def forward(self, x):
        z, indices, sizes = self.encode(x)
        return self.decode(z, indices, sizes)

    def get_logit(self, qt):
        return self.classifier(self.get_latent(qt)).squeeze(1)

    def classify_logits(self, qt):
        return self.get_logit(qt)

    def score(self, qt):
        return self.get_logit(qt)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # BUG003 fix: pin all sources of randomness for reproducible training
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def device_for_run(device_override=""):
    if device_override:
        return torch.device(device_override)
    if torch.cuda.device_count() > 1:
        return torch.device("cuda:1")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_reference_psd():
    psd_data = np.loadtxt(PSD_FILE)
    return psd_data[:, 0].astype(np.float64), psd_data[:, 1].astype(np.float64)


def load_reference_psd_o1(prepared_dir):
    psd_path = os.path.join(prepared_dir, "reference_psd.npz")
    if not os.path.exists(psd_path):
        raise FileNotFoundError(f"Missing prepared O1 PSD file: {psd_path}")
    psd_data = np.load(psd_path)
    return psd_data["freq"].astype(np.float64), psd_data["psd"].astype(np.float64)


def load_detector_asd_o1(prepared_dir, detector):
    psd_path = os.path.join(prepared_dir, f"reference_psd_{detector}.npz")
    if not os.path.exists(psd_path):
        raise FileNotFoundError(f"Missing detector-specific O1 PSD file: {psd_path}")
    psd_data = np.load(psd_path)
    psd = psd_data["psd"].astype(np.float64)
    positive = psd[np.isfinite(psd) & (psd > 0.0)]
    if len(positive) == 0:
        raise ValueError(f"Detector PSD for {detector} has no positive finite bins: {psd_path}")
    floor = float(np.median(positive) * 1e-10)
    psd_floored = np.maximum(psd, floor)
    return FrequencySeries(
        np.sqrt(psd_floored),
        f0=float(psd_data["freq"][0]),
        df=float(psd_data["freq"][1] - psd_data["freq"][0]),
    )


def infer_line_configuration(prepared_dir):
    prepared_name = os.path.basename(os.path.abspath(prepared_dir)).lower()
    if "o3" in prepared_name:
        return "o3a"
    return "o1"


def detector_line_frequencies(detector, line_configuration):
    line_freqs = list(POWERLINE_HARMONICS)
    if line_configuration == "o3a":
        line_freqs.extend(O3A_CAL_LINES[detector])
    else:
        line_freqs.extend(O1_CAL_LINES[detector])
        if detector == "L1":
            line_freqs.extend(EARLY_O1_L1_DITHER_LINES)
    return line_freqs


def apply_o1_notches(samples, detector, line_configuration="o1"):
    cleaned = np.asarray(samples, dtype=np.float64).copy()
    line_freqs = detector_line_frequencies(detector, line_configuration)
    for freq in line_freqs:
        if freq <= 0.0 or freq >= FS / 2.0:
            continue
        b, a = iirnotch(w0=freq, Q=40.0, fs=FS)
        cleaned = filtfilt(b, a, cleaned)
    return cleaned


def load_raw(directory, n_files):
    files = sorted(os.listdir(directory))[:n_files]
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")
    return np.concatenate([np.load(os.path.join(directory, f)) for f in files])


def whiten_batch(waveforms, reference_psd):
    return np.array(
        [whiten(w, fs=FS, nperseg=1024, noise_psd=reference_psd) for w in waveforms],
        dtype=np.float32,
    )


def _compute_qt_image_worker(args):
    waveform, sample_rate, frange, qrange, crop_seconds = args
    return compute_qt_image(
        waveform,
        sample_rate=sample_rate,
        frange=frange,
        qrange=qrange,
        crop_seconds=crop_seconds,
    )


def compute_center_crop_bounds(num_cols, total_seconds, crop_seconds):
    if crop_seconds is None or crop_seconds <= 0.0 or crop_seconds >= total_seconds:
        return 0, num_cols
    crop_ratio = crop_seconds / float(total_seconds)
    start = int(np.floor(0.5 * (1.0 - crop_ratio) * num_cols))
    stop = int(np.floor(0.5 * (1.0 + crop_ratio) * num_cols))
    start = max(0, min(start, num_cols))
    stop = max(start + 1, min(stop, num_cols))
    return start, stop


def compute_qt_image(
    waveform,
    sample_rate=FS,
    frange=QTRANSFORM_FRANGE,
    qrange=QTRANSFORM_QRANGE,
    crop_seconds=None,
    return_crop_bounds=False,
):
    ts = TimeSeries(np.asarray(waveform, dtype=np.float64), sample_rate=sample_rate)
    qgram = ts.q_transform(
        qrange=qrange,
        frange=frange,
        tres=0.002,
        fres=0.5,
        norm="median",
        whiten=False,
    )
    mag = np.log1p(np.abs(np.array(qgram).T)).astype(np.float32, copy=False)
    total_seconds = len(waveform) / float(sample_rate)
    start, stop = compute_center_crop_bounds(mag.shape[1], total_seconds, crop_seconds)
    mag = mag[:, start:stop]
    if return_crop_bounds:
        return mag.astype(np.float32, copy=False), start, stop, total_seconds
    return mag.astype(np.float32, copy=False)


def compute_qt_images(
    waveforms,
    sample_rate=FS,
    frange=QTRANSFORM_FRANGE,
    qrange=QTRANSFORM_QRANGE,
    crop_seconds=None,
    n_workers=QTRANSFORM_N_WORKERS,
):
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if len(waveforms) == 0:
        return np.empty((0, SPEC_SIZE[0], SPEC_SIZE[1]), dtype=np.float32)
    args = [(w, sample_rate, frange, qrange, crop_seconds) for w in waveforms]
    worker_count = max(1, min(n_workers, os.cpu_count() or 1, len(args)))
    if worker_count == 1:
        mags = [_compute_qt_image_worker(arg) for arg in args]
    else:
        with Pool(processes=worker_count) as pool:
            mags = pool.map(_compute_qt_image_worker, args)
    resized = []
    for mag in mags:
        zoom_f = SPEC_SIZE[0] / mag.shape[0]
        zoom_t = SPEC_SIZE[1] / mag.shape[1]
        resized.append(zoom(mag, (zoom_f, zoom_t), order=1).astype(np.float32))
    return min_max_norm(np.stack(resized, axis=0)).astype(np.float32)


def center_crop_waveforms(waveforms, sample_rate=FS, context_seconds=None):
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 2:
        raise ValueError(f"Expected 2D waveform array, got shape {waveforms.shape}")
    if context_seconds is None or context_seconds <= 0.0:
        return waveforms
    total_samples = waveforms.shape[1]
    target_samples = int(round(float(context_seconds) * sample_rate))
    if target_samples <= 0 or target_samples >= total_samples:
        return waveforms
    start = max(0, (total_samples - target_samples) // 2)
    stop = start + target_samples
    return waveforms[:, start:stop]


def compute_qt_tensors(whitened):
    qt_inputs = center_crop_waveforms(whitened, sample_rate=FS, context_seconds=O1_CONTEXT_SECONDS)
    qts = compute_qt_images(qt_inputs, sample_rate=FS, crop_seconds=O1_CENTER_CROP_SECONDS)
    return torch.tensor(qts).unsqueeze(1)


class NpyTensorDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.array = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        sample = np.array(self.array[idx], dtype=np.float32, copy=True)
        if sample.ndim == 2:
            sample = sample[None, ...]
        return (torch.from_numpy(sample),)


class MultiNpyTensorDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("MultiNpyTensorDataset requires at least one .npy path.")
        self.arrays = [np.load(path, mmap_mode="r") for path in self.paths]
        self.lengths = [len(arr) for arr in self.arrays]
        self.cumulative_lengths = list(accumulate(self.lengths))

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        array_idx = 0
        while idx >= self.cumulative_lengths[array_idx]:
            array_idx += 1
        prev_total = 0 if array_idx == 0 else self.cumulative_lengths[array_idx - 1]
        sample = np.array(self.arrays[array_idx][idx - prev_total], dtype=np.float32, copy=True)
        if sample.ndim == 2:
            sample = sample[None, ...]
        return (torch.from_numpy(sample),)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def maybe_print_qt_progress(processed, total, progress_state, label, progress_interval):
    progress_interval = max(1, int(progress_interval))
    next_report = progress_state.get("next_report", progress_interval)
    while processed >= next_report and next_report <= total:
        print(f"{label}: processed {next_report}/{total} QT images", flush=True)
        next_report += progress_interval
    progress_state["next_report"] = next_report
    if processed >= total and progress_state.get("final_reported") != total:
        print(f"{label}: processed {total}/{total} QT images", flush=True)
        progress_state["final_reported"] = total


def compute_qt_cache_dataset(cache_path, whitened, progress_label, progress_interval):
    whitened = center_crop_waveforms(whitened, sample_rate=FS, context_seconds=O1_CONTEXT_SECONDS)
    whitened = np.asarray(whitened, dtype=np.float32)
    total = len(whitened)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[0] == total:
            print(f"Loading QT cache from {cache_path}...", flush=True)
            return NpyTensorDataset(cache_path)
        print(f"Discarding stale QT cache with shape {cached.shape} at {cache_path}", flush=True)
        os.remove(cache_path)

    print(f"Saving QT cache to {cache_path}...", flush=True)
    ensure_parent_dir(cache_path)
    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(total, 1, SPEC_SIZE[0], SPEC_SIZE[1]),
    )
    progress_state = {}
    for start in range(0, total, QTRANSFORM_CHUNK_SIZE):
        stop = min(start + QTRANSFORM_CHUNK_SIZE, total)
        qts = compute_qt_images(whitened[start:stop], sample_rate=FS, crop_seconds=O1_CENTER_CROP_SECONDS)
        out[start:stop, 0] = qts
        maybe_print_qt_progress(stop, total, progress_state, progress_label, progress_interval)
    del out
    return NpyTensorDataset(cache_path)


def save_tensor_cache_dataset(cache_path, tensor):
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape == array.shape:
            print(f"Loading tensor cache from {cache_path}...", flush=True)
            return NpyTensorDataset(cache_path)
        print(f"Discarding stale tensor cache with shape {cached.shape} at {cache_path}", flush=True)
        os.remove(cache_path)
    print(f"Saving tensor cache to {cache_path}...", flush=True)
    ensure_parent_dir(cache_path)
    np.save(cache_path, array)
    return NpyTensorDataset(cache_path)


def load_noise_metadata(path):
    rows = []
    with open(path, newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            rows.append(row)
    return rows


def load_precomputed_noise_qt_data(qt_root_dir, detector_mode="both"):
    detector_dirs = []
    if detector_mode == "both":
        detector_dirs = ["H1", "L1"]
    elif detector_mode in ("H1", "L1"):
        detector_dirs = [detector_mode]
    else:
        raise ValueError(f"Unsupported detector mode for precomputed QT data: {detector_mode}")

    split_to_suffixes = {
        "train": ["train"],
        "val": ["val"],
        "test": ["test_a", "test_b"],
    }
    split_paths = {name: [] for name in split_to_suffixes}
    split_meta = {name: [] for name in split_to_suffixes}

    for detector in detector_dirs:
        detector_dir = os.path.join(qt_root_dir, detector)
        if not os.path.isdir(detector_dir):
            raise FileNotFoundError(f"Missing detector directory in precomputed QT dataset: {detector_dir}")
        for split_name, suffixes in split_to_suffixes.items():
            for suffix in suffixes:
                qt_path = os.path.join(detector_dir, f"{suffix}_qt.npy")
                meta_path = os.path.join(detector_dir, f"{suffix}_metadata.csv")
                if not os.path.exists(qt_path):
                    raise FileNotFoundError(f"Missing QT tensor file: {qt_path}")
                if not os.path.exists(meta_path):
                    raise FileNotFoundError(f"Missing metadata CSV: {meta_path}")
                split_paths[split_name].append(qt_path)
                split_meta[split_name].extend(load_noise_metadata(meta_path))

    return {
        "noise_qt_tr": MultiNpyTensorDataset(split_paths["train"]),
        "noise_qt_val": MultiNpyTensorDataset(split_paths["val"]),
        "noise_qt_te": MultiNpyTensorDataset(split_paths["test"]),
        "noise_train_meta": split_meta["train"],
        "noise_val_meta": split_meta["val"],
        "noise_test_meta": split_meta["test"],
        "sig_qt_tr": None,
        "sig_qt_val": None,
        "sig_qt_val_benchmark": None,
        "sig_qt_eval": None,
        "sig_val_benchmark_meta": None,
        "sig_eval_meta": None,
        "event_qt": None,
        "event_meta": None,
        "noise_sigma": None,
    }


def infer_precomputed_noise_raw_dir(cfg):
    if cfg.precomputed_noise_raw_dir and os.path.isdir(cfg.precomputed_noise_raw_dir):
        return cfg.precomputed_noise_raw_dir

    candidates = []
    if cfg.o1_data_dir:
        candidates.append(os.path.join(cfg.o1_data_dir, "raw"))

    qt_root_dir = os.path.abspath(cfg.precomputed_noise_qt_dir)
    qt_root_name = os.path.basename(qt_root_dir)
    qt_parent_dir = os.path.dirname(qt_root_dir)
    if qt_root_name == "o3a_prepared_1s_stride":
        candidates.append(os.path.join(qt_parent_dir, "o3a_prepared_4s_crop2s", "raw"))
    candidates.append(os.path.join(qt_root_dir, "raw"))

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return ""


def estimate_asd_from_metadata_interval(strain, metadata_row):
    file_gps_start = int(metadata_row["file_gps_start"])
    psd_gps_start = float(metadata_row["psd_gps_start"])
    psd_gps_end = float(metadata_row["psd_gps_end"])
    start = int(round((psd_gps_start - file_gps_start) * FS))
    stop = int(round((psd_gps_end - file_gps_start) * FS))
    quiet = np.asarray(strain[start:stop], dtype=np.float64)
    if len(quiet) == 0:
        raise ValueError(f"Empty PSD interval for metadata row: {metadata_row}")
    quiet = quiet - quiet.mean()
    ts = TimeSeries(quiet, sample_rate=FS)
    psd_series = ts.psd(fftlength=4, overlap=2, window="hann", method="median")
    freq = np.asarray(psd_series.frequencies.value, dtype=np.float64)
    psd = np.asarray(psd_series.value, dtype=np.float64)
    positive = psd[np.isfinite(psd) & (psd > 0.0)]
    if len(positive) == 0:
        raise RuntimeError("PSD estimation produced no positive finite values.")
    psd = np.maximum(psd, np.median(positive) * 1e-10)
    return FrequencySeries(
        np.sqrt(psd).astype(np.float64),
        f0=float(freq[0]),
        df=float(freq[1] - freq[0]),
    )


def whiten_and_crop_precomputed_stride_segment(segment_4s, asd, detector, notch_lines, line_configuration):
    ts = TimeSeries((np.asarray(segment_4s, dtype=np.float64) - np.mean(segment_4s)).astype(np.float64), sample_rate=FS)
    tsw = ts.whiten(asd=asd, fduration=2, highpass=20)
    arr = np.asarray(tsw.value, dtype=np.float64)
    center = len(arr) // 2
    cropped = arr[center - FS:center + FS]
    if notch_lines:
        cropped = apply_o1_notches(cropped, detector, line_configuration=line_configuration)
    cropped = cropped.astype(np.float32)
    if not np.all(np.isfinite(cropped)):
        raise ValueError(f"Non-finite samples after gwpy whitening for detector {detector}")
    return cropped


def compute_precomputed_stride_qt_batch(whitened_batch):
    qt_batch = compute_qt_images(
        whitened_batch,
        sample_rate=FS,
        frange=QTRANSFORM_FRANGE,
        qrange=QTRANSFORM_QRANGE,
        crop_seconds=None,
    )
    width = qt_batch.shape[2]
    qt_batch = qt_batch[:, :, width // 4:(3 * width) // 4]
    resized = []
    for qt_image in qt_batch:
        zoom_f = SPEC_SIZE[0] / qt_image.shape[0]
        zoom_t = SPEC_SIZE[1] / qt_image.shape[1]
        resized.append(zoom(qt_image, (zoom_f, zoom_t), order=1).astype(np.float32))
    return min_max_norm(np.stack(resized, axis=0)).astype(np.float32)


def build_precomputed_stride_injection_cache_dataset(
    cache_path,
    metadata_rows,
    raw_dir,
    signal_pool,
    seed,
    notch_lines,
    line_configuration,
    progress_label,
    progress_interval,
):
    n = len(metadata_rows)
    if n == 0:
        raise ValueError("Cannot build injections from an empty metadata split.")
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[0] == n:
            print(f"Loading injection QT cache from {cache_path}...", flush=True)
            return NpyTensorDataset(cache_path)
        print(f"Discarding stale injection QT cache with shape {cached.shape} at {cache_path}", flush=True)
        os.remove(cache_path)

    print(f"Saving injection QT cache to {cache_path}...", flush=True)
    ensure_parent_dir(cache_path)
    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, 1, SPEC_SIZE[0], SPEC_SIZE[1]),
    )

    raw_paths = {}
    for name in os.listdir(raw_dir):
        if not name.endswith(".hdf5"):
            continue
        parts = name.split("-")
        if len(parts) < 4:
            continue
        detector = parts[0][-1] + "1"
        gps_start = int(parts[-2])
        raw_paths[(detector, gps_start)] = os.path.join(raw_dir, name)

    rng = np.random.default_rng(seed)
    raw_cache = {}
    asd_cache = {}
    progress_state = {}
    segment_len = int(round(4.0 * FS))

    for start in range(0, n, QTRANSFORM_CHUNK_SIZE):
        stop = min(start + QTRANSFORM_CHUNK_SIZE, n)
        whitened_batch = []
        for row in metadata_rows[start:stop]:
            detector = row["detector"]
            file_gps_start = int(row["file_gps_start"])
            raw_path = raw_paths.get((detector, file_gps_start))
            if raw_path is None:
                raise FileNotFoundError(
                    f"Missing raw {detector} file for GPS {file_gps_start} in {raw_dir}"
                )
            strain = raw_cache.get(raw_path)
            if strain is None:
                strain, _dqmask, _injmask = load_frame(raw_path)
                raw_cache[raw_path] = strain
            psd_key = (
                detector,
                file_gps_start,
                float(row["psd_gps_start"]),
                float(row["psd_gps_end"]),
            )
            asd = asd_cache.get(psd_key)
            if asd is None:
                asd = estimate_asd_from_metadata_interval(strain, row)
                asd_cache[psd_key] = asd

            raw_start = int(round((float(row["segment_gps_start"]) - float(file_gps_start) - 1.0) * FS))
            segment_4s = np.asarray(strain[raw_start:raw_start + segment_len], dtype=np.float32)
            if len(segment_4s) != segment_len:
                raise ValueError(
                    f"Unexpected segment length {len(segment_4s)} for {detector} gps={file_gps_start} "
                    f"start={row['segment_gps_start']}"
                )

            sig_idx = int(rng.integers(len(signal_pool["H1"])))
            signal_sel = signal_pool[detector][sig_idx].astype(np.float32)
            scale, _source_detector_snr, _target_detector_snr = maybe_rescale_projected_signal_detector(
                signal_pool,
                sig_idx,
                detector,
                asd,
                rng,
            )
            if scale != 1.0:
                signal_sel = signal_sel * np.float32(scale)
            # FIX1: training injection -> random +/-0.5 s time shift
            signal_sel = place_signal_in_segment(
                signal_sel, segment_len, rng=rng, max_shift_samples=int(0.5 * FS)
            )
            injected = segment_4s + signal_sel
            whitened_batch.append(
                whiten_and_crop_precomputed_stride_segment(
                    injected,
                    asd,
                    detector,
                    notch_lines=notch_lines,
                    line_configuration=line_configuration,
                )
            )

        qts = compute_precomputed_stride_qt_batch(np.stack(whitened_batch, axis=0).astype(np.float32))
        out[start:stop, 0] = qts
        maybe_print_qt_progress(stop, n, progress_state, progress_label, progress_interval)

    del out
    return NpyTensorDataset(cache_path)


def build_precomputed_stride_benchmark_dataset(
    cache_path,
    metadata_rows,
    raw_dir,
    signal_pool,
    signal_indices,
    seed,
    notch_lines,
    line_configuration,
    progress_label,
    progress_interval,
):
    if len(signal_indices) == 0:
        return None, None

    det_to_rows = {"H1": [], "L1": []}
    for row in metadata_rows:
        detector = row.get("detector")
        if detector in det_to_rows:
            det_to_rows[detector].append(row)

    if det_to_rows["H1"] and det_to_rows["L1"]:
        benchmark_mode = "matched_pair"
        pair_count = min(len(signal_indices), len(det_to_rows["H1"]), len(det_to_rows["L1"]))
        if pair_count == 0:
            return None, None
        total_rows = 2 * pair_count
    else:
        benchmark_mode = "single_detector"
        active_detector = "H1" if det_to_rows["H1"] else "L1"
        active_rows = det_to_rows[active_detector]
        total_rows = min(len(signal_indices), len(active_rows))
        if total_rows == 0:
            return None, None

    meta_path = os.path.splitext(cache_path)[0] + "_meta.json"
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[0] == total_rows:
            print(f"Loading benchmark injection QT cache from {cache_path}...", flush=True)
            return NpyTensorDataset(cache_path), load_json(meta_path)
        print(f"Discarding stale benchmark injection QT cache at {cache_path}", flush=True)
        os.remove(cache_path)

    print(f"Saving benchmark injection QT cache to {cache_path}...", flush=True)
    ensure_parent_dir(cache_path)
    out = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=(total_rows, 1, SPEC_SIZE[0], SPEC_SIZE[1])
    )

    raw_paths = {}
    for name in os.listdir(raw_dir):
        if not name.endswith(".hdf5"):
            continue
        parts = name.split("-")
        if len(parts) < 4:
            continue
        detector = parts[0][-1] + "1"
        gps_start = int(parts[-2])
        raw_paths[(detector, gps_start)] = os.path.join(raw_dir, name)

    rng = np.random.default_rng(seed)
    raw_cache = {}
    asd_cache = {}
    progress_state = {}
    segment_len = int(round(4.0 * FS))
    benchmark_meta = []

    def load_segment_and_asd(row):
        detector = row["detector"]
        file_gps_start = int(row["file_gps_start"])
        raw_path = raw_paths.get((detector, file_gps_start))
        if raw_path is None:
            raise FileNotFoundError(
                f"Missing raw {detector} file for GPS {file_gps_start} in {raw_dir}"
            )
        strain = raw_cache.get(raw_path)
        if strain is None:
            strain, _dqmask, _injmask = load_frame(raw_path)
            raw_cache[raw_path] = strain
        psd_key = (detector, file_gps_start, float(row["psd_gps_start"]), float(row["psd_gps_end"]))
        asd = asd_cache.get(psd_key)
        if asd is None:
            asd = estimate_asd_from_metadata_interval(strain, row)
            asd_cache[psd_key] = asd

        raw_start = int(round((float(row["segment_gps_start"]) - float(file_gps_start) - 1.0) * FS))
        segment_4s = np.asarray(strain[raw_start:raw_start + segment_len], dtype=np.float32)
        if len(segment_4s) != segment_len:
            raise ValueError(
                f"Unexpected segment length {len(segment_4s)} for {detector} gps={file_gps_start} "
                f"start={row['segment_gps_start']}"
            )
        return detector, segment_4s, asd

    if benchmark_mode == "matched_pair":
        for start in range(0, pair_count, max(1, QTRANSFORM_CHUNK_SIZE // 2)):
            stop = min(start + max(1, QTRANSFORM_CHUNK_SIZE // 2), pair_count)
            whitened_batch = []
            for pair_id in range(start, stop):
                sig_idx = int(signal_indices[pair_id])
                pair_rows = {
                    "H1": det_to_rows["H1"][pair_id],
                    "L1": det_to_rows["L1"][pair_id],
                }
                pair_segments = {}
                pair_asd_by_detector = {}
                for detector in ("H1", "L1"):
                    _detector, segment_4s, asd = load_segment_and_asd(pair_rows[detector])
                    pair_segments[detector] = segment_4s
                    pair_asd_by_detector[detector] = asd

                scale, source_snr, target_snr = maybe_rescale_projected_signal_pair(
                    signal_pool,
                    sig_idx,
                    pair_asd_by_detector,
                    rng,
                )
                total_mass = float(signal_pool["total_mass"][sig_idx])
                distance_mpc = float(signal_pool["distance_mpc"][sig_idx])
                base_meta = {
                    "pair_id": int(pair_id),
                    "source_id": int(signal_pool["source_ids"][sig_idx]),
                    "total_mass": total_mass,
                    "distance_mpc": distance_mpc,
                    "mass_bin": mass_bin_label(total_mass),
                    "distance_bin": distance_bin_label(distance_mpc),
                    "scale_factor": float(scale),
                    "source_network_snr": float(source_snr) if source_snr is not None else None,
                    "target_network_snr": float(target_snr) if target_snr is not None else None,
                }

                for detector in ("H1", "L1"):
                    signal_sel = signal_pool[detector][sig_idx].astype(np.float32)
                    if scale != 1.0:
                        signal_sel = signal_sel * np.float32(scale)
                    # FIX1: benchmark/eval injection -> deterministic center (rng=None)
                    signal_sel = place_signal_in_segment(signal_sel, segment_len, rng=None)
                    injected = pair_segments[detector] + signal_sel
                    whitened_batch.append(
                        whiten_and_crop_precomputed_stride_segment(
                            injected,
                            pair_asd_by_detector[detector],
                            detector,
                            notch_lines=notch_lines,
                            line_configuration=line_configuration,
                        )
                    )
                    benchmark_meta.append({
                        **base_meta,
                        "benchmark_id": len(benchmark_meta),
                        "detector": detector,
                    })

            qts = compute_precomputed_stride_qt_batch(np.stack(whitened_batch, axis=0).astype(np.float32))
            out[2 * start:2 * stop, 0] = qts
            maybe_print_qt_progress(2 * stop, total_rows, progress_state, progress_label, progress_interval)
    else:
        for start in range(0, total_rows, QTRANSFORM_CHUNK_SIZE):
            stop = min(start + QTRANSFORM_CHUNK_SIZE, total_rows)
            whitened_batch = []
            for i in range(start, stop):
                row = active_rows[i]
                sig_idx = int(signal_indices[i])
                detector, segment_4s, asd = load_segment_and_asd(row)
                signal_sel = signal_pool[detector][sig_idx].astype(np.float32)
                scale, source_snr, target_snr = maybe_rescale_projected_signal_detector(
                    signal_pool,
                    sig_idx,
                    detector,
                    asd,
                    rng,
                )
                if scale != 1.0:
                    signal_sel = signal_sel * np.float32(scale)
                # FIX1: benchmark/eval injection -> deterministic center (rng=None)
                signal_sel = place_signal_in_segment(signal_sel, segment_len, rng=None)
                injected = segment_4s + signal_sel
                whitened_batch.append(
                    whiten_and_crop_precomputed_stride_segment(
                        injected, asd, detector, notch_lines=notch_lines, line_configuration=line_configuration
                    )
                )
                total_mass = float(signal_pool["total_mass"][sig_idx])
                distance_mpc = float(signal_pool["distance_mpc"][sig_idx])
                benchmark_meta.append({
                    "benchmark_id": i,
                    "source_id": int(signal_pool["source_ids"][sig_idx]),
                    "total_mass": total_mass,
                    "distance_mpc": distance_mpc,
                    "mass_bin": mass_bin_label(total_mass),
                    "distance_bin": distance_bin_label(distance_mpc),
                    "scale_factor": float(scale),
                    "source_network_snr": float(source_snr) if source_snr is not None else None,
                    "target_network_snr": float(target_snr) if target_snr is not None else None,
                    "detector": detector,
                })

            qts = compute_precomputed_stride_qt_batch(np.stack(whitened_batch, axis=0).astype(np.float32))
            out[start:stop, 0] = qts
            maybe_print_qt_progress(stop, total_rows, progress_state, progress_label, progress_interval)

    del out
    save_json(meta_path, benchmark_meta)
    return NpyTensorDataset(cache_path), benchmark_meta


def prepare_precomputed_noise_qt_data_with_weaksup(cfg):
    data = load_precomputed_noise_qt_data(cfg.precomputed_noise_qt_dir, detector_mode=cfg.detector_mode)
    raw_dir = infer_precomputed_noise_raw_dir(cfg)
    if not raw_dir:
        print(
            "No raw directory available for precomputed-noise injections. "
            "Proceeding with unsupervised noise-only training.",
            flush=True,
        )
        return data

    print(f"Loading projected {cfg.summary_label} signal bank from {cfg.o1_signal_bank_dir}...", flush=True)
    signal_bank = load_o1_signal_bank(cfg.o1_signal_bank_dir)
    line_configuration = infer_line_configuration(cfg.precomputed_noise_qt_dir)

    cache_root = cfg.qt_cache_dir or os.path.join(cfg.output_dir, "qt_cache_precomputed")
    data["sig_qt_tr"] = build_precomputed_stride_injection_cache_dataset(
        os.path.join(cache_root, "sig_qt_tr.npy"),
        data["noise_train_meta"],
        raw_dir,
        signal_bank,
        seed=SEED,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
        progress_label=f"{cfg.summary_label} sig_qt_tr",
        progress_interval=cfg.qt_progress_interval,
    )
    data["sig_qt_val"] = build_precomputed_stride_injection_cache_dataset(
        os.path.join(cache_root, "sig_qt_val.npy"),
        data["noise_val_meta"],
        raw_dir,
        signal_bank,
        seed=SEED + 1,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
        progress_label=f"{cfg.summary_label} sig_qt_val",
        progress_interval=cfg.qt_progress_interval,
    )
    n_benchmark = min(cfg.o1_inj_eval_count, len(data["noise_val_meta"]))
    if n_benchmark > 0:
        benchmark_signal_count = n_benchmark
        if cfg.detector_mode == "both":
            benchmark_signal_count = max(1, n_benchmark // 2)
        benchmark_signal_indices = stratified_signal_indices(signal_bank, benchmark_signal_count, seed=SEED + 11)
        sig_qt_val_benchmark, sig_val_benchmark_meta = build_precomputed_stride_benchmark_dataset(
            os.path.join(cache_root, "sig_qt_val_benchmark.npy"),
            data["noise_val_meta"],
            raw_dir,
            signal_bank,
            benchmark_signal_indices,
            seed=SEED + 11,
            notch_lines=cfg.o1_notch_lines,
            line_configuration=line_configuration,
            progress_label=f"{cfg.summary_label} sig_qt_val_benchmark",
            progress_interval=cfg.qt_progress_interval,
        )
        data["sig_qt_val_benchmark"] = sig_qt_val_benchmark
        data["sig_val_benchmark_meta"] = sig_val_benchmark_meta

    # BUG001 fix: wire sig_qt_eval into the precomputed-noise path.
    # The --use-prepared-qt path (prepare_o1_real_data_cached) builds a held-out
    # eval benchmark (sig_qt_eval / sig_eval_meta) on the TEST noise split using a
    # distinct signal selection (seed SEED + 12); the precomputed-noise path
    # previously left sig_qt_eval = None, so the downstream held-out-eval reporting
    # in main() (data["sig_qt_eval"] is not None) never fired. We now build the
    # eval benchmark on noise_test_meta to match that behavior. This is a latent
    # bug fix: it does not change training or the headline val-benchmark efficiency.
    n_eval = min(cfg.o1_inj_eval_count, len(data.get("noise_test_meta") or []))
    if n_eval > 0:
        eval_signal_count = n_eval
        if cfg.detector_mode == "both":
            eval_signal_count = max(1, n_eval // 2)
        eval_signal_indices = stratified_signal_indices(signal_bank, eval_signal_count, seed=SEED + 12)
        sig_qt_eval, sig_eval_meta = build_precomputed_stride_benchmark_dataset(
            os.path.join(cache_root, "sig_qt_eval.npy"),
            data["noise_test_meta"],
            raw_dir,
            signal_bank,
            eval_signal_indices,
            seed=SEED + 12,
            notch_lines=cfg.o1_notch_lines,
            line_configuration=line_configuration,
            progress_label=f"{cfg.summary_label} sig_qt_eval",
            progress_interval=cfg.qt_progress_interval,
        )
        data["sig_qt_eval"] = sig_qt_eval
        data["sig_eval_meta"] = sig_eval_meta
    return data


def save_noise_only_qt_outputs(
    output_dir,
    unsup_history,
    noise_test_scores,
    noise_test_meta,
    artifact_prefix="o1",
    summary_label="O1",
):
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "unsup_history.npy"), unsup_history)
    np.save(os.path.join(results_dir, f"{artifact_prefix}_noise_scores_unsup.npy"), noise_test_scores)
    pd.DataFrame({"score": noise_test_scores}).to_csv(
        os.path.join(results_dir, f"{artifact_prefix}_noise_scores_unsup.csv"),
        index=False,
    )
    if noise_test_meta:
        rows = []
        for row, score in zip(noise_test_meta, noise_test_scores):
            rows.append({**row, "score": float(score)})
        pd.DataFrame(rows).to_csv(
            os.path.join(results_dir, f"{artifact_prefix}_noise_scores_unsup_with_meta.csv"),
            index=False,
        )

    summary_lines = [
        f"{summary_label} precomputed noise-QT training completed.",
        f"test_noise_count={len(noise_test_scores)}",
        f"test_noise_mean_recon={float(noise_test_scores.mean()):.6f}",
        f"test_noise_std_recon={float(noise_test_scores.std()):.6f}",
        f"best_val_recon={float(min(unsup_history['val_total'])):.6f}",
    ]
    summary = "\n".join(summary_lines) + "\n"
    with open(os.path.join(results_dir, f"summary_{artifact_prefix}.txt"), "w") as fh:
        fh.write(summary)
    print(summary, flush=True)


def save_json(path, payload):
    ensure_parent_dir(path)
    with open(path, "w") as fout:
        json.dump(payload, fout, indent=2)


def load_json(path):
    with open(path) as fin:
        return json.load(fin)


def build_o1_injections_cache_dataset(
    cache_path,
    noise_raw,
    detectors,
    signal_pool,
    asd_by_detector,
    seed,
    notch_lines,
    line_configuration,
    progress_label,
    progress_interval,
):
    noise_raw = np.asarray(noise_raw, dtype=np.float32)
    n = len(noise_raw)
    if n == 0:
        raise ValueError("Cannot build injections from an empty noise split.")
    if os.path.exists(cache_path):
        cached = np.load(cache_path, mmap_mode="r")
        if cached.shape[0] == n:
            print(f"Loading injection QT cache from {cache_path}...", flush=True)
            return NpyTensorDataset(cache_path)
        print(f"Discarding stale injection QT cache with shape {cached.shape} at {cache_path}", flush=True)
        os.remove(cache_path)

    print(f"Saving injection QT cache to {cache_path}...", flush=True)
    ensure_parent_dir(cache_path)
    out = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, 1, SPEC_SIZE[0], SPEC_SIZE[1]),
    )
    rng = np.random.default_rng(seed)
    noise_len = noise_raw.shape[1]
    progress_state = {}
    for start in range(0, n, QTRANSFORM_CHUNK_SIZE):
        stop = min(start + QTRANSFORM_CHUNK_SIZE, n)
        injected_raw = np.empty_like(noise_raw[start:stop], dtype=np.float32)
        for local_idx, detector_name in enumerate(detectors[start:stop]):
            sig_idx = int(rng.integers(len(signal_pool["H1"])))
            signal_sel = signal_pool[detector_name][sig_idx].astype(np.float32)
            scale, _, _ = maybe_rescale_projected_signal_pair(signal_pool, sig_idx, asd_by_detector, rng)
            if scale != 1.0:
                signal_sel = signal_sel * np.float32(scale)
            # FIX1: training injection -> random +/-0.5 s time shift
            signal_sel = place_signal_in_segment(
                signal_sel, noise_len, rng=rng, max_shift_samples=int(0.5 * FS)
            )
            injected_raw[local_idx] = noise_raw[start + local_idx] + signal_sel
        whitened = whiten_batch_gwpy_o1(
            injected_raw,
            detectors[start:stop],
            asd_by_detector,
            notch_lines=notch_lines,
            line_configuration=line_configuration,
        )
        qt_inputs = center_crop_waveforms(whitened, sample_rate=FS, context_seconds=O1_CONTEXT_SECONDS)
        qts = compute_qt_images(qt_inputs, sample_rate=FS, crop_seconds=O1_CENTER_CROP_SECONDS)
        out[start:stop, 0] = qts
        maybe_print_qt_progress(stop, n, progress_state, progress_label, progress_interval)
    del out
    return NpyTensorDataset(cache_path)


def load_o1_signal_bank(bank_dir):
    npz_files = sorted(glob.glob(os.path.join(bank_dir, "**", "signals_*.npz"), recursive=True))
    if not npz_files:
        raise FileNotFoundError(f"No projected O1 signal bank files found in {bank_dir}")

    bank = {"H1": [], "L1": []}
    source_ids = []
    total_mass = []
    distance_mpc = []
    projection_modes = set()
    for npz_path in npz_files:
        chunk = np.load(npz_path)
        for detector in ("H1", "L1"):
            if detector not in chunk:
                raise KeyError(f"Missing detector array '{detector}' in {npz_path}")
            bank[detector].append(chunk[detector].astype(np.float32))

        csv_path = os.path.splitext(npz_path)[0] + ".csv"
        if os.path.exists(csv_path):
            rows = []
            with open(csv_path, newline="") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    rows.append(row)
            if rows and "projection_mode" in rows[0]:
                projection_modes.update(row["projection_mode"] for row in rows)
            source_ids.extend(int(row["source_id"]) for row in rows)
            if rows and "total_mass" in rows[0]:
                total_mass.extend(float(row["total_mass"]) for row in rows)
            elif rows and "mass1" in rows[0] and "mass2" in rows[0]:
                total_mass.extend(float(row["mass1"]) + float(row["mass2"]) for row in rows)
            else:
                total_mass.extend([float("nan")] * len(rows))
            distance_mpc.extend(float(row["distance_mpc"]) for row in rows)
        else:
            if "projectedbank" in npz_path or "projbank" in npz_path:
                projection_modes.add("projected")
            elif "intrinsicbank" in npz_path:
                projection_modes.add("intrinsic")
            start_id = len(source_ids)
            source_ids.extend(range(start_id, start_id + len(chunk["H1"])))
            total_mass.extend([float("nan")] * len(chunk["H1"]))
            distance_mpc.extend([float("nan")] * len(chunk["H1"]))

    if len(projection_modes) > 1:
        raise ValueError(f"Mixed projection modes found in signal bank {bank_dir}: {sorted(projection_modes)}")
    projection_mode = next(iter(projection_modes), "unknown")

    return {
        "H1": np.concatenate(bank["H1"], axis=0),
        "L1": np.concatenate(bank["L1"], axis=0),
        "source_ids": np.array(source_ids, dtype=np.int64),
        "total_mass": np.array(total_mass, dtype=np.float32),
        "distance_mpc": np.array(distance_mpc, dtype=np.float32),
        "projection_mode": projection_mode,
    }


def subset_signal_pool(signal_pool, mask):
    return {
        "H1": signal_pool["H1"][mask],
        "L1": signal_pool["L1"][mask],
        "source_ids": signal_pool["source_ids"][mask],
        "total_mass": signal_pool["total_mass"][mask],
        "distance_mpc": signal_pool["distance_mpc"][mask],
        "projection_mode": signal_pool.get("projection_mode", "unknown"),
    }


def find_bin_index(value, edges):
    if value < edges[0] or value > edges[-1]:
        return None
    if value == edges[-1]:
        return len(edges) - 2
    for idx in range(len(edges) - 1):
        if edges[idx] <= value < edges[idx + 1]:
            return idx
    return None


def mass_bin_label(total_mass):
    idx = find_bin_index(total_mass, MASS_BIN_EDGES)
    return None if idx is None else MASS_BIN_LABELS[idx]


def distance_bin_label(distance_mpc):
    idx = find_bin_index(distance_mpc, DISTANCE_BIN_EDGES)
    return None if idx is None else DISTANCE_BIN_LABELS[idx]


def stratified_signal_indices(signal_pool, count, seed):
    rng = np.random.default_rng(seed)
    cells = [(mi, di) for mi in range(len(MASS_BIN_LABELS)) for di in range(len(DISTANCE_BIN_LABELS))]
    indices_by_cell = {cell: [] for cell in cells}

    for idx, (total_mass, distance_mpc) in enumerate(zip(signal_pool["total_mass"], signal_pool["distance_mpc"])):
        mi = find_bin_index(float(total_mass), MASS_BIN_EDGES)
        di = find_bin_index(float(distance_mpc), DISTANCE_BIN_EDGES)
        if mi is None or di is None:
            continue
        indices_by_cell[(mi, di)].append(idx)

    selected = []
    leftovers = []
    base = count // len(cells)
    remainder = count % len(cells)
    for cell_idx, cell in enumerate(cells):
        cell_indices = np.array(indices_by_cell[cell], dtype=np.int64)
        if len(cell_indices) == 0:
            continue
        shuffled = rng.permutation(cell_indices)
        target = base + int(cell_idx < remainder)
        take = min(target, len(shuffled))
        selected.extend(shuffled[:take].tolist())
        leftovers.extend(shuffled[take:].tolist())

    if len(selected) < count:
        leftovers = rng.permutation(np.array(leftovers, dtype=np.int64)).tolist() if leftovers else []
        need = count - len(selected)
        selected.extend(leftovers[:need])

    if len(selected) < count:
        valid = np.where(np.isfinite(signal_pool["total_mass"]) & np.isfinite(signal_pool["distance_mpc"]))[0]
        if len(valid) == 0:
            raise ValueError("No valid signal-bank metadata available for stratified validation sampling.")
        fill = rng.choice(valid, size=count - len(selected), replace=True)
        selected.extend(int(x) for x in fill)

    selected = np.array(selected[:count], dtype=np.int64)
    rng.shuffle(selected)
    return selected


def build_o1_benchmark_injections(
    noise_raw,
    detectors,
    signal_pool,
    selected_indices,
    asd_by_detector,
    notch_lines=True,
    line_configuration="o1",
):
    if len(noise_raw) != len(selected_indices):
        raise ValueError("noise_raw and selected_indices must have the same length for benchmark injection construction.")

    injected_raw = np.empty_like(noise_raw, dtype=np.float32)
    benchmark_meta = []
    noise_len = noise_raw.shape[1]
    for idx, (detector_name, pool_idx) in enumerate(zip(detectors, selected_indices)):
        signal_sel = signal_pool[detector_name][pool_idx].astype(np.float32)
        scale, source_network_snr, target_network_snr = maybe_rescale_projected_signal_pair(
            signal_pool,
            int(pool_idx),
            asd_by_detector,
            np.random.default_rng(SEED + idx),
        )
        if scale != 1.0:
            signal_sel = signal_sel * np.float32(scale)
        # FIX1: benchmark/eval injection -> deterministic center (rng=None)
        signal_sel = place_signal_in_segment(signal_sel, noise_len, rng=None)
        injected_raw[idx] = noise_raw[idx] + signal_sel
        total_mass = float(signal_pool["total_mass"][pool_idx])
        distance_mpc = float(signal_pool["distance_mpc"][pool_idx])
        benchmark_meta.append(
            {
                "source_id": int(signal_pool["source_ids"][pool_idx]),
                "total_mass": total_mass,
                "distance_mpc": distance_mpc,
                "mass_bin": mass_bin_label(total_mass),
                "distance_bin": distance_bin_label(distance_mpc),
                "source_network_snr": source_network_snr,
                "target_network_snr": target_network_snr,
                "scale_factor": scale,
            }
        )

    whitened = whiten_batch_gwpy_o1(
        injected_raw,
        detectors,
        asd_by_detector,
        notch_lines=notch_lines,
        line_configuration=line_configuration,
    )
    qt = compute_qt_tensors(whitened)
    return qt, benchmark_meta


def build_o1_matched_detector_benchmark_injections(
    noise_raw,
    noise_meta,
    signal_pool,
    selected_indices,
    asd_by_detector,
    notch_lines=True,
    line_configuration="o1",
):
    det_to_noise_idx = {"H1": [], "L1": []}
    for idx, row in enumerate(noise_meta):
        detector = row["detector"]
        if detector in det_to_noise_idx:
            det_to_noise_idx[detector].append(idx)

    pair_count = min(
        len(selected_indices),
        len(det_to_noise_idx["H1"]),
        len(det_to_noise_idx["L1"]),
    )
    if pair_count == 0:
        raise ValueError("Need at least one H1 and one L1 noise segment to build matched detector benchmarks.")

    selected_indices = np.asarray(selected_indices[:pair_count], dtype=np.int64)
    injected_raw = np.empty((2 * pair_count, noise_raw.shape[1]), dtype=np.float32)
    detectors = []
    benchmark_meta = []
    noise_len = noise_raw.shape[1]

    out_idx = 0
    for pair_id, pool_idx in enumerate(selected_indices):
        scale, source_network_snr, target_network_snr = maybe_rescale_projected_signal_pair(
            signal_pool,
            int(pool_idx),
            asd_by_detector,
            np.random.default_rng(SEED + pair_id),
        )
        total_mass = float(signal_pool["total_mass"][pool_idx])
        distance_mpc = float(signal_pool["distance_mpc"][pool_idx])
        base_meta = {
            "pair_id": int(pair_id),
            "source_id": int(signal_pool["source_ids"][pool_idx]),
            "total_mass": total_mass,
            "distance_mpc": distance_mpc,
            "mass_bin": mass_bin_label(total_mass),
            "distance_bin": distance_bin_label(distance_mpc),
            "source_network_snr": source_network_snr,
            "target_network_snr": target_network_snr,
            "scale_factor": scale,
        }

        for detector_name in ("H1", "L1"):
            noise_idx = det_to_noise_idx[detector_name][pair_id]
            signal_sel = signal_pool[detector_name][pool_idx].astype(np.float32)
            if scale != 1.0:
                signal_sel = signal_sel * np.float32(scale)
            # FIX1: benchmark/eval injection -> deterministic center (rng=None)
            signal_sel = place_signal_in_segment(signal_sel, noise_len, rng=None)
            injected_raw[out_idx] = noise_raw[noise_idx] + signal_sel
            detectors.append(detector_name)
            benchmark_meta.append({**base_meta, "detector": detector_name})
            out_idx += 1

    whitened = whiten_batch_gwpy_o1(
        injected_raw,
        detectors,
        asd_by_detector,
        notch_lines=notch_lines,
        line_configuration=line_configuration,
    )
    qt = compute_qt_tensors(whitened)
    return qt, benchmark_meta


def filter_rows_and_array(rows, array, detector_mode):
    if detector_mode == "both":
        return rows, array
    keep_idx = [idx for idx, row in enumerate(rows) if row["detector"] == detector_mode]
    filtered_rows = [rows[idx] for idx in keep_idx]
    filtered_array = array[keep_idx]
    return filtered_rows, filtered_array


def load_prepared_o1_arrays(prepared_dir, detector_mode="both"):
    noise_train = np.load(os.path.join(prepared_dir, "noise_train.npy")).astype(np.float32)
    noise_val = np.load(os.path.join(prepared_dir, "noise_val.npy")).astype(np.float32)
    noise_test = np.load(os.path.join(prepared_dir, "noise_test.npy")).astype(np.float32)
    event_windows = np.load(os.path.join(prepared_dir, "event_windows.npy")).astype(np.float32)
    noise_train_meta = load_noise_metadata(os.path.join(prepared_dir, "noise_train_metadata.csv"))
    noise_val_meta = load_noise_metadata(os.path.join(prepared_dir, "noise_val_metadata.csv"))
    noise_test_meta = load_noise_metadata(os.path.join(prepared_dir, "noise_test_metadata.csv"))

    event_meta = []
    meta_path = os.path.join(prepared_dir, "event_metadata.csv")
    with open(meta_path, newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            event_meta.append(
                {
                    "event_name": row["event_name"],
                    "detector": row["detector"],
                    "event_gps": float(row["event_gps"]),
                    "window_center_gps": float(row["window_center_gps"]),
                    "offset_seconds": float(row["offset_seconds"]),
                }
            )

    noise_train_meta, noise_train = filter_rows_and_array(noise_train_meta, noise_train, detector_mode)
    noise_val_meta, noise_val = filter_rows_and_array(noise_val_meta, noise_val, detector_mode)
    noise_test_meta, noise_test = filter_rows_and_array(noise_test_meta, noise_test, detector_mode)
    event_meta, event_windows = filter_rows_and_array(event_meta, event_windows, detector_mode)

    if len(noise_train_meta) == 0 or len(noise_val_meta) == 0 or len(noise_test_meta) == 0:
        raise ValueError(f"No O1 noise samples remain after detector filtering with mode '{detector_mode}'")
    if len(event_meta) == 0:
        raise ValueError(f"No O1 event windows remain after detector filtering with mode '{detector_mode}'")

    return noise_train, noise_val, noise_test, event_windows, noise_train_meta, noise_val_meta, noise_test_meta, event_meta


def whiten_batch_gwpy_o1(waveforms, detectors, asd_by_detector, notch_lines=True, line_configuration="o1"):
    whitened = []
    for waveform, detector in zip(waveforms, detectors):
        ts = TimeSeries((waveform - np.mean(waveform)).astype(np.float64), sample_rate=FS)
        tsw = ts.whiten(asd=asd_by_detector[detector], fduration=O1_WHITEN_FDURATION, highpass=20)
        arr = np.asarray(tsw.value, dtype=np.float64)
        if notch_lines:
            arr = apply_o1_notches(arr, detector, line_configuration=line_configuration)
        arr = arr.astype(np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite samples after gwpy whitening for detector {detector}")
        whitened.append(arr)
    return np.stack(whitened, axis=0)


def compute_optimal_snr(signal, asd):
    signal = np.asarray(signal, dtype=np.float64)
    dt = 1.0 / FS
    freqs = np.fft.rfftfreq(len(signal), d=dt)
    hf = np.fft.rfft(signal) * dt
    asd_freqs = np.asarray(asd.frequencies.value, dtype=np.float64)
    psd_vals = np.asarray(asd.value, dtype=np.float64) ** 2
    psd_interp = np.interp(freqs, asd_freqs, psd_vals, left=np.inf, right=np.inf)
    valid = np.isfinite(psd_interp) & (psd_interp > 0.0) & (freqs >= 20.0)
    if not np.any(valid):
        raise ValueError("No valid PSD support for optimal-SNR computation.")
    df = freqs[1] - freqs[0]
    snr_sq = 4.0 * np.sum((np.abs(hf[valid]) ** 2) / psd_interp[valid]) * df
    return float(np.sqrt(max(snr_sq, 0.0)))


def place_signal_in_segment(signal, segment_len, rng=None, max_shift_samples=None):
    """Place signal in segment with optional random time offset.

    FIX1 (random time shift): when ``rng`` is provided, the signal is placed at
    its centered position plus a random integer offset drawn uniformly from
    [-max_shift_samples, +max_shift_samples]. This shifts the signal within the
    raw segment so that, after the deterministic 2 s context extraction and 1 s
    center crop, the signal no longer always sits at the exact center -- forcing
    the network to learn time-translation robustness.

    For training injections, callers pass ``rng=rng`` with
    ``max_shift_samples=int(0.5 * FS)`` (+/-0.5 s, which keeps the full 2 s
    context available within the 4 s raw segment). For evaluation/benchmark
    injections, callers pass ``rng=None`` so the placement is the deterministic
    center -- bit-identical to the legacy ``center_signal_in_segment`` behavior,
    keeping the baseline comparison fair.

    Args:
        signal: waveform array
        segment_len: target segment length in samples
        rng: numpy random Generator (if None, centers deterministically)
        max_shift_samples: maximum shift in samples (default: segment_len//4)
    """
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) >= segment_len:
        start = max(0, (len(signal) - segment_len) // 2)
        return signal[start:start + segment_len]
    padded = np.zeros(segment_len, dtype=np.float32)
    center_start = max(0, (segment_len - len(signal)) // 2)
    if rng is not None:
        if max_shift_samples is None:
            max_shift_samples = segment_len // 4
        shift = int(rng.integers(-max_shift_samples, max_shift_samples + 1))
        start = max(0, min(center_start + shift, segment_len - len(signal)))
    else:
        start = center_start
    padded[start:start + len(signal)] = signal
    return padded


def maybe_rescale_projected_signal_pair(signal_pool, sig_idx, asd_by_detector, rng):
    if signal_pool.get("projection_mode") != "projected":
        return 1.0, None, None

    snr_h1 = compute_optimal_snr(signal_pool["H1"][sig_idx], asd_by_detector["H1"])
    snr_l1 = compute_optimal_snr(signal_pool["L1"][sig_idx], asd_by_detector["L1"])
    network_snr = float(np.sqrt(snr_h1 ** 2 + snr_l1 ** 2))
    if network_snr <= 0.0:
        raise ValueError(f"Non-positive projected network SNR for source index {sig_idx}")
    target_snr = float(rng.uniform(*PROJECTED_TARGET_NETWORK_SNR_RANGE))
    return target_snr / network_snr, network_snr, target_snr


def maybe_rescale_projected_signal_detector(signal_pool, sig_idx, detector, asd, rng):
    if signal_pool.get("projection_mode") != "projected":
        return 1.0, None, None

    detector_snr = compute_optimal_snr(signal_pool[detector][sig_idx], asd)
    if detector_snr <= 0.0:
        raise ValueError(f"Non-positive projected detector SNR for source index {sig_idx} on {detector}")
    target_snr = float(rng.uniform(*PROJECTED_TARGET_NETWORK_SNR_RANGE))
    return target_snr / detector_snr, detector_snr, target_snr


def build_o1_injections(
    noise_raw,
    detectors,
    signal_pool,
    asd_by_detector,
    seed,
    notch_lines=True,
    return_source_ids=False,
    line_configuration="o1",
):
    rng = np.random.default_rng(seed)
    n = len(noise_raw)
    if n == 0:
        raise ValueError("Cannot build O1 injections from an empty noise split.")
    injected_raw = np.empty_like(noise_raw, dtype=np.float32)
    noise_len = noise_raw.shape[1]
    pools = {}
    selected_source_ids = []
    for detector_name in ("H1", "L1"):
        if detector_name not in signal_pool:
            raise KeyError(f"Missing detector '{detector_name}' in projected O1 signal pool")
        pools[detector_name] = signal_pool[detector_name]
        if len(pools[detector_name]) == 0:
            raise ValueError(f"Projected O1 signal pool for detector {detector_name} is empty")
    if len(pools["H1"]) != len(pools["L1"]):
        raise ValueError("Projected O1 signal pool must have aligned H1/L1 source counts for network-SNR scaling.")

    for idx, detector_name in enumerate(detectors):
        sig_idx = int(rng.integers(len(pools["H1"])))
        signal_sel = pools[detector_name][sig_idx].astype(np.float32)
        scale, _, _ = maybe_rescale_projected_signal_pair(signal_pool, sig_idx, asd_by_detector, rng)
        if scale != 1.0:
            signal_sel = signal_sel * np.float32(scale)
        if return_source_ids:
            pool_source_ids = signal_pool.get("source_ids")
            if pool_source_ids is None:
                raise KeyError("signal_pool is missing source_ids while return_source_ids=True")
            selected_source_ids.append(int(pool_source_ids[sig_idx]))
        # FIX1: training injection -> random +/-0.5 s time shift
        signal_sel = place_signal_in_segment(
            signal_sel, noise_len, rng=rng, max_shift_samples=int(0.5 * FS)
        )
        injected_raw[idx] = noise_raw[idx] + signal_sel

    whitened = whiten_batch_gwpy_o1(
        injected_raw,
        detectors,
        asd_by_detector,
        notch_lines=notch_lines,
        line_configuration=line_configuration,
    )
    qt = compute_qt_tensors(whitened)
    if return_source_ids:
        return qt, np.array(selected_source_ids, dtype=np.int64)
    return qt


def make_loader(tensor, batch_size, shuffle, num_workers):
    if isinstance(tensor, Dataset):
        dataset = tensor
    else:
        dataset = TensorDataset(tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )


def make_replacement_loader(tensor, batch_size, num_workers, num_samples, seed):
    if isinstance(tensor, Dataset):
        dataset = tensor
    else:
        dataset = TensorDataset(tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )


def compute_scores(model, loader, device, score_mode="classifier"):
    model.eval()
    scores = []
    with torch.no_grad():
        for (qt,) in loader:
            qt = qt.to(device)
            if score_mode == "classifier":
                batch_scores = model.score(qt)
            elif score_mode == "recon":
                batch_scores = compute_reconstruction_loss(model, qt)
            else:
                raise ValueError(f"Unsupported score_mode: {score_mode}")
            scores.extend(batch_scores.cpu().numpy())
    return np.array(scores)


def compute_reconstruction_loss(model, qt):
    recon = model(qt)
    return ((recon - qt) ** 2).mean(dim=[1, 2, 3])


def summarize_logits(logit_batches):
    if not logit_batches:
        return np.array([], dtype=np.float64), float("nan"), float("nan")
    logits = torch.cat(logit_batches).numpy().astype(np.float64, copy=False)
    return logits, float(logits.mean()), float(logits.std())


def compute_joint_loss(
    model,
    noise_batch,
    signal_batch,
    device,
    recon_weight,
    clf_weight,
):
    (x_n,) = noise_batch
    (x_s,) = signal_batch
    pair_count = min(len(x_n), len(x_s))
    x_n = x_n[:pair_count]
    x_s = x_s[:pair_count]
    x_n = x_n.to(device)
    x_s = x_s.to(device)
    recon_noise = compute_reconstruction_loss(model, x_n).mean()
    noise_logits = model.get_logit(x_n)
    signal_logits = model.get_logit(x_s)
    noise_targets = torch.full_like(noise_logits, LABEL_SMOOTH_NOISE)
    signal_targets = torch.full_like(signal_logits, LABEL_SMOOTH_SIGNAL)
    noise_cls_loss = WEAKSUP_CRITERION(noise_logits, noise_targets)
    signal_cls_loss = WEAKSUP_CRITERION(signal_logits, signal_targets)
    cls_loss = 0.5 * (noise_cls_loss + signal_cls_loss)
    total = recon_weight * recon_noise + clf_weight * cls_loss
    noise_acc = ((noise_logits < 0.0).float().mean()).item()
    signal_acc = ((signal_logits >= 0.0).float().mean()).item()
    return (
        total,
        recon_noise.item(),
        cls_loss.item(),
        noise_acc,
        signal_acc,
        noise_logits.detach().cpu(),
        signal_logits.detach().cpu(),
    )


def compute_margin_loss(
    model,
    noise_batch,
    signal_batch,
    device,
    margin,
    lambda_anom,
):
    (x_n,) = noise_batch
    (x_s,) = signal_batch
    pair_count = min(len(x_n), len(x_s))
    x_n = x_n[:pair_count].to(device)
    x_s = x_s[:pair_count].to(device)
    noise_scores = compute_reconstruction_loss(model, x_n)
    signal_scores = compute_reconstruction_loss(model, x_s)
    noise_loss = noise_scores.mean()
    signal_loss = signal_scores.mean()
    margin_loss = F.relu(margin * noise_loss.detach() - signal_loss)
    total = noise_loss + lambda_anom * margin_loss
    return (
        total,
        float(noise_loss.item()),
        float(signal_loss.item()),
        float(margin_loss.item()),
        noise_scores.detach().cpu(),
        signal_scores.detach().cpu(),
    )


def train_unsup_model(model, noise_train_loader, noise_val_loader, cfg, device):
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-3
    )

    history = {"train_total": [], "val_total": []}
    best_state = None
    best_val_total = float("inf")

    for epoch in range(cfg.unsup_epochs):
        model.train()
        train_losses = []
        for (qt,) in noise_train_loader:
            qt = qt.to(device)
            optimizer.zero_grad()
            loss = compute_reconstruction_loss(model, qt).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (qt,) in noise_val_loader:
                qt = qt.to(device)
                loss = compute_reconstruction_loss(model, qt).mean()
                val_losses.append(loss.item())

        train_total = float(np.mean(train_losses))
        val_total = float(np.mean(val_losses))
        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        scheduler.step(val_total)

        if val_total < best_val_total:
            best_val_total = val_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[unsup {epoch+1:02d}/{cfg.unsup_epochs}] train={train_total:.6f} val={val_total:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


def train_joint_model(
    model,
    noise_train_loader,
    noise_val_loader,
    sig_train_loader,
    sig_val_loader,
    cfg,
    device,
    val_benchmark_loader=None,
    val_benchmark_meta=None,
    phase_tag="joint",
    num_epochs=None,
):
    if num_epochs is None:
        num_epochs = cfg.epochs
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-3
    )

    history = {
        "train_total": [], "val_total": [],
        "train_recon_noise": [], "val_recon_noise": [],
        "train_cls_loss": [], "val_cls_loss": [],
        "train_noise_acc": [], "val_noise_acc": [],
        "train_signal_acc": [], "val_signal_acc": [],
        "train_noise_score_mean": [], "val_noise_score_mean": [],
        "train_noise_score_std": [], "val_noise_score_std": [],
        "train_signal_score_mean": [], "val_signal_score_mean": [],
        "train_signal_score_std": [], "val_signal_score_std": [],
        "train_latent_sep": [], "val_latent_sep": [],
        "val_inj_n_above_3sigma": [],
        "val_inj_mean_score": [],
        "val_noise_threshold_3sigma": [],
        "val_sep_k1": [],
        "val_mass_bin_n_above_3sigma": {label: [] for label in MASS_BIN_LABELS},
        "val_mass_bin_hit_rate": {label: [] for label in MASS_BIN_LABELS},
    }

    best_n_above_3sigma = -1
    best_epoch = -1
    best_val_noise = float("inf")
    best_val_total = float("inf")
    best_val_sig = float("nan")
    best_val_inj_mean = float("nan")
    best_val_sep_k1 = float("nan")
    best_val_t3 = float("nan")
    best_val_recon_noise = float("nan")
    best_val_cls_loss = float("nan")
    best_val_noise_acc = float("nan")
    best_val_signal_acc = float("nan")
    best_val_noise_score_mean = float("nan")
    best_val_noise_score_std = float("nan")
    best_val_signal_score_mean = float("nan")
    best_val_signal_score_std = float("nan")
    best_val_mass_bin_n_above_3sigma = {label: 0 for label in MASS_BIN_LABELS}
    best_val_mass_bin_hit_rate = {label: float("nan") for label in MASS_BIN_LABELS}
    best_state = None
    es_counter = 0
    consecutive_low_noise_acc = 0
    val_benchmark_mass_bins = None
    if val_benchmark_meta is not None:
        val_benchmark_mass_bins = [row["mass_bin"] for row in val_benchmark_meta]

    for epoch in range(num_epochs):
        model.train()
        tl, tr, tcl, tna, tsa = [], [], [], [], []
        train_noise_logits_batches, train_signal_logits_batches = [], []
        for noise_batch, signal_batch in zip(noise_train_loader, sig_train_loader):
            optimizer.zero_grad()
            loss, recon_noise, cls_loss, noise_acc, signal_acc, noise_logits, signal_logits = compute_joint_loss(
                model,
                noise_batch,
                signal_batch,
                device,
                cfg.recon_weight,
                cfg.clf_weight,
            )
            loss.backward()
            optimizer.step()
            tl.append(loss.item())
            tr.append(recon_noise)
            tcl.append(cls_loss)
            tna.append(noise_acc)
            tsa.append(signal_acc)
            train_noise_logits_batches.append(noise_logits)
            train_signal_logits_batches.append(signal_logits)

        model.eval()
        vl, vr, vcl, vna, vsa = [], [], [], [], []
        val_noise_logits_batches, val_signal_logits_batches = [], []
        with torch.no_grad():
            for noise_batch, signal_batch in zip(noise_val_loader, sig_val_loader):
                loss, recon_noise, cls_loss, noise_acc, signal_acc, noise_logits, signal_logits = compute_joint_loss(
                    model,
                    noise_batch,
                    signal_batch,
                    device,
                    cfg.recon_weight,
                    cfg.clf_weight,
                )
                vl.append(loss.item())
                vr.append(recon_noise)
                vcl.append(cls_loss)
                vna.append(noise_acc)
                vsa.append(signal_acc)
                val_noise_logits_batches.append(noise_logits)
                val_signal_logits_batches.append(signal_logits)

        train_total = float(np.mean(tl))
        val_total = float(np.mean(vl))
        train_recon_noise = float(np.mean(tr))
        val_recon_noise = float(np.mean(vr))
        train_cls_loss = float(np.mean(tcl))
        val_cls_loss = float(np.mean(vcl))
        train_noise_acc = float(np.mean(tna))
        val_noise_acc = float(np.mean(vna))
        train_signal_acc = float(np.mean(tsa))
        val_signal_acc = float(np.mean(vsa))
        _, train_noise_score_mean, train_noise_score_std = summarize_logits(train_noise_logits_batches)
        _, val_noise_score_mean, val_noise_score_std = summarize_logits(val_noise_logits_batches)
        _, train_signal_score_mean, train_signal_score_std = summarize_logits(train_signal_logits_batches)
        _, val_signal_score_mean, val_signal_score_std = summarize_logits(val_signal_logits_batches)
        train_latent_sep = train_signal_score_mean - train_noise_score_mean
        val_latent_sep = val_signal_score_mean - val_noise_score_mean
        val_noise_scores = compute_scores(model, noise_val_loader, device, score_mode="classifier").astype(np.float64)
        val_noise_mu = float(val_noise_scores.mean())
        val_noise_std = float(val_noise_scores.std())
        val_t3 = val_noise_mu + 3.0 * val_noise_std
        val_criterion_scores = compute_scores(model, sig_val_loader, device, score_mode="classifier").astype(np.float64)
        if val_benchmark_loader is not None:
            val_criterion_scores = compute_scores(model, val_benchmark_loader, device, score_mode="classifier").astype(np.float64)
        val_inj_n_above_3sigma = int((val_criterion_scores > val_t3).sum())
        val_inj_mean = float(val_criterion_scores.mean())
        val_sep_k1 = float(val_inj_mean - val_noise_mu)
        val_mass_bin_n_above_3sigma = {label: 0 for label in MASS_BIN_LABELS}
        val_mass_bin_hit_rate = {label: float("nan") for label in MASS_BIN_LABELS}
        if val_benchmark_mass_bins is not None:
            above = val_criterion_scores > val_t3
            for label in MASS_BIN_LABELS:
                mask = np.array([bin_label == label for bin_label in val_benchmark_mass_bins], dtype=bool)
                total = int(mask.sum())
                hits = int(above[mask].sum()) if total else 0
                val_mass_bin_n_above_3sigma[label] = hits
                val_mass_bin_hit_rate[label] = float(hits / total) if total else float("nan")

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_recon_noise"].append(train_recon_noise)
        history["val_recon_noise"].append(val_recon_noise)
        history["train_cls_loss"].append(train_cls_loss)
        history["val_cls_loss"].append(val_cls_loss)
        history["train_noise_acc"].append(train_noise_acc)
        history["val_noise_acc"].append(val_noise_acc)
        history["train_signal_acc"].append(train_signal_acc)
        history["val_signal_acc"].append(val_signal_acc)
        history["train_noise_score_mean"].append(train_noise_score_mean)
        history["val_noise_score_mean"].append(val_noise_score_mean)
        history["train_noise_score_std"].append(train_noise_score_std)
        history["val_noise_score_std"].append(val_noise_score_std)
        history["train_signal_score_mean"].append(train_signal_score_mean)
        history["val_signal_score_mean"].append(val_signal_score_mean)
        history["train_signal_score_std"].append(train_signal_score_std)
        history["val_signal_score_std"].append(val_signal_score_std)
        history["train_latent_sep"].append(train_latent_sep)
        history["val_latent_sep"].append(val_latent_sep)
        history["val_inj_n_above_3sigma"].append(val_inj_n_above_3sigma)
        history["val_inj_mean_score"].append(val_inj_mean)
        history["val_noise_threshold_3sigma"].append(val_t3)
        history["val_sep_k1"].append(val_sep_k1)
        for label in MASS_BIN_LABELS:
            history["val_mass_bin_n_above_3sigma"][label].append(val_mass_bin_n_above_3sigma[label])
            history["val_mass_bin_hit_rate"][label].append(val_mass_bin_hit_rate[label])

        scheduler.step(val_total)

        gate_enabled = cfg.checkpoint_gate_acc_min > 0.0
        gate_passed = (
            (not gate_enabled)
            or (
                val_noise_acc > cfg.checkpoint_gate_acc_min
                and val_signal_acc > cfg.checkpoint_gate_acc_min
            )
        )

        if gate_passed and (
            val_inj_n_above_3sigma > best_n_above_3sigma
            or (
                val_inj_n_above_3sigma == best_n_above_3sigma
                and val_sep_k1 > best_val_sep_k1
            )
        ):
            best_n_above_3sigma = val_inj_n_above_3sigma
            best_epoch = epoch + 1
            best_val_noise = val_noise_mu
            best_val_total = val_total
            best_val_sig = val_signal_score_mean
            best_val_inj_mean = val_inj_mean
            best_val_sep_k1 = val_sep_k1
            best_val_t3 = val_t3
            best_val_recon_noise = val_recon_noise
            best_val_cls_loss = val_cls_loss
            best_val_noise_acc = val_noise_acc
            best_val_signal_acc = val_signal_acc
            best_val_noise_score_mean = val_noise_score_mean
            best_val_noise_score_std = val_noise_score_std
            best_val_signal_score_mean = val_signal_score_mean
            best_val_signal_score_std = val_signal_score_std
            best_val_mass_bin_n_above_3sigma = dict(val_mass_bin_n_above_3sigma)
            best_val_mass_bin_hit_rate = dict(val_mass_bin_hit_rate)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter = 0
            tag = " <-- best"
        else:
            es_counter += 1
            gate_tag = ""
            if not gate_passed:
                gate_tag = (
                    f", gate fail acc_n={val_noise_acc:.3f}<= {cfg.checkpoint_gate_acc_min:.1f}"
                    if val_noise_acc <= cfg.checkpoint_gate_acc_min
                    else ""
                )
                if val_signal_acc <= cfg.checkpoint_gate_acc_min:
                    signal_gate = f"acc_s={val_signal_acc:.3f}<= {cfg.checkpoint_gate_acc_min:.1f}"
                    gate_tag = f"{gate_tag}, {signal_gate}" if gate_tag else f", gate fail {signal_gate}"
            tag = f" (no improve {es_counter}/{cfg.weaksup_es_patience}{gate_tag})"

        if cfg.noise_acc_early_stop_epochs > 0 and val_noise_acc < cfg.checkpoint_gate_acc_min:
            consecutive_low_noise_acc += 1
        else:
            consecutive_low_noise_acc = 0

        print(
            f"[{phase_tag} {epoch+1:02d}/{num_epochs}] "
            f"total={train_total:.6f}/{val_total:.6f} "
            f"recon={train_recon_noise:.6f}/{val_recon_noise:.6f} "
            f"cls={train_cls_loss:.6f}/{val_cls_loss:.6f} "
            f"acc_n={train_noise_acc:.3f}/{val_noise_acc:.3f} "
            f"acc_s={train_signal_acc:.3f}/{val_signal_acc:.3f} "
            f"logit_n={train_noise_score_mean:.6f}±{train_noise_score_std:.6f}/"
            f"{val_noise_score_mean:.6f}±{val_noise_score_std:.6f} "
            f"logit_s={train_signal_score_mean:.6f}±{train_signal_score_std:.6f}/"
            f"{val_signal_score_mean:.6f}±{val_signal_score_std:.6f} "
            f"latent_sep={train_latent_sep:.6f}/{val_latent_sep:.6f} "
            f"val>3sigma={val_inj_n_above_3sigma} "
            f"val_sep_k1={val_sep_k1:.6f} "
            f"mass_hit="
            f"{MASS_BIN_LABELS[0]}:{val_mass_bin_n_above_3sigma[MASS_BIN_LABELS[0]]}/"
            f"{sum(bin_label == MASS_BIN_LABELS[0] for bin_label in val_benchmark_mass_bins) if val_benchmark_mass_bins is not None else 0},"
            f"{MASS_BIN_LABELS[1]}:{val_mass_bin_n_above_3sigma[MASS_BIN_LABELS[1]]}/"
            f"{sum(bin_label == MASS_BIN_LABELS[1] for bin_label in val_benchmark_mass_bins) if val_benchmark_mass_bins is not None else 0},"
            f"{MASS_BIN_LABELS[2]}:{val_mass_bin_n_above_3sigma[MASS_BIN_LABELS[2]]}/"
            f"{sum(bin_label == MASS_BIN_LABELS[2] for bin_label in val_benchmark_mass_bins) if val_benchmark_mass_bins is not None else 0}"
            f"{tag}",
            flush=True,
        )

        if es_counter >= cfg.weaksup_es_patience:
            print(
                f"Early stopping at epoch {epoch+1} "
                f"(best val>3sigma: {best_n_above_3sigma}, best cls: {best_val_cls_loss:.6f})",
                flush=True,
            )
            break

        if (
            cfg.noise_acc_early_stop_epochs > 0
            and consecutive_low_noise_acc >= cfg.noise_acc_early_stop_epochs
        ):
            print(
                f"Early stopping at epoch {epoch+1} "
                f"(val_noise_acc<{cfg.checkpoint_gate_acc_min:.1f} for "
                f"{cfg.noise_acc_early_stop_epochs} consecutive epochs)",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    best_info = {
        "epoch": best_epoch,
        "criterion": "val_inj_n_above_3sigma",
        "val_inj_n_above_3sigma": best_n_above_3sigma,
        "val_inj_mean_score": best_val_inj_mean,
        "val_noise_threshold_3sigma": best_val_t3,
        "val_sep_k1": best_val_sep_k1,
        "val_recon_noise": best_val_recon_noise,
        "val_cls_loss": best_val_cls_loss,
        "val_noise_acc": best_val_noise_acc,
        "val_signal_acc": best_val_signal_acc,
        "val_noise_score_mean": best_val_noise_score_mean,
        "val_noise_score_std": best_val_noise_score_std,
        "val_d_signal": best_val_signal_score_mean,
        "val_d_signal_std": best_val_signal_score_std,
        "val_d_noise": best_val_noise_score_mean,
        "val_d_noise_std": best_val_noise_score_std,
        "val_latent_sep": best_val_signal_score_mean - best_val_noise_score_mean,
        "val_mass_bin_n_above_3sigma": best_val_mass_bin_n_above_3sigma,
        "val_mass_bin_hit_rate": best_val_mass_bin_hit_rate,
        "val_noise": best_val_noise,
        "val_total": best_val_total,
        "val_sig": best_val_sig,
        "val_sig_std": best_val_signal_score_std,
    }
    if cfg.checkpoint_gate_acc_min > 0.0:
        best_info["criterion"] = (
            "val_inj_n_above_3sigma gated by "
            f"val_noise_acc>{cfg.checkpoint_gate_acc_min:.1f} and "
            f"val_signal_acc>{cfg.checkpoint_gate_acc_min:.1f}"
        )
    return history, best_info


def train_margin_model(
    model,
    noise_train_loader,
    noise_val_loader,
    sig_train_loader,
    sig_val_loader,
    cfg,
    device,
    val_benchmark_loader=None,
    val_benchmark_meta=None,
    phase_tag="weaksup",
    num_epochs=None,
):
    if num_epochs is None:
        num_epochs = cfg.weaksup_epochs
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-3
    )

    history = {
        "train_total": [], "val_total": [],
        "train_noise": [], "val_noise": [],
        "train_sig": [], "val_sig": [],
        "val_inj_n_above_3sigma": [],
        "val_inj_mean_score": [],
        "val_noise_threshold_3sigma": [],
        "val_sep_k1": [],
        "val_mass_bin_n_above_3sigma": {label: [] for label in MASS_BIN_LABELS},
        "val_mass_bin_hit_rate": {label: [] for label in MASS_BIN_LABELS},
    }

    best_n_above_3sigma = -1
    best_epoch = -1
    best_val_noise = float("inf")
    best_val_total = float("inf")
    best_val_sig = float("nan")
    best_val_inj_mean = float("nan")
    best_val_sep_k1 = float("nan")
    best_val_t3 = float("nan")
    best_val_mass_bin_n_above_3sigma = {label: 0 for label in MASS_BIN_LABELS}
    best_val_mass_bin_hit_rate = {label: float("nan") for label in MASS_BIN_LABELS}
    best_state = None
    es_counter = 0
    val_benchmark_mass_bins = None
    if val_benchmark_meta is not None:
        val_benchmark_mass_bins = [row["mass_bin"] for row in val_benchmark_meta]

    for epoch in range(num_epochs):
        model.train()
        tl, tn, ts = [], [], []
        for noise_batch, signal_batch in zip(noise_train_loader, sig_train_loader):
            optimizer.zero_grad()
            total, noise_loss, signal_loss, margin_loss, _, _ = compute_margin_loss(
                model,
                noise_batch,
                signal_batch,
                device,
                cfg.margin,
                cfg.lambda_anom,
            )
            total.backward()
            optimizer.step()
            tl.append(total.item())
            tn.append(noise_loss)
            ts.append(signal_loss)

        model.eval()
        vl, vn, vs = [], [], []
        with torch.no_grad():
            for noise_batch, signal_batch in zip(noise_val_loader, sig_val_loader):
                total, noise_loss, signal_loss, margin_loss, _, _ = compute_margin_loss(
                    model,
                    noise_batch,
                    signal_batch,
                    device,
                    cfg.margin,
                    cfg.lambda_anom,
                )
                vl.append(total.item())
                vn.append(noise_loss)
                vs.append(signal_loss)

        train_total = float(np.mean(tl))
        val_total = float(np.mean(vl))
        train_noise = float(np.mean(tn))
        val_noise = float(np.mean(vn))
        train_sig = float(np.mean(ts))
        val_sig = float(np.mean(vs))
        val_noise_scores = compute_scores(model, noise_val_loader, device, score_mode="recon").astype(np.float64)
        val_noise_mu = float(val_noise_scores.mean())
        val_noise_std = float(val_noise_scores.std())
        val_t3 = val_noise_mu + 3.0 * val_noise_std
        val_criterion_scores = compute_scores(model, sig_val_loader, device, score_mode="recon").astype(np.float64)
        if val_benchmark_loader is not None:
            val_criterion_scores = compute_scores(model, val_benchmark_loader, device, score_mode="recon").astype(np.float64)
        val_inj_n_above_3sigma = int((val_criterion_scores > val_t3).sum())
        val_inj_mean = float(val_criterion_scores.mean())
        val_sep_k1 = float(val_inj_mean - val_noise_mu)
        val_mass_bin_n_above_3sigma = {label: 0 for label in MASS_BIN_LABELS}
        val_mass_bin_hit_rate = {label: float("nan") for label in MASS_BIN_LABELS}
        if val_benchmark_mass_bins is not None:
            above = val_criterion_scores > val_t3
            for label in MASS_BIN_LABELS:
                mask = np.array([bin_label == label for bin_label in val_benchmark_mass_bins], dtype=bool)
                total = int(mask.sum())
                hits = int(above[mask].sum()) if total else 0
                val_mass_bin_n_above_3sigma[label] = hits
                val_mass_bin_hit_rate[label] = float(hits / total) if total else float("nan")

        history["train_total"].append(train_total)
        history["val_total"].append(val_total)
        history["train_noise"].append(train_noise)
        history["val_noise"].append(val_noise)
        history["train_sig"].append(train_sig)
        history["val_sig"].append(val_sig)
        history["val_inj_n_above_3sigma"].append(val_inj_n_above_3sigma)
        history["val_inj_mean_score"].append(val_inj_mean)
        history["val_noise_threshold_3sigma"].append(val_t3)
        history["val_sep_k1"].append(val_sep_k1)
        for label in MASS_BIN_LABELS:
            history["val_mass_bin_n_above_3sigma"][label].append(val_mass_bin_n_above_3sigma[label])
            history["val_mass_bin_hit_rate"][label].append(val_mass_bin_hit_rate[label])

        scheduler.step(val_total)

        if (
            val_inj_n_above_3sigma > best_n_above_3sigma
            or (
                val_inj_n_above_3sigma == best_n_above_3sigma
                and val_sep_k1 > best_val_sep_k1
            )
        ):
            best_n_above_3sigma = val_inj_n_above_3sigma
            best_epoch = epoch + 1
            best_val_noise = val_noise_mu
            best_val_total = val_total
            best_val_sig = val_sig
            best_val_inj_mean = val_inj_mean
            best_val_sep_k1 = val_sep_k1
            best_val_t3 = val_t3
            best_val_mass_bin_n_above_3sigma = dict(val_mass_bin_n_above_3sigma)
            best_val_mass_bin_hit_rate = dict(val_mass_bin_hit_rate)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter = 0
            tag = " <-- best"
        else:
            es_counter += 1
            tag = f" (no improve {es_counter}/{cfg.weaksup_es_patience})"

        print(
            f"[{phase_tag} {epoch+1:02d}/{num_epochs}] "
            f"total={train_total:.6f}/{val_total:.6f} "
            f"noise={train_noise:.6f}/{val_noise:.6f} "
            f"sig={train_sig:.6f}/{val_sig:.6f} "
            f"val>3sigma={val_inj_n_above_3sigma} "
            f"val_sep_k1={val_sep_k1:.6f} "
            f"mass_hit="
            f"{MASS_BIN_LABELS[0]}:{val_mass_bin_n_above_3sigma[MASS_BIN_LABELS[0]]}/"
            f"{sum(bin_label == MASS_BIN_LABELS[0] for bin_label in val_benchmark_mass_bins) if val_benchmark_mass_bins is not None else 0},"
            f"{MASS_BIN_LABELS[1]}:{val_mass_bin_n_above_3sigma[MASS_BIN_LABELS[1]]}/"
            f"{sum(bin_label == MASS_BIN_LABELS[1] for bin_label in val_benchmark_mass_bins) if val_benchmark_mass_bins is not None else 0},"
            f"{MASS_BIN_LABELS[2]}:{val_mass_bin_n_above_3sigma[MASS_BIN_LABELS[2]]}/"
            f"{sum(bin_label == MASS_BIN_LABELS[2] for bin_label in val_benchmark_mass_bins) if val_benchmark_mass_bins is not None else 0}"
            f"{tag}",
            flush=True,
        )

        if es_counter >= cfg.weaksup_es_patience:
            print(
                f"Early stopping at epoch {epoch+1} "
                f"(best val>3sigma: {best_n_above_3sigma}, best total: {best_val_total:.6f})",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    best_info = {
        "epoch": best_epoch,
        "criterion": "val_inj_n_above_3sigma",
        "val_inj_n_above_3sigma": best_n_above_3sigma,
        "val_inj_mean_score": best_val_inj_mean,
        "val_noise_threshold_3sigma": best_val_t3,
        "val_sep_k1": best_val_sep_k1,
        "val_noise": best_val_noise,
        "val_total": best_val_total,
        "val_sig": best_val_sig,
        "val_d_signal": best_val_sig,
        "val_d_noise": best_val_noise,
        "val_latent_sep": best_val_sig - best_val_noise,
        "val_mass_bin_n_above_3sigma": best_val_mass_bin_n_above_3sigma,
        "val_mass_bin_hit_rate": best_val_mass_bin_hit_rate,
    }
    return history, best_info


def evaluate_model(model, noise_eval_loader, mdc_eval_loader, is_imbh, device, score_mode):
    noise_scores = compute_scores(model, noise_eval_loader, device, score_mode=score_mode)
    signal_scores = compute_scores(model, mdc_eval_loader, device, score_mode=score_mode)

    mu = float(noise_scores.mean())
    std = float(noise_scores.std())
    t3 = mu + 3.0 * std
    t5 = mu + 5.0 * std

    eff3 = 100.0 * float((signal_scores > t3).mean())
    eff5 = 100.0 * float((signal_scores > t5).mean())
    eff3_imbh = 100.0 * float((signal_scores[is_imbh] > t3).mean())
    eff5_imbh = 100.0 * float((signal_scores[is_imbh] > t5).mean())

    labels = np.concatenate([np.zeros(len(noise_scores)), np.ones(len(signal_scores))])
    scores = np.concatenate([noise_scores, signal_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    return {
        "eff3": eff3,
        "eff5": eff5,
        "eff3_imbh": eff3_imbh,
        "eff5_imbh": eff5_imbh,
        "auc": roc_auc,
        "noise_mean": mu,
        "noise_std": std,
        "t3": t3,
        "t5": t5,
        "noise_scores": np.asarray(noise_scores, dtype=np.float32),
        "signal_scores": np.asarray(signal_scores, dtype=np.float32),
    }


def load_mdc(reference_psd, noise_sigma, cfg):
    strain_files = sorted(
        f for f in glob.glob(os.path.join(MDC_DIR, "mdc1_BH_batch*.npy"))
        if "_snr" not in f and "_catidx" not in f
    )
    snr_files = sorted(glob.glob(os.path.join(MDC_DIR, "mdc1_BH_batch*_snr.npy")))

    all_strains, all_snrs, all_catidxs = [], [], []
    for sf, snrf in zip(strain_files, snr_files):
        idxf = sf.replace(".npy", "_catidx.npy")
        strains = np.load(sf)
        snrs = np.load(snrf).flatten()
        n = min(len(strains), len(snrs))
        all_strains.append(strains[:n])
        all_snrs.append(snrs[:n])
        if os.path.exists(idxf):
            all_catidxs.append(np.load(idxf).flatten()[:n])

    all_strains = np.concatenate(all_strains, axis=0)
    all_snrs = np.concatenate(all_snrs, axis=0)
    all_catidxs = np.concatenate(all_catidxs, axis=0) if all_catidxs else None

    noise_mdc = load_raw(NOISE_DIR, cfg.n_files)
    noise_mdc = downsample_waveforms(noise_mdc, FS_NOISE, FS)

    mdc_strains = all_strains
    if mdc_strains.shape[1] == load_raw(NOISE_DIR, 1).shape[1]:
        mdc_strains = downsample_waveforms(mdc_strains, FS_NOISE, FS)

    seg_len = mdc_strains.shape[1]
    noise_idx = np.random.default_rng(SEED).choice(len(noise_mdc), size=len(mdc_strains), replace=True)
    noise_segs = noise_mdc[noise_idx]
    if noise_segs.shape[1] >= seg_len:
        noise_segs = noise_segs[:, :seg_len]
    else:
        noise_segs = np.pad(noise_segs, ((0, 0), (0, seg_len - noise_segs.shape[1])))

    mdc_injected = mdc_strains + noise_segs
    mdc_whitened = whiten_batch(mdc_injected, reference_psd)
    mdc_qt_tensor = compute_qt_tensors(mdc_whitened)

    catalog = pd.read_csv(os.path.join(MDC_DIR, "mdc1_BH_catalog.csv"))
    if all_catidxs is not None:
        catalog = catalog[catalog["idx"].isin(all_catidxs)].reset_index(drop=True)
        idx_order = {v: i for i, v in enumerate(all_catidxs)}
        catalog = catalog.iloc[catalog["idx"].map(idx_order).argsort()].reset_index(drop=True)
    else:
        catalog = catalog.iloc[:len(all_strains)].reset_index(drop=True)

    total_mass = (catalog["mz1"].values + catalog["mz2"].values) / (1 + catalog["z"].values)
    snr_et = catalog["snrET_Opt"].values
    is_imbh = total_mass > 100

    return mdc_qt_tensor, snr_et, total_mass, is_imbh


def prepare_data(cfg, reference_psd):
    print("Loading raw waveforms...", flush=True)
    noise_raw = load_raw(NOISE_DIR, cfg.n_files)
    noise_raw = downsample_waveforms(noise_raw, FS_NOISE, FS)

    signal_raw = load_raw(SIGNAL_DIR, cfg.n_files)
    noise_len = noise_raw.shape[1]
    sig_len = signal_raw.shape[1]
    if sig_len > noise_len:
        signal_raw = signal_raw[:, :noise_len]
    elif sig_len < noise_len:
        signal_raw = np.pad(signal_raw, ((0, 0), (0, noise_len - sig_len)))

    rng = np.random.default_rng(SEED)
    noise_idx = rng.permutation(len(noise_raw))
    sig_idx = rng.permutation(len(signal_raw))
    n_noise_te = int(len(noise_raw) * 0.2)
    n_sig_te = int(len(signal_raw) * 0.2)

    noise_train_raw = noise_raw[noise_idx[n_noise_te:]]
    noise_test_raw = noise_raw[noise_idx[:n_noise_te]]
    signal_train_raw = signal_raw[sig_idx[n_sig_te:]]
    signal_test_raw = signal_raw[sig_idx[:n_sig_te]]

    inj_idx_tr = rng.choice(len(noise_train_raw), size=len(signal_train_raw), replace=True)
    inj_idx_te = rng.choice(len(noise_test_raw), size=len(signal_test_raw), replace=True)
    signal_train_inj = signal_train_raw + noise_train_raw[inj_idx_tr]
    signal_test_inj = signal_test_raw + noise_test_raw[inj_idx_te]

    print("Whitening train/test splits...", flush=True)
    noise_w_tr = whiten_batch(noise_train_raw, reference_psd)
    noise_w_te = whiten_batch(noise_test_raw, reference_psd)
    sig_w_tr = whiten_batch(signal_train_inj, reference_psd)
    sig_w_te = whiten_batch(signal_test_inj, reference_psd)

    noise_sigma = float(noise_w_tr[:200].std())
    print(f"Noise sigma (whitened): {noise_sigma:.6f}", flush=True)

    print("Computing Q-transforms...", flush=True)
    noise_qt_tr = compute_qt_tensors(noise_w_tr)
    noise_qt_te = compute_qt_tensors(noise_w_te)
    sig_qt_tr = compute_qt_tensors(sig_w_tr)
    sig_qt_te = compute_qt_tensors(sig_w_te)

    return {
        "noise_qt_tr": noise_qt_tr,
        "noise_qt_te": noise_qt_te,
        "sig_qt_tr": sig_qt_tr,
        "sig_qt_te": sig_qt_te,
        "noise_sigma": noise_sigma,
    }


def prepare_o1_real_data(cfg, reference_psd):
    print(f"Loading prepared {cfg.summary_label} arrays from {cfg.o1_data_dir}...", flush=True)
    (
        noise_train_raw,
        noise_val_raw,
        noise_test_raw,
        event_windows_raw,
        noise_train_meta,
        noise_val_meta,
        noise_test_meta,
        event_meta,
    ) = load_prepared_o1_arrays(cfg.o1_data_dir, detector_mode=cfg.detector_mode)
    asd_by_detector = {det: load_detector_asd_o1(cfg.o1_data_dir, det) for det in ("H1", "L1")}
    line_configuration = infer_line_configuration(cfg.o1_data_dir)
    noise_train_det = [row["detector"] for row in noise_train_meta]
    noise_val_det = [row["detector"] for row in noise_val_meta]
    noise_test_det = [row["detector"] for row in noise_test_meta]
    event_det = [row["detector"] for row in event_meta]
    print(f"Whitening {cfg.summary_label} train/validation/test noise with gwpy + detector ASD...", flush=True)
    noise_w_tr = whiten_batch_gwpy_o1(
        noise_train_raw, noise_train_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )
    noise_w_val = whiten_batch_gwpy_o1(
        noise_val_raw, noise_val_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )
    noise_w_te = whiten_batch_gwpy_o1(
        noise_test_raw, noise_test_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )

    print(f"Whitening {cfg.summary_label} event evaluation windows with gwpy + detector ASD...", flush=True)
    event_w = whiten_batch_gwpy_o1(
        event_windows_raw, event_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )

    noise_sigma = float(noise_w_tr[: min(200, len(noise_w_tr))].std())
    print(f"{cfg.summary_label} noise sigma (whitened): {noise_sigma:.6f}", flush=True)

    print(f"Computing Q-transforms for {cfg.summary_label} data...", flush=True)
    noise_qt_tr = compute_qt_tensors(noise_w_tr)
    noise_qt_val = compute_qt_tensors(noise_w_val)
    noise_qt_te = compute_qt_tensors(noise_w_te)
    event_qt = compute_qt_tensors(event_w)

    sig_qt_tr = None
    sig_qt_val = None
    sig_qt_val_benchmark = None
    sig_val_benchmark_meta = None
    sig_qt_eval = None
    sig_eval_meta = None
    print(f"Loading projected {cfg.summary_label} signal bank from {cfg.o1_signal_bank_dir}...", flush=True)
    signal_bank = load_o1_signal_bank(cfg.o1_signal_bank_dir)
    unique_source_ids = np.unique(signal_bank["source_ids"])
    rng = np.random.default_rng(SEED)
    perm_source_ids = rng.permutation(unique_source_ids)
    n_sig_eval = max(1, min(cfg.o1_inj_eval_count, int(len(perm_source_ids) * 0.1)))
    n_sig_val = max(1, int(len(perm_source_ids) * 0.2))
    n_sig_train = max(1, len(perm_source_ids) - n_sig_val - n_sig_eval)
    train_source_ids = set(int(x) for x in perm_source_ids[:n_sig_train])
    val_source_ids = set(int(x) for x in perm_source_ids[n_sig_train:n_sig_train + n_sig_val])
    eval_source_ids = set(int(x) for x in perm_source_ids[n_sig_train + n_sig_val:])
    source_ids = signal_bank["source_ids"]
    train_mask = np.array([sid in train_source_ids for sid in source_ids], dtype=bool)
    val_mask = np.array([sid in val_source_ids for sid in source_ids], dtype=bool)
    eval_mask = np.array([sid in eval_source_ids for sid in source_ids], dtype=bool)
    signal_train_raw = subset_signal_pool(signal_bank, train_mask)
    signal_val_raw = subset_signal_pool(signal_bank, val_mask)
    signal_eval_raw = subset_signal_pool(signal_bank, eval_mask)
    print(f"Building {cfg.summary_label} weak-supervision injections...", flush=True)
    sig_qt_tr = build_o1_injections(
        noise_train_raw,
        noise_train_det,
        signal_train_raw,
        asd_by_detector,
        seed=SEED,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
    )
    sig_qt_val = build_o1_injections(
        noise_val_raw,
        noise_val_det,
        signal_val_raw,
        asd_by_detector,
        seed=SEED + 1,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
    )
    val_benchmark_count = cfg.o1_inj_eval_count
    val_benchmark_indices = stratified_signal_indices(
        signal_val_raw,
        val_benchmark_count,
        seed=SEED + 11,
    )
    sig_qt_val_benchmark, sig_val_benchmark_meta = build_o1_matched_detector_benchmark_injections(
        noise_val_raw,
        noise_val_meta,
        signal_val_raw,
        val_benchmark_indices,
        asd_by_detector,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
    )
    for idx, row in enumerate(sig_val_benchmark_meta):
        row["benchmark_id"] = idx
    eval_benchmark_indices = stratified_signal_indices(
        signal_eval_raw,
        cfg.o1_inj_eval_count,
        seed=SEED + 12,
    )
    sig_qt_eval, sig_eval_meta = build_o1_matched_detector_benchmark_injections(
        noise_test_raw,
        noise_test_meta,
        signal_eval_raw,
        eval_benchmark_indices,
        asd_by_detector,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
    )
    for idx, row in enumerate(sig_eval_meta):
        row["benchmark_id"] = idx

    return {
        "noise_qt_tr": noise_qt_tr,
        "noise_qt_val": noise_qt_val,
        "noise_qt_te": noise_qt_te,
        "sig_qt_tr": sig_qt_tr,
        "sig_qt_val": sig_qt_val,
        "sig_qt_val_benchmark": sig_qt_val_benchmark,
        "sig_val_benchmark_meta": sig_val_benchmark_meta,
        "sig_qt_eval": sig_qt_eval,
        "sig_eval_meta": sig_eval_meta,
        "event_qt": event_qt,
        "event_meta": event_meta,
        "noise_sigma": noise_sigma,
    }


def o1_real_cache_manifest(cfg):
    return {
        "summary_label": cfg.summary_label,
        "data_dir": os.path.abspath(cfg.o1_data_dir),
        "signal_bank_dir": os.path.abspath(cfg.o1_signal_bank_dir),
        "detector_mode": cfg.detector_mode,
        "notch_lines": bool(cfg.o1_notch_lines),
        "context_seconds": float(O1_CONTEXT_SECONDS),
        "qt_center_crop_seconds": float(O1_CENTER_CROP_SECONDS),
        "spec_size": list(SPEC_SIZE),
        "qtransform_frange": list(QTRANSFORM_FRANGE),
        "qtransform_qrange": list(QTRANSFORM_QRANGE),
        "inj_eval_count": int(cfg.o1_inj_eval_count),
    }


def load_o1_real_cache(cfg, cache_dir):
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing QT cache manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    expected = o1_real_cache_manifest(cfg)
    if manifest != expected:
        raise ValueError(
            "QT cache manifest does not match current preprocessing config.\n"
            f"cache={manifest}\nexpected={expected}"
        )

    data = {
        "noise_qt_tr": NpyTensorDataset(os.path.join(cache_dir, "noise_qt_tr.npy")),
        "noise_qt_val": NpyTensorDataset(os.path.join(cache_dir, "noise_qt_val.npy")),
        "noise_qt_te": NpyTensorDataset(os.path.join(cache_dir, "noise_qt_te.npy")),
        "sig_qt_tr": NpyTensorDataset(os.path.join(cache_dir, "sig_qt_tr.npy")),
        "sig_qt_val": NpyTensorDataset(os.path.join(cache_dir, "sig_qt_val.npy")),
        "sig_qt_val_benchmark": (
            NpyTensorDataset(os.path.join(cache_dir, "sig_qt_val_benchmark.npy"))
            if os.path.exists(os.path.join(cache_dir, "sig_qt_val_benchmark.npy"))
            else None
        ),
        "sig_qt_eval": (
            NpyTensorDataset(os.path.join(cache_dir, "sig_qt_eval.npy"))
            if os.path.exists(os.path.join(cache_dir, "sig_qt_eval.npy"))
            else None
        ),
        "event_qt": NpyTensorDataset(os.path.join(cache_dir, "event_qt.npy")),
        "sig_val_benchmark_meta": (
            load_json(os.path.join(cache_dir, "sig_val_benchmark_meta.json"))
            if os.path.exists(os.path.join(cache_dir, "sig_val_benchmark_meta.json"))
            else None
        ),
        "sig_eval_meta": (
            load_json(os.path.join(cache_dir, "sig_eval_meta.json"))
            if os.path.exists(os.path.join(cache_dir, "sig_eval_meta.json"))
            else None
        ),
        "event_meta": load_json(os.path.join(cache_dir, "event_meta.json")),
        "noise_test_meta": (
            load_json(os.path.join(cache_dir, "noise_test_meta.json"))
            if os.path.exists(os.path.join(cache_dir, "noise_test_meta.json"))
            else []
        ),
        "noise_sigma": float(load_json(os.path.join(cache_dir, "noise_sigma.json"))["noise_sigma"]),
    }
    return data


def prepare_o1_real_data_cached(cfg):
    cache_dir = cfg.qt_cache_dir or os.path.join(cfg.output_dir, "qt_cache")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if cfg.use_prepared_qt:
        print(f"Loading prepared QT cache from {cache_dir}...", flush=True)
        return load_o1_real_cache(cfg, cache_dir)

    print(f"Loading prepared {cfg.summary_label} arrays from {cfg.o1_data_dir}...", flush=True)
    (
        noise_train_raw,
        noise_val_raw,
        noise_test_raw,
        event_windows_raw,
        noise_train_meta,
        noise_val_meta,
        noise_test_meta,
        event_meta,
    ) = load_prepared_o1_arrays(cfg.o1_data_dir, detector_mode=cfg.detector_mode)
    asd_by_detector = {det: load_detector_asd_o1(cfg.o1_data_dir, det) for det in ("H1", "L1")}
    line_configuration = infer_line_configuration(cfg.o1_data_dir)
    noise_train_det = [row["detector"] for row in noise_train_meta]
    noise_val_det = [row["detector"] for row in noise_val_meta]
    noise_test_det = [row["detector"] for row in noise_test_meta]
    event_det = [row["detector"] for row in event_meta]

    print(
        f"Whitening {cfg.summary_label} train/validation/test noise from full prepared 4 s segments...",
        flush=True,
    )
    noise_w_tr = whiten_batch_gwpy_o1(
        noise_train_raw, noise_train_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )
    noise_w_val = whiten_batch_gwpy_o1(
        noise_val_raw, noise_val_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )
    noise_w_te = whiten_batch_gwpy_o1(
        noise_test_raw, noise_test_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )
    print(f"Whitening {cfg.summary_label} event evaluation windows from full prepared 4 s segments...", flush=True)
    event_w = whiten_batch_gwpy_o1(
        event_windows_raw, event_det, asd_by_detector, notch_lines=cfg.o1_notch_lines, line_configuration=line_configuration
    )

    noise_sigma = float(noise_w_tr[: min(200, len(noise_w_tr))].std())
    print(f"{cfg.summary_label} noise sigma (whitened): {noise_sigma:.6f}", flush=True)

    noise_qt_tr = compute_qt_cache_dataset(
        os.path.join(cache_dir, "noise_qt_tr.npy"),
        noise_w_tr,
        progress_label=f"{cfg.summary_label} noise_qt_tr",
        progress_interval=cfg.qt_progress_interval,
    )
    noise_qt_val = compute_qt_cache_dataset(
        os.path.join(cache_dir, "noise_qt_val.npy"),
        noise_w_val,
        progress_label=f"{cfg.summary_label} noise_qt_val",
        progress_interval=cfg.qt_progress_interval,
    )
    noise_qt_te = compute_qt_cache_dataset(
        os.path.join(cache_dir, "noise_qt_te.npy"),
        noise_w_te,
        progress_label=f"{cfg.summary_label} noise_qt_te",
        progress_interval=cfg.qt_progress_interval,
    )
    event_qt = compute_qt_cache_dataset(
        os.path.join(cache_dir, "event_qt.npy"),
        event_w,
        progress_label=f"{cfg.summary_label} event_qt",
        progress_interval=cfg.qt_progress_interval,
    )

    print(f"Loading projected {cfg.summary_label} signal bank from {cfg.o1_signal_bank_dir}...", flush=True)
    signal_bank = load_o1_signal_bank(cfg.o1_signal_bank_dir)
    unique_source_ids = np.unique(signal_bank["source_ids"])
    rng = np.random.default_rng(SEED)
    perm_source_ids = rng.permutation(unique_source_ids)
    n_sig_eval = max(1, min(cfg.o1_inj_eval_count, int(len(perm_source_ids) * 0.1)))
    n_sig_val = max(1, int(len(perm_source_ids) * 0.2))
    n_sig_train = max(1, len(perm_source_ids) - n_sig_val - n_sig_eval)
    train_source_ids = set(int(x) for x in perm_source_ids[:n_sig_train])
    val_source_ids = set(int(x) for x in perm_source_ids[n_sig_train:n_sig_train + n_sig_val])
    eval_source_ids = set(int(x) for x in perm_source_ids[n_sig_train + n_sig_val:])
    source_ids = signal_bank["source_ids"]
    train_mask = np.array([sid in train_source_ids for sid in source_ids], dtype=bool)
    val_mask = np.array([sid in val_source_ids for sid in source_ids], dtype=bool)
    eval_mask = np.array([sid in eval_source_ids for sid in source_ids], dtype=bool)
    signal_train_raw = subset_signal_pool(signal_bank, train_mask)
    signal_val_raw = subset_signal_pool(signal_bank, val_mask)
    signal_eval_raw = subset_signal_pool(signal_bank, eval_mask)

    sig_qt_tr = build_o1_injections_cache_dataset(
        os.path.join(cache_dir, "sig_qt_tr.npy"),
        noise_train_raw,
        noise_train_det,
        signal_train_raw,
        asd_by_detector,
        seed=SEED,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
        progress_label=f"{cfg.summary_label} sig_qt_tr",
        progress_interval=cfg.qt_progress_interval,
    )
    sig_qt_val = build_o1_injections_cache_dataset(
        os.path.join(cache_dir, "sig_qt_val.npy"),
        noise_val_raw,
        noise_val_det,
        signal_val_raw,
        asd_by_detector,
        seed=SEED + 1,
        notch_lines=cfg.o1_notch_lines,
        line_configuration=line_configuration,
        progress_label=f"{cfg.summary_label} sig_qt_val",
        progress_interval=cfg.qt_progress_interval,
    )
    sig_qt_val_benchmark = None
    sig_val_benchmark_meta = None
    sig_qt_eval = None
    sig_eval_meta = None
    available_noise_detectors = {row["detector"] for row in noise_val_meta + noise_test_meta}
    if {"H1", "L1"}.issubset(available_noise_detectors):
        val_benchmark_indices = stratified_signal_indices(signal_val_raw, cfg.o1_inj_eval_count, seed=SEED + 11)
        sig_qt_val_benchmark_tensor, sig_val_benchmark_meta = build_o1_matched_detector_benchmark_injections(
            noise_val_raw,
            noise_val_meta,
            signal_val_raw,
            val_benchmark_indices,
            asd_by_detector,
            notch_lines=cfg.o1_notch_lines,
            line_configuration=line_configuration,
        )
        sig_qt_val_benchmark = save_tensor_cache_dataset(
            os.path.join(cache_dir, "sig_qt_val_benchmark.npy"),
            sig_qt_val_benchmark_tensor,
        )
        for idx, row in enumerate(sig_val_benchmark_meta):
            row["benchmark_id"] = idx

        eval_benchmark_indices = stratified_signal_indices(signal_eval_raw, cfg.o1_inj_eval_count, seed=SEED + 12)
        sig_qt_eval_tensor, sig_eval_meta = build_o1_matched_detector_benchmark_injections(
            noise_test_raw,
            noise_test_meta,
            signal_eval_raw,
            eval_benchmark_indices,
            asd_by_detector,
            notch_lines=cfg.o1_notch_lines,
            line_configuration=line_configuration,
        )
        sig_qt_eval = save_tensor_cache_dataset(
            os.path.join(cache_dir, "sig_qt_eval.npy"),
            sig_qt_eval_tensor,
        )
        for idx, row in enumerate(sig_eval_meta):
            row["benchmark_id"] = idx

    save_json(manifest_path, o1_real_cache_manifest(cfg))
    save_json(os.path.join(cache_dir, "noise_sigma.json"), {"noise_sigma": noise_sigma})
    save_json(os.path.join(cache_dir, "event_meta.json"), event_meta)
    save_json(os.path.join(cache_dir, "noise_test_meta.json"), noise_test_meta)
    save_json(os.path.join(cache_dir, "sig_val_benchmark_meta.json"), sig_val_benchmark_meta)
    save_json(os.path.join(cache_dir, "sig_eval_meta.json"), sig_eval_meta)

    return {
        "noise_qt_tr": noise_qt_tr,
        "noise_qt_val": noise_qt_val,
        "noise_qt_te": noise_qt_te,
        "sig_qt_tr": sig_qt_tr,
        "sig_qt_val": sig_qt_val,
        "sig_qt_val_benchmark": sig_qt_val_benchmark,
        "sig_val_benchmark_meta": sig_val_benchmark_meta,
        "sig_qt_eval": sig_qt_eval,
        "sig_eval_meta": sig_eval_meta,
        "event_qt": event_qt,
        "event_meta": event_meta,
        "noise_test_meta": noise_test_meta,
        "noise_sigma": noise_sigma,
    }


def save_outputs(output_dir, unsup_history, weaksup_history, unsup_metrics, weaksup_metrics):
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "unsup_history.npy"), unsup_history)
    np.save(os.path.join(results_dir, "weaksup_history.npy"), weaksup_history)
    np.save(os.path.join(results_dir, "unsup_metrics.npy"), unsup_metrics)
    np.save(os.path.join(results_dir, "weaksup_metrics.npy"), weaksup_metrics)

    table = (
        "\n"
        "===========================================================================\n"
        "Model                  Mode       3σ all   5σ all   3σ IMBH   5σ IMBH      AUC\n"
        "───────────────────────────────────────────────────────────────────────────\n"
        f"baseline_cae           unsup      {unsup_metrics['eff3']:>6.1f}%   {unsup_metrics['eff5']:>6.1f}%"
        f"   {unsup_metrics['eff3_imbh']:>6.1f}%   {unsup_metrics['eff5_imbh']:>6.1f}%   {unsup_metrics['auc']:.4f}\n"
        f"baseline_cae           weaksup    {weaksup_metrics['eff3']:>6.1f}%   {weaksup_metrics['eff5']:>6.1f}%"
        f"   {weaksup_metrics['eff3_imbh']:>6.1f}%   {weaksup_metrics['eff5_imbh']:>6.1f}%   {weaksup_metrics['auc']:.4f}\n"
        "===========================================================================\n"
    )
    with open(os.path.join(results_dir, "summary.txt"), "w") as fh:
        fh.write(table)
    print(table, flush=True)

    for name, metrics in (("unsup", unsup_metrics), ("weaksup", weaksup_metrics)):
        np.save(os.path.join(results_dir, f"noise_scores_{name}.npy"), metrics["noise_scores"])
        np.save(os.path.join(results_dir, f"mdc_signal_scores_{name}.npy"), metrics["signal_scores"])


def evaluate_o1_events(model, noise_eval_loader, noise_eval_meta, event_qt, event_meta, device, score_mode):
    noise_scores = compute_scores(model, noise_eval_loader, device, score_mode=score_mode)
    event_loader = make_loader(event_qt, batch_size=256, shuffle=False, num_workers=0)
    event_scores = compute_scores(model, event_loader, device, score_mode=score_mode)

    mu = float(noise_scores.mean())
    std = float(noise_scores.std())
    t3 = mu + 3.0 * std
    t5 = mu + 5.0 * std

    grouped = {}
    for row, score in zip(event_meta, event_scores):
        key = (row["event_name"], row["detector"])
        grouped.setdefault(key, []).append({**row, "score": float(score)})

    event_results = []
    for (event_name, detector), rows in sorted(grouped.items()):
        best = max(rows, key=lambda item: item["score"])
        center = min(rows, key=lambda item: abs(item["offset_seconds"]))
        event_results.append(
            {
                "event_name": event_name,
                "detector": detector,
                "event_gps": best["event_gps"],
                "best_window_center_gps": best["window_center_gps"],
                "best_offset_seconds": best["offset_seconds"],
                "best_score": best["score"],
                "center_score": center["score"],
                "above_3sigma": int(best["score"] > t3),
                "above_5sigma": int(best["score"] > t5),
            }
        )

    aligned_results = []
    grouped_by_event = {}
    for row, score in zip(event_meta, event_scores):
        event_name = row["event_name"]
        offset = float(row["offset_seconds"])
        detector = row["detector"]
        grouped_by_event.setdefault(event_name, {}).setdefault(offset, {})[detector] = {**row, "score": float(score)}

    for event_name, offset_map in sorted(grouped_by_event.items()):
        ranked_offsets = []
        for offset, per_detector in offset_map.items():
            total_score = float(sum(item["score"] for item in per_detector.values()))
            ranked_offsets.append((total_score, offset, per_detector))
        if not ranked_offsets:
            continue
        _, shared_offset, shared_rows = max(ranked_offsets, key=lambda item: (item[0], -abs(item[1])))
        for detector, row in sorted(shared_rows.items()):
            detector_rows = grouped[(event_name, detector)]
            detector_best = max(detector_rows, key=lambda item: item["score"])
            center = min(detector_rows, key=lambda item: abs(item["offset_seconds"]))
            aligned_results.append(
                {
                    "event_name": event_name,
                    "detector": detector,
                    "event_gps": row["event_gps"],
                    "shared_window_center_gps": row["window_center_gps"],
                    "shared_offset_seconds": row["offset_seconds"],
                    "shared_score": row["score"],
                    "shared_above_3sigma": int(row["score"] > t3),
                    "shared_above_5sigma": int(row["score"] > t5),
                    "network_score_sum": float(sum(item["score"] for item in shared_rows.values())),
                    "network_detector_count": len(shared_rows),
                    "detector_best_window_center_gps": detector_best["window_center_gps"],
                    "detector_best_offset_seconds": detector_best["offset_seconds"],
                    "detector_best_score": detector_best["score"],
                    "center_score": center["score"],
                }
            )

    return {
        "noise_mean": mu,
        "noise_std": std,
        "t3": t3,
        "t5": t5,
        "noise_scores": np.asarray(noise_scores, dtype=np.float32),
        "noise_rows": [
            {
                **row,
                "score": float(score),
                "above_3sigma": int(float(score) > t3),
                "above_5sigma": int(float(score) > t5),
            }
            for row, score in zip(noise_eval_meta, noise_scores)
        ],
        "rows": event_results,
        "aligned_rows": aligned_results,
    }


def evaluate_o1_injections(model, noise_eval_loader, noise_eval_meta, sig_qt, sig_meta, device, score_mode):
    noise_scores = compute_scores(model, noise_eval_loader, device, score_mode=score_mode)
    sig_loader = make_loader(sig_qt, batch_size=256, shuffle=False, num_workers=0)
    sig_scores = compute_scores(model, sig_loader, device, score_mode=score_mode)

    # BUG007 (noise-split methodology) verification:
    #   - The threshold noise here is the dedicated TEST split (noise_qt_te), which is
    #     a disjoint split from the training noise (noise_qt_tr) and from the noise that
    #     the benchmark signals are injected onto (noise_val). Train/val/test are built
    #     as disjoint GPS splits, so there is no leakage between threshold noise,
    #     training noise, and efficiency-measurement noise.
    #   - The hypothesized "1 s stride over 2 s windows = 50% overlap" inflation does NOT
    #     occur in this dataset: the prepared test windows have a minimum intra-file
    #     segment-start stride of 2.0 s, and the network input is the central 2 s context
    #     (1 s center crop) of each 4 s raw segment. Adjacent 2 s context windows therefore
    #     at most touch (>=2 s apart) and never overlap, so the effective sample size is
    #     not inflated and the mu/std threshold is not artificially tightened.
    #   Conclusion: no overlap found -> non-overlapping threshold == overlapping threshold.
    #   The threshold is left unchanged (no silent modification).
    mu = float(noise_scores.mean())
    std = float(noise_scores.std())
    t3 = mu + 3.0 * std
    t5 = mu + 5.0 * std

    rows = []
    for meta, score in zip(sig_meta, sig_scores):
        rows.append(
            {
                **meta,
                "score": float(score),
                "above_3sigma": int(float(score) > t3),
                "above_5sigma": int(float(score) > t5),
            }
        )

    scores = np.asarray(sig_scores, dtype=np.float64)
    return {
        "noise_mean": mu,
        "noise_std": std,
        "t3": t3,
        "t5": t5,
        "noise_scores": np.asarray(noise_scores, dtype=np.float32),
        "noise_rows": [
            {
                **row,
                "score": float(score),
                "above_3sigma": int(float(score) > t3),
                "above_5sigma": int(float(score) > t5),
            }
            for row, score in zip(noise_eval_meta, noise_scores)
        ],
        "signal_scores": np.asarray(sig_scores, dtype=np.float32),
        "rows": rows,
        "count": int(len(rows)),
        "mean_score": float(scores.mean()) if len(scores) else float("nan"),
        "median_score": float(np.median(scores)) if len(scores) else float("nan"),
        "max_score": float(scores.max()) if len(scores) else float("nan"),
        "n_above_3sigma": int(sum(row["above_3sigma"] for row in rows)),
        "n_above_5sigma": int(sum(row["above_5sigma"] for row in rows)),
    }


def summarize_noise_only_metrics(model, noise_eval_loader, noise_eval_meta, device, score_mode):
    noise_scores = compute_scores(model, noise_eval_loader, device, score_mode=score_mode)
    mu = float(noise_scores.mean())
    std = float(noise_scores.std())
    t3 = mu + 3.0 * std
    t5 = mu + 5.0 * std
    return {
        "noise_mean": mu,
        "noise_std": std,
        "t3": t3,
        "t5": t5,
        "noise_scores": np.asarray(noise_scores, dtype=np.float32),
        "noise_rows": [
            {
                **row,
                "score": float(score),
                "above_3sigma": int(float(score) > t3),
                "above_5sigma": int(float(score) > t5),
            }
            for row, score in zip(noise_eval_meta, noise_scores)
        ],
        "rows": [],
    }


def save_o1_outputs(
    output_dir,
    unsup_history,
    unsup_metrics,
    weaksup_history=None,
    weaksup_metrics=None,
    unsup_inj_metrics=None,
    weaksup_inj_metrics=None,
    weaksup_best_info=None,
    artifact_prefix="o1",
    summary_label="O1",
):
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "unsup_history.npy"), unsup_history)
    np.save(os.path.join(results_dir, f"unsup_{artifact_prefix}_metrics.npy"), unsup_metrics)
    if weaksup_history is not None:
        np.save(os.path.join(results_dir, "weaksup_history.npy"), weaksup_history)
    if weaksup_metrics is not None:
        np.save(os.path.join(results_dir, f"weaksup_{artifact_prefix}_metrics.npy"), weaksup_metrics)
    if weaksup_best_info is not None:
        with open(os.path.join(results_dir, "weaksup_best.json"), "w") as fh:
            json.dump(weaksup_best_info, fh, indent=2, sort_keys=True)

    def save_detector_split_artifacts(base_name, rows, include_score_array=False):
        if not rows:
            return
        df = pd.DataFrame(rows)
        if "detector" not in df.columns:
            return
        for detector, det_df in df.groupby("detector", sort=True):
            det_suffix = str(detector)
            det_csv_path = os.path.join(results_dir, f"{base_name}_{det_suffix}.csv")
            det_df.to_csv(det_csv_path, index=False)
            if include_score_array and "score" in det_df.columns:
                np.save(
                    os.path.join(results_dir, f"{base_name}_{det_suffix}.npy"),
                    det_df["score"].to_numpy(dtype=np.float32),
                )

    for name, metrics in (("unsup", unsup_metrics), ("weaksup", weaksup_metrics)):
        if metrics is None:
            continue
        np.save(os.path.join(results_dir, f"{artifact_prefix}_noise_scores_{name}.npy"), metrics["noise_scores"])
        pd.DataFrame({"score": metrics["noise_scores"]}).to_csv(
            os.path.join(results_dir, f"{artifact_prefix}_noise_scores_{name}.csv"),
            index=False,
        )
        save_detector_split_artifacts(
            f"{artifact_prefix}_noise_scores_{name}",
            metrics.get("noise_rows"),
            include_score_array=True,
        )
        csv_path = os.path.join(results_dir, f"{artifact_prefix}_event_scores_{name}.csv")
        pd.DataFrame(metrics["rows"]).to_csv(csv_path, index=False)
        aligned_csv_path = os.path.join(results_dir, f"{artifact_prefix}_event_scores_{name}_aligned.csv")
        pd.DataFrame(metrics.get("aligned_rows", [])).to_csv(aligned_csv_path, index=False)
        save_detector_split_artifacts(
            f"{artifact_prefix}_event_scores_{name}",
            metrics["rows"],
            include_score_array=False,
        )
    for name, metrics in (("unsup", unsup_inj_metrics), ("weaksup", weaksup_inj_metrics)):
        if metrics is None:
            continue
        np.save(os.path.join(results_dir, f"{artifact_prefix}_injected_benchmark_scores_{name}.npy"), metrics["signal_scores"])
        csv_path = os.path.join(results_dir, f"{artifact_prefix}_injected_benchmark_{name}.csv")
        pd.DataFrame(metrics["rows"]).to_csv(csv_path, index=False)
        save_detector_split_artifacts(
            f"{artifact_prefix}_injected_benchmark_{name}",
            metrics["rows"],
            include_score_array=True,
        )

    summary_lines = [
        f"================================ {summary_label} Event Summary ================================",
        "mode      event       det   best_score   center_score   best_dt[s]   >3sigma   >5sigma",
        "----------------------------------------------------------------------------------",
    ]
    for name, metrics in (("unsup", unsup_metrics), ("weaksup", weaksup_metrics)):
        if metrics is None:
            continue
        for row in metrics["rows"]:
            summary_lines.append(
                f"{name:<9}{row['event_name']:<12}{row['detector']:<6}"
                f"{row['best_score']:>11.6f}{row['center_score']:>14.6f}"
                f"{row['best_offset_seconds']:>13.2f}{row['above_3sigma']:>10d}{row['above_5sigma']:>10d}"
            )
        aligned_rows = metrics.get("aligned_rows") or []
        if aligned_rows:
            summary_lines.append(f"{name} aligned event reporting (shared offset across available detectors):")
            summary_lines.append(
                "mode      event       det   shared_score  center_score shared_dt[s] det_best_dt[s] >3sigma   >5sigma"
            )
            summary_lines.append(
                "-----------------------------------------------------------------------------------------------"
            )
            for row in aligned_rows:
                summary_lines.append(
                    f"{name:<9}{row['event_name']:<12}{row['detector']:<6}"
                    f"{row['shared_score']:>13.6f}{row['center_score']:>13.6f}"
                    f"{row['shared_offset_seconds']:>13.2f}{row['detector_best_offset_seconds']:>15.2f}"
                    f"{row['shared_above_3sigma']:>10d}{row['shared_above_5sigma']:>10d}"
                )
    summary_lines.append("==================================================================================")
    summary_lines.append(
        f"Noise thresholds from validation noise: 3sigma={unsup_metrics['t3']:.6f} 5sigma={unsup_metrics['t5']:.6f}"
    )
    for name, metrics in (("unsup", unsup_inj_metrics), ("weaksup", weaksup_inj_metrics)):
        if metrics is None:
            continue
        summary_lines.append(
            f"{name} injected benchmark: N={metrics['count']} mean={metrics['mean_score']:.6f} "
            f"median={metrics['median_score']:.6f} max={metrics['max_score']:.6f} "
            f">3sigma={metrics['n_above_3sigma']} >5sigma={metrics['n_above_5sigma']}"
        )
    if weaksup_best_info is not None:
        summary_lines.append(
            "weaksup best checkpoint: "
            f"epoch={weaksup_best_info['epoch']} "
            f"criterion={weaksup_best_info['criterion']} "
            f"val_inj>3sigma={weaksup_best_info['val_inj_n_above_3sigma']} "
            f"val_sep_k1={weaksup_best_info['val_sep_k1']:.6f} "
            f"val_noise={weaksup_best_info['val_noise']:.6f} "
            f"val_total={weaksup_best_info['val_total']:.6f} "
            f"val_d_signal={weaksup_best_info['val_d_signal']:.6f} "
            f"val_d_noise={weaksup_best_info['val_d_noise']:.6f} "
            f"val_latent_sep={weaksup_best_info['val_latent_sep']:.6f}"
        )
        summary_lines.append(
            "weaksup best mass-bin hits: "
            + " ".join(
                f"{label}={weaksup_best_info['val_mass_bin_n_above_3sigma'][label]}"
                f" ({weaksup_best_info['val_mass_bin_hit_rate'][label]:.3f})"
                for label in MASS_BIN_LABELS
            )
        )

    summary = "\n".join(summary_lines) + "\n"
    with open(os.path.join(results_dir, f"summary_{artifact_prefix}.txt"), "w") as fh:
        fh.write(summary)
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Best baseline + weak supervision GW pipeline.")
    parser.add_argument("--dataset-mode", choices=["synthetic", "o1_real"], default="synthetic")
    parser.add_argument("--detector-mode", choices=["both", "H1", "L1"], default="both")
    parser.add_argument("--training-mode", choices=["joint", "two_stage_logit", "margin_loss"], default="joint")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--recon-weight", type=float, default=0.3)
    parser.add_argument("--clf-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--unsup-epochs", type=int, default=10)
    parser.add_argument("--weaksup-epochs", type=int, default=10)
    parser.add_argument("--weaksup-es-patience", type=int, default=10)
    parser.add_argument("--checkpoint-gate-acc-min", type=float, default=CHECKPOINT_GATE_ACC_MIN)
    parser.add_argument("--noise-acc-early-stop-epochs", type=int, default=NOISE_ACC_EARLY_STOP_EPOCHS)
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--lambda-anom", type=float, default=2.0)
    parser.add_argument("--n-files", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--o1-data-dir", type=str, default=O1_PREPARED_DIR)
    parser.add_argument("--o1-signal-bank-dir", type=str, default=O1_SIGNAL_BANK_DIR)
    parser.add_argument("--o1-inj-eval-count", type=int, default=100)
    parser.add_argument("--disable-o1-notch-lines", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--use-prepared-qt", action="store_true")
    parser.add_argument("--qt-cache-dir", type=str, default="")
    parser.add_argument(
        "--precomputed-noise-qt-dir",
        type=str,
        default="",
        help="Directory containing per-detector precomputed quiet-noise QT tensors, e.g. data/o3a_prepared_1s_stride",
    )
    parser.add_argument(
        "--precomputed-noise-raw-dir",
        type=str,
        default="",
        help="Optional raw-frame directory used to build injected samples on top of a precomputed noise-QT dataset.",
    )
    parser.add_argument("--qt-progress-interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.getcwd(), "outputs_best_baseline"))
    parser.add_argument("--artifact-prefix", type=str, default="o1")
    parser.add_argument("--summary-label", type=str, default="O1")
    args = parser.parse_args()

    cfg = Config(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        recon_weight=args.recon_weight,
        clf_weight=args.clf_weight,
        epochs=args.epochs,
        weaksup_es_patience=args.weaksup_es_patience,
        n_files=args.n_files,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        dataset_mode=args.dataset_mode,
        o1_data_dir=args.o1_data_dir,
        o1_signal_bank_dir=args.o1_signal_bank_dir,
        o1_inj_eval_count=args.o1_inj_eval_count,
        o1_notch_lines=not args.disable_o1_notch_lines,
        detector_mode=args.detector_mode,
        device=args.device,
        training_mode=args.training_mode,
        unsup_epochs=args.unsup_epochs,
        weaksup_epochs=args.weaksup_epochs,
        checkpoint_gate_acc_min=args.checkpoint_gate_acc_min,
        noise_acc_early_stop_epochs=args.noise_acc_early_stop_epochs,
        margin=args.margin,
        lambda_anom=args.lambda_anom,
        artifact_prefix=args.artifact_prefix,
        summary_label=args.summary_label,
        qt_cache_dir=args.qt_cache_dir,
        prepare_only=args.prepare_only,
        use_prepared_qt=args.use_prepared_qt,
        qt_progress_interval=args.qt_progress_interval,
        precomputed_noise_qt_dir=args.precomputed_noise_qt_dir,
        precomputed_noise_raw_dir=args.precomputed_noise_raw_dir,
    )

    set_seed(SEED)
    device = device_for_run(cfg.device)
    print(f"Device: {device}", flush=True)
    print(f"Output dir: {cfg.output_dir}", flush=True)
    print(f"Training mode: {cfg.training_mode}", flush=True)
    if cfg.dataset_mode == "o1_real":
        print(f"Detector mode: {cfg.detector_mode}", flush=True)
        print(f"O1 notch lines: {cfg.o1_notch_lines}", flush=True)
        print(f"O1 signal bank dir: {cfg.o1_signal_bank_dir}", flush=True)
        if cfg.qt_cache_dir:
            print(f"QT cache dir: {cfg.qt_cache_dir}", flush=True)
        if cfg.precomputed_noise_qt_dir:
            print(f"Precomputed noise QT dir: {cfg.precomputed_noise_qt_dir}", flush=True)
        if cfg.precomputed_noise_raw_dir:
            print(f"Precomputed noise raw dir: {cfg.precomputed_noise_raw_dir}", flush=True)
    elif cfg.detector_mode != "both":
        raise ValueError("Detector-specific training is only supported for --dataset-mode o1_real")

    noise_only_qt_mode = bool(cfg.precomputed_noise_qt_dir)
    if cfg.dataset_mode == "o1_real":
        if noise_only_qt_mode:
            data = prepare_precomputed_noise_qt_data_with_weaksup(cfg)
        elif cfg.use_prepared_qt:
            data = prepare_o1_real_data_cached(cfg)
        else:
            reference_psd = load_reference_psd_o1(cfg.o1_data_dir)
            print(f"Loaded {cfg.summary_label} reference PSD from {cfg.o1_data_dir}", flush=True)
            data = prepare_o1_real_data_cached(cfg)
        if cfg.prepare_only and not noise_only_qt_mode:
            print(f"Prepared reusable QT cache for {cfg.summary_label}. Exiting due to --prepare-only.", flush=True)
            return
    else:
        reference_psd = load_reference_psd()
        print(f"Loaded reference PSD from {PSD_FILE}", flush=True)
        data = prepare_data(cfg, reference_psd)

    noise_train_loader = make_loader(data["noise_qt_tr"], cfg.batch_size, True, cfg.num_workers)
    noise_val_tensor = data["noise_qt_val"] if cfg.dataset_mode == "o1_real" else data["noise_qt_te"]
    noise_test_tensor = data["noise_qt_te"]
    noise_val_loader = make_loader(noise_val_tensor, cfg.batch_size, False, cfg.num_workers)
    sig_train_tensor = data["sig_qt_tr"]
    sig_val_tensor = data["sig_qt_val"] if cfg.dataset_mode == "o1_real" else data["sig_qt_te"]
    unsup_only_mode = sig_train_tensor is None or sig_val_tensor is None
    sig_train_loader = None
    sig_val_loader = None
    joint_noise_train_loader = None
    if not unsup_only_mode:
        sig_train_loader = make_loader(sig_train_tensor, cfg.batch_size, True, cfg.num_workers)
        sig_val_loader = make_loader(sig_val_tensor, cfg.batch_size, False, cfg.num_workers)
        joint_noise_train_loader = make_replacement_loader(
            data["noise_qt_tr"],
            cfg.batch_size,
            cfg.num_workers,
            num_samples=len(sig_train_tensor),
            seed=SEED,
        )
        print(
            "Joint noise sampling: "
            f"replacement=True num_samples={len(sig_train_tensor)} "
            f"from train_noise={len(data['noise_qt_tr'])}",
            flush=True,
        )
    else:
        print(
            "Signal QT tensors are unavailable for this dataset. "
            "Proceeding with unsupervised noise-only training.",
            flush=True,
        )

    output_models_dir = os.path.join(cfg.output_dir, "models")
    os.makedirs(output_models_dir, exist_ok=True)
    unsup_ckpt = os.path.join(output_models_dir, "baseline_cae_unsup.pt")
    joint_ckpt = os.path.join(output_models_dir, "baseline_cae_joint.pt")
    weaksup_ckpt = os.path.join(output_models_dir, "baseline_cae_weaksup.pt")
    weaksup_best_ckpt = os.path.join(output_models_dir, "baseline_cae_weaksup_best.pt")
    model = BaselineCAE(dropout=cfg.dropout).to(device)
    unsup_history = None
    joint_history = None
    unsup_eval_model = None
    val_benchmark_loader = None
    if data.get("sig_qt_val_benchmark") is not None:
        val_benchmark_loader = make_loader(
            data["sig_qt_val_benchmark"], cfg.batch_size, False, cfg.num_workers
        )
    if args.eval_only:
        if unsup_only_mode:
            if not os.path.exists(unsup_ckpt):
                raise FileNotFoundError(f"Missing checkpoint for --eval-only: {unsup_ckpt}")
            model.load_state_dict(torch.load(unsup_ckpt, map_location=device))
            print(f"Loaded {unsup_ckpt} for evaluation", flush=True)
            unsup_eval_model = model
            weaksup_best_info = None
        else:
            eval_joint_ckpt = weaksup_best_ckpt if os.path.exists(weaksup_best_ckpt) else joint_ckpt
            if not os.path.exists(eval_joint_ckpt):
                raise FileNotFoundError(f"Missing checkpoint for --eval-only: {eval_joint_ckpt}")
            model.load_state_dict(torch.load(eval_joint_ckpt, map_location=device))
            print(f"Loaded {eval_joint_ckpt} for evaluation", flush=True)
            if os.path.exists(unsup_ckpt):
                unsup_eval_model = BaselineCAE(dropout=cfg.dropout).to(device)
                unsup_eval_model.load_state_dict(torch.load(unsup_ckpt, map_location=device))
                print(f"Loaded {unsup_ckpt} for unsupervised evaluation", flush=True)
            weaksup_best_info = None
    else:
        if unsup_only_mode:
            print("Training unsupervised baseline_cae...", flush=True)
            unsup_history = train_unsup_model(model, noise_train_loader, noise_val_loader, cfg, device)
            torch.save(model.state_dict(), unsup_ckpt)
            print(f"Saved {unsup_ckpt}", flush=True)
            unsup_eval_model = BaselineCAE(dropout=cfg.dropout).to(device)
            unsup_eval_model.load_state_dict(torch.load(unsup_ckpt, map_location=device))
        elif cfg.training_mode in ("two_stage_logit", "margin_loss"):
            print("Training unsupervised baseline_cae...", flush=True)
            unsup_history = train_unsup_model(model, noise_train_loader, noise_val_loader, cfg, device)
            torch.save(model.state_dict(), unsup_ckpt)
            print(f"Saved {unsup_ckpt}", flush=True)
            unsup_eval_model = BaselineCAE(dropout=cfg.dropout).to(device)
            unsup_eval_model.load_state_dict(torch.load(unsup_ckpt, map_location=device))
            print(f"Fine-tuning O1 weak supervision from {unsup_ckpt}", flush=True)
            if cfg.training_mode == "two_stage_logit":
                joint_history, weaksup_best_info = train_joint_model(
                    model,
                    joint_noise_train_loader,
                    noise_val_loader,
                    sig_train_loader,
                    sig_val_loader,
                    cfg,
                    device,
                    val_benchmark_loader=val_benchmark_loader,
                    val_benchmark_meta=data.get("sig_val_benchmark_meta"),
                    phase_tag="weaksup",
                    num_epochs=cfg.weaksup_epochs,
                )
            else:
                joint_history, weaksup_best_info = train_margin_model(
                    model,
                    joint_noise_train_loader,
                    noise_val_loader,
                    sig_train_loader,
                    sig_val_loader,
                    cfg,
                    device,
                    val_benchmark_loader=val_benchmark_loader,
                    val_benchmark_meta=data.get("sig_val_benchmark_meta"),
                    phase_tag="weaksup",
                    num_epochs=cfg.weaksup_epochs,
                )
        else:
            print("Training joint baseline_cae...", flush=True)
            joint_history, weaksup_best_info = train_joint_model(
                model,
                joint_noise_train_loader,
                noise_val_loader,
                sig_train_loader,
                sig_val_loader,
                cfg,
                device,
                val_benchmark_loader=val_benchmark_loader,
                val_benchmark_meta=data.get("sig_val_benchmark_meta"),
                phase_tag="joint",
                num_epochs=cfg.epochs,
            )
            torch.save(model.state_dict(), joint_ckpt)
            print(f"Saved {joint_ckpt}", flush=True)
        torch.save(model.state_dict(), weaksup_best_ckpt)
        print(f"Saved {weaksup_best_ckpt}", flush=True)
        torch.save(model.state_dict(), weaksup_ckpt)
        print(f"Saved {weaksup_ckpt}", flush=True)

    noise_eval_loader = make_loader(noise_test_tensor, cfg.batch_size, False, cfg.num_workers)
    unsup_model = unsup_eval_model if unsup_eval_model is not None else model
    weaksup_score_mode = "classifier" if cfg.training_mode in ("joint", "two_stage_logit") else "recon"

    if cfg.dataset_mode == "o1_real":
        if unsup_only_mode:
            noise_scores = compute_scores(unsup_model, noise_eval_loader, device, score_mode="recon").astype(np.float32)
            save_noise_only_qt_outputs(
                cfg.output_dir,
                unsup_history,
                noise_scores,
                data.get("noise_test_meta"),
                artifact_prefix=cfg.artifact_prefix,
                summary_label=cfg.summary_label,
            )
            return
        if data.get("event_qt") is not None and data.get("event_meta") is not None:
            unsup_metrics = evaluate_o1_events(
                unsup_model,
                noise_eval_loader,
                data["noise_test_meta"],
                data["event_qt"],
                data["event_meta"],
                device,
                score_mode="recon",
            )
            weaksup_metrics = evaluate_o1_events(
                model,
                noise_eval_loader,
                data["noise_test_meta"],
                data["event_qt"],
                data["event_meta"],
                device,
                score_mode=weaksup_score_mode,
            )
        else:
            unsup_metrics = summarize_noise_only_metrics(
                unsup_model,
                noise_eval_loader,
                data["noise_test_meta"],
                device,
                score_mode="recon",
            )
            weaksup_metrics = summarize_noise_only_metrics(
                model,
                noise_eval_loader,
                data["noise_test_meta"],
                device,
                score_mode=weaksup_score_mode,
            )
        unsup_inj_metrics = None
        if data["sig_qt_eval"] is not None and data["sig_eval_meta"] is not None:
            unsup_inj_metrics = evaluate_o1_injections(
                unsup_model, noise_eval_loader, data["noise_test_meta"], data["sig_qt_eval"], data["sig_eval_meta"], device, score_mode="recon"
            )
        weaksup_inj_metrics = None
        if data["sig_qt_eval"] is not None and data["sig_eval_meta"] is not None:
            weaksup_inj_metrics = evaluate_o1_injections(
                model, noise_eval_loader, data["noise_test_meta"], data["sig_qt_eval"], data["sig_eval_meta"], device, score_mode=weaksup_score_mode
            )
        save_o1_outputs(
            cfg.output_dir,
            unsup_history,
            unsup_metrics,
            joint_history,
            weaksup_metrics,
            unsup_inj_metrics,
            weaksup_inj_metrics,
            weaksup_best_info,
            artifact_prefix=cfg.artifact_prefix,
            summary_label=cfg.summary_label,
        )
        return

    mdc_qt_tensor, snr_et, total_mass, is_imbh = load_mdc(reference_psd, data["noise_sigma"], cfg)
    _ = snr_et, total_mass
    mdc_eval_loader = make_loader(mdc_qt_tensor, cfg.batch_size, False, cfg.num_workers)
    unsup_metrics = evaluate_model(model, noise_eval_loader, mdc_eval_loader, is_imbh, device, score_mode="recon")
    weaksup_metrics = evaluate_model(model, noise_eval_loader, mdc_eval_loader, is_imbh, device, score_mode="classifier")
    save_outputs(cfg.output_dir, None, joint_history, unsup_metrics, weaksup_metrics)


if __name__ == "__main__":
    main()

"""P7 shared library: the 5-seed MEAN ENSEMBLE deploy arm.

Round-2 review item 5 / [[gw-o4b-anomalies-resolved]]: the deployed seed0 arm is
the OUTLIER seed on the O4b L1 glitch family (family median g -0.67 vs
-1.9/-3.0/-3.0/-3.4 for seeds 1-4). The fix is to adopt the 5-seed MEAN of the
deploy seeds as THE deploy arm. Every place the v4 pipeline used
arm_deploy_seed0, the ensemble run uses mean_{s=0..4} logit_s.

This module only SCORES (the 5 seed arms in p1v42/arm_deploy_seed{0..4}.pt are
already trained and frozen). Tiles are built ONCE per segment and scored by all
5 seeds; the per-tile arithmetic mean of the logits is the ensemble g. QT /
whitening helpers mirror the frozen arm script byte-for-byte (CTX=2.0,
Pool(16), zoom to 256x128, min_max_norm).
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from multiprocessing import Pool
from scipy.ndimage import zoom

import os
ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/improved")
import improved_pipeline as ip

LR = f"{ROOT}/lr_cascade"
P1V4 = f"{LR}/p1v42"
FS = ip.FS
N_SEEDS = 5


class GlitchArm(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [1, 16, 32, 64, 128]
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]),
                          nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x).squeeze(-1)


def load_seeds(device):
    arms = []
    for k in range(N_SEEDS):
        a = GlitchArm().to(device)
        a.load_state_dict(torch.load(f"{P1V4}/arm_deploy_seed{k}.pt", map_location=device))
        a.eval()
        arms.append(a)
    return arms


def build_qt(white):
    """Identical to the frozen arm script's build_qt (CTX=2.0)."""
    qi = ip.center_crop_waveforms(np.asarray(white, np.float32), sample_rate=FS,
                                  context_seconds=2.0)
    args = [(w, FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0) for w in qi]
    with Pool(16) as p:
        mags = p.map(ip._compute_qt_image_worker, args)
    return ip.min_max_norm(np.stack([zoom(m, (256 / m.shape[0], 128 / m.shape[1]), order=1)
                                     for m in mags]).astype(np.float32)).astype(np.float32)


def score_all_seeds(arms, qts, device):
    """Return (n_seeds, n_tiles) logits and the per-tile mean ensemble (n_tiles,)."""
    gs = np.zeros((len(arms), len(qts)), np.float64)
    with torch.no_grad():
        for k, a in enumerate(arms):
            out = []
            for (xb,) in DataLoader(TensorDataset(torch.from_numpy(qts[:, None])), batch_size=128):
                out.extend(a(xb.to(device)).cpu().numpy())
            gs[k] = np.asarray(out, np.float64)
    return gs, gs.mean(0)

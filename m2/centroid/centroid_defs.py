"""MADGRAV2 item 1: candidate spectral-centroid definitions computed on the frozen model-input tile.

Tile geometry (verified 2026-09-17 against improved_pipeline.compute_qt_image + madgrav_qt_build/tiles.py):
  gwpy q_transform(frange=(10,1291), qrange=(4,64), tres=0.002, fres=0.5, norm='median') -> log1p(|Q|), transposed to
  (freq, time), centre-cropped to 1.0 s (500 cols), zoomed to (256, 128), per-tile min-max normalised to [0,1].
  Row i  -> f = 10 + (i+0.5) * 1281/256 Hz  (5.004 Hz/row, linear).  Col j -> t = (j+0.5) * 1.0/128 s, centre at col 64.
Frozen (MADGRAV1) centroid: energy-weighted mean frequency of the whitened-strain PSD, central 0.5 s, band 20-400 Hz,
morphology gate cH < 190.79 Hz and cL < 190.79 Hz. Here the same recipe is approximated on the tile ('old_proxy') and
compared with excess-only definitions that use the full 10-1291 Hz axis.
"""
import numpy as np

NF, NT = 256, 128
F0, F1 = 10.0, 1291.0
F_AXIS = F0 + (np.arange(NF) + 0.5) * (F1 - F0) / NF          # Hz per row centre
T_AXIS = (np.arange(NT) + 0.5) / NT                            # s, tile spans 1.0 s

def _cols(window_s):
    half = int(round(window_s * NT / 2.0)); c = NT // 2
    return slice(c - half, c + half)

def _band(fmin, fmax):
    return (F_AXIS >= fmin) & (F_AXIS <= fmax)

def _wmean_f(w, band):
    """w: (N, NF, NT) non-negative weights -> energy-weighted mean frequency over rows in band."""
    wf = w[:, band, :].sum(2)                                   # (N, nband)
    num = (wf * F_AXIS[band][None, :]).sum(1); den = wf.sum(1)
    out = num / np.where(den > 0, den, np.nan)
    return out

def old_proxy(tiles, window_s=0.5, fmin=20.0, fmax=400.0):
    """Frozen recipe transplanted onto the tile: all pixels, central window, 20-400 Hz, weights = tile value."""
    t = tiles[:, :, _cols(window_s)]
    w = np.zeros_like(tiles); w[:, :, _cols(window_s)] = t
    return _wmean_f(w, _band(fmin, fmax))

def excess_pct(tiles, pct=99.0, window_s=1.0, fmin=F0, fmax=F1):
    """Excess-only: pixels above the per-tile `pct` percentile (background level), weights = value - threshold."""
    thr = np.percentile(tiles.reshape(len(tiles), -1), pct, axis=1)[:, None, None]
    w = np.clip(tiles - thr, 0, None)
    if window_s < 1.0:
        m = np.zeros(NT, bool); m[_cols(window_s)] = True; w = w * m[None, None, :]
    return _wmean_f(w, _band(fmin, fmax))

def excess_bgsub(tiles, window_s=0.13, fmin=F0, fmax=F1, nsig=2.0):
    """Row-wise background subtraction: per-row median and MAD over the full 1 s, keep residual > nsig*MAD in the
    central coherence-length window (0.13 s default), weights = residual."""
    med = np.median(tiles, axis=2, keepdims=True)
    mad = np.median(np.abs(tiles - med), axis=2, keepdims=True) * 1.4826 + 1e-6
    r = tiles - med - nsig * mad
    w = np.clip(r, 0, None)
    m = np.zeros(NT, bool); m[_cols(window_s)] = True; w = w * m[None, None, :]
    return _wmean_f(w, _band(fmin, fmax))

def peak_f(tiles, window_s=0.5):
    """Frequency of the loudest pixel in the central window."""
    t = tiles[:, :, _cols(window_s)]
    idx = t.reshape(len(t), -1).argmax(1) // t.shape[2]
    return F_AXIS[idx]

DEFS = {
    "old_proxy":        lambda x: old_proxy(x),
    "old_proxy_full":   lambda x: old_proxy(x, fmin=F0, fmax=F1),
    "exc99":            lambda x: excess_pct(x, 99.0),
    "exc99_5":          lambda x: excess_pct(x, 99.5),
    "exc99_c05":        lambda x: excess_pct(x, 99.0, window_s=0.5),
    "exc99_b400":       lambda x: excess_pct(x, 99.0, fmin=20.0, fmax=400.0),
    "bgsub013":         lambda x: excess_bgsub(x, 0.13),
    "bgsub05":          lambda x: excess_bgsub(x, 0.5),
    "peak05":           lambda x: peak_f(x, 0.5),
}

def compute_all(tiles, chunk=512):
    tiles = np.asarray(tiles)
    out = {k: [] for k in DEFS}
    for i in range(0, len(tiles), chunk):
        x = tiles[i:i + chunk].astype(np.float32)
        for k, fn in DEFS.items():
            out[k].append(fn(x))
    return {k: np.concatenate(v) for k, v in out.items()}

# ---- connected excess cluster: the excess region that contains the loudest pixel -------------------------------
from scipy import ndimage as _ndi

def cluster_centroid(tiles, pct=99.0, window_s=1.0, fmin=F0, fmax=F1):
    """Threshold at the per-tile `pct` percentile, label 8-connected components, keep the component holding the
    loudest pixel inside the central window; return its weighted mean frequency (weights = value - threshold).
    Robust to scattered noise pixels elsewhere in the tile."""
    n = len(tiles); thr = np.percentile(tiles.reshape(n, -1), pct, axis=1)
    out = np.full(n, np.nan); band = _band(fmin, fmax); cs = _cols(window_s)
    st = np.ones((3, 3), bool)
    for i in range(n):
        t = tiles[i]; m = t > thr[i]
        if not m.any(): continue
        lab, k = _ndi.label(m, structure=st)
        sub = t[:, cs]; r, c = np.unravel_index(sub.argmax(), sub.shape); c += cs.start
        l = lab[r, c]
        if l == 0: continue
        comp = (lab == l) & band[:, None]
        w = np.where(comp, t - thr[i], 0.0)
        wf = w.sum(1); den = wf.sum()
        if den > 0: out[i] = (wf * F_AXIS).sum() / den
    return out

DEFS.update({
    "exc99_f500":   lambda x: excess_pct(x, 99.0, fmax=500.0),
    "exc99_5_f500": lambda x: excess_pct(x, 99.5, fmax=500.0),
    "clu98":        lambda x: cluster_centroid(x, 98.0),
    "clu99":        lambda x: cluster_centroid(x, 99.0),
    "clu99_5":      lambda x: cluster_centroid(x, 99.5),
    "clu99_f500":   lambda x: cluster_centroid(x, 99.0, fmax=500.0),
})

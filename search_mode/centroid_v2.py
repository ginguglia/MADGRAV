"""MADGRAV2 spectral centroid v2 (adopted 2026-09-17): the energy-weighted mean frequency of the connected excess region
of the frozen model-input tile that contains the tile's loudest pixel (searched over the full 1.0 s).
  tile: (256 freq rows x 128 time cols), min-max normalised QT (log1p |Q|), rows linear 10-1291 Hz (5.004 Hz/row), 1.0 s.
  threshold: per-tile 98th percentile; connectivity: 8-neighbour; weights: value - threshold; band: full tile.
Gate (MADGRAV2 morphology gate): a window passes iff centroid2 < CEN2_CUT in BOTH detectors; CEN2_CUT = 220 Hz =
p99 of the centroid over triggering injections (sigma_net >= 4) of the 2026-09-17 flat-in-log Mtot 15-800 population
(m2/positives/analysis_posm2.md). The v1 strain-PSD centroid (20-400 Hz, 0.5 s) is kept in the metas as 'centroid' for
reproduction of MADGRAV1 but no longer enters the statistic when SM_CEN_V1 is unset.
Pilot and validation: MADGRAV2/m2/centroid (pilot_summary.md, cut_scan.md, fam532_*.md)."""
import os
import numpy as np
from scipy import ndimage as _ndi

CEN2_CUT = float(os.environ.get("SM_CEN2_CUT", "220.0"))
CEN2_PCT = 98.0
NF, NT = 256, 128
F0, F1 = 10.0, 1291.0
F_AXIS = F0 + (np.arange(NF) + 0.5) * (F1 - F0) / NF
_ST = np.ones((3, 3), bool)
_CS = slice(0, NT)   # loudest-pixel search window: full tile (pilot definition clu98, window_s=1.0)

def cluster_centroid(tiles, pct=CEN2_PCT):
    """tiles: (N, 256, 128) float. Returns (N,) centroid in Hz; NaN when the tile has no excess (all-equal tile)."""
    t_all = np.asarray(tiles, np.float32); n = len(t_all)
    thr = np.percentile(t_all.reshape(n, -1), pct, axis=1); out = np.full(n, np.nan)
    for i in range(n):
        t = t_all[i]; m = t > thr[i]
        if not m.any(): continue
        lab, _ = _ndi.label(m, structure=_ST)
        sub = t[:, _CS]; r, c = np.unravel_index(int(sub.argmax()), sub.shape); l = lab[r, c + _CS.start]
        if l == 0: continue
        w = np.where(lab == l, t - thr[i], 0.0); wf = w.sum(1); den = wf.sum()
        if den > 0: out[i] = float((wf * F_AXIS).sum() / den)
    return out

def cen2_pass(cH2, cL2, cut=None):
    """Morphology gate v2: both legs below the cut; an undefined centroid (NaN) passes (nothing to veto on)."""
    c = CEN2_CUT if cut is None else cut
    cH2 = np.asarray(cH2, float); cL2 = np.asarray(cL2, float)
    return ~((np.isfinite(cH2) & (cH2 >= c)) | (np.isfinite(cL2) & (cL2 >= c)))

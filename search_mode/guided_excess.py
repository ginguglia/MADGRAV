"""MADGRAV2.1 guided second-leg excess ("gx") idea 2026-09-19, pilot coh_study/pilot_guided_excess.py (+2 held-out segments).

Given the two re-whitened 1 s coherence-series rows of a pair (H1, L1; coh_rewhiten.flatten output = what
driver_search_multi.series_for returns under SM_COH_REWHITEN=1), find the LOUDER leg's time-frequency excess box and measure
the OTHER leg's energy excess above its own local noise inside that box. A floored single-detector sigma hides an SNR~4 leg;
inside a box of <=6 cells fixed by the loud leg the look-elsewhere is tiny, so a few-sigma local excess is significant.
Pilot (1 % per-pair false alarm, eff by weak-leg SNR 3-5 / 5-7): gx 17-32 % / 43-56 % vs re-whitened coherence 1-3 % / 6-8 %;
loud time-slid background 0/2500 above the noise p99; low-frequency narrow-band glitch pairs 0/14. Box tightness is the whole
game (loose boxes destroy it); the statistic is combined with coherence THROUGH the LR (a naive z-sum is worse than gx alone).

Definition (all parameters env-overridable, defaults = validated "cap6" variant):
  x = row / hann(4096) (undo the flatten taper), keep [SM_GX_CROP) s (default 0.2-0.8: taper >= 0.35);
  STFT: hann SM_GX_NPERSEG=256 (62 ms), hop SM_GX_HOP=32 (7.8 ms), rows SM_GX_BAND=40,480 Hz (16 Hz bins -> 28 rows);
  Q = |STFT|^2 / per-row median over time (noise: exponential, median 1, mean MU=1/ln2);
  box(row) = cells with Q > SM_GX_Q0=12 within +-SM_GX_TWIN=0.12 s of the row's peak cell, the SM_GX_NMAX=6 loudest of them
             (fallback when none: the 3x3 block around the peak);
  z(other in box) = (sum_box Q_other - K*MU) / (MU*sqrt(K));   gx = z of the WEAKER leg (smaller Q max) in the LOUDER leg's box.
  LR feature = min(gx, SM_GX_CAP=15): the cap bounds the chance tail of noise pairs (noise p99.9 ~15) and makes the prune
  prefilter's upper bound exact (fit_lr pins beta_gx >= 0, so loglr at gx=CAP >= loglr at the real gx).
Activation: SM_GX=1 (requires SM_COH_REWHITEN=1: gx is defined on the flattened rows). Default off = 7-feature LR, bit-identical.
Every consumer (LR refit noise/injections, zero-lag triggers, background prune rescore, merge setup, inject.py) computes gx
through gx_vec / prepare+gx_pairs here, so foreground, background and injections are defined identically. No side effects at import."""
import os, numpy as np
from scipy.signal import get_window
ON = os.environ.get("SM_GX", "0") == "1"
if ON and os.environ.get("SM_COH_REWHITEN", "0") != "1":
    raise SystemExit("[guided_excess] SM_GX=1 requires SM_COH_REWHITEN=1 (gx is defined on the re-whitened series rows)")
FS = 4096; SLEN = 4096
NPER = int(os.environ.get("SM_GX_NPERSEG", "256")); HOP = int(os.environ.get("SM_GX_HOP", "32"))
F0, F1 = (float(x) for x in os.environ.get("SM_GX_BAND", "40,480").split(","))
C0, C1 = (float(x) for x in os.environ.get("SM_GX_CROP", "0.2,0.8").split(","))
Q0 = float(os.environ.get("SM_GX_Q0", "12")); NMAX = int(os.environ.get("SM_GX_NMAX", "6")); TWIN = float(os.environ.get("SM_GX_TWIN", "0.12"))
CAP = float(os.environ.get("SM_GX_CAP", "15"))
MU = 1.0 / np.log(2.0)
_HANN = np.hanning(SLEN); _W = get_window("hann", NPER); _I0, _I1 = int(C0 * FS), int(C1 * FS)
_FQ = np.fft.rfftfreq(NPER, 1.0 / FS); _FM = (_FQ >= F0) & (_FQ <= F1); NF = int(_FM.sum()); NT = (_I1 - _I0 - NPER) // HOP + 1
TW = int(TWIN * FS / HOP); NB = max(NMAX, 9)
_FRAME_IDX = np.arange(NPER)[None, :] + HOP * np.arange(NT)[:, None]      # [NT, NPER] sample indices into the cropped row
QCHUNK = int(os.environ.get("SM_GX_QCHUNK", "4096"))                        # rows per STFT chunk (bounds transient memory)
PARAMS = dict(nperseg=NPER, hop=HOP, band=(F0, F1), crop=(C0, C1), q0=Q0, nmax=NMAX, twin=TWIN, cap=CAP, nf=NF, nt=NT)

def qmap(rows):
    """rows [n, 4096] flattened series -> Q [n, NF, NT] float32 (per-row median-normalised STFT power, band rows only)."""
    rows = np.asarray(rows, np.float64); n = len(rows); out = np.empty((n, NF, NT), np.float32)
    for c0 in range(0, n, QCHUNK):
        x = rows[c0:c0 + QCHUNK, _I0:_I1] / _HANN[None, _I0:_I1]           # undo the taper on the kept part only (hann(0)=0)
        fr = x[:, _FRAME_IDX] * _W[None, None, :]                            # [m, NT, NPER]
        P = np.abs(np.fft.rfft(fr, axis=-1)) ** 2                            # [m, NT, 129]
        P = P[:, :, _FM].transpose(0, 2, 1)                                  # [m, NF, NT]
        med = np.median(P, axis=2, keepdims=True) + 1e-30
        out[c0:c0 + QCHUNK] = (P / med).astype(np.float32)
    return out

def boxes(Q):
    """Q [n, NF, NT] -> flat cell indices [n, NB] (int64, -1 = unused) of each row's excess box (see module doc)."""
    n = len(Q); flat = Q.reshape(n, -1); jm = flat.argmax(1); jf, jt = np.divmod(jm, NT)
    tm = np.abs(np.arange(NT)[None, :] - jt[:, None]) <= TW                  # [n, NT] time window about the peak
    m = (Q > Q0) & tm[:, None, :]
    qm = np.where(m, Q, -np.inf).reshape(n, -1)
    top = np.argpartition(-qm, NMAX - 1, axis=1)[:, :NMAX]                   # NMAX loudest excess cells (all of them when fewer)
    valid = np.isfinite(np.take_along_axis(qm, top, axis=1))
    idx = np.full((n, NB), -1, np.int64); idx[:, :NMAX] = np.where(valid, top, -1)
    none = ~valid.any(1)                                                     # no excess cell: 3x3 block around the peak
    if none.any():
        df, dt = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij"); df = df.ravel(); dt = dt.ravel()
        ff = jf[none][:, None] + df[None, :]; tt = jt[none][:, None] + dt[None, :]
        ok = (ff >= 0) & (ff < NF) & (tt >= 0) & (tt < NT)
        idx[none] = np.where(ok, ff * NT + tt, -1)
    return idx

def excess_in(Qflat, idx):
    """z-score of the summed Q of Qflat [n, NF*NT] over the box cells idx [n, NB] (-1 padded)."""
    ok = idx >= 0; K = ok.sum(1).astype(np.float64)
    S = np.where(ok, np.take_along_axis(Qflat, np.maximum(idx, 0), axis=1), 0.0).sum(1)
    return (S - K * MU) / (MU * np.sqrt(np.maximum(K, 1)))

def prepare(rows):
    """Per-row precompute for many pairs sharing rows: dict(Q=[n, NF*NT] float32, qmax=[n], idx=[n, NB])."""
    Q = qmap(rows); n = len(Q)
    return dict(Q=Q.reshape(n, -1), qmax=Q.reshape(n, -1).max(1), idx=boxes(Q))

def gx_pairs(PA, ia, PB, ib):
    """gx for pairs (PA row ia[k], PB row ib[k]) from prepare() outputs; identical to gx_vec on the gathered rows."""
    ia = np.asarray(ia, np.int64); ib = np.asarray(ib, np.int64)
    if len(ia) == 0: return np.empty(0)
    QA = PA["Q"][ia]; QB = PB["Q"][ib]
    zB = excess_in(QB, PA["idx"][ia]); zA = excess_in(QA, PB["idx"][ib])
    return np.where(PA["qmax"][ia] >= PB["qmax"][ib], zB, zA)

def gx_vec(A, B):
    """A, B [m, 4096] aligned flattened rows -> gx [m] (uncapped; feats() applies the cap)."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    if len(A) == 0: return np.empty(0)
    PA = prepare(A); PB = prepare(B); k = np.arange(len(A))
    return gx_pairs(PA, k, PB, k)

def cap(gx): return np.minimum(np.asarray(gx, np.float64), CAP)

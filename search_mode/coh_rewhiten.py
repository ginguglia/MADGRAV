"""MADGRAV2 re-whitened coherence series (2026-09-18).

The frozen whitening uses a reference ASD whose shape is far from the O4 noise (measured 2026-09-18: whitened noise
power per Hz at 140-500 Hz is ~17-19x that at 60-140 Hz). The BA coherence (1 s, 20-500 Hz, symmetric norm) is then
dominated by noise where high-mass signals have no power and carries ~no signal information (injections at SNR<=40 sit
on the noise median). Dividing every cached 1 s series by a per-(segment, detector) noise reference spectrum restores a
flat noise floor: injection efficiency above the noise 99th percentile at SNR 15 goes 1% -> 77%, loud glitch pairs stay
at coherence ~0 (amplitude mismatch), random background survivors move DOWN (median z 1.9 -> 0.9).

Activation: SM_COH_REWHITEN=1. Reference spectra live in SM_COHREF_DIR as <name>_<det>_cohref.npy (median |rfft| of
Hann-tapered cached rows, floored at 1e-3 x in-band median, +inf outside 20-500 Hz so those bins vanish). Build them once
with build_coh_ref.py; get_ref() builds on the fly from the series cache if a file is missing. Everything downstream
(driver_search_multi.series_for -> coh_vec / coh_from_fft, refit_lr_m2, driver_blindscan noise sample + triggers +
prune rescore, inject.py) sees the re-whitened rows, so foreground, background and injections are defined identically.
No side effects at import."""
import os, numpy as np
FS = 4096; SLEN = 4096; BAND = (20.0, 500.0)
REWHITEN = os.environ.get("SM_COH_REWHITEN", "0") == "1"
REF_DIR = os.environ.get("SM_COHREF_DIR") or (f"{os.environ['SM_SERIESCACHE']}/_cohref" if os.environ.get("SM_SERIESCACHE") else "search_mode/_cohref")
NREF = int(os.environ.get("SM_COHREF_ROWS", "512"))          # rows per (segment, det) used for the reference spectrum
_FQ = np.fft.rfftfreq(SLEN, 1.0 / FS); INBAND = (_FQ >= BAND[0]) & (_FQ <= BAND[1]); _HANN = np.hanning(SLEN)
_refs = {}

def ref_path(name, det): return os.path.join(REF_DIR, f"{name}_{det}_cohref.npy")

def build_ref(rows2d):
    """Reference amplitude spectrum from a stack of cached 1 s series rows (already band-limited + demeaned)."""
    rows2d = np.asarray(rows2d, np.float64); assert rows2d.shape[1] == SLEN, rows2d.shape
    S = np.median(np.abs(np.fft.rfft(rows2d * _HANN[None, :], axis=1)), axis=0)
    S = np.maximum(S, np.median(S[INBAND]) * 1e-3); S[~INBAND] = np.inf
    return S

def build_ref_from_cache(name, det, cache_npy, save=True):
    mm = np.load(cache_npy, mmap_mode="r"); n = mm.shape[0]
    ii = np.unique(np.linspace(0, n - 1, min(NREF, n)).astype(int))          # evenly spaced rows: robust to local glitches
    S = build_ref(np.array(mm[ii]))
    if save:
        os.makedirs(REF_DIR, exist_ok=True); tmp = ref_path(name, det) + ".tmp.npy"; np.save(tmp, S); os.replace(tmp, ref_path(name, det))
    return S

def get_ref(name, det, cache_npy=None):
    k = (name, det)
    if k in _refs: return _refs[k]
    p = ref_path(name, det)
    if os.path.exists(p): S = np.load(p)
    elif cache_npy and os.path.exists(cache_npy): S = build_ref_from_cache(name, det, cache_npy)
    else: raise FileNotFoundError(f"[coh_rewhiten] no reference spectrum {p} and no series cache to build it from")
    assert S.shape == (SLEN // 2 + 1,), S.shape
    _refs[k] = S; return S

def flatten(rows2d, S):
    """Divide each row's spectrum by the reference S; bins outside the band (S=inf) vanish. Returns demeaned float64 rows."""
    rows2d = np.asarray(rows2d, np.float64)
    X = np.fft.rfft(rows2d * _HANN[None, :], axis=1) / S[None, :]
    a = np.fft.irfft(X, n=SLEN, axis=1); a -= a.mean(1, keepdims=True)
    return a

def coh_vec_np(A, B, lag):
    """BA symmetric-norm coherence, numpy path identical to driver_search_multi.coh_vec (no GPU)."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    if len(A) == 0: return np.empty(0)
    L = A.shape[1]; cc = np.fft.irfft(np.fft.rfft(A, axis=1) * np.conj(np.fft.rfft(B, axis=1)), n=L, axis=1)
    win = np.concatenate([cc[:, :lag + 1], cc[:, -lag:]], axis=1)
    return 2.0 * np.abs(win).max(1) / ((A * A).sum(1) + (B * B).sum(1) + 1e-30)


# ---------------------------------------------------------------------------------------------------------------------
# MADGRAV 2.2 coherent statistic v2 (2026-09-27, prototype coh/coh_proto_v3-v5 on all 86 O4b events): SM_COH_V2=1 replaces the
# BA symmetric-norm cross-correlation by a two-detector constrained-likelihood coherent energy in the time-frequency domain.
# Per aligned 1 s pair (rows already re-whitened/flattened, 20-500 Hz): STFT (62.5 ms Hann, 50 % overlap), H1-L1 delay loop
# +-10 ms in 1 ms steps applied to L1 as an FFT phase shift, cluster = the N pixels of largest min(|a|,|b|) (a one-detector
# transient cannot win the selection), E_coh = 2 |sum a b*| over the cluster, rho_coh = sqrt(max over delay of E_coh) with the
# pixel power normalised to unit noise (constant per row set from the row's own median pixel power / ln 2). Returned in place
# of coh; default SM_COH_V2 unset -> nothing changes.
V2 = os.environ.get("SM_COH_V2", "0") == "1"
V2_NPIX = int(os.environ.get("SM_COH_V2_NPIX", "32")); V2_TAU_MS = float(os.environ.get("SM_COH_V2_TAU", "10")); V2_STEP_MS = 1.0
_NPERSEG, _NOVER = 256, 128
_W = np.hanning(_NPERSEG + 1)[:-1]                                     # periodic Hann, as scipy.signal.windows.hann(sym=False)
_HOP = _NPERSEG - _NOVER; _NCOL = (SLEN - _NPERSEG) // _HOP + 1
_FR = np.fft.rfftfreq(_NPERSEG, 1.0 / FS); _KEEP = (_FR >= BAND[0]) & (_FR <= BAND[1])
_TAUS = np.arange(-V2_TAU_MS, V2_TAU_MS + 1e-9, V2_STEP_MS) * 1e-3
_FQ_FULL = np.fft.rfftfreq(SLEN, 1.0 / FS)

def _stft_rows(X):
    """X [m, SLEN] -> [m, nf_band, ncol] complex; frames start at 0, hop 128, no padding (matches scipy stft boundary=None)."""
    idx = np.arange(_NPERSEG)[None, :] + _HOP * np.arange(_NCOL)[:, None]              # [ncol, nperseg]
    fr = X[:, idx] * _W[None, None, :]                                                    # [m, ncol, nperseg]
    Z = np.fft.rfft(fr, axis=2)[:, :, _KEEP]                                              # [m, ncol, nf]
    return np.transpose(Z, (0, 2, 1))                                                     # [m, nf, ncol]

def _unit(Z):
    """Normalise each row's pixels so the median pixel power / ln 2 (= chi^2_2 mean) is 1 -> energies in SNR^2 units."""
    p = np.median(np.abs(Z) ** 2, axis=(1, 2)) / np.log(2.0)
    return Z / np.sqrt(p + 1e-30)[:, None, None]

def coh_v2_np(A, B):
    """A, B: [m, SLEN] aligned re-whitened rows -> rho_coh [m] (float64)."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64); m = len(A)
    if m == 0: return np.empty(0)
    ZA = _unit(_stft_rows(A)); FB = np.fft.rfft(B, axis=1)
    best = np.zeros(m)
    for tau in _TAUS:
        Bs = np.fft.irfft(FB * np.exp(-2j * np.pi * _FQ_FULL * tau)[None, :], n=SLEN, axis=1)
        ZB = _unit(_stft_rows(Bs))
        P = np.minimum(np.abs(ZA), np.abs(ZB)).reshape(m, -1)                             # selection statistic per pixel
        k = min(V2_NPIX, P.shape[1]); sel = np.argpartition(P, -k, axis=1)[:, -k:]      # cluster indices per row
        a = np.take_along_axis(ZA.reshape(m, -1), sel, axis=1); b = np.take_along_axis(ZB.reshape(m, -1), sel, axis=1)
        E = 2.0 * np.abs(np.sum(a * np.conj(b), axis=1))
        best = np.maximum(best, E)
    return np.sqrt(best)

"""Rebuild a run's reference ASD with a LEAKAGE-FREE estimator -> reference_psd_fixed_{H1,L1}.npz.

The accepted builder (build_o3b_prep.py, replicating prepare_o1_data.estimate_reference_psd) cuts a
4 s window and then runs `welch(w, fs=4096, nperseg=1024)` -- a 0.25 s sub-FFT, 4 Hz bins. A 0.25 s
Hann window cannot resolve the seismic wall, so sub-20 Hz power leaks up into the 20-50 Hz band: the
resulting reference is wrong by up to 136x in ASD at 20-35 Hz (O4b H1), while being correct to ~1.8x
above 200 Hz. Two changes, both matching the local estimator the ASD veto already uses
(asd_consistency: fftlength=4, overlap=2, method="median"):

  nperseg 1024 -> WN   (full 4 s transform, 0.25 Hz bins -- resolves the wall instead of leaking it)
  mean        -> median (robust to the glitches the RMS pre-cut does not catch)

Writes to reference_psd_FIXED_*.npz, NEVER over reference_psd_*.npz: the frozen search config must
stay byte-identical. Only the VT layer (an explicit optimal-SNR integral against the ASD) consumes
the fixed file; the search whitens and then min-max-normalises, which is why it is insensitive to
the error (measured: net_loc/net = 0.92-1.02 across all runs and masses).

Run: SM_STRAIN=/scratch/.../strain_o4b_full SM_PREP_OUT=.../data/o4b_search_prep build_prep_fixed.py [nwin]
"""
import glob
import os
import sys

import numpy as np
from scipy.signal import welch

FS = 4096
WN = 4 * FS
NPERSEG = WN                  # was 1024 -- the whole defect
STRAIN = os.environ["SM_STRAIN"]
OUT = os.environ["SM_PREP_OUT"]
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
NFILE = int(os.environ.get("SM_PREP_NFILE", "200"))   # 0 = every file (the accepted builder's I/O)
# The drop-loudest-N% pre-cut was needed when the estimator averaged with MEAN. With MEDIAN
# averaging it is redundant -- the median is already robust to the loud tail -- and it biases the
# reference OPTIMISTIC, because it describes a clean-data floor rather than the data analysed.
# SM_PREP_DROP=0 gives the unbiased convention; SM_PREP_TAG names the output so both can coexist.
DROP_FRAC = float(os.environ.get("SM_PREP_DROP", "0.10"))
TAG = os.environ.get("SM_PREP_TAG", "fixed")
RNG = np.random.default_rng(20260706)      # same seed as the accepted builder: same windows


def det_psd(det):
    files = sorted(glob.glob(f"{STRAIN}/*_{det}.npz"))
    assert files, f"no {det} strain in {STRAIN}"
    # I/O: reading one .npz costs the WHOLE ~200 MB strain array, so the accepted builder paid
    # 280+ GB per detector to collect 2 windows per file. Sample a calendar-STRATIFIED subset of
    # files and take more windows from each instead: same window count, same spread (files are
    # sorted by GPS), ~7x less I/O. NFILE=0 keeps every file.
    if NFILE and len(files) > NFILE:
        files = [files[i] for i in np.linspace(0, len(files) - 1, NFILE).round().astype(int)]
    per_file = max(1, NWIN // len(files))
    rms, psds = [], []
    for f in files:
        strain = np.load(f)["strain"]
        n = (len(strain) - WN) // WN
        if n <= 0:
            continue
        starts = RNG.choice(n, size=min(per_file, n), replace=False) * WN
        for s in starts:
            w = strain[s:s + WN].astype(np.float64)
            if not np.all(np.isfinite(w)):
                continue
            fr, p = welch(w, fs=FS, nperseg=NPERSEG)
            psds.append(p); rms.append(float(np.sqrt(np.mean(w * w))))
    psds = np.asarray(psds); rms = np.asarray(rms)
    keep = (rms <= np.quantile(rms, 1.0 - DROP_FRAC)) if DROP_FRAC > 0 else np.ones(len(rms), bool)
    # MEDIAN BIAS CORRECTION. nperseg == the window length, so each window contributes ONE
    # periodogram, and periodogram values are exponentially distributed about the true PSD: their
    # MEDIAN estimates ln(2) x the mean, not the mean. Without this the reference ASD is optimistic
    # by 1/sqrt(ln 2) = 1.2011 in amplitude. Verified 2026-08-31 against gwpy's method="median"
    # (which applies the same correction internally): gwpy/ours 1.17 -> 0.97-0.98, flat in frequency.
    psd = np.median(psds[keep], axis=0) / np.log(2.0)
    print(f"  {det}: {len(files)} files, {len(psds)} windows, kept {keep.sum()}", flush=True)
    return fr.astype(np.float64), psd.astype(np.float64)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[prep-fixed] strain={STRAIN} -> {OUT} (Welch nperseg={NPERSEG} = {NPERSEG/FS:.0f}s, MEDIAN, drop={DROP_FRAC:.0%}, tag={TAG}, {NWIN} win over {NFILE or 'all'} files)", flush=True)
    for det in ("H1", "L1"):
        fr, psd = det_psd(det)
        np.savez(f"{OUT}/reference_psd_{TAG}_{det}.npz", freq=fr, psd=psd)
        old = f"{OUT}/reference_psd_{det}.npz"
        msg = ""
        if os.path.exists(old):
            z = np.load(old); fo, po = z["freq"], z["psd"]
            for lo, hi in ((20, 35), (50, 100), (200, 400)):
                m = (fr >= lo) & (fr < hi) & (psd > 0); mo = (fo >= lo) & (fo < hi) & (po > 0)
                msg += f"  {lo}-{hi}Hz old/new={np.sqrt(np.median(po[mo]))/np.sqrt(np.median(psd[m])):.2f}"
        print(f"  wrote reference_psd_{TAG}_{det}.npz{msg}", flush=True)
    print("[prep-fixed] DONE", flush=True)


if __name__ == "__main__":
    main()

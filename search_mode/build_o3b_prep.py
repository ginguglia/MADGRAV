"""Build an O3b-matched reference ASD (reference_psd_{H1,L1}.npz) -> data/o3b_search_prep/.

Replicates improved/prepare_o1_data.estimate_reference_psd EXACTLY: Welch (nperseg=1024, fs=4096),
mean over noise segments. Only the DATA source differs (O3b strain instead of O3a). Samples many 4s
non-overlapping windows spread across the O3b strain files, drops the loudest (event/glitch) windows
by per-window RMS before averaging so the reference tracks the typical O3b noise floor.

Run: SM_STRAIN=/scratch/.../strain_o3b_full python build_o3b_prep.py [n_windows_per_det]
"""
import os, sys, glob
import numpy as np
from scipy.signal import welch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


FS = 4096
WN = 4 * FS
NPERSEG = 1024
STRAIN = os.environ.get("SM_STRAIN", MADGRAV_SCRATCH + "/strain_o3b_full")
OUT = os.environ.get("SM_PREP_OUT", MADGRAV_ROOT + "/data/o3b_search_prep")
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
DROP_FRAC = 0.10   # drop the loudest 10% of windows (events/glitches) before averaging
RNG = np.random.default_rng(20260706)


def det_psd(det):
    files = sorted(glob.glob(f"{STRAIN}/*_{det}.npz"))
    assert files, f"no {det} strain in {STRAIN}"
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
            fr, p = welch(w, fs=FS, nperseg=NPERSEG)
            psds.append(p); rms.append(float(np.sqrt(np.mean(w * w))))
    psds = np.asarray(psds); rms = np.asarray(rms)
    keep = rms <= np.quantile(rms, 1.0 - DROP_FRAC)
    psd = psds[keep].mean(axis=0)
    print(f"  {det}: {len(files)} files, {len(psds)} windows, kept {keep.sum()} (dropped loudest {DROP_FRAC:.0%})", flush=True)
    return fr.astype(np.float64), psd.astype(np.float64)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[o3b-prep] strain={STRAIN} -> {OUT}  (Welch nperseg={NPERSEG}, mean, drop loudest {DROP_FRAC:.0%})", flush=True)
    for det in ("H1", "L1"):
        fr, psd = det_psd(det)
        np.savez(f"{OUT}/reference_psd_{det}.npz", freq=fr, psd=psd)
        print(f"  wrote {OUT}/reference_psd_{det}.npz  (median ASD {np.sqrt(np.median(psd[psd>0])):.2e})", flush=True)
    print("[o3b-prep] DONE", flush=True)


if __name__ == "__main__":
    main()

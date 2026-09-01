#!/usr/bin/env python
"""STEP 4 (O3 leg): run-median reference PSDs for O3a/O3b from the GWOSC
release strain itself (full-run downloads on scratch, uniform file coverage).

WHY NOT A RELEASE TARBALL: verified 2026-08-12 against the zenodo file list
(record 7890437) - the GWTC-3 sensitivity release ships injection hdf5 only;
monthly PSD products first appeared with the O4 releases (psds-o4ab.tar.gz).
The GWOSC strain IS the release product, so "release-grade" for O3 = median
Welch over windows sampled uniformly across the full analyzed run. This
replaces the biased as-run refs' construction on two axes:
  * MEDIAN across windows (the as-run builds used a MEAN with only a 10%
    loudest-RMS drop - heavy-tail-sensitive exactly at low f), and
  * uniform whole-run sampling (the original O3a ref was event-clustered).

Grid: Welch nperseg=1024 @ fs=4096 -> df=4 Hz, 513 bins on [0,2048] Hz -- the
identical grid of the existing reference_psd_*.npz (asserted).

Outputs (originals untouched):
  data/o3{a,b}_search_prep/reference_psd_release_{H1,L1}.npz
  search_mode/pastro_final/o3_release_psd_report.json  (current/release band
  ratios, window/month coverage, glitchy-window fraction diagnostic)

Env knobs (smoke tests): SM_NWIN (default 4000/det), SM_MAXFILES (default 400
evenly-strided files/det), SM_PSD_OUT (redirect output prep root), SM_RUNS.
Run: madgrav-venv python build_release_psds_o3.py
"""
import glob
import json
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
FS = 4096
WN = 4 * FS
NPERSEG = 1024
NWIN = int(os.environ.get("SM_NWIN", "4000"))
MAXFILES = int(os.environ.get("SM_MAXFILES", "400"))
OUTROOT = os.environ.get("SM_PSD_OUT", f"{MG}/data")
RUNS = os.environ.get("SM_RUNS", "o3a,o3b").split(",")
STRAIN = {"o3a": f"{SC}/strain_o3a_full", "o3b": f"{SC}/strain_o3b_full"}
RNG = np.random.default_rng(20260812)
GPS0_UNIX = 315964800  # GPS epoch in unix secs (leap secs ~18 s << month)
# HP AMENDMENT (decision 2026-08-13, see campaign/asd_rootcause_note.md):
# zero-phase 15 Hz high-pass before Welch. Without it every window shares a
# leakage floor from the sub-15 Hz seismic wall that a MEDIAN cannot remove
# (O3b L1 implied BNS range 85 -> 124 Mpc under this filter). The analysis
# band starts at 20 Hz (>= 1.3x cutoff; order-8 attenuation at 20 Hz < 0.4%).
HP_SOS = butter(8, 15.0, btype="highpass", fs=FS, output="sos")


def month_of(path):
    """UTC year_month from the GPS start encoded in <run>_<gps>_<det>.npz."""
    from datetime import datetime, timezone
    gps = int(os.path.basename(path).split("_")[1])
    t = datetime.fromtimestamp(GPS0_UNIX + gps, tz=timezone.utc)
    return f"{t.year}_{t.month:02d}"


def det_psd(run, det):
    files = sorted(glob.glob(f"{STRAIN[run]}/*_{det}.npz"))
    assert files, f"no {det} strain in {STRAIN[run]}"
    if len(files) > MAXFILES:  # even stride keeps whole-run coverage
        files = [files[i] for i in
                 np.linspace(0, len(files) - 1, MAXFILES).astype(int)]
    per_file = max(1, int(np.ceil(NWIN / len(files))))
    psds, months = [], []
    for fp in files:
        strain = np.load(fp)["strain"]
        n = (len(strain) - WN) // WN
        if n <= 0:
            continue
        starts = RNG.choice(n, size=min(per_file, n), replace=False) * WN
        mon = month_of(fp)
        for s in starts:
            w = sosfiltfilt(HP_SOS, strain[s:s + WN].astype(np.float64))
            fr, p = welch(w, fs=FS, nperseg=NPERSEG)
            psds.append(p)
            months.append(mon)
    psds = np.asarray(psds)
    med = np.median(psds, axis=0)
    # diagnostic only (median needs no glitch drop): windows >10x median band
    band = (fr >= 20) & (fr <= 60)
    glitchy = float(np.mean(psds[:, band].mean(1) > 10 * med[band].mean()))
    mon_counts = {m: int(c) for m, c in
                  zip(*np.unique(months, return_counts=True))}
    return fr.astype(np.float64), med.astype(np.float64), len(files), \
        len(psds), glitchy, mon_counts


def main():
    report = {}
    for run in RUNS:
        outdir = f"{OUTROOT}/{run}_search_prep"
        os.makedirs(outdir, exist_ok=True)
        for det in ("H1", "L1"):
            fr, med, nf, nw, glitchy, mons = det_psd(run, det)
            cur = np.load(f"{MG}/data/{run}_search_prep/reference_psd_{det}.npz")
            assert len(cur["freq"]) == len(fr) and \
                float(cur["freq"][1] - cur["freq"][0]) == float(fr[1] - fr[0]), \
                f"{run} {det}: grid mismatch vs existing reference"
            out = f"{outdir}/reference_psd_release_{det}.npz"
            np.savez(out, freq=fr, psd=med)
            low = (fr >= 20) & (fr <= 60)
            mid = (fr >= 120) & (fr <= 500)
            r_low = cur["psd"][low] / med[low]
            r_mid = cur["psd"][mid] / med[mid]
            print(f"[{run} {det}] files={nf} windows={nw} months={len(mons)} "
                  f"glitchy>10x={glitchy:.3f} | current/release 20-60 Hz "
                  f"median {np.median(r_low):.2f}x max {np.max(r_low):.2f}x | "
                  f"120-500 Hz median {np.median(r_mid):.2f}x -> {out}",
                  flush=True)
            report[f"{run}_{det}"] = dict(
                files=nf, windows=nw, months=mons, glitchy_frac=glitchy,
                ratio_2060_median=float(np.median(r_low)),
                ratio_2060_max=float(np.max(r_low)),
                ratio_120500_median=float(np.median(r_mid)),
                method="median Welch nperseg=1024, 15 Hz order-8 zero-phase "
                       "high-pass before Welch (HP amendment 2026-08-13), "
                       "uniform-strided full-run GWOSC strain (no O3 PSD "
                       "tarball exists in the GWTC-3 release; verified "
                       "zenodo 7890437 file list)")
    json.dump(report, open(f"{MG}/search_mode/pastro_final/"
                           "o3_release_psd_report.json", "w"), indent=1)
    print("[done] report -> o3_release_psd_report.json", flush=True)


if __name__ == "__main__":
    main()

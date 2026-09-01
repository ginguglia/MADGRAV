#!/usr/bin/env python
"""STEP 3a: run-median reference PSDs for O4a/O4b from the GWTC-5 release
monthly PSDs (psds-o4ab.tar.gz), on the pipeline df=4 Hz grid.

Method: each monthly 0.25-Hz PSD is band-averaged (mean over the 16 native
bins covering [f-2, f+2) Hz) onto the 513-bin 0-2048 Hz grid of the existing
reference_psd_*.npz, then the per-bin MEDIAN across the run's months is taken
(registered fix 2026-08-11; robust to any single bad month). Written as NEW
files reference_psd_release_{H1,L1}.npz next to the originals - the original
event-clustered reference_psd_*.npz are NOT touched (frozen-pipeline inputs).

Diagnostics printed: (i) current/release PSD ratio at 20-60 Hz per run+det
(expect ~9-30x per the 2026-08-11 referee-critique verification, i.e. the
event-clustered bias); (ii) median vs our-exposure-weighted-mean spread
(months are weighted equally by the median; this bounds what weighting could
change).
Run: madgrav-venv python build_release_psds_o4.py
"""
import glob
import gzip
import json
import os

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
PSDDIR = MADGRAV_EXTDATA + "/gwtc5_sensitivity/psds"
RUN_MONTHS = {
    "o4a": [f"2023_{m:02d}" for m in range(5, 13)] + ["2024_01"],
    "o4b": [f"2024_{m:02d}" for m in range(3, 13)] + ["2025_01"],
}
DETS = {"H1": "H", "L1": "L"}
GPS_EPOCH_2000 = 630720013  # not needed; month weighting uses UTC from seg GPS


def month_files(run, det):
    out = []
    for mon in RUN_MONTHS[run]:
        hits = sorted(glob.glob(f"{PSDDIR}/psd-O4-{mon}_v1-{DETS[det]}.*.gz"))
        hits = [h for h in hits if not os.path.basename(h).startswith("._")]
        if hits:
            out.append((mon, hits[0]))
    return out


def load_monthly(path):
    op = gzip.open(path, "rt")
    with op as fh:
        first = fh.readline()
        delim = "," if "," in first else None
        arr = np.loadtxt(fh, delimiter=delim)
    return arr[:, 0], arr[:, 1]


def band_average(f_native, p_native, f_grid, df):
    out = np.empty(len(f_grid))
    for i, fc in enumerate(f_grid):
        sel = (f_native >= fc - df / 2) & (f_native < fc + df / 2)
        out[i] = p_native[sel].mean() if sel.any() else np.nan
    return out


def our_month_exposure(run):
    """Wall-clock seconds of our analyzed coincident segments per UTC month."""
    from datetime import datetime, timedelta, timezone
    segj = json.load(open(f"{SC}/{run}_full_coincident.json"))
    gps0 = datetime(1980, 1, 6, tzinfo=timezone.utc)  # GPS epoch, leap secs ~18 s << month
    w = {}
    for s in segj["segments"]:
        t = gps0 + timedelta(seconds=float(s[0]))
        mon = f"{t.year}_{t.month:02d}"
        w[mon] = w.get(mon, 0.0) + float(s[2])
    return w


def main():
    ref = np.load(f"{MG}/data/o4a_search_prep/reference_psd_H1.npz")
    f_grid = ref["freq"].astype(float)
    df = float(f_grid[1] - f_grid[0])
    report = {}
    for run in ("o4a", "o4b"):
        expo = our_month_exposure(run)
        for det in ("H1", "L1"):
            mf = month_files(run, det)
            assert len(mf) >= 8, f"{run} {det}: only {len(mf)} monthly PSDs found"
            stack, wts = [], []
            for mon, path in mf:
                fn, pn = load_monthly(path)
                stack.append(band_average(fn, pn, f_grid, df))
                wts.append(expo.get(mon, 0.0))
            stack = np.array(stack)
            wts = np.array(wts)
            med = np.nanmedian(stack, axis=0)
            wmean = (np.nansum(stack * wts[:, None], axis=0) / wts.sum()
                     if wts.sum() > 0 else np.full_like(med, np.nan))
            out = f"{MG}/data/{run}_search_prep/reference_psd_release_{det}.npz"
            np.savez(out, freq=f_grid, psd=med)
            cur = np.load(f"{MG}/data/{run}_search_prep/reference_psd_{det}.npz")["psd"]
            band = (f_grid >= 20) & (f_grid <= 60)
            ratio = cur[band] / med[band]
            mid = (f_grid >= 120) & (f_grid <= 500)
            ratio_mid = cur[mid] / med[mid]
            wm_dev = np.nanmax(np.abs(wmean[band & (med > 0)] /
                                      med[band & (med > 0)] - 1))
            print(f"[{run} {det}] months={len(mf)} (exposure-covered "
                  f"{sum(1 for _, w in zip(mf, wts) if w > 0)}); "
                  f"current/release 20-60 Hz: median {np.median(ratio):.1f}x "
                  f"max {ratio.max():.1f}x | 120-500 Hz median "
                  f"{np.median(ratio_mid):.2f}x | expo-wmean vs median "
                  f"20-60 Hz max dev {wm_dev * 100:.0f}% -> {out}")
            report[f"{run}_{det}"] = dict(
                months=len(mf), ratio_2060_median=float(np.median(ratio)),
                ratio_2060_max=float(ratio.max()),
                ratio_120500_median=float(np.median(ratio_mid)),
                wmean_vs_median_2060_maxdev=float(wm_dev))
    json.dump(report, open(f"{MG}/search_mode/pastro_final/"
                           "release_psd_report.json", "w"), indent=1)
    print("[done] report -> release_psd_report.json")


if __name__ == "__main__":
    main()

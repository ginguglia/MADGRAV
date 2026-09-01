#!/usr/bin/env python
"""Forensic autopsy of the biased O4 as-run reference ASDs (2026-08-13).

Reproduces build_o3b_prep.py's exact window sample (same RNG seed 20260706,
same sequential H1-then-L1 draw order over the sorted strain files), keeps the
per-window PSDs and their source segments, then tests the poisoned-mean
hypothesis against the observed spectral shape:

  A. reproduction gate: mean over kept windows must match the archived as-run
     reference_psd npz (validates that the recomputed sample IS the sample).
  B. estimator swap: median over the same windows vs the release run-median
     refs -> if the median lands near release, the sample was fine and the
     MEAN was the poison vector.
  C. concentration: how few windows carry the 20-60 Hz band mean (share of
     top 0.1/1/5% of windows; count needed for 50%/90% of the band excess).
  D. per-segment leave-one-out: drop each segment's windows from the mean;
     report the largest movers at 20-60 Hz.
  E. drop-criterion check: fraction of the top-1% band-loud windows that the
     broadband-RMS 10% drop actually removed (the escape route).

Out: pastro_final/asd_forensics.{json,txt}
Run (CPU sbatch): python asd_forensics.py o4a o4b
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.signal import welch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


FS = 4096
WN = 4 * FS
NPERSEG = 1024
NWIN = 4000
DROP_FRAC = 0.10
SC = MADGRAV_SCRATCH
MR = MADGRAV_ROOT
HERE = f"{MR}/search_mode/pastro_final"
BANDS = [(20, 40), (40, 60), (60, 120), (120, 500)]


def band_ratio(freq, p_num, p_den, lo, hi):
    m = (freq >= lo) & (freq < hi) & (p_den > 0)
    return float(np.sqrt(np.mean(p_num[m]) / np.mean(p_den[m])))  # ASD ratio


def collect(run, rng):
    """Replicates build_o3b_prep.det_psd sampling exactly, keeping provenance."""
    out = {}
    for det in ("H1", "L1"):
        files = sorted(glob.glob(f"{SC}/strain_{run}_full/*_{det}.npz"))
        per_file = max(1, NWIN // len(files))
        rms, psds, segs = [], [], []
        freq = None
        for f in files:
            strain = np.load(f)["strain"]
            n = (len(strain) - WN) // WN
            if n <= 0:
                continue
            starts = rng.choice(n, size=min(per_file, n), replace=False) * WN
            seg = os.path.basename(f).rsplit("_", 1)[0]
            for s in starts:
                w = strain[s:s + WN].astype(np.float64)
                freq, p = welch(w, fs=FS, nperseg=NPERSEG)
                psds.append(p)
                rms.append(float(np.sqrt(np.mean(w * w))))
                segs.append(seg)
        out[det] = (freq, np.asarray(psds), np.asarray(rms), np.asarray(segs))
        print(f"  [{run} {det}] {len(files)} files, {len(psds)} windows", flush=True)
    return out


def analyze(run, det, freq, psds, rms, segs):
    R = {}
    keep = rms <= np.quantile(rms, 1.0 - DROP_FRAC)
    mean_kept = psds[keep].mean(axis=0)

    # A. reproduction gate vs archived as-run ref
    ref = np.load(f"{MR}/data/{run}_search_prep/reference_psd_{det}.npz")
    rel = np.abs(mean_kept - ref["psd"]) / np.maximum(ref["psd"], 1e-60)
    R["repro_max_reldiff"] = float(rel.max())
    R["repro_ok"] = bool(rel.max() < 1e-6)

    rel_ref = np.load(f"{MR}/data/{run}_search_prep_release/reference_psd_{det}.npz")
    med_all = np.median(psds, axis=0)
    R["asd_ratio_vs_release"] = {
        "asrun_mean": {f"{lo}-{hi}": band_ratio(freq, mean_kept, rel_ref["psd"], lo, hi)
                       for lo, hi in BANDS},
        "recomputed_median": {f"{lo}-{hi}": band_ratio(freq, med_all, rel_ref["psd"], lo, hi)
                              for lo, hi in BANDS},
    }

    # C. concentration of the 20-60 Hz band mean over kept windows
    bm = (freq >= 20) & (freq < 60)
    bp = psds[keep][:, bm].mean(axis=1)          # per-window band power
    order = np.argsort(bp)[::-1]
    tot = bp.sum()
    med_band = np.median(psds[:, bm].mean(axis=1))
    excess = tot - len(bp) * med_band            # band power above a flat median floor
    csum = np.cumsum(bp[order] - med_band)
    R["band2060_concentration"] = {
        "share_top_0.1pct": float(bp[order[:max(1, len(bp) // 1000)]].sum() / tot),
        "share_top_1pct": float(bp[order[:max(1, len(bp) // 100)]].sum() / tot),
        "share_top_5pct": float(bp[order[:max(1, len(bp) // 20)]].sum() / tot),
        "n_windows_for_50pct_excess": int(np.searchsorted(csum, 0.5 * excess) + 1) if excess > 0 else None,
        "n_windows_for_90pct_excess": int(np.searchsorted(csum, 0.9 * excess) + 1) if excess > 0 else None,
        "n_kept": int(keep.sum()),
    }

    # D. per-segment leave-one-out on the kept-mean, 20-60 Hz ASD ratio vs release
    base = band_ratio(freq, mean_kept, rel_ref["psd"], 20, 60)
    loo = {}
    for seg in np.unique(segs[keep]):
        m = keep & (segs != seg)
        loo[seg] = band_ratio(freq, psds[m].mean(axis=0), rel_ref["psd"], 20, 60)
    movers = sorted(loo.items(), key=lambda kv: kv[1])[:10]
    R["loo_20_60"] = {"baseline_asd_ratio": base,
                      "top_movers": [{"seg": s, "ratio_without": v,
                                      "drop": base - v} for s, v in movers]}

    # E. did the broadband-RMS drop catch the band-loud windows?
    band_all = psds[:, bm].mean(axis=1)
    top1 = band_all >= np.quantile(band_all, 0.99)
    R["drop_criterion"] = {
        "frac_top1pct_bandloud_dropped": float((top1 & ~keep).sum() / max(top1.sum(), 1)),
        "note": "1.0 would mean the broadband-RMS drop removed all band-loud windows",
    }
    return R


def main():
    runs = sys.argv[1:] or ["o4a", "o4b"]
    rep = {}
    lines = ["AS-RUN REFERENCE-ASD AUTOPSY (poisoned-mean test)", ""]
    for run in runs:
        rng = np.random.default_rng(20260706)   # module-level seed: H1 then L1
        data = collect(run, rng)
        for det in ("H1", "L1"):
            freq, psds, rms, segs = data[det]
            R = analyze(run, det, freq, psds, rms, segs)
            rep[f"{run}_{det}"] = R
            r = R["asd_ratio_vs_release"]
            lines += [
                f"[{run} {det}] repro {'OK' if R['repro_ok'] else 'FAILED (max rel '+format(R['repro_max_reldiff'],'.2e')+')'}",
                "  ASD/release  " + "  ".join(f"{b}: mean {r['asrun_mean'][b]:.2f} med {r['recomputed_median'][b]:.2f}"
                                              for b in r["asrun_mean"]),
                f"  20-60 concentration: top1% carries {R['band2060_concentration']['share_top_1pct']:.0%} "
                f"of band power; {R['band2060_concentration']['n_windows_for_90pct_excess']} windows "
                f"= 90% of excess (of {R['band2060_concentration']['n_kept']})",
                f"  worst-seg LOO: " + ", ".join(f"{m['seg']} -> {m['ratio_without']:.2f}"
                                                 for m in R["loo_20_60"]["top_movers"][:3]),
                f"  RMS-drop caught {R['drop_criterion']['frac_top1pct_bandloud_dropped']:.0%} of top-1% band-loud windows",
                "",
            ]
    json.dump(rep, open(f"{HERE}/asd_forensics.json", "w"), indent=1)
    with open(f"{HERE}/asd_forensics.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

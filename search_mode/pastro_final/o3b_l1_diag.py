#!/usr/bin/env python
"""O3b-L1 release-grade PSD diagnosis (2026-08-13): why does the
release-grade (median-Welch) O3b L1 PSD imply an 85 Mpc BNS range against a
published ~133 Mpc?

Tests, per detector (L1 primary, H1 contrast):
  A. residual leakage: median PSD over the same window population computed
     raw vs 15 Hz zero-phase high-passed; implied BNS range for both; band
     ratios raw/HP (leakage affects every window alike, so a MEDIAN does not
     remove it - the O4 mechanism at smaller amplitude).
  B. storm-epoch drag: per-UTC-month median PSDs (raw + HP) and implied
     ranges across the run; does the winter microseism season drag the
     full-run median, and where does the full-run number sit in the monthly
     spread?
Out: pastro_final/o3b_l1_diag.{json,txt}
"""
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
FS = 4096
WN = 4 * FS
NPERSEG = 1024
MAXFILES = 300
PER_FILE = 10
RNG = np.random.default_rng(20260813)
GPS0_UNIX = 315964800
SOS = butter(8, 15.0, btype="highpass", fs=FS, output="sos")
BANDS = [(20, 40), (40, 60), (60, 120), (120, 500)]

G = 6.674e-11
C = 299792458.0
MSUN = 1.989e30
MPC = 3.0857e22
MC = 1.219 * MSUN


def bns_range(fr, psd, f_lo=20.0, f_hi=1570.0):
    m = (fr >= f_lo) & (fr <= f_hi) & (psd > 0)
    A2 = (5.0 / 24.0) * np.pi ** (-4.0 / 3.0) * C ** 2 * (G * MC / C ** 3) ** (5.0 / 3.0)
    integ = np.trapezoid(A2 * fr[m] ** (-7.0 / 3.0) / psd[m], fr[m])
    return float(np.sqrt(4.0 * integ) / 8.0 / 2.264 / MPC)


def month_of(path):
    gps = int(os.path.basename(path).split("_")[1])
    t = datetime.fromtimestamp(GPS0_UNIX + gps, tz=timezone.utc)
    return f"{t.year}_{t.month:02d}"


def main():
    rep = {}
    lines = ["O3B RELEASE-GRADE PSD DIAGNOSIS (raw vs 15 Hz HP, monthly medians)", ""]
    for det in ("L1", "H1"):
        files = sorted(glob.glob(f"{SC}/strain_o3b_full/*_{det}.npz"))
        if len(files) > MAXFILES:
            files = [files[i] for i in
                     np.linspace(0, len(files) - 1, MAXFILES).astype(int)]
        raw_p, hp_p, months = [], [], []
        fr = None
        for fp in files:
            strain = np.load(fp)["strain"]
            n = (len(strain) - WN) // WN
            if n <= 0:
                continue
            starts = RNG.choice(n, size=min(PER_FILE, n), replace=False) * WN
            mon = month_of(fp)
            for s in starts:
                w = strain[s:s + WN].astype(np.float64)
                fr, p = welch(w, fs=FS, nperseg=NPERSEG)
                raw_p.append(p)
                _, p2 = welch(sosfiltfilt(SOS, w), fs=FS, nperseg=NPERSEG)
                hp_p.append(p2)
                months.append(mon)
        raw_p, hp_p, months = np.asarray(raw_p), np.asarray(hp_p), np.asarray(months)
        med_raw, med_hp = np.median(raw_p, 0), np.median(hp_p, 0)
        stored = np.load(f"{MG}/data/o3b_search_prep/reference_psd_release_{det}.npz")

        def br(pn, pd, lo, hi):
            m = (fr >= lo) & (fr < hi) & (pd > 0)
            return float(np.sqrt(np.mean(pn[m]) / np.mean(pd[m])))

        R = {"n_files": len(files), "n_windows": int(len(raw_p)),
             "range_stored_release": bns_range(fr, stored["psd"]),
             "range_median_raw": bns_range(fr, med_raw),
             "range_median_hp15": bns_range(fr, med_hp),
             "asd_ratio_raw_over_hp": {f"{lo}-{hi}": br(med_raw, med_hp, lo, hi)
                                       for lo, hi in BANDS},
             "monthly": {}}
        for mon in sorted(np.unique(months)):
            sel = months == mon
            R["monthly"][mon] = {
                "n": int(sel.sum()),
                "range_raw": bns_range(fr, np.median(raw_p[sel], 0)),
                "range_hp15": bns_range(fr, np.median(hp_p[sel], 0)),
            }
        rep[det] = R
        lines += [
            f"[{det}] {len(files)} files / {len(raw_p)} windows; implied BNS range: "
            f"stored-release {R['range_stored_release']:.1f}  recomputed-raw-median "
            f"{R['range_median_raw']:.1f}  HP15-median {R['range_median_hp15']:.1f} Mpc",
            "  ASD leakage residual (raw/HP): " +
            "  ".join(f"{b}: {v:.2f}" for b, v in R["asd_ratio_raw_over_hp"].items()),
            "  monthly (raw -> HP15 Mpc): " +
            "  ".join(f"{m}: {d['range_raw']:.0f}->{d['range_hp15']:.0f}"
                      for m, d in R["monthly"].items()),
            "",
        ]
    json.dump(rep, open(f"{HERE}/o3b_l1_diag.json", "w"), indent=1)
    with open(f"{HERE}/o3b_l1_diag.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

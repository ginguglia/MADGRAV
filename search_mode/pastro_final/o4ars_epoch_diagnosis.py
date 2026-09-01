#!/usr/bin/env python
"""o4ars late-run-cliff diagnosis (2026-08-14).

The o4ars fg preview lost 11/14 as-run O4a detections, all at GPS >=
1376883066 (~2023-08-24 onward). Three checks against the corrected
(release run-median) reference PSDs used by the rescan:

PART 1  Three-way overlay on one pre-cliff and one post-cliff segment:
        A  local median-Welch of the segment (1-s windows, average='median')
           -> ground truth for that epoch
        C  as-run reference   data/o4a_search_prep/reference_psd_{det}.npz
        D  release run-median data/o4ars_search_prep/reference_psd_{det}.npz
        If D/A is ~1 pre-cliff but far from 1 post-cliff, the run-median
        reference diverges from late-epoch reality.

PART 2  Monthly-bundle audit: which GWTC-5 monthly PSDs exist per det,
        our coincident-segment exposure per month, and median vs
        exposure-weighted-mean spread (what equal-month weighting hides).

DESIGN NOTE (successor gates, mandated 2026-08-14): any future pilot
or viability verdict MUST use stratified epoch coverage — pilot segments
sampled across the run's full calendar (e.g. one per month or per
sensitivity epoch), never a convenience block from one end. The 3e/3G/3H
O4a pilot drew all six segments from 2023-05-24..2023-07-24 and its
"rescan-viable" verdict silently failed to cover the five late months
where the o4ars fg preview then lost 11/14 as-run detections. PART 3
below is the matching per-epoch reference gate; the epoch-stratified
pilot requirement is its injection-side twin. Both are gates: a pilot or
prep that cannot demonstrate per-epoch coverage fails, regardless of its
run-average numbers.

PART 3  Epoch-resolved BNS-range gate (successor of prep_sanity_gate.py):
        implied 1.4+1.4 sky-averaged SNR-8 range per MONTH from the release
        monthly PSDs, vs the single implied range of the as-run and release
        references, vs the local-Welch ranges of the two PART-1 segments.

Run: madgrav-venv python o4ars_epoch_diagnosis.py   (login node, ~2 min)
"""
import glob
import gzip
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

import json
import os

import numpy as np
from scipy.signal import welch

MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
HERE = f"{MG}/search_mode/pastro_final"
PSDDIR = MADGRAV_EXTDATA + "/gwtc5_sensitivity/psds"
FIGDIR = f"{MG}/figures/o4ars_diag"
FS = 4096
SEGS = {"pre": "o4a_1369681145",    # 2023-06-01, GW230601 re-fires in o4ars
        "post": "o4a_1384773767"}   # 2023-11-23, GW231123 ABSENT in o4ars
O4A_MONTHS = [f"2023_{m:02d}" for m in range(5, 13)] + ["2024_01"]
DETS = {"H1": "H", "L1": "L"}
BANDS = {"20-60": (20, 60), "60-120": (60, 120), "120-500": (120, 500),
         "500-1000": (500, 1000)}

# BNS range constants (prep_sanity_gate.py conventions)
G = 6.674e-11
C_LIGHT = 299792458.0
MSUN = 1.989e30
MPC = 3.0857e22
MC = 1.219 * MSUN
F_LO, F_HI = 20.0, 1570.0
SKY_FACTOR = 2.264


def bns_range_mpc(f, S):
    m = (f >= F_LO) & (f <= F_HI) & (S > 0) & np.isfinite(S)
    f, S = f[m], S[m]
    A2 = (5.0 / 24.0) * np.pi ** (-4.0 / 3.0) * C_LIGHT ** 2 * \
        (G * MC / C_LIGHT ** 3) ** (5.0 / 3.0)
    integ = np.trapezoid(A2 * f ** (-7.0 / 3.0) / S, f)
    return np.sqrt(4.0 * integ) / 8.0 / SKY_FACTOR / MPC


def band_avg(f_native, p_native, f_grid, df):
    out = np.empty(len(f_grid))
    for i, fc in enumerate(f_grid):
        s = (f_native >= fc - df / 2) & (f_native < fc + df / 2)
        out[i] = p_native[s].mean() if s.any() else np.nan
    return out


def load_monthly(path):
    with gzip.open(path, "rt") as fh:
        first = fh.readline()
        delim = "," if "," in first else None
        arr = np.loadtxt(fh, delimiter=delim)
    return arr[:, 0], arr[:, 1]


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    ref0 = np.load(f"{MG}/data/o4a_search_prep/reference_psd_H1.npz")
    f4 = ref0["freq"].astype(float)
    df = float(f4[1] - f4[0])
    report = {}

    # ---------------- PART 1: three-way overlay ----------------
    print("=" * 72)
    print("PART 1: three-way overlay (A=local median-Welch, C=as-run ref, "
          "D=release run-median)")
    curves = {}
    p1 = {}
    for tag, seg in SEGS.items():
        for det in ("H1", "L1"):
            strain = np.load(f"{SC}/strain_o4a_full/{seg}_{det}.npz")[
                "strain"].astype(np.float64)
            fA, pA = welch(strain, fs=FS, nperseg=FS, average="median")
            A = band_avg(fA, pA, f4, df)
            C = np.load(f"{MG}/data/o4a_search_prep/"
                        f"reference_psd_{det}.npz")["psd"]
            D = np.load(f"{MG}/data/o4ars_search_prep/"
                        f"reference_psd_{det}.npz")["psd"]
            curves[(tag, det)] = dict(A=A, C=C, D=D)
            rep = {"segment": seg, "hours": len(strain) / FS / 3600}
            print(f"[{tag} {det}] {seg} ({rep['hours']:.1f} h)  "
                  f"band medians C/A, D/A:")
            for name, (lo, hi) in BANDS.items():
                s = (f4 >= lo) & (f4 <= hi) & (A > 0)
                rep[name] = {"asrun_C_over_A": float(np.median(C[s] / A[s])),
                             "release_D_over_A": float(np.median(D[s] / A[s]))}
                print(f"   {name:>9}: C/A {rep[name]['asrun_C_over_A']:8.2f}  "
                      f"D/A {rep[name]['release_D_over_A']:8.2f}")
            p1[f"{tag}_{det}"] = rep
    report["part1_overlay"] = p1

    # ---------------- PART 2: monthly bundle audit ----------------
    print("=" * 72)
    print("PART 2: monthly PSD bundle audit (O4a)")
    segj = json.load(open(f"{SC}/o4a_full_coincident.json"))
    from datetime import datetime, timedelta, timezone
    gps0 = datetime(1980, 1, 6, tzinfo=timezone.utc)
    expo = {}
    for s in segj["segments"]:
        t = gps0 + timedelta(seconds=float(s[0]))
        mon = f"{t.year}_{t.month:02d}"
        expo[mon] = expo.get(mon, 0.0) + float(s[2])
    p2 = {"exposure_s": expo, "months": {}}
    monthly = {}          # (det, mon) -> band-averaged psd on f4
    for det in ("H1", "L1"):
        found = []
        for mon in O4A_MONTHS:
            hits = sorted(glob.glob(f"{PSDDIR}/psd-O4-{mon}_v1-"
                                    f"{DETS[det]}.*.gz"))
            hits = [h for h in hits if not
                    os.path.basename(h).startswith("._")]
            if hits:
                fn, pn = load_monthly(hits[0])
                monthly[(det, mon)] = band_avg(fn, pn, f4, df)
                found.append(mon)
        missing = [m for m in O4A_MONTHS if m not in found]
        p2["months"][det] = {"found": found, "missing": missing}
        stack = np.array([monthly[(det, m)] for m in found])
        wts = np.array([expo.get(m, 0.0) for m in found])
        med = np.nanmedian(stack, axis=0)
        wmean = np.nansum(stack * wts[:, None], axis=0) / wts.sum()
        Dref = np.load(f"{MG}/data/o4ars_search_prep/"
                       f"reference_psd_{det}.npz")["psd"]
        band = (f4 >= 20) & (f4 <= 60) & (med > 0)
        rebuild_dev = float(np.nanmax(np.abs(med[band] / Dref[band] - 1)))
        wm_dev = float(np.nanmax(np.abs(wmean[band] / med[band] - 1)))
        p2["months"][det].update(
            rebuilt_median_vs_ref_2060_maxdev=rebuild_dev,
            expo_wmean_vs_median_2060_maxdev=wm_dev)
        print(f"[{det}] months found {len(found)}/9  missing={missing or '-'}  "
              f"rebuilt-median vs shipped ref (20-60 Hz) max dev "
              f"{rebuild_dev*100:.1f}%  expo-wmean vs median max dev "
              f"{wm_dev*100:.0f}%")
    zero = [m for m in O4A_MONTHS if expo.get(m, 0.0) == 0.0]
    print("exposure per month [h]: " +
          "  ".join(f"{m}:{expo.get(m,0)/3600:.0f}" for m in O4A_MONTHS))
    if zero:
        print(f"months with ZERO analyzed exposure: {zero} "
              f"(equal-weighted by the median anyway)")
    report["part2_audit"] = p2

    # ---------------- PART 3: epoch-resolved BNS ranges ----------------
    print("=" * 72)
    print("PART 3: epoch-resolved implied BNS ranges [Mpc]")
    p3 = {}
    for det in ("H1", "L1"):
        C = np.load(f"{MG}/data/o4a_search_prep/"
                    f"reference_psd_{det}.npz")["psd"]
        D = np.load(f"{MG}/data/o4ars_search_prep/"
                    f"reference_psd_{det}.npz")["psd"]
        r_asrun = bns_range_mpc(f4, C)
        r_rel = bns_range_mpc(f4, D)
        months = {m: bns_range_mpc(f4, monthly[(det, m)])
                  for m in O4A_MONTHS if (det, m) in monthly}
        locals_ = {tag: bns_range_mpc(f4, curves[(tag, det)]["A"])
                   for tag in SEGS}
        p3[det] = {"asrun_ref": r_asrun, "release_ref": r_rel,
                   "monthly": months, "local_segments": locals_}
        print(f"[{det}] as-run ref {r_asrun:6.1f}   release ref {r_rel:6.1f}   "
              f"local pre-seg {locals_['pre']:6.1f}   "
              f"local post-seg {locals_['post']:6.1f}")
        line = "   monthly: " + "  ".join(
            f"{m[2:]}:{r:.0f}" for m, r in months.items())
        print(line)
        worst = max(months.items(),
                    key=lambda kv: abs(np.log(r_rel / kv[1])))
        print(f"   release-ref vs monthly ratio range "
              f"{min(r_rel/r for r in months.values()):.2f}-"
              f"{max(r_rel/r for r in months.values()):.2f} "
              f"(worst month {worst[0]})")
    report["part3_ranges"] = p3

    json.dump(report, open(f"{HERE}/o4ars_epoch_diagnosis.json", "w"),
              indent=1)

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = {"A": "#0072B2", "C": "#D55E00", "D": "#009E73"}
    labels = {"A": "local median-Welch (ground truth)",
              "C": "as-run reference", "D": "release run-median (o4ars)"}
    for i, tag in enumerate(("pre", "post")):
        for j, det in enumerate(("H1", "L1")):
            ax = axes[i][j]
            band = (f4 >= 20) & (f4 <= 1000)
            for key in ("C", "D", "A"):
                ax.loglog(f4[band], curves[(tag, det)][key][band], lw=1.5,
                          color=colors[key], label=labels[key])
            ax.set_title(f"{det} — {SEGS[tag]} ({tag}-cliff)")
            ax.grid(alpha=0.25, which="both", lw=0.4)
            if i == 1:
                ax.set_xlabel("frequency [Hz]")
            if j == 0:
                ax.set_ylabel(r"PSD [strain$^2$/Hz]")
    axes[0][0].legend(fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGDIR}/o4ars_epoch_diagnosis.{ext}", dpi=160)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    xm = np.arange(len(O4A_MONTHS))
    for ax, det in zip(axes2, ("H1", "L1")):
        r = p3[det]
        vals = [r["monthly"].get(m, np.nan) for m in O4A_MONTHS]
        ax.plot(xm, vals, "o-", color="#0072B2", label="monthly release PSD")
        ax.axhline(r["release_ref"], color="#009E73", lw=1.5,
                   label="release run-median ref")
        ax.axhline(r["asrun_ref"], color="#D55E00", lw=1.5, ls="--",
                   label="as-run ref")
        ax.plot([1], [r["local_segments"]["pre"]], "s", ms=9,
                color="#CC79A7", label="local Welch pre-seg (Jun)")
        ax.plot([6], [r["local_segments"]["post"]], "D", ms=9,
                color="#E69F00", label="local Welch post-seg (Nov)")
        ax.set_xticks(xm, [m[2:] for m in O4A_MONTHS], rotation=45)
        ax.set_title(f"{det} implied BNS range across O4a")
        ax.set_ylabel("sky-avg SNR-8 range [Mpc]")
        ax.grid(alpha=0.25, lw=0.4)
    axes2[0].legend(fontsize=7.5)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(f"{FIGDIR}/o4ars_epoch_ranges.{ext}", dpi=160)
    print(f"[done] -> {HERE}/o4ars_epoch_diagnosis.json + "
          f"{FIGDIR}/o4ars_epoch_diagnosis.png + o4ars_epoch_ranges.png")


if __name__ == "__main__":
    main()

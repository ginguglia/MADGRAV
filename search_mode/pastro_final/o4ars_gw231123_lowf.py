#!/usr/bin/env python
"""GW231123-segment low-frequency check (item 2, 2026-08-14).

Focused 20-120 Hz overlay for o4a_1384773767 (hosts GW231123, ABSENT from
the o4ars fg preview): local median-Welch of the segment (ground truth)
vs the corrected (release run-median) reference actually used by the
rescan whitening, plus the whitening-error ratio S_local/S_ref per 4-Hz
bin. If the global ~2-10x low-f under-estimate of the run-median ref is
the mechanism killing late-run events, it must show here.

Run: madgrav-venv python o4ars_gw231123_lowf.py   (login node, ~1 min)
"""
import json

import numpy as np
from scipy.signal import welch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
SEG = "o4a_1384773767"
FS = 4096


def band_avg(fn, pn, f4, df):
    out = np.empty(len(f4))
    for i, fc in enumerate(f4):
        s = (fn >= fc - df / 2) & (fn < fc + df / 2)
        out[i] = pn[s].mean() if s.any() else np.nan
    return out


def main():
    ref0 = np.load(f"{MG}/data/o4ars_search_prep/reference_psd_H1.npz")
    f4 = ref0["freq"].astype(float)
    df = float(f4[1] - f4[0])
    band = (f4 >= 20) & (f4 <= 120)
    rep = {}
    curves = {}
    for det in ("H1", "L1"):
        strain = np.load(f"{SC}/strain_o4a_full/{SEG}_{det}.npz")[
            "strain"].astype(np.float64)
        fA, pA = welch(strain, fs=FS, nperseg=FS, average="median")
        A = band_avg(fA, pA, f4, df)
        D = np.load(f"{MG}/data/o4ars_search_prep/"
                    f"reference_psd_{det}.npz")["psd"]
        curves[det] = (A, D)
        r = A[band] / D[band]
        rep[det] = {
            "ratio_local_over_ref_2060_median": float(
                np.median((A / D)[(f4 >= 20) & (f4 <= 60)])),
            "ratio_local_over_ref_60120_median": float(
                np.median((A / D)[(f4 >= 60) & (f4 <= 120)])),
            "ratio_20120_max": float(np.nanmax(r)),
            "ratio_20120_max_freq": float(f4[band][int(np.nanargmax(r))]),
            "per_bin_20120": {f"{f4[band][i]:.0f}": float(r[i])
                              for i in range(len(r))},
        }
        print(f"[{det}] {SEG}: S_local/S_ref median 20-60 Hz "
              f"{rep[det]['ratio_local_over_ref_2060_median']:.2f}x, "
              f"60-120 Hz {rep[det]['ratio_local_over_ref_60120_median']:.2f}x,"
              f" max {rep[det]['ratio_20120_max']:.2f}x @ "
              f"{rep[det]['ratio_20120_max_freq']:.0f} Hz")
    json.dump(rep, open(f"{MG}/search_mode/pastro_final/"
                        "o4ars_gw231123_lowf.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7),
                             gridspec_kw={"height_ratios": [2.2, 1]},
                             sharex=True)
    for j, det in enumerate(("H1", "L1")):
        A, D = curves[det]
        ax = axes[0][j]
        ax.semilogy(f4[band], A[band], lw=1.6, color="#0072B2",
                    label="local median-Welch (ground truth)")
        ax.semilogy(f4[band], D[band], lw=1.6, color="#009E73",
                    label="corrected ref (release run-median)")
        ax.set_title(f"{det} — {SEG} (hosts GW231123)")
        ax.grid(alpha=0.25, which="both", lw=0.4)
        if j == 0:
            ax.set_ylabel(r"PSD [strain$^2$/Hz]")
            ax.legend(fontsize=8)
        axr = axes[1][j]
        axr.plot(f4[band], (A / D)[band], "o-", ms=3, color="#D55E00")
        axr.axhline(1.0, color="k", lw=0.8, ls=":")
        axr.set_xlabel("frequency [Hz]")
        axr.grid(alpha=0.25, lw=0.4)
        if j == 0:
            axr.set_ylabel(r"$S_{\rm local}/S_{\rm ref}$")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{MG}/figures/o4ars_diag/gw231123_lowf.{ext}", dpi=160)
    print("[done] -> pastro_final/o4ars_gw231123_lowf.json + "
          "figures/o4ars_diag/gw231123_lowf.png")


if __name__ == "__main__":
    main()

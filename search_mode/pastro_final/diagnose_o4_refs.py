#!/usr/bin/env python
"""Diagnosis of the current O4 reference PSDs (2026-08-12).

Overlay, per detector, 20-1000 Hz, on the known-good O4b science segment
o4b_1400806212 (16.4 h, hosts no detection):
  A fresh median-Welch  : nperseg=4096 (1 s, 1 Hz), average='median',
                          whole segment -> ground truth for that segment
  B builder-replica     : build_o3b_prep recipe on the SAME data - random 4-s
                          windows, welch nperseg=1024 (0.25 s Hann) MEAN,
                          drop loudest 10% by RMS      -> isolates ESTIMATOR
  C current reference   : data/o4b_search_prep/reference_psd_{det}.npz
  D release run-median  : reference_psd_release_{det}.npz (step 3a)
B/A = estimator-induced bias on good data; C/B = data/sampling residual;
A/D = segment-vs-run + convention residual.
Run: madgrav-venv python diagnose_o4_refs.py
"""
import json

import numpy as np
from scipy.signal import welch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
SEG = MADGRAV_SCRATCH + "/strain_o4b_full/o4b_1400806212"
FS = 4096
RNG = np.random.default_rng(20260812)

BANDS = {"20-60": (20, 60), "60-120": (60, 120), "120-500": (120, 500),
         "500-1000": (500, 1000)}


def band_avg_to4(f_native, p_native, f4):
    out = np.empty(len(f4))
    for i, fc in enumerate(f4):
        s = (f_native >= fc - 2) & (f_native < fc + 2)
        out[i] = p_native[s].mean() if s.any() else np.nan
    return out


def main():
    ref = np.load(f"{MG}/data/o4b_search_prep/reference_psd_H1.npz")
    f4 = ref["freq"].astype(float)
    curves, report = {}, {}
    for det in ("H1", "L1"):
        strain = np.load(f"{SEG}_{det}.npz")["strain"].astype(np.float64)
        dur_h = len(strain) / FS / 3600
        # A: fresh median-Welch, 1-s windows
        fA, pA = welch(strain, fs=FS, nperseg=FS, average="median")
        A = band_avg_to4(fA, pA, f4)
        # B: builder replica on the same data
        WN = 4 * FS
        n = (len(strain) - WN) // WN
        starts = RNG.choice(n, size=min(4000, n), replace=False) * WN
        psds, rms = [], []
        for s in starts:
            w = strain[s:s + WN]
            fB, p = welch(w, fs=FS, nperseg=1024)
            psds.append(p); rms.append(float(np.sqrt(np.mean(w * w))))
        psds, rms = np.asarray(psds), np.asarray(rms)
        keep = rms <= np.quantile(rms, 0.90)
        B = psds[keep].mean(axis=0)
        # C, D
        C = np.load(f"{MG}/data/o4b_search_prep/reference_psd_{det}.npz")["psd"]
        D = np.load(f"{MG}/data/o4b_search_prep/reference_psd_release_{det}.npz")["psd"]
        curves[det] = dict(A=A, B=B, C=C, D=D)
        rep = {"segment_hours": dur_h}
        for name, (lo, hi) in BANDS.items():
            s = (f4 >= lo) & (f4 <= hi)
            rep[name] = {
                "estimator_bias_B_over_A": float(np.median(B[s] / A[s])),
                "data_residual_C_over_B": float(np.median(C[s] / B[s])),
                "current_vs_truth_C_over_A": float(np.median(C[s] / A[s])),
                "release_vs_truth_D_over_A": float(np.median(D[s] / A[s])),
            }
        report[det] = rep
        print(f"[{det}] seg {dur_h:.1f} h; medians per band "
              f"(B/A=estimator, C/B=data, C/A=current-vs-truth, D/A=release-vs-truth):")
        for name in BANDS:
            r = rep[name]
            print(f"   {name:>9}: B/A {r['estimator_bias_B_over_A']:8.1f}  "
                  f"C/B {r['data_residual_C_over_B']:6.2f}  "
                  f"C/A {r['current_vs_truth_C_over_A']:9.1f}  "
                  f"D/A {r['release_vs_truth_D_over_A']:6.2f}")

    json.dump(report, open(f"{HERE}/o4_ref_diagnosis.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    colors = {"A": "#0072B2", "B": "#E69F00", "C": "#D55E00", "D": "#009E73"}
    labels = {"A": "fresh median-Welch (good seg, 1 s)",
              "B": "builder replica (same seg, 0.25 s Hann, mean)",
              "C": "current reference (o4b_search_prep)",
              "D": "release run-median (step 3a)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, det in zip(axes, ("H1", "L1")):
        band = (f4 >= 20) & (f4 <= 1000)
        for key in ("C", "B", "D", "A"):
            ax.loglog(f4[band], curves[det][key][band], lw=1.6,
                      color=colors[key], label=labels[key])
        ax.set_title(f"{det} - O4b segment 1400806212")
        ax.set_xlabel("frequency [Hz]")
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_ylabel(r"PSD [strain$^2$/Hz]")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{MG}/figures/vt_compare/o4_ref_diagnosis.{ext}", dpi=160)
    print(f"[done] -> figures/vt_compare/o4_ref_diagnosis.png + o4_ref_diagnosis.json")


if __name__ == "__main__":
    main()

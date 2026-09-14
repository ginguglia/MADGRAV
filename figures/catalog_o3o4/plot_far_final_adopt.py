#!/usr/bin/env python
"""Paper figure: calibrated FAR (w/ 90% Poisson CI) vs source-frame total mass for the ADOPTED
46 MADGRAV detections, O3a/O3b/O4a/O4b. UL90 arrows for the N=0 events.

Statistic frozen 2026-08-31: lnLambda-channel per-arm FAR against the foreground-EXCLUDED
time-slide background, multiplied by the per-run null-calibration factor K. Selection imported
from adopted_set.py so this figure and Table I cannot disagree.

Supersedes plot_far_final_x1.py, which drew the 48-detection trials=1 set on the superseded
(uncalibrated, sigma_net-inclusive, inclusive-background) FAR axis.

Writes far_final_o3o4_adopt.* -- the x1 and accepted files are NOT touched.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import adopted_set

rows = adopted_set.load()

def flt(x):
    try: return float(x)
    except Exception: return None

# Okabe-Ito, one hue per run (fixed order) -- identical to the accepted figure
STYLE = {"O3a": dict(c="#0072B2", m="o"), "O3b": dict(c="#009E73", m="s"),
         "O4a": dict(c="#E69F00", m="D"), "O4b": dict(c="#D55E00", m="^")}

plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(4.4, 3.3))

n_plotted = 0
no_mass = [r["name"] for r in rows if not flt(r["mtot"])]
n_ul = 0

for run in ["O3a", "O3b", "O4a", "O4b"]:
    st = STYLE[run]
    rr = [r for r in rows if r["run"] == run and flt(r["mtot"])]   # mass axis needs a published Mtot
    meas = [r for r in rr if r["N"] > 0]
    xs = [flt(r["mtot"]) for r in meas]
    ys = [r["far"] for r in meas]
    ci = [adopted_set.poisson_ci(r["N"], r["T"], r["ke"]) for r in meas]
    lo = [r["far"] - c[0] for r, c in zip(meas, ci)]
    hi = [c[1] - r["far"] for r, c in zip(meas, ci)]
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt=st["m"], ms=5, color=st["c"], mec="white", mew=0.6,
                elinewidth=1.0, capsize=0, ls="none", zorder=3,
                label=run)
    # UL90 arrows for N=0 (no background family louder than the candidate)
    for r in rr:
        if r["N"] == 0:
            x, u = flt(r["mtot"]), r["ul90"]
            ax.errorbar([x], [u], fmt=st["m"], ms=5, color=st["c"], mec="white", mew=0.6, zorder=3)
            ax.annotate("", xy=(x, u * 0.45), xytext=(x, u),
                        arrowprops=dict(arrowstyle="-|>", color=st["c"], lw=1.2), zorder=3)
            n_ul += 1
    n_plotted += len(rr)

# per-event annotations removed 2026-09-01 (user).

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(2.5, 320)
ax.axhline(1.0, color="#b8bcc2", lw=1.0, ls="--", zorder=1)
from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xticks([3, 10, 30, 100, 300])
ax.set_xlabel(r"Source-frame total mass  $M_{\mathrm{tot}}$  [$M_\odot$]")
ax.set_ylabel(r"Calibrated false-alarm rate  [yr$^{-1}$]")
ax.grid(True, which="major", color="#e6e8eb", lw=0.7, zorder=0)
# full box: Figs 3 and 4 use closed frames, so these match (2026-09-01)
leg = ax.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#d0d3d7",
                fontsize=6.5, labelspacing=0.3, borderpad=0.4)
leg.set_zorder(5)

# y-range: keep the 90% lower bounds and the threshold line visible, set from the data
ylo = min([adopted_set.poisson_ci(r["N"], r["T"], r["ke"])[0] for r in rows if r["N"] > 0]
          + [r["ul90"] * 0.45 for r in rows if r["N"] == 0])
ax.set_ylim(ylo * 0.6, 1.8)
ax.text(2.7, 1.22, "detection threshold (1 yr$^{-1}$)", fontsize=7, color="#8a9096")

fig.tight_layout()
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

for ext in ("png", "pdf"):
    fig.savefig(f"{__import__('os').environ.get('SM_FIG_OUT', MADGRAV_ROOT + '/figures/far_final_o3o4_adopt')}.{ext}",
                dpi=170, bbox_inches="tight")
print(f"wrote figures/far_final_o3o4_adopt.png/.pdf  ({n_plotted}/{len(rows)} plotted, "
      f"{n_ul} UL arrows)")
print(f"  calibrated FAR range: {min(r['far'] for r in rows if r['N']>0):.4f} - "
      f"{max(r['far'] for r in rows):.3f}")
print(f"  N=0 events: {sorted((r['name'], round(r['ul90'],4), round(r['T'],1)) for r in rows if r['N']==0)}")
if no_mass:
    print("NOT plotted (no published source-frame Mtot): " + ", ".join(no_mass))

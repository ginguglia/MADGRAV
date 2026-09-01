"""Four-panel VT figure: efficiency and <VT>, each shown in BOTH mass frames.

Rows = quantity (efficiency, <VT>); columns = mass frame (detector, source). Small multiples
rather than one panel with two x-scales: the two frames are different measures of the same
injections and must not share an axis.

Why both frames. The relabel layer assigns every injection an implied distance and hence a
redshift, so its SOURCE-frame mass M_det/(1+z) is lower than its detector-frame mass. VT is
reported in the source frame (that is the astrophysical statement), but efficiency is most
naturally read in the detector frame -- it is a property of what the pipeline saw. Showing one of
each made the panels non-comparable; showing only the source frame hides that the source-frame
efficiency carries a SELECTION effect at high mass (a high source-frame bin is fed either by a
high detector-frame mass at high z or a moderate one at low z, and the low-z ones are louder and
preferentially detected -- which is what produces the upturn in the top source-frame bin).

Each panel is masked with the N_eff support of ITS OWN frame (NEFF_MIN=300); the frames drain
differently, so borrowing one mask for the other would mask the wrong bins.

Usage: SM_VT_SUF=_x1cnnfixveto fig_vt_frames.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
FIGDIR = f"{MG}/figures/vt_search"
RUNS = ("O3a", "O3b", "O4a", "O4b")
# Okabe-Ito, unchanged from fig_vt_paper.py so the two figures read as one system. Validated
# 2026-08-31: worst-pair OKLab dE x100 = 15.6 normal / 11.4 protan / 11.0 deutan / 8.6 tritan,
# all above threshold; marker shape carries identity independently of hue.
STYLE = {"O3a": dict(c="#0072B2", m="o"), "O3b": dict(c="#009E73", m="s"),
         "O4a": dict(c="#E69F00", m="D"), "O4b": dict(c="#D55E00", m="^")}
NEFF_MIN = 300.0
SUF = os.environ.get("SM_VT_SUF", "")

rel = json.load(open(f"{HERE}/vt_relabel_comoving{SUF}.json"))
eff_j = json.load(open(f"{HERE}/eff_srcframe{SUF}.json"))
nef = json.load(open(f"{HERE}/neff_srcframe{SUF}.json"))
edges = np.array(rel["mass_edges"], float)
mids = 0.5 * (edges[1:] + edges[:-1])


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def masked(vals, support):
    v = arr(vals).copy()
    v[arr(support) < NEFF_MIN] = np.nan
    return v


DATA = {}
for r in RUNS:
    R = rel["runs"][r]
    s_sup = nef["neff_srcframe"][r]["support"]
    d_sup = nef["neff_detframe"][r]["support"]
    DATA[r] = {
        ("eff", "detector"): masked(eff_j[r]["eff_detframe"], d_sup),
        ("eff", "source"): masked(eff_j[r]["eff_srcframe"], s_sup),
        ("vt", "detector"): masked(R["vt_comoving_gpc3yr"], d_sup),
        ("vt", "source"): masked(R["vt_comoving_srcframe_gpc3yr"], s_sup),
    }

plt.rcParams.update({"font.size": 10.5, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex="col")
for j, frame in enumerate(("detector", "source")):
    for i, q in enumerate(("eff", "vt")):
        ax = axes[i][j]
        for r in RUNS:
            st = STYLE[r]
            ax.plot(mids, DATA[r][(q, frame)], color=st["c"], marker=st["m"], ms=4.5, lw=1.6,
                    label=r, clip_on=False)
        if q == "vt":
            stack = np.array([DATA[r][(q, frame)] for r in RUNS])
            tot = np.where(np.any(np.isnan(stack), axis=0), np.nan, np.nansum(stack, axis=0))
            ax.plot(mids, tot, color="0.25", lw=2.2, label="O3+O4 total")
            ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(edges[0], edges[-1] if frame == "detector" else 320.0)
    axes[0][j].set_title(f"{frame} frame", fontsize=11)
    axes[1][j].set_xlabel(r"$M_{\rm tot}$ [$M_\odot$]")

# shared scale within each row: the columns are only comparable if the y-axes match
for i, q in enumerate(("eff", "vt")):
    lo = min(np.nanmin(DATA[r][(q, f)]) for r in RUNS for f in ("detector", "source"))
    hi = max(np.nanmax(DATA[r][(q, f)]) for r in RUNS for f in ("detector", "source"))
    for j in range(2):
        axes[i][j].set_ylim((0, hi * 1.08) if q == "eff" else (lo * 0.5, hi * 6))
axes[0][0].set_ylabel("volume-averaged efficiency\n(FAR $<1$/yr)")
axes[1][0].set_ylabel(r"$\langle VT\rangle$ [Gpc$^3$ yr, comoving]")
axes[0][0].legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
axes[1][0].legend(frameon=False, fontsize=9, ncol=2, loc="lower center")
fig.tight_layout()
os.makedirs(FIGDIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIGDIR}/vt_vs_mass_frames{SUF}.{ext}", dpi=200, bbox_inches="tight")

print("last plotted bin centre per panel:")
for q in ("eff", "vt"):
    for f in ("detector", "source"):
        k = max(int(np.max(np.where(np.isfinite(DATA[r][(q, f)]))[0])) for r in RUNS)
        print(f"  {q:<4} {f:<9} -> {mids[k]:.0f} Msun")
print(f"[fig_vt_frames] -> {FIGDIR}/vt_vs_mass_frames{SUF}.pdf/.png")

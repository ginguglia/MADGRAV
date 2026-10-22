#!/usr/bin/env python
"""MADGRAV 2.2 <VT> in the layout of the paper's fig:vt right panel (2026-10-06).

Two panels on identical axes: left = MADGRAV 2.2 (13 source-frame bins, 10-400 Msun), right = the adopted paper pipeline
(10 bins, 20-400), each with the four runs and the O3+O4 total; the paper total is repeated as a grey dashed reference on the
left. Same run colours/markers as fig_vt_paper.py. The paper's efficiency panel is not reproduced (its source-frame efficiency
file does not exist for 2.2). Out: figures_m2/vt22_paper.{pdf,png}.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

REL22 = os.environ.get("VT22_JSON", MADGRAV_SCRATCH + "/madgrav2/vt22/vt_relabel_comoving_m22.json")
REL1 = MADGRAV_SCRATCH + "/MADGRAV1/search_mode/pastro_final/vt_relabel_comoving_x1cnnadopt48fveto.json"
FIGDIR = os.environ.get("MG_FIGDIR", MADGRAV_SCRATCH + "/madgrav2/figures_m2")
RUNS = ("O3a", "O3b", "O4a", "O4b")
STYLE = {"O3a": dict(c="#0072B2", m="o"), "O3b": dict(c="#009E73", m="s"), "O4a": dict(c="#E69F00", m="D"), "O4b": dict(c="#D55E00", m="^")}


def load(path):
    d = json.load(open(path)); e = np.array(d["mass_edges"], float); mids = 0.5 * (e[1:] + e[:-1])
    vt = {r: np.array([np.nan if x is None else x for x in d["runs"][r]["vt_comoving_srcframe_gpc3yr"]], float) for r in RUNS}
    arr = np.array([vt[r] for r in RUNS]); tot = np.where(np.any(np.isnan(arr), 0), np.nan, np.nansum(arr, 0))
    return e, mids, vt, tot


def main():
    e22, m22, vt22, tot22 = load(REL22); e1, m1, vt1, tot1 = load(REL1)
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, mids, vt, tot, name in ((axA, m22, vt22, tot22, "MADGRAV 2.2"), (axB, m1, vt1, tot1, "MADGRAV (paper, adopted)")):
        for r in RUNS:
            st = STYLE[r]; ax.plot(mids, vt[r], color=st["c"], marker=st["m"], ms=4.5, lw=1.6, label=r)
        ax.plot(mids, tot, color="0.25", lw=2.2, label="O3+O4 total")
        ax.set_title(f"Sensitive volume-time, {name}", fontsize=10.5)
        ax.set_xlabel(r"$M_{\rm tot}$ (source frame) [$M_\odot$]"); ax.set_yscale("log"); ax.grid(alpha=0.25, lw=0.5); ax.set_xlim(0, 400)
    axA.plot(m1, tot1, color="0.25", lw=1.4, ls="--", label="paper O3+O4 total (reference)")
    axA.set_ylabel(r"$\langle VT\rangle$ [Gpc$^3$ yr, comoving]")
    axA.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=2); axB.legend(frameon=False, fontsize=8.5, loc="lower left")
    axA.set_ylim(1e-3, 30)
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"{FIGDIR}/vt22_paper.{ext}", dpi=200, bbox_inches="tight")
    print("total O3+O4 Gpc^3 yr on the paper bins (20-400): 2.2 = %.2f, paper = %.2f" % (np.nansum(tot22[2:]), np.nansum(tot1)))
    print(f"[fig] -> {FIGDIR}/vt22_paper.{{pdf,png}}")


if __name__ == "__main__":
    main()

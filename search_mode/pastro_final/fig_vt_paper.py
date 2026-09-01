#!/usr/bin/env python
"""Paper Fig. (fig:vt), amended 2026-08-15: efficiency (detector-frame bins,
unchanged) | COMOVING relabeled <VT> vs SOURCE-frame Mtot per run + total.

Numerator = vt_relabel_comoving.json:vt_comoving_srcframe_gpc3yr (four-epoch
figure-of-record numerator; float64 masks, O4 release-relabeled, FlatLCDM
67.9/0.3065, V_max = int dVc/(1+z), wall-clock T). Support-N_eff >= 300 mask
on source-frame bins taken from vt_compare_pipelines.json (neff_srcframe);
a summed curve is masked wherever any component run is masked.
Replaces vt_search.py's Euclidean proxy panel in the paper (that figure is
kept as figures/vt_search/vt_vs_mass_search.* for provenance).
Out: figures/vt_search/vt_vs_mass_paper.{pdf,png} + vt_paper_numbers.json
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
FIGDIR = f"{MG}/figures/vt_search"
RUNS = ("O3a", "O3b", "O4a", "O4b")
STYLE = {"O3a": dict(c="#0072B2", m="o"), "O3b": dict(c="#009E73", m="s"),
         "O4a": dict(c="#E69F00", m="D"), "O4b": dict(c="#D55E00", m="^")}
NEFF_MIN = 300.0
SUF = os.environ.get("SM_VT_SUF", "")   # "" = accepted as-run, "_x1" = trials=1

rel = json.load(open(f"{HERE}/vt_relabel_comoving{SUF}.json"))
# support mask: the CNN campaign recomputes its own N_eff (extended O3a); otherwise the
# harmonized cross-pipeline file supplies it.
import os.path as _op
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

_nf = f"{HERE}/neff_srcframe{SUF}.json"
cmp_ = json.load(open(_nf)) if _op.exists(_nf) else json.load(open(f"{HERE}/vt_compare_pipelines.json"))
edges = np.array(rel["mass_edges"], float); mids = 0.5 * (edges[1:] + edges[:-1])
vt, eff, T = {}, {}, {}
for r in RUNS:
    R = rel["runs"][r]
    v = np.array([np.nan if x is None else x for x in R["vt_comoving_srcframe_gpc3yr"]], float)
    _e = cmp_["neff_srcframe"][r]
    neff = np.array(_e["support"] if isinstance(_e, dict) else _e, float)
    v[neff < NEFF_MIN] = np.nan
    vt[r] = v; T[r] = R["T_obs_yr"]
    # PANEL CONSISTENCY (2026-08-31). Two defects fixed together, both "the panels are not on a
    # common footing": (1) eff_covered is binned in DETECTOR-frame mass while VT is rebinned to
    # SOURCE frame, so the same x position meant different masses on the left and right; (2) the
    # N_eff support mask was applied to VT only, so the left panel plotted points at masses the
    # right panel had already judged unsupported (365 vs 295 Msun reach). Use the source-frame
    # efficiency when it exists and mask it identically. Absent the file, the old behaviour and the
    # old axis label are reproduced exactly, so earlier suffixes rebuild byte-identically.
    _ef = f"{HERE}/eff_srcframe{SUF}.json"
    if _op.exists(_ef):
        e_ = np.array([np.nan if x is None else x
                       for x in json.load(open(_ef))[r]["eff_srcframe"]], float)
        e_[neff < NEFF_MIN] = np.nan
        eff[r] = e_
        EFF_FRAME = "source"
    else:
        eff[r] = np.array([np.nan if x is None else x for x in R["eff_covered"]], float)
        EFF_FRAME = "detector"
arr = np.array([vt[r] for r in RUNS]); anymask = np.any(np.isnan(arr), axis=0)
tot = np.where(anymask, np.nan, np.nansum(arr, axis=0))
o3 = np.where(np.isnan(vt["O3a"]) | np.isnan(vt["O3b"]), np.nan, vt["O3a"] + vt["O3b"])

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
fig, (axE, axV) = plt.subplots(1, 2, figsize=(8.8, 3.4))
for r in RUNS:
    st = STYLE[r]
    axE.plot(mids, eff[r], color=st["c"], marker=st["m"], ms=4.5, lw=1.6, label=r, clip_on=False)
    axV.plot(mids, vt[r], color=st["c"], marker=st["m"], ms=4.5, lw=1.6, label=r)
axV.plot(mids, tot, color="0.25", lw=2.2, label="O3+O4 total")
axE.set_xlabel(rf"$M_{{\rm tot}}$ ({EFF_FRAME} frame) [$M_\odot$]"); axE.set_ylabel("volume-averaged efficiency")
axE.set_ylim(0, None); axE.legend(frameon=False, fontsize=9)
axV.set_xlabel(r"$M_{\rm tot}$ (source frame) [$M_\odot$]")
axV.set_ylabel(r"$\langle VT\rangle$ [Gpc$^3$ yr, comoving]")
axV.set_yscale("log"); axV.legend(frameon=False, fontsize=9, loc="lower left")
for ax in (axE, axV):
    ax.grid(alpha=0.25, lw=0.5); ax.set_xlim(edges[0], edges[-1])
axE.set_title("Search efficiency (FAR$<1$/yr)", fontsize=10)
axV.set_title("Sensitive volume–time (comoving, relabeled)", fontsize=10)
fig.tight_layout()
os.makedirs(FIGDIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIGDIR}/vt_vs_mass_paper{SUF}.{ext}", dpi=200, bbox_inches="tight")

def pk(a): i = int(np.nanargmax(a)); return float(a[i]), float(mids[i])
num = dict(T_obs_yr=T, per_run_vt=({r: vt[r].tolist() for r in RUNS}), o3=o3.tolist(), total=tot.tolist(),
           peak_o3a=pk(vt["O3a"]), peak_o3=pk(o3), peak_total=pk(tot),
           masked_bins={r: [int(k) for k in np.where(np.array(cmp_["neff_srcframe"][r]["support"]
                             if isinstance(cmp_["neff_srcframe"][r], dict) else cmp_["neff_srcframe"][r])
                             < NEFF_MIN)[0]] for r in RUNS})
json.dump(num, open(f"{HERE}/vt_paper_numbers{SUF}.json", "w"), indent=1)
print(json.dumps({k: num[k] for k in ("T_obs_yr", "peak_o3a", "peak_o3", "peak_total", "masked_bins")}, indent=1))
print(f"[fig_vt_paper] -> {FIGDIR}/vt_vs_mass_paper{SUF}.pdf/.png")

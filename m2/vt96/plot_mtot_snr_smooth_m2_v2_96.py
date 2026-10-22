#!/usr/bin/env python
"""Mass-SNR recovery plane, MADGRAV2 campaign detections -- smooth rendering.

2026-10-19 DECK COPY (scratchpad, read-only on the repos): stars = the frozen version-2 set, the 96 catalogue rows of
detections_v2_96.csv, matched by name to hl_snr_o3o4_m2_crop96.json for Mtot/snr_hl. One star style, no highlight layer,
no footer; per-run title N = version-2 count in the run (CSV), P/M as computed below from the 2.2 seed-20261004 field (vt22, m22).

MADGRAV2 VARIANT of mtot_snr_sim_experiment/plot_mtot_snr_smooth.py. Only the star layer
changes: it is the MADGRAV2 gx-campaign detection set (build_m2_eventset.py).

TWO CAVEATS, both drawn on the figure:
  * the viridis recovery field is still the MADGRAV1 adopted injection campaign. MADGRAV2
    injections exist (inj_<run>_m2) but have not been scored through pastro_final, so the
    field shows where MADGRAV1 could see, with MADGRAV2's detections on top of it.
  * o4a is still running (CNN 1754431 / merge 1754432), so that panel carries no stars.

Same data and same viridis gradient as mtot_snr_sim_adopted, but: mass binning
refined from 10 to 14 log-spaced bins (genuine extra resolution -- 27k-57k
injections per run support it), and the field interpolated bicubically in
(log Mtot, log SNR) before filling, so the cell edges are no longer visible.

The SNR resolution is NOT improved: the campaign has 9 discrete injected SNR
levels, so vertical structure between them is interpolation, not measurement.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.interpolate import RegularGridInterpolator


def isotonic(y, w):
    """Monotone non-decreasing least-squares fit (pool adjacent violators), weight w.

    Detection probability cannot fall with injected SNR at fixed mass, so the monotone
    fit is the correct estimator for a column of the recovery grid -- not a smoothing
    choice. Without it, sparse low-mass cells (only ~2.7% of injections land below
    Mtot 32) produce ~2 sigma dips that the interpolator renders as closed contours
    looking like real structure. Measured 2026-09-04: an 0.91 dip at SNR 60 in both
    O3a and O4a with recovery back at 0.99-1.00 at SNR 80, i.e. no mechanism.
    """
    y = np.asarray(y, float).copy(); w = np.asarray(w, float).copy()
    lvl = [[i] for i in range(len(y))]
    i = 0
    while i < len(y) - 1:
        if y[i] <= y[i + 1] + 1e-12:
            i += 1
            continue
        wn = w[i] + w[i + 1]
        y[i] = (w[i] * y[i] + w[i + 1] * y[i + 1]) / wn if wn > 0 else y[i]
        w[i] = wn
        lvl[i] = lvl[i] + lvl[i + 1]
        y = np.delete(y, i + 1); w = np.delete(w, i + 1); del lvl[i + 1]
        if i > 0:
            i -= 1
    out = np.empty(sum(len(g) for g in lvl))
    for v, g in zip(y, lvl):
        out[g] = v
    return out
from scipy.ndimage import gaussian_filter

import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MG = MADGRAV_SCRATCH + "/MADGRAV1"
PF = f"{MG}/search_mode/pastro_final"
CAT = f"{MG}/figures/catalog_o3o4"
TAG = os.environ.get("MG_FIG_TAG", "x1cnnadopt48fveto")   # set by injhi_finalize.sh

C_ALL, C_CWB, C_MAD = "#d4d8dd", "#0072B2", "#D55E00"
# 16 log-spaced bins from 10 Msun: the unified inj_full campaign carries a low-mass
# stratum whose SOURCE-frame reach is ~8-20, so binning from 16 (the old floor, set by
# the stellar bank's 20 Msun detector-frame limit divided by 1+z) would hide the new
# coverage and keep excluding the sub-16 detections.
MEDGES = np.geomspace(10., 400., 17)
MIN_N = 40
XLO = 10.0           # axis left; the bank floors both components at 10 Msun
YBOT = 5.0           # axis bottom; vt_relabel_comoving drops the SNR-5 level for O3a/O3b,
                     # so the lowest MEASURED level is per-run and marked, not assumed

M2 = MADGRAV_SCRATCH + "/madgrav2"
real = json.load(open(os.environ.get("MG_HL_JSON", f"{M2}/figures_m2/hl_snr_o3o4_m2_crop96.json")))
# 2026-10-19: the star set is the CSV (96 catalogue rows), not the json flag; the json 'mad' flag is only cross-checked.
import csv
CSV = os.environ.get("MG_DET_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "detections_v2_96.csv"))
ALIAS = {"GW190412": "GW190412_053044", "GW190521": "GW190521_030229"}
_rows = [r for r in csv.DictReader(open(CSV)) if r["name"].startswith("GW")]      # non-catalogue rows (gps<...>) excluded
CSV_SET = {ALIAS.get(r["name"], r["name"]) for r in _rows}
N_RUN = {}
for r in _rows:
    N_RUN[r["run"].lower()] = N_RUN.get(r["run"].lower(), 0) + 1
_jmad = {n for n, v in real.items() if v["mad"]}
print(f"CSV catalogue rows: {len(CSV_SET)} per run {N_RUN}; json mad flags: {len(_jmad)}; "
      f"csv-not-json {sorted(CSV_SET - _jmad)} json-not-csv {sorted(_jmad - CSV_SET)}", flush=True)
for n, v in real.items():
    v["mad"] = n in CSV_SET
UNPLOTTABLE = {run: [n for n, v in real.items() if v["mad"] and v["run"].lower() == run
               and not (v.get("plotted", True) and v["coincident"] and v.get("mtot") is not None)] for run in ("o3a", "o3b", "o4a", "o4b")}
PENDING = set(filter(None, os.environ.get("MG_PENDING_RUNS", "").split(",")))   # nothing pending: all four runs frozen
real = {n: v for n, v in real.items() if v.get("plotted", True) and v["coincident"]}

plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.7,
                     "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), sharex=True, sharey=True)
SUPPORT, SUPPORT_LO, MEAS = {}, {}, {}        # measured SNR ceiling / floor, per-cell mask
fig.subplots_adjust(right=0.87, hspace=0.16, wspace=0.07)

mc = np.sqrt(MEDGES[:-1] * MEDGES[1:])          # log bin centres
for ax, run in zip(axes.ravel(), ("o3a", "o3b", "o4a", "o4b")):
    # MG_FIELD_DIR / MG_FIELD_TAG (2026-10-04): read the MADGRAV 2.2 injection field (m2/vt22: inj_scored_<run>_<tag>.npz +
    # relabel_inj_<run>_<tag>.npz in one directory) instead of the MADGRAV1 pastro_final field. Unset = MADGRAV1 field as before.
    _FD = os.environ.get("MG_FIELD_DIR", f"{M2}/vt22"); _FT = os.environ.get("MG_FIELD_TAG", "m22")   # 2.2 field, seed 20261004 (as figure_cap.sbatch)
    z = np.load(f"{_FD}/inj_scored_{run}_{_FT}.npz")
    # z["mtot"] is DETECTOR-frame (the bank bounds, 20-400). The real events are in
    # source-frame mass, so use the relabel's msrc and keep only relabelled entries --
    # median z ~ 0.3-0.37, so the two frames differ by ~30% at the median.
    rl = np.load(f"{_FD}/relabel_inj_{run}_{_FT}.npz")
    keep = rl["kept"]
    d, s, m = z["det_frac"][keep], z["net_snr"][keep], rl["msrc"][keep]
    # 2026-10-05: optional per-bank equal weighting (MG_BANK_EQUAL=1 + vt22/bank_tag_<run>_<tag>.npz from m2/vt22/bank_tags22.py).
    # The low-mass bank (10-22 Msun det-frame) is oversampled x10 and ~1.5x less detectable than the main bank at equal mass+SNR,
    # so the pooled mean printed the bank seam at 22 Msun as a shoulder at 25-35 Msun source frame in every panel.
    _bt = None
    if os.environ.get("MG_BANK_EQUAL", "0") == "1" and os.path.exists(f"{_FD}/bank_tag_{run}_{_FT}.npz"):
        _bt = np.load(f"{_FD}/bank_tag_{run}_{_FT}.npz")["tag"][keep]
    SNR = np.unique(s)                       # whatever levels this campaign actually has

    grid = np.full((len(SNR), len(mc)), np.nan)
    for i, sv in enumerate(SNR):
        for j, (a, b) in enumerate(zip(MEDGES[:-1], MEDGES[1:])):
            k = (s == sv) & (m >= a) & (m < b)
            if k.sum() >= MIN_N:
                if _bt is not None:
                    _means = [d[k & (_bt == t)].mean() for t in np.unique(_bt[k]) if (k & (_bt == t)).sum() >= MIN_N // 4]
                    grid[i, j] = float(np.mean(_means)) if _means else d[k].mean()
                else:
                    grid[i, j] = d[k].mean()
    # Support is judged PER CELL, not as a mass rectangle. A rectangle needed every SNR
    # level to clear MIN_N in a mass bin, which fails wherever one level is intrinsically
    # truncated: vt_relabel keeps an injection only if net_snr * c_inj >= RHO_TH, and c_inj
    # is source-dependent, so the lowest SNR row covers only part of the mass range. That
    # made O3a report zero support above Mtot 16 despite 122,850 injections. An event is
    # comparable iff ITS OWN (mass bin, SNR level) cell is measured.
    # isotonic fit down each mass column before interpolation (see isotonic() docstring)
    cnt = np.zeros_like(grid)
    for i, sv in enumerate(SNR):
        for j, (a, b) in enumerate(zip(MEDGES[:-1], MEDGES[1:])):
            cnt[i, j] = ((s == sv) & (m >= a) & (m < b)).sum()
    MEAS[run] = cnt >= MIN_N          # per-cell measured mask, before any gap filling
    for j in range(grid.shape[1]):
        col, wc = grid[:, j], cnt[:, j]
        ok = ~np.isnan(col)
        if ok.sum() > 1:
            grid[ok, j] = isotonic(col[ok], np.maximum(wc[ok], 1.0))

    # fill any gap along mass by nearest valid, so the interpolator has no holes
    for i in range(len(SNR)):
        row = grid[i]
        if np.isnan(row).all():
            grid[i] = 0.0
        elif np.isnan(row).any():
            ok = ~np.isnan(row)
            grid[i] = np.interp(np.arange(len(row)), np.flatnonzero(ok), row[ok])

    # bicubic interpolation in log-log, clamped at the measured envelope
    gx, gy = np.log10(mc), np.log10(SNR)
    YTOP = float(os.environ.get("MG_FIG_YTOP", 85)) or max(30.0, 1.12 * SNR[-1])
    itp = RegularGridInterpolator((gy, gx), grid, method="linear",
                                  bounds_error=False, fill_value=None)
    fx = np.linspace(np.log10(XLO), np.log10(MEDGES[-1]), 520)
    fy = np.linspace(np.log10(YBOT), np.log10(YTOP), 380)
    FX, FY = np.meshgrid(fx, fy)
    Z = itp(np.column_stack([FY.ravel(), FX.ravel()])).reshape(FX.shape)
    Z = np.clip(gaussian_filter(Z, sigma=(6, 8), mode="nearest"), 0, 1)
    # monotone guard on the extrapolated band: above the top measured SNR the field
    # may only rise, never dip below what was actually measured at SNR 25
    itop = int(np.searchsorted(fy, np.log10(SNR[-1])))
    if itop < len(fy):
        Z[itop:] = np.maximum.accumulate(np.maximum(Z[itop:], Z[itop - 1]), axis=0)
    ibot = int(np.searchsorted(fy, np.log10(SNR[0])))          # and may only fall below it
    if ibot > 0:
        Z[:ibot] = np.minimum.accumulate(
            np.minimum(Z[:ibot], Z[ibot])[::-1], axis=0)[::-1]

    cf = ax.contourf(10 ** fx, 10 ** fy, Z, levels=np.linspace(0, 1, 61),
                     cmap="viridis", vmin=0, vmax=1, zorder=1, antialiased=True)
    cf.set_edgecolor("face")                      # kill contour seams in the PDF
    ax.contour(10 ** fx, 10 ** fy, Z, levels=[0.5], colors="white",
               linewidths=1.3, zorder=2)
    ax.contour(10 ** fx, 10 ** fy, Z, levels=[0.9], colors="white",
               linewidths=1.0, linestyles="dashed", zorder=2)

    R = {n: v for n, v in real.items() if v["run"].lower() == run}
    ax.scatter([v["mtot"] for v in R.values()], [v["snr_hl"] for v in R.values()],
               marker="o", s=12, facecolor=C_ALL, edgecolor="#6b7075",
               linewidths=0.35, alpha=0.9, zorder=3, label="GWTC confident")
    cw = [v for v in R.values() if v["cwb"]]
    ax.scatter([v["mtot"] for v in cw], [v["snr_hl"] for v in cw], marker="s", s=30,
               facecolor="none", edgecolors=C_CWB, linewidths=1.05, zorder=4, label="cWB")
    md = [v for v in R.values() if v["mad"]]
    # SM_FIG_NOMAD=1: talk build -- draw the population and the recovery field only, no
    # MADGRAV stars (the detections are revealed on the next slide). Counts still printed.
    NOMAD = os.environ.get("SM_FIG_NOMAD") == "1" or run in PENDING
    if run in PENDING:
        ax.text(0.5, 0.5, "MADGRAV2 o4a\nstill running", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="white", zorder=7,
                bbox=dict(fc="#3b3b3b", ec="none", alpha=0.55, pad=6))
    if not NOMAD:
        ax.scatter([v["mtot"] for v in md], [v["snr_hl"] for v in md], marker="*", s=105,
                   facecolor=C_MAD, edgecolors="white", linewidths=0.7, zorder=5,
                   label=os.environ.get("MG_FIG_LABEL", "MADGRAV 2 version 2 (96)"))
        print(f"[{run}] stars drawn: {len(md)}" + (f"  (+{len(UNPLOTTABLE[run])} without published mass, not drawable: {UNPLOTTABLE[run]})" if UNPLOTTABLE[run] else ""), flush=True)

    if UNPLOTTABLE[run]:
        ax.text(0.015, 0.035, f"+{len(UNPLOTTABLE[run])} detection without published mass (not drawn)", transform=ax.transAxes,
                ha="left", va="center", fontsize=6.8, color="#3b3b3b", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))
    # Events louder than the axis top are drawn ON the top edge with an up-caret and
    # counted, so no detection is silently dropped (GW250114_082203 sits at SNR 78.6).
    off = sorted((v for v in R.values() if v["snr_hl"] > YTOP), key=lambda v: -v["snr_hl"])
    if off:
        yo = YTOP * 0.955
        if NOMAD:
            ax.scatter([v["mtot"] for v in off], [yo] * len(off), marker="o", s=12,
                       facecolor=C_ALL, edgecolor="#6b7075", linewidths=0.35, alpha=0.9, zorder=6)
        else:
            ax.scatter([v["mtot"] for v in off], [yo] * len(off), marker="*", s=105,
                       facecolor=[C_MAD if v["mad"] else C_ALL for v in off],
                       edgecolors="white", linewidths=0.7, zorder=6)
        ax.scatter([v["mtot"] for v in off], [yo * 1.045] * len(off), marker="^", s=16,
                   color="#3b3b3b", zorder=6)
        lab = (f"{len(off)} above scale, to SNR {off[0]['snr_hl']:.0f}"
               if len(off) > 1 else f"SNR {off[0]['snr_hl']:.0f}")
        ax.text(0.985, 0.035, lab, transform=ax.transAxes, ha="right", va="center",
                fontsize=6.8, color="#3b3b3b", zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))

    # "found" and "predicted" MUST be counted over the same event set, otherwise the
    # title compares a total over all detections against a sum over supported cells only.
    pred = nsup = 0; found = 0
    # Support means the SAME thing here as the drawn dashed boundary: mass inside MSUP
    # (every SNR level clears MIN_N) AND the cell measured. Using the gap-filled grid
    # instead would count events the figure's own boundary declares unsupported.
    for v in R.values():
        if not (MEDGES[0] <= v["mtot"] < MEDGES[-1]):
            continue
        j = int(np.digitize(v["mtot"], MEDGES) - 1)
        i = int(np.abs(SNR - min(v["snr_hl"], SNR[-1])).argmin())
        if MEAS[run][i, j] and not np.isnan(grid[i, j]):
            pred += grid[i, j]; nsup += 1
            found += bool(v["mad"])
    n_excl = len(md) - found            # adopted detections outside supported cells

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(XLO, 400); ax.set_ylim(YBOT, YTOP)
    if _bt is not None:                                   # mark the bank seam (22 Msun det-frame -> source frame at the local median z)
        _zs = rl["z"][keep][(z["mtot"][keep] >= 21) & (z["mtot"][keep] < 23)]
        _seam = 22.0 / (1.0 + (float(np.median(_zs)) if len(_zs) else 0.1))
        ax.axvline(_seam, color="white", lw=0.8, ls=(0, (1, 2.5)), alpha=0.85)
        ax.text(_seam * 1.04, YBOT * 1.12, "bank seam", color="white", fontsize=6.5, rotation=90, va="bottom", alpha=0.9)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks([10, 15, 20, 30, 50, 100, 200, 400])
    ax.set_yticks([t for t in (5, 6, 8, 10, 15, 20, 25, 30, 40, 60, 80, 100) if t <= YTOP])
    SUPPORT[run] = SNR[-1]; SUPPORT_LO[run] = SNR[0]
    ax.tick_params(which="both", direction="out", labelsize=8.5, color="#9aa0a6")   # outward ticks on the white page: must not be white
    _x = f"  +{n_excl} unsupp." if n_excl else ""
    _rl = run.upper()[:2] + run.upper()[2:].lower()      # O3a, not O3A (paper style)
    # SM_FIG_NOTITLE=1: run label only, for the paper build -- the found/predicted
    # counts live in tab:cmp there. They are still printed below either way.
    n_v2 = N_RUN.get(run, 0)
    print(f"[{_rl}] native: {found} found / {pred:.1f} predicted  of {nsup}{_x}  | title uses N = version-2 count {n_v2}", flush=True)
    print(f"[{_rl}] {found} found / {pred:.1f} predicted  of {nsup}{_x}"
          + ("   (MADGRAV2 pending)" if run in PENDING else ""), flush=True)
    if os.environ.get("SM_FIG_NOTITLE") == "1":
        ax.set_title(_rl, fontsize=9.5, pad=5)
    else:
        ax.set_title((f"{_rl}   pending" if run in PENDING else
                      f"{_rl}   {n_v2} found / {pred:.1f} predicted  of {nsup}"),
                     fontsize=9.5, pad=5)
    for sp in ax.spines.values():
        sp.set_color("#9aa0a6")

for ax in axes[1]:
    ax.set_xlabel(r"Source-frame total mass  $M_{\mathrm{tot}}$  [$M_\odot$]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"H1+L1 network SNR")
for _ax in axes.ravel():
    _ax.plot([], [], color="#3b3b3b", lw=1.3, label="50% recovery")
    _ax.plot([], [], color="#3b3b3b", lw=1.0, ls="dashed", label="90% recovery")
    break
# upper centre: the high-recovery plateau carries no markers, so the legend covers no data
# MG_FIG_LEGLOC: "upper center" (default, anchored) or any matplotlib loc string, e.g. "upper left"
_legloc = os.environ.get("MG_FIG_LEGLOC", "upper left")
_legkw = dict(bbox_to_anchor=(0.5, 0.99)) if _legloc == "upper center" else {}
_h, _l = axes[0, 0].get_legend_handles_labels()
leg = axes[0, 0].legend(_h, _l, loc=_legloc, ncol=1, fontsize=7.5, framealpha=0.92,
                        facecolor="white", edgecolor="#d0d3d7", borderpad=0.45,
                        handletextpad=0.4, **_legkw)
leg.set_zorder(6)

cax = fig.add_axes([0.89, 0.13, 0.019, 0.74])
cb = fig.colorbar(cf, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1.0])
cb.set_label(os.environ.get("MG_FIG_CBLABEL", "injection recovery probability at FAR < 1/yr\n(MADGRAV 2.2 campaign, seed 20261004)"), fontsize=8.5)
cb.ax.tick_params(labelsize=8.5)
cb.solids.set_edgecolor("face")

out = os.environ.get("MG_FIG_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2_recovery_plane_v2_96"))
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print(f"wrote {out}.png/.pdf")

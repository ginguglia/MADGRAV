#!/usr/bin/env python
"""Cumulative inverse-FAR ("loudest events") plot for the adopted MADGRAV statistic.

For every CNN-gate-passing zero-lag candidate of the four main runs (the as-run triggers of
blindscan.json with max(s_HM, s_LM) > 0.5), recompute the ADOPTED false-alarm rate:

    lnLambda channel only (SM_LR_ONLY=1), sigma_net < 10.6 upper veto on candidates and background
    (SM_NETMAX=10.6), glitch-arm gate g_net >= threshold on candidates and background (SM_GNET),
    foreground-excluded background (pairs within +/-4 s of any detection removed), per-run trials
    factor K from the null calibration (ke_gnet.json):

        FAR = K * N_excl / T_bg ,   UL90 = K * (N_excl + 2.3026) / T_bg   (N_excl = 0 -> upper limit only)

The counting is the exact code of inclusive_exclusive_far.py; before anything is drawn the script asserts
that it reproduces every row of far_lronly_g106_gnet.csv (the published 48) bit-for-bit.

The figure then shows, per run and combined, the number of zero-lag candidates with inverse FAR >= x
against the background expectation N_bg(x) = T_fg / x (T_fg = analysed zero-lag time, same convention as
pastro_final.py) with its central 1, 2, 3 sigma Poisson bands. Candidates with N_excl = 0 have only a
lower limit on IFAR and are drawn as right-pointing arrows at 1/UL90.

Run with the adopted environment:
    SM_LR_ONLY=1 SM_NETMAX=10.6 SM_GNET=-4.02 python search_mode/ifar_plot.py
Outputs: figures/ifar_o3o4_gnet.{pdf,png}, figures/ifar_combined_gnet.{pdf,png} (all in-search candidates),
         figures/ifar_combined_gnet_48.{pdf,png} (only the 48 candidates of the published table),
         figures/catalog_o3o4/ifar_candidates_gnet.csv
"""
import csv, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson

import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

sys.path.insert(0, MADGRAV_SCRATCH + "/MADGRAV1/search_mode")
import successor_stat as S
import inclusive_exclusive_far as IEF

MG = MADGRAV_SCRATCH + "/MADGRAV1"
DET = f"{MG}/details/successor_statistic"
SC = MADGRAV_SCRATCH
FIG = f"{MG}/figures"
REF_CSV = os.environ.get("SM_FAR_LR_CSV", f"{MG}/figures/catalog_o3o4/far_lronly_g106_gnet.csv")
KE = json.load(open(os.environ.get("SM_KE_JSON", f"{DET}/ke_gnet.json")))
DET_CSV = f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"
OUT_CSV = f"{MG}/figures/catalog_o3o4/ifar_candidates_gnet.csv"
NETMAX = IEF.NETMAX
assert IEF.LR_ONLY and NETMAX is not None and S.GNET is not None, \
    "run with SM_LR_ONLY=1 SM_NETMAX=10.6 SM_GNET=<threshold> (the adopted configuration)"
RUNS = ["O3a", "O3b", "O4a", "O4b"]
COL = {"O3a": "#1f77b4", "O3b": "#2ca02c", "O4a": "#e6a817", "O4b": "#d62728"}


def t_fg(bg, run):
    segj = {s[3]: s[2] for s in json.load(open(f"{SC}/{run.lower()}_full_coincident.json"))["segments"]}
    return sum(segj.get(n, 0.0) for n in bg.seg_names) * 0.5 / 3.1557e7     # ANALYZED_FRAC, as pastro_final.py


def main():
    ref = {(r["run"], r["name"]): r for r in csv.DictReader(open(REF_CSV))}
    dets = [r for r in csv.DictReader(open(DET_CSV))]
    rows, T = [], {}
    for run in RUNS:
        bg = S.Background(run, f"{DET}/bg_veto_{run.lower()}.npz", verbose=False)
        T[run] = t_fg(bg, run)
        # foreground-excluded mask: identical construction to inclusive_exclusive_far.main()
        touch = np.zeros(len(bg.ll), bool)
        for r in [d for d in dets if d["run"] == run]:
            Sx = bg.seg_ix[r["seg"]]; g = float(r["gps"])
            touch |= (bg.hseg == Sx) & (np.abs(bg.gpsH - g) <= 4.0)
            touch |= (bg.lseg == Sx) & (np.abs(bg.gpsL - g) <= 4.0)
        trig = json.load(open(f"{S.RUNS[run]}/blindscan.json"))["triggers"]
        cands = [t for t in trig if t.get("cnn_hm") is not None and max(t["cnn_hm"], t["cnn_lm"]) > S.GATE]
        K = KE[run.lower()]
        print(f"[ifar:{run}] {len(cands)} gate-passing zero-lag candidates, T_fg={T[run]:.4f} yr, K={K}", flush=True)
        for t in cands:
            c = dict(seg=t["seg"], loglr=float(t["loglr"]), net=float(t["net"]), cnn_hm=float(t["cnn_hm"]), cnn_lm=float(t["cnn_lm"]))
            gn, gH, gL = IEF.cand_gnet(run, t["seg"], float(t["gps"]))
            gpass = bool(gn >= S.GNET)
            netpass = bool(c["net"] < NETMAX)
            fi, ui, Ni = IEF.far_of(*IEF.counts(bg, c, None))
            fe, ue, Ne = IEF.far_of(*IEF.counts(bg, c, touch))
            name = t.get("matches_known") or ""
            det_row = next((d for d in dets if d["run"] == run and d["seg"] == t["seg"] and abs(float(d["gps"]) - float(t["gps"])) <= 2.0), None)
            if det_row is not None:
                name = det_row["name"]
                k = (run, name)
                assert k in ref, k
                for a, b in (("far_incl", fi), ("far_excl", fe), ("ul90_incl", ui), ("ul90_excl", ue)):
                    assert abs(float(ref[k][a]) - b) <= 1e-9 * max(1.0, abs(b)), (k, a, ref[k][a], b)
                assert int(float(ref[k]["gnet_pass"])) == int(gpass), (k, "gnet_pass")
            far, ul = K * fe, K * ue
            rows.append(dict(run=run, name=name, gps=float(t["gps"]), seg=t["seg"], loglr=c["loglr"], net=c["net"],
                             cnn_hm=c["cnn_hm"], cnn_lm=c["cnn_lm"], gnet=gn, gnet_pass=int(gpass), netmax_pass=int(netpass),
                             N_incl=Ni, N_excl=Ne, K=K, far_incl_cal=K * fi, far_cal=far, ul90_cal=ul,
                             ifar_yr=(1.0 / far if far > 0 else float("inf")), ifar_lower_yr=1.0 / ul,
                             in_search=int(gpass and netpass), detection=int(gpass and netpass and far < 1.0 and ul < 1.0),
                             published=int(det_row is not None)))
    n_ref = sum(r["published"] for r in rows)
    assert n_ref == len(ref), (n_ref, len(ref))
    print(f"GATE PASS: all {n_ref} published FARs and gate flags reproduced exactly", flush=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ------------------------------------------------------------------ figure
    def panel(ax, sel, Tfg, title):
        ifar = np.array([r["ifar_yr"] if r["N_excl"] > 0 else r["ifar_lower_yr"] for r in sel])
        lim = np.array([r["N_excl"] == 0 for r in sel])
        xs = np.logspace(-3.5, 4.0, 600)
        mu = Tfg / xs
        for k, a in ((3, 0.10), (2, 0.18), (1, 0.28)):
            p = 0.5 * (1 + math.erf(k / np.sqrt(2)))
            lo, hi = poisson.ppf(1 - p, mu), poisson.ppf(p, mu)
            ax.fill_between(xs, np.maximum(lo, 0.3), np.maximum(hi, 0.3), color="0.4", alpha=a, lw=0, label=f"{k}$\\sigma$ Poisson band")
        ax.plot(xs, np.maximum(mu, 0.3), "k--", lw=1.2, label="background $T_{\\rm fg}/x$")
        order = np.argsort(-ifar); xi = ifar[order]; y = np.arange(1, len(xi) + 1)
        ax.step(np.r_[xi, 1e-4], np.r_[y, y[-1]] if len(y) else [0.3, 0.3], where="post", color="k", lw=1.0, alpha=0.6)
        runs = np.array([r["run"] for r in sel])
        for run in RUNS:
            m = (runs[order] == run) & ~lim[order]
            if m.any():
                ax.plot(xi[m], y[m], "o", ms=5, color=COL[run], mec="k", mew=0.4, label=f"{run} candidates" if title == "all runs" else None)
            m = (runs[order] == run) & lim[order]
            if m.any():
                ax.plot(xi[m], y[m], ">", ms=6, color=COL[run], mec="k", mew=0.4)
        ax.axvline(1.0, color="0.6", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-2, 3e3); ax.set_ylim(0.3, max(60, 3 * len(sel) if sel else 60))
        nd = sum(r["detection"] for r in sel)
        ax.set_title(f"{title}: {len(sel)} candidates, {nd} at FAR$<1\\,$yr$^{{-1}}$ ($T_{{\\rm fg}}$={Tfg:.3f} yr)", fontsize=10)
        ax.set_xlabel("inverse false-alarm rate $x$ [yr]"); ax.set_ylabel("candidates with IFAR $\\geq x$")
        ax.grid(alpha=0.25, which="both", lw=0.4)

    used = [r for r in rows if r["in_search"]]
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, run in zip(axs.ravel(), RUNS):
        panel(ax, [r for r in used if r["run"] == run], T[run], run)
    axs[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("MADGRAV zero-lag candidates vs time-slide background expectation (adopted statistic)\n"
                 "points: candidates passing all gates; arrows: N=0, IFAR lower limit at 1/UL90; bands: Poisson 1, 2, 3$\\sigma$", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/ifar_o3o4_gnet.{ext}", dpi=170)
    fig2, ax = plt.subplots(figsize=(6.4, 5.0))
    panel(ax, used, sum(T.values()), "all runs")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(f"{FIG}/ifar_combined_gnet.{ext}", dpi=170)
    # combined figure restricted to the 48 candidates of the published table (far_lronly_g106_gnet.csv)
    pub = [r for r in used if r["published"]]
    fig3, ax = plt.subplots(figsize=(6.4, 5.0))
    panel(ax, pub, sum(T.values()), "all runs")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig3.tight_layout()
    for ext in ("pdf", "png"):
        fig3.savefig(f"{FIG}/ifar_combined_gnet_48.{ext}", dpi=170)
    # summary to stdout
    for run in RUNS + ["all"]:
        sel = used if run == "all" else [r for r in used if r["run"] == run]
        Tf = sum(T.values()) if run == "all" else T[run]
        sub = [r for r in sel if not r["detection"]]
        print(f"{run:4s} in-search {len(sel):3d}  detections {sum(r['detection'] for r in sel):3d}  N=0 {sum(r['N_excl']==0 for r in sel):3d}  "
              f"sub-threshold {len(sub):2d} (expected noise above IFAR=1yr: {Tf:.3f}; observed FAR<1: {sum(r['far_cal']<1 for r in sel)})")
    vet = [r for r in rows if not r["in_search"]]
    print(f"removed before the plot: {sum(1 for r in vet if not r['netmax_pass'])} by sigma_net>=10.6, {sum(1 for r in vet if r['netmax_pass'] and not r['gnet_pass'])} by the glitch-arm gate")


if __name__ == "__main__":
    main()

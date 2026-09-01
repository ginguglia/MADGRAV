#!/usr/bin/env python
"""Fig. 3/4/5 self-consistency (banked 2026-08-15; run by paper_search/build_paper.sh).
Verifies, from the JSON products the figures were drawn from, that
  F1  Fig.3 (vt_paper_numbers.json) per-run == Fig.5 (vt_compare_pipelines.json) per-run,
      values AND mask; both == vt_relabel_comoving.json:vt_comoving_srcframe_gpc3yr on unmasked bins;
  F2  Fig.3 O3 sum / total == Fig.5 vt_o3 / vt_total (mask propagated: any component masked -> point masked);
  F3  Fig.4 (vt_fourepoch_ratio.json) ratio x pipeline VT (vt_pipelines_target_zc.json) == the same numerator
      on every plotted bin, and every plotted point has pipeline N_eff >= 300;
  F4  the numerator-support mask (from vt_compare_pipelines.json neff_srcframe.support < 300) is a subset of
      Fig.4's unplotted set (no plotted Fig.4 point sits on a support-starved numerator bin);
  F5  identical mass edges, cosmology string, per-run T_obs across the three products.
Exit 1 on any failure (build_paper.sh treats that as a hard stop).
"""
import json, sys, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

H = MADGRAV_ROOT + "/search_mode/pastro_final"
FE = MADGRAV_ROOT + "/figures/vt_fourepoch/vt_fourepoch_ratio.json"
RUNS = ("O3a", "O3b", "O4a", "O4b"); NEFF = 300.0
rel = json.load(open(f"{H}/vt_relabel_comoving.json")); paper = json.load(open(f"{H}/vt_paper_numbers.json"))
cmp_ = json.load(open(f"{H}/vt_compare_pipelines.json")); fe = json.load(open(FE)); pl = json.load(open(f"{H}/vt_pipelines_target_zc.json"))
f = lambda a: np.array([np.nan if v is None else v for v in a], float)
fails = []
def chk(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond: fails.append(msg)
print("[F5] conventions")
chk(rel["mass_edges"] == cmp_["mass_edges"] == pl["mass_edges"], "mass edges identical (rel/cmp/pipelines)")
chk(rel["cosmology"] == cmp_.get("cosmology"), f"cosmology string identical: {rel['cosmology']}")
for r in RUNS:
    chk(abs(paper["T_obs_yr"][r] - cmp_["runs"][r]["T_obs_yr"]) < 1e-12 and abs(rel["runs"][r]["T_obs_yr"] - paper["T_obs_yr"][r]) < 1e-12
        and abs(pl["runs"][r]["T_ours_yr"] - paper["T_obs_yr"][r]) < 1e-9, f"T_obs {r} identical across Fig3/Fig5/relabel/pipelines ({paper['T_obs_yr'][r]:.6f})")
print("[F1] Fig.3 == Fig.5 per run == numerator")
support_mask = {}
for r in RUNS:
    num = f(rel["runs"][r]["vt_comoving_srcframe_gpc3yr"]); p3 = f(paper["per_run_vt"][r]); p5 = f(cmp_["runs"][r]["vt_gpc3yr"])
    m = np.isnan(p3); support_mask[r] = np.array(cmp_["neff_srcframe"][r]["support"], float) < NEFF
    chk(np.array_equal(m, np.isnan(p5)) and np.allclose(p3[~m], p5[~m]), f"{r}: Fig3 == Fig5 (values+mask), masked bins {np.where(m)[0].tolist()}")
    chk(np.allclose(p3[~m], num[~m]), f"{r}: unmasked Fig3 == relabel numerator")
    chk(np.array_equal(m, support_mask[r]), f"{r}: mask == support N_eff<{NEFF:.0f} rule")
print("[F2] summed curves")
chk(np.array_equal(np.isnan(f(paper["o3"])), np.isnan(f(cmp_["vt_o3_gpc3yr"]))) and np.allclose(np.nan_to_num(f(paper["o3"])), np.nan_to_num(f(cmp_["vt_o3_gpc3yr"]))), "Fig3 O3 == Fig5 vt_o3")
chk(np.array_equal(np.isnan(f(paper["total"])), np.isnan(f(cmp_["vt_total_gpc3yr"]))) and np.allclose(np.nan_to_num(f(paper["total"])), np.nan_to_num(f(cmp_["vt_total_gpc3yr"]))), "Fig3 total == Fig5 vt_total")
anym = np.any([np.isnan(f(paper["per_run_vt"][r])) for r in RUNS], axis=0)
chk(np.array_equal(np.isnan(f(paper["total"])), anym), "total masked exactly where any component masked")
print("[F3/F4] Fig.4 ratios reconstruct the numerator; N_eff; support subset")
for r in RUNS:
    num = f(rel["runs"][r]["vt_comoving_srcframe_gpc3yr"])
    for pipe, rat in fe["ratios"][r].items():
        if "band" in pipe: continue
        P = pl["runs"][r]["pipelines"][pipe]; v = f(P["vt_gpc3yr"]); neff = f(P["neff"]); rat = f(rat); sel = np.isfinite(rat)
        chk(np.allclose(rat[sel] * v[sel], num[sel], rtol=1e-6), f"{r} {pipe}: ratio x VT_pipe == numerator on {sel.sum()} bins")
        chk(np.all(neff[sel] >= NEFF), f"{r} {pipe}: all plotted points N_eff>={NEFF:.0f}")
        chk(not np.any(sel & support_mask[r]), f"{r} {pipe}: no plotted point on a support-starved numerator bin")
print(f"[check_fig_consistency] {'ALL PASS' if not fails else f'{len(fails)} FAILURE(S)'}")
sys.exit(1 if fails else 0)

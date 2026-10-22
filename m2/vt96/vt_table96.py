"""vt96: VT table in the format of m2/vt22/VT_TABLE_M22_20261005.md with three pipelines (MADGRAV1 adopted, 2.2, version 2 = 96 recipe)
and the ratios 96/M1 and 96/2.2 per bin, per run and in total (20-400 Msun). Inputs: the three vt_relabel_comoving JSONs. Output: VT_TABLE_M96_<date>.md."""
import json, datetime, os, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

REL96 = os.environ.get("VT96_JSON", MADGRAV_SCRATCH + "/madgrav2/vt96/vt_relabel_comoving_m96.json")
REL22 = MADGRAV_SCRATCH + "/madgrav2/vt22/vt_relabel_comoving_m22.json"
REL1 = MADGRAV_SCRATCH + "/MADGRAV1/search_mode/pastro_final/vt_relabel_comoving_x1cnnadopt48fveto.json"
OUT = os.environ.get("VT96_TABLE", f"{MADGRAV_ROOT}/m2/vt96/VT_TABLE_M96_{datetime.date.today():%Y%m%d}.md")
RUNS = ("O3a", "O3b", "O4a", "O4b")
def load(p):
    d = json.load(open(p)); e = np.array(d["mass_edges"], float)
    vt = {r: np.array([np.nan if x is None else x for x in d["runs"][r]["vt_comoving_srcframe_gpc3yr"]], float) for r in RUNS}
    return d, e, vt
d96, e96, v96 = load(REL96); d22, e22, v22 = load(REL22); d1, e1, v1 = load(REL1)
assert np.array_equal(e96, e22), (e96, e22)
def to_m1(v, e):   # fine bins (10,15,20,30,40,...) -> MADGRAV1 bins (20,40,60,...): drop <20, sum 20-30 + 30-40
    i20 = int(np.where(e == 20.0)[0][0]); i40 = int(np.where(e == 40.0)[0][0]); return np.concatenate([[np.nansum(v[i20:i40])], v[i40:]])
lab = lambda e: [f"{int(a)}-{int(b)}" for a, b in zip(e[:-1], e[1:])]
def rt(a, b): return "nan" if not (np.isfinite(a) and np.isfinite(b) and b > 0) else f"{a/b:.2f}"
L = [f"# MADGRAV 2 version 2 (frozen 96 recipe) sensitive volume-time, comoving, source-frame mass bins ({datetime.date.today()})", "",
     "Field: m2/vt96 campaign (seed 20261004, paired injection-by-injection with the 2.2 field), frozen version-2 recipe (net floor 2.75 + zero-lag gates, _off5 arm -2.05, "
     "coincident-crop CNN/S3 gate, S3 0.5357 + HGB ranker, LM rule 0.0382, local-ASD veto with the plateau rule (feature clause ON in O3, OFF in O4; net clause everywhere), "
     "net-sigma guard (injection form: raw loglr > 0 AND lm > 0.5), loglr cap bypass); detection = FAR < 1/yr (point and UL90). No null-calibration K factor. "
     "Relabel: vt_relabel_comoving22.py, rho_th = 5, FlatLambdaCDM(H0=67.9, Om0=0.3065), Vmax=int dVc/(1+z), fixed2 ASD (SM_VT_FIXED_ASD=1, like-for-like with both references).",
     f"References: 2.2 = {REL22} (VT_TABLE_M22_20261005.md); MADGRAV1 = {REL1} (adopted 47-detection pipeline, bins from 20 Msun).", "",
     "VT in Gpc^3 yr (comoving, source frame). Ratios on the MADGRAV1 bins (fine bins 20-30 and 30-40 summed).", ""]
tot96 = np.zeros(len(e1) - 1); tot22 = np.zeros(len(e1) - 1); tot1 = np.zeros(len(e1) - 1); per_run = []
for r in RUNS:
    R96 = d96["runs"][r]; R22 = d22["runs"][r]; R1 = d1["runs"][r]
    L.append(f"## {r}   T_obs 96 = {R96['T_obs_yr']:.4f} yr, n_inj {R96['n_inj']}, kept_frac mean {np.nanmean(R96['kept_frac']):.4f}; 2.2 T_obs = {R22['T_obs_yr']:.4f} yr; MADGRAV1 T_obs = {R1['T_obs_yr']:.4f} yr"); L.append("")
    L.append("| Mtot bin [Msun] | VT 96 (fine bins) | eff 96 | VT 2.2 (fine bins) | eff 2.2 | 96/2.2 |"); L.append("|---|---|---|---|---|---|")
    for j, b in enumerate(lab(e96)):
        L.append(f"| {b} | {v96[r][j]:.4f} | {R96['eff_covered'][j]:.3f} | {v22[r][j]:.4f} | {R22['eff_covered'][j]:.3f} | {rt(v96[r][j], v22[r][j])} |")
    L.append(""); m96 = to_m1(v96[r], e96); m22 = to_m1(v22[r], e22); m1 = v1[r]; tot96 += m96; tot22 += m22; tot1 += m1
    L.append("| Mtot bin [Msun] | VT 96 | VT 2.2 | VT MADGRAV1 | 96/M1 | 96/2.2 |"); L.append("|---|---|---|---|---|---|")
    for j, b in enumerate(lab(e1)): L.append(f"| {b} | {m96[j]:.4f} | {m22[j]:.4f} | {m1[j]:.4f} | {rt(m96[j], m1[j])} | {rt(m96[j], m22[j])} |")
    s96, s22, s1 = np.nansum(m96), np.nansum(m22), np.nansum(m1); per_run.append((r, s96, s22, s1))
    L.append(f"| **20-400 total** | **{s96:.4f}** | **{s22:.4f}** | **{s1:.4f}** | **{rt(s96, s1)}** | **{rt(s96, s22)}** |"); L.append("")
L.append("## Sum over the four runs (MADGRAV1 bins)"); L.append("")
L.append("| Mtot bin [Msun] | VT 96 | VT 2.2 | VT MADGRAV1 | 96/M1 | 96/2.2 |"); L.append("|---|---|---|---|---|---|")
for j, b in enumerate(lab(e1)): L.append(f"| {b} | {tot96[j]:.4f} | {tot22[j]:.4f} | {tot1[j]:.4f} | {rt(tot96[j], tot1[j])} | {rt(tot96[j], tot22[j])} |")
L.append("")
L.append(f"Total VT 20-400 Msun, four runs: 96 recipe = {tot96.sum():.3f} Gpc^3 yr, 2.2 = {tot22.sum():.3f} Gpc^3 yr, MADGRAV1 = {tot1.sum():.3f} Gpc^3 yr; "
         f"ratios 96/M1 = {tot96.sum()/tot1.sum():.2f}, 96/2.2 = {tot96.sum()/tot22.sum():.2f}, 2.2/M1 = {tot22.sum()/tot1.sum():.2f}.")
L.append("Per run 20-400 Msun (96/M1, 96/2.2): " + "; ".join(f"{r} {s96:.3f} vs {s22:.3f} vs {s1:.3f} ({rt(s96, s1)}, {rt(s96, s22)})" for r, s96, s22, s1 in per_run) + ".")
L.append("")
L.append("Caveats: bins below ~25 Msun straddle the two-bank seam (m2/vt22/bank_tags22.py). The high-mass bins (>260 Msun) rest on MADGRAV1 VT of 0.27 and 0.002 Gpc^3 yr "
         "(a handful of recovered injections); quote them as 'MADGRAV1 had essentially no reach there', not as a factor. The 96 recipe carries the local-ASD veto and the "
         "injection form of the net-sigma guard (no Grad-CAM-crop clause for injections: conservative); the 2.2 field carried neither. No null-calibration K factor in either.")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-4:])); print(f"[table] -> {OUT}")

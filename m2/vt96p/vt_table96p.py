"""vt96p: VT table of the frozen 96 recipe at the search's own placement (vt96p) beside the grid-placement field (vt96), MADGRAV1 and the 2.2
total; ratios 96p/96 and 96p/M1 per bin, per run and in total (20-400 Msun). Format of m2/vt96/vt_table96.py. Output: VT_TABLE_M96P_<date>.md."""
import json, datetime, os, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

RELP = os.environ.get("VT96P_JSON", MADGRAV_SCRATCH + "/madgrav2/vt96/vt_relabel_comoving_m96p.json")
REL96 = MADGRAV_SCRATCH + "/madgrav2/vt96/vt_relabel_comoving_m96.json"
REL22 = MADGRAV_SCRATCH + "/madgrav2/vt22/vt_relabel_comoving_m22.json"
REL1 = MADGRAV_SCRATCH + "/MADGRAV1/search_mode/pastro_final/vt_relabel_comoving_x1cnnadopt48fveto.json"
OUT = os.environ.get("VT96P_TABLE", f"{MADGRAV_ROOT}/m2/vt96p/VT_TABLE_M96P_{datetime.date.today():%Y%m%d}.md")
RUNS = ("O3a", "O3b", "O4a", "O4b")
def load(p):
    d = json.load(open(p)); e = np.array(d["mass_edges"], float)
    vt = {r: np.array([np.nan if x is None else x for x in d["runs"][r]["vt_comoving_srcframe_gpc3yr"]], float) for r in RUNS}
    return d, e, vt
dP, eP, vP = load(RELP); d96, e96, v96 = load(REL96); d22, e22, v22 = load(REL22); d1, e1, v1 = load(REL1)
assert np.array_equal(eP, e96) and np.array_equal(e96, e22), (eP, e96, e22)
def to_m1(v, e):   # fine bins (10,15,20,30,40,...) -> MADGRAV1 bins (20,40,60,...): drop <20, sum 20-30 + 30-40
    i20 = int(np.where(e == 20.0)[0][0]); i40 = int(np.where(e == 40.0)[0][0]); return np.concatenate([[np.nansum(v[i20:i40])], v[i40:]])
lab = lambda e: [f"{int(a)}-{int(b)}" for a, b in zip(e[:-1], e[1:])]
def rt(a, b): return "nan" if not (np.isfinite(a) and np.isfinite(b) and b > 0) else f"{a/b:.2f}"
L = [f"# MADGRAV 2 version 2 (frozen 96 recipe) sensitive volume-time at the search's own placement -- vt96p ({datetime.date.today()})", "",
     "Field: m2/vt96p campaign = the vt96 field (seed 20261004, paired injection-by-injection with vt96 and the 2.2 field) re-injected with SM_INJ_PLACE=prod: the merger sits at a "
     "uniform offset in [-0.5, 0.5) s from the centre of a tile of the fixed 1 s search grid and the scored windows are that tile and its neighbour on the merger's side "
     "(peak = argmax net over the two = the search's cluster representative). vt96/vt22 scored three 0.25 s-spaced windows around the merger (argmax net), which flatters the "
     "frozen 1 s search near the net floors. Recipe, scoring (far96.py) and relabel (vt_relabel_comoving22.py, rho_th = 5, fixed2 ASD) identical to vt96; detection = FAR < 1/yr; no K factor.",
     f"References: vt96 = {REL96}; 2.2 = {REL22}; MADGRAV1 = {REL1} (adopted 47-detection pipeline, bins from 20 Msun).", "",
     "VT in Gpc^3 yr (comoving, source frame). Ratios on the MADGRAV1 bins (fine bins 20-30 and 30-40 summed).", ""]
totP = np.zeros(len(e1) - 1); tot96 = np.zeros(len(e1) - 1); tot22 = np.zeros(len(e1) - 1); tot1 = np.zeros(len(e1) - 1); per_run = []
for r in RUNS:
    RP = dP["runs"][r]; R96 = d96["runs"][r]; R1 = d1["runs"][r]
    L.append(f"## {r}   T_obs = {RP['T_obs_yr']:.4f} yr, n_inj {RP['n_inj']}, kept_frac mean {np.nanmean(RP['kept_frac']):.4f} (vt96 n_inj {R96['n_inj']}); MADGRAV1 T_obs = {R1['T_obs_yr']:.4f} yr"); L.append("")
    L.append("| Mtot bin [Msun] | VT 96p (fine bins) | eff 96p | VT 96 grid (fine bins) | eff 96 grid | 96p/96 |"); L.append("|---|---|---|---|---|---|")
    for j, b in enumerate(lab(eP)):
        L.append(f"| {b} | {vP[r][j]:.4f} | {RP['eff_covered'][j]:.3f} | {v96[r][j]:.4f} | {R96['eff_covered'][j]:.3f} | {rt(vP[r][j], v96[r][j])} |")
    L.append(""); mP = to_m1(vP[r], eP); m96 = to_m1(v96[r], e96); m22 = to_m1(v22[r], e22); m1 = v1[r]; totP += mP; tot96 += m96; tot22 += m22; tot1 += m1
    L.append("| Mtot bin [Msun] | VT 96p | VT 96 grid | VT MADGRAV1 | 96p/96 | 96p/M1 |"); L.append("|---|---|---|---|---|---|")
    for j, b in enumerate(lab(e1)): L.append(f"| {b} | {mP[j]:.4f} | {m96[j]:.4f} | {m1[j]:.4f} | {rt(mP[j], m96[j])} | {rt(mP[j], m1[j])} |")
    sP, s96, s22, s1 = np.nansum(mP), np.nansum(m96), np.nansum(m22), np.nansum(m1); per_run.append((r, sP, s96, s22, s1))
    L.append(f"| **20-400 total** | **{sP:.4f}** | **{s96:.4f}** | **{s1:.4f}** | **{rt(sP, s96)}** | **{rt(sP, s1)}** |"); L.append("")
L.append("## Sum over the four runs (MADGRAV1 bins)"); L.append("")
L.append("| Mtot bin [Msun] | VT 96p | VT 96 grid | VT 2.2 | VT MADGRAV1 | 96p/96 | 96p/M1 | 96p/2.2 |"); L.append("|---|---|---|---|---|---|---|---|")
for j, b in enumerate(lab(e1)): L.append(f"| {b} | {totP[j]:.4f} | {tot96[j]:.4f} | {tot22[j]:.4f} | {tot1[j]:.4f} | {rt(totP[j], tot96[j])} | {rt(totP[j], tot1[j])} | {rt(totP[j], tot22[j])} |")
L.append("")
L.append(f"Total VT 20-400 Msun, four runs: 96p (search placement) = {totP.sum():.3f} Gpc^3 yr, 96 grid = {tot96.sum():.3f}, 2.2 = {tot22.sum():.3f}, MADGRAV1 = {tot1.sum():.3f} Gpc^3 yr; "
         f"ratios 96p/96 = {totP.sum()/tot96.sum():.3f}, 96p/M1 = {totP.sum()/tot1.sum():.2f}, 96p/2.2 = {totP.sum()/tot22.sum():.2f} (96 grid/M1 was {tot96.sum()/tot1.sum():.2f}).")
L.append("Per run 20-400 Msun (96p / 96 grid / M1; 96p/96, 96p/M1): " + "; ".join(f"{r} {sP:.3f} / {s96:.3f} / {s1:.3f} ({rt(sP, s96)}, {rt(sP, s1)})" for r, sP, s96, s22, s1 in per_run) + ".")
L.append("")
L.append("Caveats: as VT_TABLE_M96 (bank seam below ~25 Msun; >260 Msun bins rest on near-zero MADGRAV1 VT). MADGRAV1's own VT field was scored with the same three-window "
         "convention, so the 96p/M1 ratio mixes a placement correction on one side only; the like-for-like comparison with MADGRAV1 is 96/M1, the honest absolute number for the frozen search is 96p.")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-4:])); print(f"[table] -> {OUT}")

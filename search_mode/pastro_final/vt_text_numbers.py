"""Numbers quoted in the paper's VT paragraph, from a vt_relabel_comoving json. Usage: vt_text_numbers.py <suffix>"""
import json, sys, csv, numpy as np
from scipy.stats import binom
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

HERE = MADGRAV_ROOT + "/search_mode/pastro_final"; CAT = MADGRAV_ROOT + "/figures/catalog_o3o4"
suf = sys.argv[1]; d = json.load(open(f"{HERE}/vt_relabel_comoving{suf}.json")); e = np.array(d["mass_edges"]); R = d["runs"]
runs = ["O3a", "O3b", "O4a", "O4b"]
vt = {r: np.array(R[r]["vt_comoving_srcframe_gpc3yr"]) for r in runs}; eff = {r: np.array(R[r]["eff_covered"]) for r in runs}
T = {r: R[r]["T_obs_yr"] for r in runs}; zmed = {r: np.array(R[r]["z_trunc_median"]) for r in runs}
tot = sum(vt[r] for r in runs); o3 = vt["O3a"] + vt["O3b"]; i = int(np.argmax(tot))
print(f"mass edges {e[0]:.0f}-{e[-1]:.0f}; total VT peak {tot[i]:.2f} Gpc3 yr at {e[i]:.0f}-{e[i+1]:.0f} (centre {0.5*(e[i]+e[i+1]):.0f}); O3a+O3b peak {o3.max():.2f}")
print("efficiency peaks (volume-averaged, eff_covered max):", {r: round(float(np.nanmax(eff[r])), 3) for r in runs})
print("O4 over O3 efficiency at peak bin:", round(float((eff['O4a'][i]+eff['O4b'][i])/(eff['O3a'][i]+eff['O3b'][i])), 2))
ratio = vt["O4a"] / vt["O3a"]; print("O4a/O3a VT ratio per bin:", np.round(ratio, 2).tolist())
r_vt = vt["O4b"][i] / vt["O3a"][i]; r_eff = eff["O4b"][i] / eff["O3a"][i]; r_T = T["O4b"] / T["O3a"]; r_vol = r_vt / r_eff / r_T
print(f"O4b/O3a at peak bin: VT {r_vt:.2f} = eff {r_eff:.2f} x volume {r_vol:.2f} x exposure {r_T:.2f}")
print("median z at peak bin:", {r: round(float(zmed[r][i]), 2) for r in runs}, " T_obs yr:", {r: round(T[r], 3) for r in runs})
# catalog-mass-weighted shares
cat = [r for r in csv.DictReader(open(f"{CAT}/merged_plot_v2.csv"))]
m = np.array([float(r["total_mass_source"]) for r in cat if r["total_mass_source"].strip()])
n_b = np.histogram(m, bins=e)[0]
w = {r: float(np.nansum(np.nan_to_num(vt[r]) * n_b)) for r in runs}; W = sum(w.values())
sys.path.insert(0, CAT); import adopted_set   # observed detections per run = the adopted set (honours SM_FAR_LR_CSV / SM_KE_JSON)
ad = adopted_set.load(); det = {r: sum(1 for x in ad if x["run"] == r) for r in runs}; nd = sum(det.values())
p4 = (w["O4a"] + w["O4b"]) / W; k4 = det["O4a"] + det["O4b"]
P4 = min(1.0, 2 * min(binom.cdf(k4, nd, p4), 1 - binom.cdf(k4 - 1, nd, p4)))
r33 = w["O3a"] / w["O3b"]; p3 = r33 / (1 + r33); k3 = det["O3a"]; n3 = det["O3a"] + det["O3b"]
P3 = min(1.0, 2 * min(binom.cdf(k3, n3, p3), 1 - binom.cdf(k3 - 1, n3, p3)))
print(f"catalog-weighted predicted O4 share {100*p4:.0f}% vs observed {k4}/{nd}={100*k4/nd:.0f}% (two-sided binomial P={P4:.2f}); "
      f"O3a:O3b predicted {r33:.2f} vs observed {k3}:{n3-k3} (P={P3:.2f}); detections per run {det}")

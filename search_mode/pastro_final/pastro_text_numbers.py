"""Numbers quoted in the paper for p_astro, from pastro_final<suffix>.json restricted to the adopted set.
Usage: SM_KE_JSON=... SM_FAR_LR_CSV=... pastro_text_numbers.py <suffix>   (K.Ln limit uses SM_KE_JSON or ke_adopted)"""
import json, sys, os, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

HERE = MADGRAV_ROOT + "/search_mode/pastro_final"; MG = MADGRAV_ROOT
sys.path.insert(0, f"{MG}/figures/catalog_o3o4"); import adopted_set
suf = sys.argv[1]; d = json.load(open(f"{HERE}/pastro_final{suf}.json"))
KE = json.load(open(os.environ.get("SM_KE_JSON", f"{MG}/details/successor_statistic/ke_adopted.json")))
ad = {(x["run"], x["name"]) for x in adopted_set.load()}
rows = [x for x in d if (x["run"], x["matches_known"]) in ad]; assert len(rows) == len(ad), (len(rows), len(ad))
runs = ["O3a", "O3b", "O4a", "O4b"]
first = {r: next(x for x in d if x["run"] == r) for r in runs}
print("Ln (pinned):", {r: round(first[r]["Ln"], 3) for r in runs}); print("Ls (fitted):", {r: round(first[r]["Ls"], 2) for r in runs})
print("free fit Ln:", {r: round(first[r]["Ln_free"], 2) for r in runs}, " Ls_free:", {r: round(first[r]["Ls_free"], 2) for r in runs})
p = np.array([x["p_astro"] for x in rows]); pf = np.array([x["p_astro_freeLn"] for x in rows])
print(f"adopted {len(rows)}: min p_astro {p.min():.4f}; n>0.90 {int((p>0.90).sum())}; n>=0.99 {int((p>=0.99).sum())}; n>=0.98 {int((p>=0.98).sum())}; min under free Ln {pf.min():.4f}")
low = sorted(rows, key=lambda x: x["p_astro"])[:4]; print("lowest four:", [(x["matches_known"], round(x["p_astro"], 4)) for x in low])
pk = np.array([(x["p_astro"] / (1 - x["p_astro"])) / ((x["p_astro"] / (1 - x["p_astro"])) + KE[x["run"].lower()]) for x in rows])
print(f"K*Ln conservative limit: n>=0.95 {int((pk>=0.95).sum())}; lowest:", [(x["matches_known"], round(v, 2)) for v, x in sorted(zip(pk, rows), key=lambda t: t[0])[:4]])

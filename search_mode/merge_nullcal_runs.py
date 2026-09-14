"""Merge per-run perarm_nullcal outputs (SM_RUNS=<run> SM_TAG_EXTRA=_<run>) into the combined-file format and
recompute the pooled K/E from the per-run K and E sums. Usage: merge_nullcal_runs.py <tag>  (e.g. excldet_netmax_lronly_gnet)"""
import json, sys, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

DET = MADGRAV_ROOT + "/details/successor_statistic"; tag = sys.argv[1]
parts = {r: json.load(open(f"{DET}/perarm_nullcal_{tag}_{r.lower()}.json")) for r in ("O3a", "O3b", "O4a", "O4b")}
out = dict(xs=parts["O3a"]["xs"], ns=parts["O3a"]["ns"], n_sample_per_fold=parts["O3a"]["n_sample_per_fold"], runs={}, pooled={})
for r, d in parts.items(): out["runs"][r] = d["runs"][r]
for n in map(str, out["ns"]):
    K = sum(np.array(out["runs"][r][n]["K"]) for r in out["runs"]); E = sum(np.array(out["runs"][r][n]["E"]) for r in out["runs"])
    ke = (K / E).tolist(); out["pooled"][n] = dict(K_over_E=ke, pass_at_1=bool(0.5 <= ke[0] <= 1.5))
json.dump(out, open(f"{DET}/perarm_nullcal_{tag}.json", "w"), indent=1)
print("merged ->", f"perarm_nullcal_{tag}.json", {r: round(out["runs"][r]["1.0"]["K_over_E"][0], 2) for r in out["runs"]}, "pooled", round(out["pooled"]["1.0"]["K_over_E"][0], 2))

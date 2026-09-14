"""ke_gnet.json from the gated null calibration: K/E at x = 1/yr, trials n = 1.0, per run, rounded to 0.01 exactly as
ke_adopted.json was taken from perarm_nullcal_excldet_netmax_lronly.json (2026-08-31)."""
import json
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

DET = MADGRAV_ROOT + "/details/successor_statistic"
d = json.load(open(f"{DET}/perarm_nullcal_excldet_netmax_lronly_gnet.json"))
ke = {run.lower(): round(d["runs"][run]["1.0"]["K_over_E"][0], 2) for run in ("O3a", "O3b", "O4a", "O4b")}
old = json.load(open(f"{DET}/ke_adopted.json"))
json.dump(ke, open(f"{DET}/ke_gnet.json", "w"))
print("ke_gnet   ", ke); print("ke_adopted", old)
for r in ("O3a", "O3b", "O4a", "O4b"): print(f"  {r}: K/E grid (x=1,.5,.2,.1,.05,.02,.01) = {[round(v,2) for v in d['runs'][r]['1.0']['K_over_E']]}")

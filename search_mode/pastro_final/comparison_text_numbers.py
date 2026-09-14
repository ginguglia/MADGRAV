"""Digest of the numbers quoted in Sec. comparison: VT ratios vs pipelines by mass range (from the four-epoch json),
and recovery vs SNR for Mtot>=20 (from inj_scored). Usage: comparison_text_numbers.py <suffix_m20> <tag_for_inj_scored>"""
import json, sys, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MG = MADGRAV_ROOT; HERE = f"{MG}/search_mode/pastro_final"
suf, tag = sys.argv[1], sys.argv[2]
d = json.load(open(f"{MG}/figures/vt_fourepoch/vt_fourepoch_ratio{suf}.json")); mids = np.array(d["mass_mids"])
def rng(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)]); return f"{v.min():.2f}--{v.max():.2f}" if len(v) else "n/a"
for run, P in d["ratios"].items():
    mf = [p for p in P if p != "cWB"]
    for lo, hi in ((20, 100), (100, 160), (160, 200)):
        sel = (mids > lo) & (mids < hi)
        mfv = [P[p][i] for p in mf for i in np.nonzero(sel)[0]]; cw = [P["cWB"][i] for i in np.nonzero(sel)[0]]
        per = {p: rng([P[p][i] for i in np.nonzero(sel)[0]]) for p in P}
        print(f"{run} {lo}-{hi}: MF {rng(mfv)}  cWB {rng(cw)}   per pipeline {per}")
print("\nrecovery at FAR<1/yr vs SNR level, Mtot>=20 (detector frame), w0-weighted, per run:")
for run in ("o3a", "o3b", "o4a", "o4b"):
    z = np.load(f"{HERE}/inj_scored_{run}_{tag}.npz"); m = z["mtot"] >= 20; snr = z["net_snr"]; df = z["det_frac"]
    lv = sorted(set(np.round(snr[m], 3))); s = " ".join(f"{int(round(l))}:{df[m & (np.round(snr,3)==l)].mean():.2f}" for l in lv if l <= 25)
    print(f"  {run}: {s}")

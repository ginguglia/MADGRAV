#!/usr/bin/env python
"""Build the trials=1 detection table: figures/catalog_o3o4/madgrav_far_final_x1.csv (48 rows).

Statistic (as used, and as the paper describes it): per candidate, for each channel c in
{lnLambda, net-sigma} take the SMALLER of the two per-arm background counts (HM, LM), take the
channel that minimises it, and divide by that fold's background livetime:

    FAR = min_c min(N_c^HM, N_c^LM) / T_f          [single counting -- no Bonferroni factor]

Justification (independent of the yield, and measured before it): the arm and channel axes are
strongly correlated -- measured N_eff arm 1.27/1.52, channel 1.13, all well below 2
(null_calibration/decomp_homogeneity). A x2-per-axis Bonferroni is therefore not derivable.

Detection criterion (pre-registered, unchanged): FAR < 1/yr AND its Poisson 90% upper limit < 1/yr.

Gate: the same counting machinery must first reproduce all 44 rows of the frozen as-run
madgrav_far_final.csv exactly (per-arm x2 and channel xN restored). Fails loud if not.

Membership = the four runs' detections.json, MINUS the candidates the local-ASD consistency veto
rejects, PLUS the four candidates the as-run trials factor had held above 1/yr. NOTE detections.json
is PRE-veto for O3b but POST-veto for O3a/O4a/O4b, so the O3b vetoed candidates must be removed here.

Writes a NEW file; the accepted madgrav_far_final.csv is never modified.
"""
import json, csv, sys, os
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from far_repro_check import perarm_counts, RUNS, CSV
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
DST = f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"
CATF = f"{MG}/figures/master.json"

# candidates held out by the as-run trials factor; all four are local-ASD veto KEEP (jobs 1695335-7)
PROMOTED = {"O3a": [1242442967.0], "O4a": [1372768865.0], "O4b": [1412214831.0, 1396420146.0]}
# local-ASD veto rejects (6/6 unmatched; none is a catalog event)
ASD_VETOED = [1258868518.0, 1267878086.0, 1258995493.0, 1398052082.0, 1379864500.0, 1249205578.0]

cat = json.load(open(CATF))
def meta(g):
    c = [r for r in cat if abs(r["gps"] - g) < 3.0]
    return (c[0]["name"], c[0].get("mtot"), c[0].get("cwb"), c[0].get("snr")) if c else (None, None, None, None)


def counts_for(bg, seg_ix, seg_fold, far_live, d):
    six = seg_ix[d["seg"]]; fo = int(seg_fold[six]); T = float(far_live[fo])
    return perarm_counts(bg, six, fo, d["loglr"], d["net"], d["cnn_hm"], d["cnn_lm"]), T


def main():
    # ---------- GATE: reproduce the frozen 44 with the as-run factors ----------
    frozen = list(csv.DictReader(open(CSV)))
    n_ok = n_bad = 0
    caches = {}
    for run, out in RUNS.items():
        z = np.load(f"{out}/bg_cache_{run.lower()}.npz", allow_pickle=False)
        caches[run] = (dict((k, z[k]) for k in ("hseg","lseg","fold","fam","loglr","net","cnn_hm","cnn_lm")),
                       {nm: i for i, nm in enumerate(z["seg_names"])}, z["seg_fold"], z["far_live"])
        bg, seg_ix, seg_fold, far_live = caches[run]
        dets = json.load(open(f"{out}/detections.json"))
        for r in frozen:
            if r["run"] != run: continue
            m = [d for d in dets if abs(d["loglr"]-float(r["loglr"]))<1e-9 and abs(d["net"]-float(r["net"]))<1e-9]
            assert len(m) == 1, f"{run} {r['name']}: {len(m)} matches"
            c, T = counts_for(bg, seg_ix, seg_fold, far_live, m[0])
            lr = 2*min(c[0], c[1])/T; nt = 2*min(c[2], c[3])/T
            ch = "net-sigma" if nt < lr else "loglr"
            far = 2*min(lr, nt); N = min((c[0], c[1]) if ch == "loglr" else (c[2], c[3]))
            ok = (abs(far-float(r["far"]))<1e-9 and ch == r["channel"]
                  and N == int(float(r["N_bg"])) and abs(T-float(r["livetime_yr"]))<5e-4)
            n_ok += ok; n_bad += (not ok)
    if n_bad:
        raise SystemExit(f"GATE FAIL: {n_bad} of {n_ok+n_bad} frozen rows not reproduced -- refusing to build")
    print(f"[gate] PASS: {n_ok}/{n_ok} frozen as-run rows reproduced exactly")

    # ---------- build the trials=1 table ----------
    rows = []; dropped = []
    for run, out in RUNS.items():
        bg, seg_ix, seg_fold, far_live = caches[run]
        cands = [dict(d, _src="as-run") for d in json.load(open(f"{out}/detections.json"))]
        trig = json.load(open(f"{out}/blindscan.json"))["triggers"]
        for g in PROMOTED.get(run, []):
            t = [x for x in trig if abs(x["gps"]-g) < 0.6]
            assert len(t) == 1, f"{run} {g}: {len(t)} triggers"
            cands.append(dict(t[0], _src="trials-promoted"))
        for d in cands:
            if any(abs(d["gps"]-v) < 0.6 for v in ASD_VETOED):
                dropped.append((run, d["gps"])); continue
            c, T = counts_for(bg, seg_ix, seg_fold, far_live, d)
            lr = min(c[0], c[1])/T; nt = min(c[2], c[3])/T          # trials = 1
            ch = "net-sigma" if nt < lr else "loglr"
            N = min((c[0], c[1]) if ch == "loglr" else (c[2], c[3]))
            far = N/T
            nch = 2                                                  # both channels defined for all 48
            far_asrun = nch*2*min(lr, nt)
            nm, mtot, cwb, snr = meta(d["gps"])
            assert nm, f"{run} {d['gps']}: no catalog match"
            rows.append(dict(run=run, name=nm, gps=d["gps"], seg=d["seg"],
                             net=d["net"], loglr=d["loglr"], cnn_hm=d["cnn_hm"], cnn_lm=d["cnn_lm"],
                             channel=ch, N_bg=N, livetime_yr=round(T, 3), trials=1.0, far=far,
                             far_lo90=(chi2.ppf(0.05, 2*N)/2/T if N > 0 else 0.0),
                             far_hi90=chi2.ppf(0.95, 2*N+2)/2/T,
                             far_ul90=chi2.ppf(0.90, 2*N+2)/2/T,
                             ifar=(1.0/far if far > 0 else float("inf")),
                             mtot=mtot, snr_cat=snr, cwb=cwb, source=d["_src"], far_asrun=far_asrun,
                             detection_and_ul=bool(far < 1.0 and chi2.ppf(0.90, 2*N+2)/2/T < 1.0)))
    order = {"O3a": 0, "O3b": 1, "O4a": 2, "O4b": 3}
    rows.sort(key=lambda r: (order[r["run"]], r["far"], r["name"]))
    assert all(r["detection_and_ul"] for r in rows), "a row fails far<1 AND UL90<1"
    assert len({r["name"] for r in rows}) == len(rows), "duplicate event name"
    with open(DST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(f"[build] wrote {DST}: {len(rows)} detections  {dict(Counter(r['run'] for r in rows))}")
    print(f"[build] ASD-vetoed excluded: {len(dropped)}   trials-promoted included: "
          f"{[r['name'] for r in rows if r['source']=='trials-promoted']}")


if __name__ == "__main__":
    main()

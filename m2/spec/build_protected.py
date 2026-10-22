"""Selection file for the PROTECTED + SIDEBAND render (PREREG_SPECIALISTS_20260923 §4a/b, §6): pairs never trained on.
  det      : the 67 MADGRAV2 catalogue detections (search_out_<run>_m2_gx/detections_lm.json) as ZERO-LAG pairs:
             H1 and L1 at the same segment and window index (argmin |gps - detection gps|).
  sideband : the 2,996 counted MADGRAV2 FAR-family pairs (neg_sideband.csv, A2-corr), looked up in survivors_bg.json.
  top50    : the 50 highest-lnL S0-passing background pairs that reach the specialists (all runs pooled); a pair also in the
             sideband keeps source 'sideband' and gets top50=1.
Every row carries the production CNN values (cnn_hm, cnn_lm), so the renderer's tripwire re-checks the crop path on them.
Output: madgrav2_spec/protected_sel.npz in the render_negatives.py schema (+ name, top50)."""
import os, sys, json, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M = MADGRAV_SCRATCH + "/madgrav2"; OUT = MADGRAV_SCRATCH + "/madgrav2_spec"
STREAMS = {"o3a": f"{M}/streams_o3a_full", "o3b": f"{M}/streams_o3b_full", "o4a": f"{M}/streams_o4a_full", "o4b": f"{M}/streams_o4b_full_cat2"}
ARM_MIN, CEN2, NET_MIN = -2.05, 220.0, 3.0
mc = {}
def meta(run, seg, det):
    k = (run, seg, det)
    if k not in mc: z = np.load(f"{STREAMS[run]}/{seg}_{det}_meta.npz"); mc[k] = {f: np.asarray(z[f], float) for f in ("gps", "sigma", "g2", "centroid2")}
    return mc[k]
R = {k: [] for k in ("run", "H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm", "g2_H", "g2_L", "c2_H", "c2_L", "source", "name", "top50")}
def add(run, sH, iH, sL, iL, loglr, hm, lm, src, name, top):
    a, b = meta(run, sH, "H1"), meta(run, sL, "L1")
    for k, v in (("run", run), ("H1_seg", sH), ("L1_seg", sL), ("iH", iH), ("iL", iL), ("gps_H", a["gps"][iH]), ("gps_L", b["gps"][iL]), ("sigma_H", a["sigma"][iH]),
                 ("sigma_L", b["sigma"][iL]), ("loglr", loglr), ("cnn_hm", hm), ("cnn_lm", lm), ("g2_H", a["g2"][iH]), ("g2_L", b["g2"][iL]), ("c2_H", a["centroid2"][iH]),
                 ("c2_L", b["centroid2"][iL]), ("source", src), ("name", name), ("top50", top)): R[k].append(v)
side = {}
for r in csv.DictReader(open(f"{OUT}/neg_sideband.csv")): side[(r["run"], round(float(r["gps_H"]), 1), round(float(r["gps_L"]), 1))] = 1
top_pool = []
for run in ("o3a", "o3b", "o4a", "o4b"):
    for d in json.load(open(f"{M}/search_out_{run}_m2_gx/detections_lm.json")):
        g = meta(run, d["seg"], "H1")["gps"]; i = int(np.argmin(np.abs(g - d["gps"])))
        assert abs(g[i] - d["gps"]) < 1.0 and abs(meta(run, d["seg"], "L1")["gps"][i] - d["gps"]) < 1.0, (run, d["seg"], d["gps"])
        add(run, d["seg"], i, d["seg"], i, d.get("loglr") or np.nan, d["cnn_hm"], d["cnn_lm"], "det", d.get("matches_known") or f"{run}_{d['gps']:.0f}", 0)
    S = json.load(open(f"{M}/search_out_{run}_m2_gx/survivors_bg.json"))["survivors"]; nside = 0
    for r in S:
        key = (run, round(float(r["gps_H"]), 1), round(float(r["gps_L"]), 1))
        if key in side:
            add(run, r["H1_seg"], int(r["iH"]), r["L1_seg"], int(r["iL"]), r.get("loglr") or np.nan, r["cnn_hm"], r["cnn_lm"], "sideband", "", 0); nside += 1; side[key] = 2
        elif r.get("loglr") is not None and max(r["cnn_hm"], r["cnn_lm"]) > 0.5:
            a, b = meta(run, r["H1_seg"], "H1"), meta(run, r["L1_seg"], "L1"); iH, iL = int(r["iH"]), int(r["iL"])
            sH, sL = a["sigma"][iH], b["sigma"][iL]; lead = a["g2"][iH] if sH >= sL else b["g2"][iL]
            if lead >= ARM_MIN and (sH + sL) / np.sqrt(2) >= NET_MIN and not (a["centroid2"][iH] >= CEN2 or b["centroid2"][iL] >= CEN2):
                top_pool.append((float(r["loglr"]), run, r["H1_seg"], iH, r["L1_seg"], iL, r["cnn_hm"], r["cnn_lm"]))
    del S; mc.clear(); print(f"[protected] {run}: detections {sum(1 for x, s in zip(R['run'], R['source']) if x == run and s == 'det')}, sideband {nside}", flush=True)
top_pool.sort(key=lambda t: -t[0]); side_ll = sorted([(l, i) for i, (l, s) in enumerate(zip(R["loglr"], R["source"])) if s == "sideband"], key=lambda t: -t[0])
# top-50 over sideband + non-sideband S0-passing admitted pairs, pooled
cand = [(l, "side", i) for l, i in side_ll[:50]] + [(t[0], "pool", t) for t in top_pool[:50]]; cand.sort(key=lambda t: -t[0]); cand = cand[:50]
for l, kind, x in cand:
    if kind == "side": R["top50"][x] = 1
    else: add(x[1], x[2], x[3], x[4], x[5], x[0], x[6], x[7], "top50", "", 1)
missing = [k for k, v in side.items() if v == 1]
A = {k: np.array(v) for k, v in R.items()}; A["iH"] = A["iH"].astype(int); A["iL"] = A["iL"].astype(int)
A["fold"] = np.array([QU.fold_of(int(g // 86400)) for g in A["gps_H"]]); A["s0_pass"] = (np.maximum(A["cnn_hm"], A["cnn_lm"]) > 0.5).astype(int)
np.savez(f"{OUT}/protected_sel.npz", **A)
print(f"[protected] {len(A['run'])} pairs: det {int((A['source']=='det').sum())}, sideband {int((A['source']=='sideband').sum())} (not found {len(missing)}), "
      f"top50 {int(A['top50'].sum())} of which in sideband {int(((A['source']=='sideband')&(A['top50']==1)).sum())}; per run " + " ".join(f"{r}:{int((A['run']==r).sum())}" for r in np.unique(A["run"])), flush=True)

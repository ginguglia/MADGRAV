"""Selection of catalogue-matched ZERO-LAG triggers that reach the specialists but were VETOED by S0 (max(cnn_hm,cnn_lm) <= 0.5)
in the MADGRAV2 campaign (search_out_<run>_m2_gx/blindscan_lm.json triggers). Catalogue events are quarantined from all
specialist training, so any new model scores them cleanly. Output: madgrav2_spec/vetoed_sel.npz (render_negatives.py schema)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M = MADGRAV_SCRATCH + "/madgrav2"; OUT = MADGRAV_SCRATCH + "/madgrav2_spec"
STREAMS = {"o3a": f"{M}/streams_o3a_full", "o3b": f"{M}/streams_o3b_full", "o4a": f"{M}/streams_o4a_full", "o4b": f"{M}/streams_o4b_full_cat2"}
R = {k: [] for k in ("run", "H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm", "source", "name", "top50", "net")}
for run in ("o3a", "o3b", "o4a", "o4b"):
    dets = {d.get("matches_known") for d in json.load(open(f"{M}/search_out_{run}_m2_gx/detections_lm.json"))}
    for t in json.load(open(f"{M}/search_out_{run}_m2_gx/blindscan_lm.json"))["triggers"]:
        if not t.get("matches_known") or t["matches_known"] in dets or max(t.get("cnn_hm") or 0, t.get("cnn_lm") or 0) > 0.5: continue
        a = np.load(f"{STREAMS[run]}/{t['seg']}_H1_meta.npz"); b = np.load(f"{STREAMS[run]}/{t['seg']}_L1_meta.npz"); i = int(np.argmin(np.abs(a["gps"] - t["gps"])))
        assert abs(a["gps"][i] - t["gps"]) < 1 and abs(b["gps"][i] - t["gps"]) < 1
        for k, v in (("run", run), ("H1_seg", t["seg"]), ("L1_seg", t["seg"]), ("iH", i), ("iL", i), ("gps_H", a["gps"][i]), ("gps_L", b["gps"][i]), ("sigma_H", a["sigma"][i]),
                     ("sigma_L", b["sigma"][i]), ("loglr", t.get("loglr") if t.get("loglr") is not None else np.nan), ("cnn_hm", t["cnn_hm"]), ("cnn_lm", t["cnn_lm"]),
                     ("source", "vetoed_cat"), ("name", t["matches_known"]), ("top50", 0), ("net", t.get("net") or np.nan)): R[k].append(v)
A = {k: np.array(v) for k, v in R.items()}; A["iH"] = A["iH"].astype(int); A["iL"] = A["iL"].astype(int)
A["fold"] = np.array([QU.fold_of(int(g // 86400)) for g in A["gps_H"]]); A["s0_pass"] = np.zeros(len(A["run"]), int)
np.savez(f"{OUT}/vetoed_sel.npz", **A); print(f"[vetoed] {len(A['run'])} pairs: " + " ".join(f"{r}:{int((A['run']==r).sum())}" for r in np.unique(A["run"])), flush=True)

"""Selection of the NON-CATALOGUE candidates of the MADGRAV2 gx merges that the adopted net-sigma LM>0.5 rule (detections_lm)
removed or kept (base detections.json vs detections_lm.json), as zero-lag pairs; all are quarantine anchors (detections*.json),
so any new model scores them cleanly. Output: madgrav2_spec/noncat_sel.npz (render_negatives.py schema)."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M = MADGRAV_SCRATCH + "/madgrav2"; OUT = MADGRAV_SCRATCH + "/madgrav2_spec"
STREAMS = {"o3a": f"{M}/streams_o3a_full", "o3b": f"{M}/streams_o3b_full", "o4a": f"{M}/streams_o4a_full", "o4b": f"{M}/streams_o4b_full_cat2"}
R = {k: [] for k in ("run", "H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm", "source", "name", "top50", "net", "far_base", "far_lm", "channel")}
for run in ("o3a", "o3b", "o4a", "o4b"):
    base = {f"{d['gps']:.0f}": d for d in json.load(open(f"{M}/search_out_{run}_m2_gx/detections.json")) if not d.get("matches_known")}
    lm = {f"{d['gps']:.0f}": d for d in json.load(open(f"{M}/search_out_{run}_m2_gx/detections_lm.json")) if not d.get("matches_known")}
    for g, d in {**base, **lm}.items():
        a = np.load(f"{STREAMS[run]}/{d['seg']}_H1_meta.npz"); b = np.load(f"{STREAMS[run]}/{d['seg']}_L1_meta.npz"); i = int(np.argmin(np.abs(a["gps"] - d["gps"])))
        assert abs(a["gps"][i] - d["gps"]) < 1 and abs(b["gps"][i] - d["gps"]) < 1
        for k, v in (("run", run), ("H1_seg", d["seg"]), ("L1_seg", d["seg"]), ("iH", i), ("iL", i), ("gps_H", a["gps"][i]), ("gps_L", b["gps"][i]), ("sigma_H", a["sigma"][i]), ("sigma_L", b["sigma"][i]),
                     ("loglr", d.get("loglr") if d.get("loglr") is not None else np.nan), ("cnn_hm", d["cnn_hm"]), ("cnn_lm", d["cnn_lm"]), ("source", "noncat"), ("name", f"{run}_{g}"), ("top50", 0),
                     ("net", d.get("net") or np.nan), ("far_base", base[g]["best_far"] if g in base else np.nan), ("far_lm", lm[g]["best_far"] if g in lm else np.nan), ("channel", d.get("channel") or "")): R[k].append(v)
A = {k: np.array(v) for k, v in R.items()}; A["iH"] = A["iH"].astype(int); A["iL"] = A["iL"].astype(int)
A["fold"] = np.array([QU.fold_of(int(g // 86400)) for g in A["gps_H"]]); A["s0_pass"] = np.ones(len(A["run"]), int)
np.savez(f"{OUT}/noncat_sel.npz", **A); print(f"[noncat] {len(A['run'])} candidates: " + " ".join(f"{n}(base {fb:.3g}, lm {fl:.3g})" for n, fb, fl in zip(A["name"], A["far_base"], A["far_lm"])), flush=True)

"""Selection of the 22 ARM-VETOED catalogue events of the 2026-09-28 stage-of-loss ledger (DECISIONS_WHITENING.md 20:30): the stream
window pair (same index, 1 s stride) with the largest net within +-1.5 s of the catalogue GPS, in the render_negatives.py schema, so the
S3 specialist can score the windows the arm removed. Output: madgrav2_spec/armvet_sel.npz."""
import os, sys, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

M = MADGRAV_SCRATCH + "/madgrav2"; OUT = MADGRAV_SCRATCH + "/madgrav2_spec"
STREAMS = {"o3a": f"{M}/streams_o3a_full", "o3b": f"{M}/streams_o3b_full", "o4a": f"{M}/streams_o4a_full", "o4b": f"{M}/streams_o4b_full_cat2"}
EV = {"o3a": ["GW190929_012149"], "o3b": ["GW191127_050227", "GW200220_124850"],
      "o4a": ["GW230702_185453", "GW230709_122727", "GW230805_034249", "GW230814_061920", "GW230819_171910", "GW230930_110730", "GW231001_140220", "GW231005_021030", "GW231129_081745", "GW231206_233134", "GW231206_233901", "GW240107_013215"],
      "o4b": ["GW240824_205609", "GW240922_142106", "GW240924_000316", "GW240930_035959", "GW241006_015333", "GW241116_151753", "GW250109_010541"]}
G = {}
for f in glob.glob(MADGRAV_EXTDATA + "/gwosc_eventapi/GWTC-*.json"):
    for k, e in json.load(open(f))["events"].items():
        if e.get("GPS"): G[k.split("-v")[0]] = float(e["GPS"])
R = {k: [] for k in ("run", "H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm", "source", "name", "top50", "net", "far_base", "far_lm", "channel", "g2_H", "g2_L")}
for run, names in EV.items():
    segs = {os.path.basename(f).rsplit("_", 2)[0]: np.load(f)["gps"] for f in glob.glob(f"{STREAMS[run]}/*_H1_meta.npz")}
    for n in names:
        g = G[n]; s = next(k for k, gg in segs.items() if gg[0] - 1 <= g <= gg[-1] + 1)
        a = np.load(f"{STREAMS[run]}/{s}_H1_meta.npz"); b = np.load(f"{STREAMS[run]}/{s}_L1_meta.npz"); k = np.flatnonzero(np.abs(a["gps"] - g) < 1.5)
        net = (a["sigma"][k] + b["sigma"][k]) / np.sqrt(2); i = int(k[np.argmax(net)])
        for kk, v in (("run", run), ("H1_seg", s), ("L1_seg", s), ("iH", i), ("iL", i), ("gps_H", a["gps"][i]), ("gps_L", b["gps"][i]), ("sigma_H", a["sigma"][i]), ("sigma_L", b["sigma"][i]),
                      ("loglr", np.nan), ("cnn_hm", np.nan), ("cnn_lm", np.nan), ("source", "armvet"), ("name", n), ("top50", 0), ("net", float(net.max())), ("far_base", np.nan), ("far_lm", np.nan), ("channel", "arm-vetoed"),
                      ("g2_H", a["g2"][i]), ("g2_L", b["g2"][i])): R[kk].append(v)
A = {k: np.array(v) for k, v in R.items()}; A["iH"] = A["iH"].astype(int); A["iL"] = A["iL"].astype(int)
A["fold"] = np.array([QU.fold_of(int(g // 86400)) for g in A["gps_H"]]); A["s0_pass"] = np.zeros(len(A["run"]), int)
np.savez(f"{OUT}/armvet_sel.npz", **A); print(f"[armvet] {len(A['run'])} windows: " + " ".join(f"{n}(net {nt:.2f}, g2 {gh:.2f}/{gl:.2f})" for n, nt, gh, gl in zip(A["name"], A["net"], A["g2_H"], A["g2_L"])), flush=True)

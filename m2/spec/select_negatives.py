"""Specialist-retrain NEGATIVES, stage 1 = SELECTION (CPU; PREREG_SPECIALISTS_20260923 §2, §3, A2).
Pool per run = every CNN-scored time-slid background pair of the MADGRAV2 campaign (search_out_<run>_m2_gx/survivors_bg.json),
restricted to what REACHES the specialists in production:
  louder-leg arm-v2 g2 >= -2.05 (streams meta g2 at the pair's window indices), centroid2 < 220 Hz in both legs, sigma_net >= 3.
S0 = the June gate as run (cnn_hm / cnn_lm in the file): pass iff max > 0.5.
SIDEBAND (A2 corrected): the union over detections (detections_lm.json) of the background pairs that SET each detection's FAR in
  driver_blindscan_merge: own channel, per-arm dominance, the arm with the smaller count (see code). Sideband pairs are never trained on, and their LEGS join the
  quarantine (+-64 s, same detector), so every other slid pair built on the same glitch leg is excluded too.
Exclusions: quarantined legs (quarantine.py + sideband legs); CROSS-FOLD pairs (fold(day_H) != fold(day_L); §7.1).
Draw: ALL remaining S0-passing pairs + S0-failing pairs sampled uniformly (per run, proportional to the run's failing pool) up to
N_TOTAL = 300000. Seed fixed. Output /data/.../madgrav2_spec/neg_select.npz (+ neg_sideband.csv, census printed)."""
import os, sys, json, glob, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M = MADGRAV_SCRATCH + "/madgrav2"; OUT = MADGRAV_SCRATCH + "/madgrav2_spec"; os.makedirs(OUT, exist_ok=True)
STREAMS = {"o3a": f"{M}/streams_o3a_full", "o3b": f"{M}/streams_o3b_full", "o4a": f"{M}/streams_o4a_full", "o4b": f"{M}/streams_o4b_full_cat2"}
N_TOTAL = int(os.environ.get("NEG_TOTAL", "300000")); SEED = 20260923; ARM_MIN, CEN2, NET_MIN, GT = -2.05, 220.0, 3.0, 0.5
rng = np.random.default_rng(SEED)
meta_cache = {}
def meta(run, seg, det):
    k = (run, seg, det)
    if k not in meta_cache:
        z = np.load(f"{STREAMS[run]}/{seg}_{det}_meta.npz"); meta_cache[k] = (np.asarray(z["g2"], float), np.asarray(z["centroid2"], float))
    return meta_cache[k]
P = {}; side_rows = []
for run in ("o3a", "o3b", "o4a", "o4b"):
    d = json.load(open(f"{M}/search_out_{run}_m2_gx/survivors_bg.json"))["survivors"]
    det = json.load(open(f"{M}/search_out_{run}_m2_gx/detections_lm.json"))
    n = len(d); cols = {k: np.empty(n, object if k in ("H1_seg", "L1_seg") else float) for k in
                        ("H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm", "g2_H", "g2_L", "c2_H", "c2_L")}
    for i, r in enumerate(d):
        for k in ("H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "cnn_hm", "cnn_lm"): cols[k][i] = r[k]
        cols["loglr"][i] = r["loglr"] if r.get("loglr") is not None else -np.inf
        gH, cH = meta(run, r["H1_seg"], "H1"); gL, cL = meta(run, r["L1_seg"], "L1")
        cols["g2_H"][i] = gH[r["iH"]]; cols["c2_H"][i] = cH[r["iH"]]; cols["g2_L"][i] = gL[r["iL"]]; cols["c2_L"][i] = cL[r["iL"]]
    del d; meta_cache.clear()
    c = {k: (v.astype(float) if k not in ("H1_seg", "L1_seg") else v) for k, v in cols.items()}
    net = (c["sigma_H"] + c["sigma_L"]) / np.sqrt(2); lead = np.where(c["sigma_H"] >= c["sigma_L"], c["g2_H"], c["g2_L"])
    cen_ok = ~((np.isfinite(c["c2_H"]) & (c["c2_H"] >= CEN2)) | (np.isfinite(c["c2_L"]) & (c["c2_L"] >= CEN2)))
    reach = (lead >= ARM_MIN) & cen_ok & (net >= NET_MIN); s0 = np.maximum(c["cnn_hm"], c["cnn_lm"]) > GT
    # SIDEBAND (A2 corrected): for each detection, the background pairs that SET its FAR, exactly as driver_blindscan_merge
    # counts them: in the detection's own channel (loglr: louder in lnLambda; net-sigma: louder in sigma_net AND S0 pass AND
    # cnn_lm > 0.0382), dominated per arm (HM: cnn_hm >= det's; LM: cnn_lm >= det's); the arm with the SMALLER count is the one
    # the per-arm FAR uses, so its set is taken. Union over detections. (Fold and self-segment guards ignored -> slight superset.)
    side = np.zeros(n, bool); hm, lm = c["cnn_hm"], c["cnn_lm"]
    for x in det:
        if x.get("channel") == "loglr": base_ = reach & (c["loglr"] >= float(x["loglr"]))
        else: base_ = reach & s0 & (lm > 0.0382) & (net >= float(x["net"]))
        mh = base_ & (hm >= float(x["cnn_hm"])); ml = base_ & (lm >= float(x["cnn_lm"]))
        side |= mh if mh.sum() <= ml.sum() else ml
    fH = np.array([QU.fold_of(int(g // 86400)) for g in c["gps_H"]]); fL = np.array([QU.fold_of(int(g // 86400)) for g in c["gps_L"]])
    P[run] = dict(c=c, net=net, reach=reach, s0=s0, side=side, fH=fH, fL=fL)
    for i in np.where(side)[0]: side_rows.append(dict(run=run, gps_H=c["gps_H"][i], gps_L=c["gps_L"][i], loglr=c["loglr"][i], sigma_net=net[i]))
    print(f"[{run}] scored {n} | reach specialists {reach.sum()} (S0 pass {(reach&s0).sum()}) | sideband {side.sum()} "
          f"(FAR-setting sets of {len(det)} detections) | fold-pure {(fH==fL).mean():.2f}", flush=True)
sc = f"{OUT}/neg_sideband.csv"
with open(sc, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "gps_H", "gps_L", "loglr", "sigma_net"]); w.writeheader(); w.writerows(side_rows)
Q = QU.build(extra_csv=[sc])
sel = {}; fail_pool = {}
for run, p in P.items():
    c = p["c"]; qH = QU.quarantined(Q, "H1", c["gps_H"]); qL = QU.quarantined(Q, "L1", c["gps_L"])
    ok = p["reach"] & ~p["side"] & ~qH & ~qL & (p["fH"] == p["fL"])
    sel[run] = np.where(ok & p["s0"])[0]; fail_pool[run] = np.where(ok & ~p["s0"])[0]
    print(f"[{run}] after quarantine+sideband-legs+fold-purity: S0-pass {len(sel[run])}, S0-fail pool {len(fail_pool[run])} "
          f"(quarantined H {(p['reach']&qH).sum()} L {(p['reach']&qL).sum()})", flush=True)
n_pass = sum(len(v) for v in sel.values()); n_fill = max(0, N_TOTAL - n_pass); tot_fail = sum(len(v) for v in fail_pool.values())
out = {k: [] for k in ("run", "H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm",
                       "g2_H", "g2_L", "c2_H", "c2_L", "fold", "s0_pass")}
for run, p in P.items():
    k = int(round(n_fill * len(fail_pool[run]) / max(1, tot_fail)))
    pick = rng.choice(fail_pool[run], size=min(k, len(fail_pool[run])), replace=False) if k else np.array([], int)
    for idx, flag in ((sel[run], 1), (np.sort(pick), 0)):
        for f_ in ("H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "loglr", "cnn_hm", "cnn_lm", "g2_H", "g2_L", "c2_H", "c2_L"):
            out[f_] += list(p["c"][f_][idx])
        out["run"] += [run] * len(idx); out["fold"] += list(p["fH"][idx]); out["s0_pass"] += [flag] * len(idx)
    print(f"[{run}] drawn: {len(sel[run])} S0-pass + {len(pick)} S0-fail", flush=True)
arr = {k: np.array(v) for k, v in out.items()}; arr["iH"] = arr["iH"].astype(int); arr["iL"] = arr["iL"].astype(int)
np.savez(f"{OUT}/neg_select.npz", **arr, seed=SEED)
print(f"[neg-select] {len(arr['run'])} negatives (S0-pass {int(arr['s0_pass'].sum())}) -> {OUT}/neg_select.npz; "
      f"folds {np.bincount(arr['fold'])}; sideband pairs {len(side_rows)} -> {sc}", flush=True)

"""Login-node pre-check (pure json, no tables): far96's post-merge logic (channel, local-ASD veto from the merge-log local values, guard, cap
bypass, 96 set) on the production fields of blindscan_v2.json, before far_tables exist. Section A (FAR arithmetic) is not covered here."""
import sys, json, glob, os, re, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

sys.path.insert(0, MADGRAV_ROOT + "/m2/vt96"); import far96 as F
S3_THR = 0.5356733202934265; FLOOR = 3.0; NSF = 3.0
for run in ("o3a", "o3b", "o4a", "o4b"):
    out = f"{MADGRAV_SCRATCH}/madgrav2/search_out_{run}_m2_gx_off5_crop"; cnn_on, mf = F.clause_on(run)
    bs = json.load(open(f"{out}/blindscan_v2.json")); trig = bs["triggers"]; live = {0: bs["far_live_yr"]["0"], 1: bs["far_live_yr"]["1"]}
    pre = {}; R = {}
    for t in trig:
        r = dict(t); r["stat"] = float(t["loglr"]); r["below_floor"] = t.get("far_lr_per_yr") is None
        ch = [(t.get("far_lr_perarm"), t.get("far_lr_perarm_ul90")), (t.get("far_net_perarm"), t.get("far_net_perarm_ul90"))]; ch = [c for c in ch if c[0] is not None]
        r["detected"] = bool(ch) and (len(ch) * min(ch, key=lambda c: c[0])[0] < 1 and len(ch) * min(ch, key=lambda c: c[0])[1] < 1)
        R[float(t["gps"])] = (r, t)
        if r["detected"]: pre[float(t["gps"])] = (r, t)
    logs = sorted(glob.glob(f"{F.CHAIN_LOGS}/crop-{run}-merge-v2_*.log"), key=os.path.getmtime)
    rx = re.compile(r"\[asd-veto\]\s+(\S+)\s+([\d.]+)\s+\((loglr|net-sigma)\)\s+net\s+([-\d.]+)->([-\d.]+)\s+cnn\s+([\d.]+)/([\d.]+)->([\d.]+)/([\d.]+)\s+(KEEP|VETO\(([^)]*)\))")
    rows = {}
    for ln in open(logs[-1]):
        m = rx.search(ln)
        if m: rows[float(m.group(2))] = dict(chan=m.group(3), net_loc=float(m.group(5)), hm_loc=float(m.group(8)), lm_loc=float(m.group(9)), keep=m.group(10) == "KEEP", reason=m.group(11) or "keep")
    nb = 0; kept = set(); nchan = 0
    for gps, (r, t) in pre.items():
        p = rows[gps]; ch = F.chan_of(r)
        if ch != p["chan"]: nchan += 1
        keep, reason = F.asd_veto(ch, cnn_on, p["net_loc"], p["hm_loc"], p["lm_loc"], p["hm_loc"] if ch == "loglr" else np.nan, S3_THR, NSF)
        if keep != p["keep"]: nb += 1; print(f"   {run} veto mismatch {t.get('matches_known') or t['seg']} {gps} {ch}: mine {keep}/{reason} log {p['keep']}/{p['reason']} {p}")
        if p["keep"]: kept.add(gps)
    dv = json.load(open(f"{out}/detections_v2.json")); dvg = {float(d["gps"]) for d in dv}
    dn = json.load(open(f"{out}/detections_v2_nsguard.json")); dng = {float(d["gps"]) for d in dn}
    cap = {}
    for gps, (r, t) in R.items():
        adm, far, ul = F.cap_bypass(run, r, live[int(t["fold"])])
        if adm and gps not in dng: cap[gps] = (t.get("matches_known") or t["seg"], round(far, 3), round(ul, 3))
    names = {d["matches_known"] for d in dn if d.get("matches_known")} | {v[0] for v in cap.values() if v[0].startswith("GW")}
    c96 = json.load(open(F.CROP96)); ref = {n for n, v in c96.items() if v.get("mad") and v["run"].lower() == run}
    print(f"[precheck] {run}: clause {'ON' if cnn_on else 'OFF'} ({mf:.4f}); pre-veto {len(pre)} (log rows {len(rows)}); channel mismatches {nchan}; veto mismatches {nb}; kept {len(kept)} vs detections_v2 {len(dvg)} symdiff {len(kept ^ dvg)}; cap rows {list(cap.values())}; final set {len(names)} vs crop96 {len(ref)} symdiff {sorted(names ^ ref)}")

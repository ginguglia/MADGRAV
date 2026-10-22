"""vt96 (2026-10-19): the frozen MADGRAV 2 version-2 recipe at scoring time, on flat FAR tables (far_tables_<run>.npz from
export_far_tables96.py = the production _v2 merge inputs of the crop re-pass), applied to
  (a) the merge's own zero-lag triggers -> identity check against blindscan_v2.json / detections_v2.json / detections_v2_nsguard.json /
      the 96 set (figures_m2/hl_snr_o3o4_m2_crop96.json)            : far96.py validate --run R --tables F --outdir SM_OUT
  (b) injection files (inj_<run>_m2_96, SM_CROP=coinc, local-ASD columns) -> inj_scored_<run>_m96.npz : far96.py score ...

Merge arithmetic (loglr channel: stat >= floor 3 and S3 >= S3_THR, single-arm S3 family count; net-sigma channel: net >= 3, June
max(hm,lm) >= 0.5, cnn_lm > 0.0382, per-arm counts; best = n_open x min; detected iff point and UL90 < 1/yr) is m2/vt22/far22.Tables,
unchanged by the crop re-pass. Added here, in the production order (driver_blindscan_merge_hook.py, netsig_guard.py, DECISIONS_CAP_BYPASS.md):
  1. local-ASD veto on the merge detections, channel = the one that set best_far (ties -> loglr):
       loglr rows     : feature clause ON (plateau max_frac > 0.04: o3a, o3b) -> keep iff s3_loc >= S3_THR; OFF (o4a, o4b) -> keep.
       net-sigma rows : feature clause ON and max(hm_loc, lm_loc) < 0.5 -> veto(cnn); else net_loc < 3.0 -> veto(net<floor); else keep.
     (the net clause is unconditional; a loglr row has no net clause, as in asd_consistency.survives)
  2. net-sigma guard on the surviving net-sigma rows, EXACTLY the production rule (netsig_guard.py): keep iff (Grad-CAM-crop June pass
     [max(cnn_hm_gc, cnn_lm_gc) >= 0.5, no LM floor in this clause] OR raw loglr > 0) AND cnn_lm > 0.5 (cnn_lm = coincident-crop LM).
     Injections carry cnn_hm_gc/cnn_lm_gc (inject96.py, Grad-CAM placement, same models and crops) -> same rule, same inputs.
  3. loglr cap bypass (post-merge, not vetoed, not guarded): a trigger whose loglr gate fails (S3 < S3_THR) with stat >= L_cap[run] is
     admitted at FAR = 2 x max(ungated per-trigger FAR n_louder_L1fam/T, cell FAR[run]) and the UL90 analogue, both < 1/yr.
     L_cap / cell FAR per run from DECISIONS_CAP_BYPASS.md section 5 (slides, gate-failing cell FAR <= 0.2/yr on the 0.001 grid).
Written 2026-10-19."""
import os, sys, json, glob, re, time, argparse, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

sys.path.insert(0, MADGRAV_ROOT + "/m2/vt22")
from far22 import Tables, ul90                                  # the merge's FAR arithmetic on flat arrays (validated 2026-10-04/05)

L_CAP = {"o4b": 22.986, "o4a": 22.991, "o3a": 22.991, "o3b": 22.985}
CELL_FAR = {"o4b": 0.152, "o4a": 0.105, "o3a": 0.143, "o3b": 0.170}
PLATEAU_THR = float(os.environ.get("SM_ASD_PLATEAU_THR", "0.04")); GUARD_LM = 0.5; GLITCH = 0.5
CHAIN_LOGS = MADGRAV_ROOT + "/m2/popb_cap/occlusion/chain/logs"
CROP96 = MADGRAV_SCRATCH + "/madgrav2/figures_m2/hl_snr_o3o4_m2_crop96.json"

def clause_on(run):
    pj = os.environ.get("SM_ASD_PLATEAU_JSON", f"{MADGRAV_ROOT}/m2/asdveto/plateau_{run}.json")
    mf = float(json.load(open(pj))["max_frac"]); return bool(mf > PLATEAU_THR), mf

def chan_of(r):
    """== merge_hook._chan on a trigger/result dict (PERARM): None if no channel open."""
    a, b = r.get("far_lr_perarm"), r.get("far_net_perarm")
    if a is None and b is None: return None
    if a is None: return "net-sigma"
    if b is None: return "loglr"
    return "net-sigma" if b < a else "loglr"

def asd_veto(chan, cnn_on, net_loc, hm_loc, lm_loc, s3_loc, s3_thr, floor):
    """(keep, reason) == merge_hook local-ASD block + asd_consistency.survives, with the S3 substitution on loglr rows."""
    if chan == "loglr":
        if not cnn_on: return True, "keep(feature clause off)"
        if not np.isfinite(s3_loc): return True, "keep(unmeasured)"
        return (True, "keep") if s3_loc >= s3_thr else (False, "cnn")
    if cnn_on and not (max(hm_loc, lm_loc) >= GLITCH): return False, "cnn"
    if not (net_loc >= floor): return False, "net<floor"
    return True, ("keep" if cnn_on or max(hm_loc, lm_loc) >= GLITCH else "keep(feature clause off)")

def guard_keep(hm_gc, lm_gc, lnl_raw, lm_coinc):
    """== netsig_guard.rule(gpass, loglr_raw, lm): gpass = max(Grad-CAM-crop hm, lm) >= 0.5 (no LM floor); lm = coincident-crop LM > 0.5."""
    gpass = bool(np.isfinite(hm_gc) and np.isfinite(lm_gc) and max(hm_gc, lm_gc) >= GLITCH)
    return bool((gpass or lnl_raw > 0) and lm_coinc > GUARD_LM)

def cap_bypass(run, r, flt):
    """(admitted, far, ul90) of the loglr cap bypass for a trigger result r (stat in r['stat'])."""
    if r.get("below_floor", True) or not r.get("is_glitch_lr", True) or r["stat"] < L_CAP[run]: return False, None, None
    n1 = r["n_louder_L1fam"]; far = 2 * max(n1 / flt, CELL_FAR[run]); ul = 2 * max(ul90(n1, flt), CELL_FAR[run])
    return bool(far < 1.0 and ul < 1.0), far, ul

def _cmp(name, a, b, tol, bad):
    if (a is None) != (b is None): bad.append((name, a, b)); return
    if a is None: return
    if abs(float(a) - float(b)) > tol * max(1.0, abs(float(b))): bad.append((name, a, b))

# ---------------------------------------------------------------------------------------------------------------------------- validate
def validate(args):
    run = args.run; T = Tables(args.tables); out = args.outdir; cnn_on, mf = clause_on(run)
    bs = json.load(open(f"{out}/blindscan_v2.json")); trig = bs["triggers"]
    assert abs(bs["far_live_yr"]["0"] - T.live[0]) < 1e-9 and abs(bs["far_live_yr"]["1"] - T.live[1]) < 1e-9, ("livetime", bs["far_live_yr"], T.live)
    assert abs(bs["netsig_lm_min"] - T.lm_min) < 1e-12 and abs(bs["netsig_floor"] - T.netsig_floor) < 1e-12 and abs(bs["net_cut"] - T.net_cut) < 1e-12
    print(f"[vt96-validate] {run}: tables {args.tables}; floor {T.floor} s3_thr {T.s3_thr:.6f} lm_min {T.lm_min} netsig_floor {T.netsig_floor} net_cut {T.net_cut} zl_gates {T.zl_gates} arm {T.arm_rule}; feature clause {'ON' if cnn_on else 'OFF'} (plateau {mf:.4f} vs {PLATEAU_THR})")
    FIELDS = ("n_louder_L1fam", "far_lr_per_yr", "far_lr_ul90", "far_lr_perarm", "far_lr_perarm_ul90", "n_louder_netfam_cnn", "far_net_per_yr", "far_net_ul90", "far_net_hm", "far_net_lm", "far_net_perarm", "far_net_perarm_ul90")
    nbad = 0; worst = []; pre = {}; R = {}
    for t in trig:
        g = int(t["fold"]); si = int(t["si"]); assert T.seg_name[si] == t["seg"], (si, t["seg"])
        r = T.trigger(g, si, float(t["loglr"]), t.get("s3"), float(t["net"]), t.get("cnn_hm"), t.get("cnn_lm")); r["stat"] = float(t["loglr"])
        bad = []
        for k in FIELDS: _cmp(k, r.get(k), t.get(k), 1e-9, bad)
        if bad: nbad += 1; worst.append((t["seg"], t["gps"], t.get("matches_known", ""), bad[:4]))
        R[float(t["gps"])] = (r, t)
        if r["detected"]: pre[float(t["gps"])] = (r, t)
    print(f"[vt96-validate] A. merge FAR fields: {len(trig)} triggers, {nbad} with any mismatch; pre-veto detections reproduced: {len(pre)}")
    for w in worst[:10]: print("   MISMATCH", w)
    # ---- B. local-ASD veto: the production decisions from the merge log (local values at 2 decimals) ----
    logs = sorted(glob.glob(f"{CHAIN_LOGS}/crop-{run}-merge-v2_*.log"), key=os.path.getmtime)
    rx = re.compile(r"\[asd-veto\]\s+(\S+)\s+([\d.]+)\s+\((loglr|net-sigma)\)\s+net\s+([-\d.]+)->([-\d.]+)\s+cnn\s+([\d.]+)/([\d.]+)->([\d.]+)/([\d.]+)\s+(KEEP|VETO\(([^)]*)\))")
    rows = {}
    for ln in open(logs[-1]):
        m = rx.search(ln)
        if m: rows[float(m.group(2))] = dict(name=m.group(1), chan=m.group(3), net_loc=float(m.group(5)), hm_loc=float(m.group(8)), lm_loc=float(m.group(9)), keep=m.group(10) == "KEEP", reason=m.group(11) or "keep")
    nb_chan = nb_veto = namb = 0; kept_prod = set(); kept_mine = set()
    for gps, (r, t) in pre.items():
        p = rows.get(gps)
        if p is None: nb_veto += 1; print(f"   NO LOG ROW for pre-veto detection {t['seg']} {gps}"); continue
        ch = chan_of(r)
        if ch != p["chan"]: nb_chan += 1; print(f"   CHANNEL mismatch {t['seg']} {gps}: mine {ch} log {p['chan']}")
        s3_loc = p["hm_loc"] if ch == "loglr" else np.nan
        keep, reason = asd_veto(ch, cnn_on, p["net_loc"], p["hm_loc"], p["lm_loc"], s3_loc, T.s3_thr, T.netsig_floor)
        amb = (ch == "net-sigma" and (abs(p["net_loc"] - T.netsig_floor) < 0.006 or abs(max(p["hm_loc"], p["lm_loc"]) - GLITCH) < 0.006)) or (ch == "loglr" and abs(p["hm_loc"] - T.s3_thr) < 0.006)
        if keep != p["keep"]:
            if amb: namb += 1; print(f"   veto AMBIGUOUS at 2 decimals {t['seg']} {gps} ({ch}): mine {keep}/{reason} log {p['keep']}/{p['reason']} net_loc {p['net_loc']} hm/lm_loc {p['hm_loc']}/{p['lm_loc']}")
            else: nb_veto += 1; print(f"   VETO mismatch {t['seg']} {gps} ({ch}): mine {keep}/{reason} log {p['keep']}/{p['reason']} net_loc {p['net_loc']} hm/lm_loc {p['hm_loc']}/{p['lm_loc']}")
        if p["keep"]: kept_prod.add(gps)
        if keep: kept_mine.add(gps)
    dv = json.load(open(f"{out}/detections_v2.json")); dv_gps = {float(d["gps"]) for d in dv}
    nb_dv = len(dv_gps ^ kept_prod)
    print(f"[vt96-validate] B. local-ASD veto: {len(pre)} pre-veto rows, log rows {len(rows)}; channel mismatches {nb_chan}; decision mismatches {nb_veto} (ambiguous at 2 decimals {namb}); production survivors {len(kept_prod)} == detections_v2.json {len(dv_gps)} (sym diff {nb_dv})")
    # ---- C. net-sigma guard, production form (== netsig_guard.py) and injection form ----
    P = json.load(open(f"{out.replace('_crop', '_nf275')}/blindscan_p205.json"))["triggers"]; pg = np.array([t["gps"] for t in P])
    Tc = json.load(open(f"{out}/blindscan_p205.json"))["triggers"]; tg = np.array([t["gps"] for t in Tc])
    keep_prod = set(); keep_inj = set(); n_ns = 0
    for d in dv:
        gps = float(d["gps"])
        if d.get("channel") != "net-sigma": keep_prod.add(gps); keep_inj.add(gps); continue
        n_ns += 1; j = int(np.argmin(np.abs(pg - gps))); p = P[j] if abs(pg[j] - gps) < 1.5 else None
        gpass = (max(p["cnn_hm"], p["cnn_lm"]) >= GLITCH) if p else None
        jc = int(np.argmin(np.abs(tg - gps))); raw = Tc[jc]["loglr"] if abs(tg[jc] - gps) < 1.5 else d["loglr"]
        ok_prod = ((bool(gpass) or raw > 0) and d["cnn_lm"] > GUARD_LM) if p else True
        ok_inj = guard_keep(p["cnn_hm"] if p else np.nan, p["cnn_lm"] if p else np.nan, raw, d["cnn_lm"]) if p else True   # far96's own function on the same inputs
        if ok_prod: keep_prod.add(gps)
        if ok_inj: keep_inj.add(gps)
    dn = json.load(open(f"{out}/detections_v2_nsguard.json")); dn_gps = {float(d["gps"]) for d in dn}
    nb_guard = len(dn_gps ^ keep_prod); extra = keep_prod - keep_inj
    print(f"[vt96-validate] C. net-sigma guard: {n_ns} net-sigma rows of {len(dv)}; production form keeps {len(keep_prod)} == detections_v2_nsguard.json {len(dn_gps)} (sym diff {nb_guard}); "
          f"far96.guard_keep on the same inputs differs on {len(extra)} rows (must be 0)")
    nb_guard += len(extra)
    # ---- D. cap bypass + the 96 set ----
    cap = {}
    for gps, (r, t) in R.items():
        adm, far, ul = cap_bypass(run, r, T.live[int(t["fold"])])
        if adm and gps not in dn_gps: cap[gps] = (t.get("matches_known") or t["seg"], far, ul)
    names_final = {d["matches_known"] for d in dn if d.get("matches_known")} | {v[0] for v in cap.values() if v[0].startswith("GW")}
    c96 = json.load(open(CROP96)); ref = {n for n, v in c96.items() if v.get("mad") and v["run"].lower() == run}
    from collections import Counter   # the crop96 table uses short names for GW190412 / GW190521: compare the multiset of 8-char prefixes (collision-safe)
    cf, cr = Counter(n[:8] for n in names_final), Counter(n[:8] for n in ref); nb_96 = sum(abs(cf[k] - cr[k]) for k in set(cf) | set(cr))
    print(f"[vt96-validate] D. cap bypass (L_cap {L_CAP[run]}, cell {CELL_FAR[run]}/yr): {len(cap)} row(s) {[(v[0], round(v[1],3), round(v[2],3)) for v in cap.values()]}; "
          f"final catalogue set {len(names_final)} vs crop96 table {len(ref)} (prefix multiset diff {nb_96}; name-level diff {sorted(names_final ^ ref)})")
    ok = nbad == 0 and nb_chan == 0 and nb_veto == 0 and nb_dv == 0 and nb_guard == 0 and nb_96 == 0
    print(f"[vt96-validate] {run}: " + ("PASS" if ok else "FAIL") + f" (FAR mismatches {nbad}, veto {nb_veto}+{namb} ambiguous, detections_v2 {nb_dv}, guard {nb_guard}, set96 {nb_96})", flush=True)
    return 0 if ok else 1

# ---------------------------------------------------------------------------------------------------------------------------- score
RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1")   # == m2/arm/s3off5/score_rank_off5.py / apply_stat_nf.py
def score(args):
    import joblib
    MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]
    for _p in ("search_mode", "improved", "spectrogram_cascade"):
        _ap = os.path.join(MADGRAV_ROOT, _p)
        if _ap not in sys.path: sys.path.insert(0, _ap)
    os.chdir(MADGRAV_ROOT); import driver_search_multi as M
    run = args.run; T = Tables(args.tables); t0 = time.time(); cnn_on, mf = clause_on(run)
    lrm = np.load(args.lr_model or T.lr_model); mdl = {g: (lrm[f"mu{g}"], lrm[f"sd{g}"], lrm[f"be{g}"]) for g in (0, 1)}
    models = joblib.load(args.ranker or f"{MADGRAV_SCRATCH}/madgrav2_spec/rank/models_{run}.joblib"); assert set(models) == {0, 1}
    print(f"[vt96-score] {run}: feature clause {'ON' if cnn_on else 'OFF'} (plateau {mf:.4f}); L_cap {L_CAP[run]} cell {CELL_FAR[run]}; lr {args.lr_model or T.lr_model}; GX.ON={M.GX.ON} cap {M.GX.CAP}", flush=True)
    def rule_pass(gH, gL, sH, sL):
        if T.arm_rule == "none": return np.ones(len(gH), bool)
        name, thr = T.arm_rule.split(":"); thr = float(thr)
        if name == "max": v = np.where(sH >= sL, gH, gL)
        elif name == "mean": v = 0.5 * (gH + gL)
        elif name == "both": v = np.minimum(gH, gL)
        elif name == "ramp": v = np.clip(sH / 3.0, 0, 1) * gH + np.clip(sL / 3.0, 0, 1) * gL
        else: raise SystemExit(T.arm_rule)
        return v >= thr
    files = [f for d in args.inj_dirs for f in sorted(glob.glob(f"{d}/*_inj.npz"))]
    if not files: raise SystemExit(f"no *_inj.npz in {args.inj_dirs}")
    KEYS = ("mtot", "net_snr", "is_um", "ev", "fold", "net", "lnl", "stat", "s3", "cnn_hm", "cnn_lm", "cnn_hm_gc", "cnn_lm_gc", "net_loc", "hm_loc", "lm_loc", "s3_loc", "trig", "lr_open", "net_open",
            "far_lr", "far_net", "best_far", "best_ul90", "channel", "det_merge", "veto_keep", "veto_reason", "det_veto", "guard_keep", "det_guard", "cap_adm", "far_cap", "detected", "far_final", "n_lr", "nhm", "nlm", "gate_fail")
    cols = {k: [] for k in KEYS}; n_nonfinite = 0; n_unmeasured = 0
    for f in files:
        z = np.load(f); name = os.path.basename(f).replace("_inj.npz", ""); si = T.name2i[name]; g = int(T.fold[si]); n = len(z["net"])
        for k in ("cnn_hm", "cnn_lm", "s3", "net_loc", "hm_loc", "lm_loc", "s3_loc", "cnn_hm_gc", "cnn_lm_gc"):
            if k not in z.files: raise SystemExit(f"{f}: column {k} missing (inject with m2/vt96/inject96.py: SM_INJ_CNN=1 SM_INJ_S3=1 SM_INJ_ASDVETO=1 SM_CROP=coinc)")
        hm_gc = np.asarray(z["cnn_hm_gc"], float); lm_gc = np.asarray(z["cnn_lm_gc"], float)
        sH, sL, coh, gx = (np.asarray(z[k], float) for k in ("sigH", "sigL", "coh", "gx")); cH2, cL2, gH2, gL2 = (np.asarray(z[k], float) for k in ("cenH2", "cenL2", "gH2", "gL2"))
        cH1, cL1, gH1, gL1 = (np.asarray(z[k], float) for k in ("cenH", "cenL", "gH", "gL")); net = np.asarray(z["net"], float)
        hm = np.asarray(z["cnn_hm"], float); lm = np.asarray(z["cnn_lm"], float); s3 = np.asarray(z["s3"], float)
        net_loc, hm_loc, lm_loc, s3_loc = (np.asarray(z[k], float) for k in ("net_loc", "hm_loc", "lm_loc", "s3_loc"))
        # trigger formation: build_setup zero-lag rule (net > NET_CUT 2.75, centroid-v2 + arm gates) + merge pair rule on the pair itself
        trig = net > T.net_cut
        if T.zl_gates: trig &= M.cen_gate(cH2, cL2) & M.arm_gate(gH2, gL2, sH, sL)
        trig &= rule_pass(gH2, gL2, sH, sL)
        fin = np.isfinite(coh) & np.isfinite(gx) & np.isfinite(sH) & np.isfinite(sL); n_nonfinite += int((trig & ~fin).sum()); trig &= fin
        lnl = np.full(n, np.nan); stat = np.full(n, np.nan)
        if trig.any():
            F = M.feats(sH[trig], sL[trig], coh[trig], cH2[trig], cL2[trig], gH2[trig], gL2[trig], gx=gx[trig]); mu, sd, be = mdl[1 - g]
            lnl[trig] = M.loglr(mu, sd, be, F)
            X = np.column_stack([sH, sL, coh, gx, cH2, cL2, gH2, gL2, cH1, cL1, gH1, gL1])[trig]
            p = np.mean([m_.predict_proba(X)[:, 1] for m_ in models[1 - g]], 0)
            stat[trig] = np.where(lnl[trig] < 3.0, lnl[trig], 3.0 + 20.0 * p)      # == apply_stat_nf.py (FLOOR 3, SCALE 20)
        for i in range(n):
            if not trig[i]:
                r = dict(far_lr_perarm=None, far_net_perarm=None, best_far=None, best_ul90=None, channel=None, detected=False, is_glitch_lr=True, is_glitch=True, netsig_lm_vetoed=False, n_louder_L1fam_cnn=None, far_net_hm=None, far_net_lm=None, below_floor=True)
                nhm = nlm = None
            else:
                r = T.trigger(g, si, float(stat[i]), float(s3[i]), float(net[i]), float(hm[i]), float(lm[i])); r["stat"] = float(stat[i])
                nhm = r["far_net_hm"] * T.live[g] if r["far_net_hm"] is not None else None; nlm = r["far_net_lm"] * T.live[g] if r["far_net_lm"] is not None else None
            ch = chan_of(r) if trig[i] else None
            # 1. local-ASD veto on merge detections
            vk, vr = True, ""
            if r["detected"]:
                if ch == "net-sigma" and not np.isfinite(net_loc[i]): n_unmeasured += 1
                vk, vr = asd_veto(ch, cnn_on, float(net_loc[i]), float(hm_loc[i]), float(lm_loc[i]), float(s3_loc[i]), T.s3_thr, T.netsig_floor)
            det_veto = bool(r["detected"] and vk)
            # 2. net-sigma guard, production rule on the injection's own Grad-CAM-crop scores
            gk = True
            if det_veto and ch == "net-sigma": gk = guard_keep(float(hm_gc[i]), float(lm_gc[i]), float(lnl[i]), float(lm[i]))
            det_guard = bool(det_veto and gk)
            # 3. cap bypass (post-merge, independent of veto and guard)
            adm, far_cap, ul_cap = cap_bypass(run, r, T.live[g]) if trig[i] else (False, None, None)
            detected = bool(det_guard or adm)
            far_final = (r["best_far"] if det_guard else (far_cap if adm else np.inf))
            cols["mtot"].append(float(z["mtot"][i])); cols["net_snr"].append(float(z["net_snr"][i])); cols["is_um"].append(int(z["is_um"][i])); cols["ev"].append(name); cols["fold"].append(g)
            cols["net"].append(float(net[i])); cols["lnl"].append(float(lnl[i])); cols["stat"].append(float(stat[i])); cols["s3"].append(float(s3[i])); cols["cnn_hm"].append(float(hm[i])); cols["cnn_lm"].append(float(lm[i]))
            cols["cnn_hm_gc"].append(float(hm_gc[i])); cols["cnn_lm_gc"].append(float(lm_gc[i])); cols["net_loc"].append(float(net_loc[i])); cols["hm_loc"].append(float(hm_loc[i])); cols["lm_loc"].append(float(lm_loc[i])); cols["s3_loc"].append(float(s3_loc[i]))
            cols["trig"].append(bool(trig[i])); cols["lr_open"].append(bool(trig[i] and not r["is_glitch_lr"])); cols["net_open"].append(bool(trig[i] and r["far_net_perarm"] is not None))
            cols["far_lr"].append(r["far_lr_perarm"] if r["far_lr_perarm"] is not None else np.inf); cols["far_net"].append(r["far_net_perarm"] if r["far_net_perarm"] is not None else np.inf)
            cols["best_far"].append(r["best_far"] if r["best_far"] is not None else np.inf); cols["best_ul90"].append(r["best_ul90"] if r["best_ul90"] is not None else np.inf)
            cols["channel"].append(ch or ""); cols["det_merge"].append(bool(r["detected"])); cols["veto_keep"].append(bool(vk)); cols["veto_reason"].append(vr); cols["det_veto"].append(det_veto)
            cols["guard_keep"].append(bool(gk)); cols["det_guard"].append(det_guard); cols["cap_adm"].append(bool(adm)); cols["far_cap"].append(far_cap if far_cap is not None else np.inf)
            cols["detected"].append(detected); cols["far_final"].append(float(far_final)); cols["n_lr"].append(r["n_louder_L1fam_cnn"] if r["n_louder_L1fam_cnn"] is not None else -1)
            cols["nhm"].append(int(round(nhm)) if nhm is not None else -1); cols["nlm"].append(int(round(nlm)) if nlm is not None else -1)
            cols["gate_fail"].append("" if trig[i] else ("nonfinite" if not fin[i] else "trig"))
        d_ = np.array(cols["detected"][-n:]); dm = np.array(cols["det_merge"][-n:]); dg = np.array(cols["det_guard"][-n:]); ca = np.array(cols["cap_adm"][-n:])
        print(f"[vt96-score] {name} fold {g}: {n} inj, trig {int(trig.sum())}, merge-det {int(dm.sum())} -> after veto+guard {int(dg.sum())}, cap-admitted {int((ca & ~dg).sum())} -> detected {int(d_.sum())} ({(time.time()-t0)/60:.1f} min)", flush=True)
    A = {k: np.array(v) for k, v in cols.items()}
    lev = np.unique(A["net_snr"]); mids = (lev[1:] + lev[:-1]) / 2; dr = np.diff(np.concatenate([[lev[0]], mids, [lev[-1]]])) if len(lev) > 1 else np.ones(1)
    w = lev ** -4.0 * dr; w /= w.sum(); w_of = {float(s): float(x) for s, x in zip(lev, w)}; A["w0"] = np.array([w_of[float(s)] for s in A["net_snr"]])
    A["det_frac"] = A["detected"].astype(float); A["far_mean"] = A["far_final"]
    meta = dict(npair=1, veto_applied=1, feature_clause_on=int(cnn_on), plateau_max_frac=mf, l_cap=L_CAP[run], cell_far=CELL_FAR[run], guard="(gradcam max(hm,lm)>=0.5 OR raw loglr>0) AND lm>0.5 (production rule, netsig_guard.py)",
                tables=args.tables, lr_model=str(args.lr_model or T.lr_model), n_nonfinite_triggers=n_nonfinite, n_unmeasured_local=n_unmeasured,
                inj_dirs=np.array(args.inj_dirs), files=np.array(files), built=time.strftime("%F %T"), snr_levels=lev,
                recipe="version 2 (96): floor 2.75 + zl gates, arm max:-2.05, coinc crop CNN/S3, S3 gate 0.5357 + HGB ranker, LM rule 0.0382, local-ASD veto (net clause; feature clause by plateau), net-sigma guard, cap bypass")
    np.savez(args.out, **A, **meta)
    for s in lev:
        m = A["net_snr"] == s
        print(f"[vt96-score]  SNR {s:5.1f}: {int(m.sum())} inj, trig {A['trig'][m].mean():.3f}, merge-det {A['det_merge'][m].mean():.3f}, vetoed {np.mean(A['det_merge'][m] & ~A['veto_keep'][m]):.3f}, guard-removed {np.mean(A['det_veto'][m] & ~A['guard_keep'][m]):.3f}, cap-added {np.mean(A['cap_adm'][m] & ~A['det_guard'][m]):.3f}, det_frac {A['det_frac'][m].mean():.3f} (loglr {np.mean((A['channel'][m]=='loglr')&A['detected'][m]):.3f}, net-sigma {np.mean((A['channel'][m]=='net-sigma')&A['detected'][m]):.3f})")
    print(f"[vt96-score] wrote {args.out}: {len(A['det_frac'])} injections, {int(A['det_frac'].sum())} detected; non-finite triggers dropped {n_nonfinite}; net-sigma detections without local columns {n_unmeasured} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate"); v.add_argument("--run", required=True); v.add_argument("--tables", required=True); v.add_argument("--outdir", required=True)
    s = sp.add_parser("score"); s.add_argument("--run", required=True); s.add_argument("--tables", required=True); s.add_argument("--inj-dirs", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--lr-model", default=""); s.add_argument("--ranker", default="")
    a = ap.parse_args(); sys.exit(validate(a) if a.cmd == "validate" else score(a))

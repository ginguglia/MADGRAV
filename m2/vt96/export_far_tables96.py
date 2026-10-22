"""vt96 (2026-10-19): export the FAR-setting tables of one run's production _v2 merge (crop re-pass: nf275 shards, s3_crop bank, rank_crop stat) as flat arrays. Copy of m2/vt22/export_far_tables.py with VT96_OUT; arithmetic unchanged.

Mirrors m2/rank/driver_blindscan_merge_hook.py main() lines 105-190 + the CNN shard cache (lines 260-271) VERBATIM in its
arithmetic (same build_setup, same shard merge, same STAT hook, same S3 hook, same arm rule), then stops: nothing is ranked
here. Output = everything the per-trigger FAR of that merge depends on, so far22.py can reproduce the merge's FARs for the
zero-lag triggers (validation) and compute them for injections (the 2.2 recovery field).

Env: the _off5 campaign env of the run (vt22_stage.sbatch sources it), SM_STAT_HOOK, SM_S3_HOOK, SM_S3_THR, SM_ARM_RULE,
SM_NETSIG_LM_MIN, VT96_OUT. Written 2026-10-04."""
import os, sys, glob, json, time, numpy as np
MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

for _p in ("search_mode", "improved", "spectrogram_cascade"):
    _ap = os.path.join(MADGRAV_ROOT, _p)
    if _ap not in sys.path: sys.path.insert(0, _ap)
os.chdir(MADGRAV_ROOT)
import _pb_shard_common as C
import driver_search_multi as M
t0 = time.time(); RUN = os.environ["SM_RUN"]; OUTD = os.environ["VT96_OUT"]; os.makedirs(OUTD, exist_ok=True)
SHARD_DIR = os.environ["SM_SHARD_DIR"]; NSHARD = int(os.environ["SM_NSHARD"])
RULE = os.environ.get("SM_ARM_RULE", "none"); S3_THR = float(os.environ["SM_S3_THR"]); LM_MIN = float(os.environ.get("SM_NETSIG_LM_MIN", "0"))

def load_partials():                                   # == merge_hook.load_partials (hard-fail on a missing shard)
    parts = []; missing = []
    for k in range(NSHARD):
        f = f"{SHARD_DIR}/shard_{k}.npz"
        if os.path.exists(f): parts.append(np.load(f, allow_pickle=False))
        else: missing.append(k)
    if missing: raise SystemExit(f"[vt96-export] FATAL: shards {missing} missing of {NSHARD}")
    return parts
def merge(parts):                                      # == merge_hook.merge
    bg = {0: [], 1: []}; famN = {}; face_live = {0: 0.0, 1: 0.0}
    for p in parts:
        for g, key in ((0, "bg0"), (1, "bg1")):
            for row in p[key]: bg[g].append((float(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])))
        for kk, vv in zip(p["fam_keys"], p["fam_vals"]):
            g = int(kk[0]); kky = (g, (int(kk[1]), int(kk[2]))); net = float(vv[0]); rep = (net, int(vv[1]), int(vv[2]), int(vv[3]), int(vv[4]))
            if kky not in famN or net > famN[kky][0]: famN[kky] = rep
        fl = p["face_live"]; face_live[0] += float(fl[0]); face_live[1] += float(fl[1])
    return bg, famN, face_live

S = C.build_setup(); segs = S["segs"]; fold = S["fold"]; floor = S["floor"]
L1key = lambda bi, iL: C.L1key(segs, bi, iL)
print(f"[vt96-export] {RUN}: setup done, {len(segs)} segs, floor {floor} ({(time.time()-t0)/60:.1f} min)", flush=True)
bg, famN, face_live = merge(load_partials())
# ---- STAT hook (merge lines 113-131) ----
_h = np.load(os.environ["SM_STAT_HOOK"], allow_pickle=True)
_bk = {(str(a), int(b), str(c), int(d)): float(s) for a, b, c, d, s in zip(_h["bank_H1_seg"], _h["bank_iH"], _h["bank_L1_seg"], _h["bank_iL"], _h["bank_stat"])}
_MISS = set()
for g in bg:
    new = []
    for (ll, ai, iH, bi, iL) in bg[g]:
        k = (segs[ai]["name"], int(iH), segs[bi]["name"], int(iL))
        if k in _bk: new.append((_bk[k], ai, iH, bi, iL))
        else: new.append((ll, ai, iH, bi, iL)); _MISS.add((int(ai), int(iH), int(bi), int(iL)))
    bg[g] = new
# ---- S3 hook (merge lines 133-137) ----
_z = np.load(os.environ["SM_S3_HOOK"], allow_pickle=True)
_s3b = {(str(a_), int(b_), str(c_), int(d_)): float(v_) for a_, b_, c_, d_, v_, k_ in zip(_z["H1_seg"], _z["iH"], _z["L1_seg"], _z["iL"], _z["s3"], _z["kind"]) if k_ == "bank"}
# ---- arm rule (merge lines 163-178) ----
def _rule_pass(gH, gL, sH, sL):
    if RULE == "none": return True
    name, thr = RULE.split(":"); thr = float(thr); gH, gL, sH, sL = (float(x) for x in (gH, gL, sH, sL))
    if name == "max": v = gH if sH >= sL else gL
    elif name == "mean": v = 0.5 * (gH + gL)
    elif name == "both": v = min(gH, gL)
    elif name == "ramp": v = min(1.0, max(0.0, sH / 3.0)) * gH + min(1.0, max(0.0, sL / 3.0)) * gL
    else: raise SystemExit(f"unknown SM_ARM_RULE {RULE}")
    return v >= thr
def _pair_ok(ai, iH, bi, iL): A_, B_ = segs[ai], segs[bi]; return _rule_pass(A_["gH"][iH], B_["gL"][iL], A_["sH"][iH], B_["sL"][iL])
n_bg0 = sum(len(v) for v in bg.values()); n_fam0 = len(famN)
if RULE != "none":
    bg = {g: [r for r in bg[g] if _pair_ok(r[1], r[2], r[3], r[4])] for g in bg}
    famN = {k: v for k, v in famN.items() if _pair_ok(v[1], v[2], v[3], v[4])}
_left = [r for g in bg for r in bg[g] if (int(r[1]), int(r[2]), int(r[3]), int(r[4])) in _MISS]
if _left: raise SystemExit(f"[vt96-export] {len(_left)} bank rows after the arm rule have no ranker statistic")
far_live = {g: face_live[g] * M.ANALYZED_FRAC / M.YR for g in (0, 1)}
print(f"[vt96-export] arm rule {RULE}: bg {n_bg0}->{sum(len(v) for v in bg.values())}, fams {n_fam0}->{len(famN)}; livetime {far_live}", flush=True)
# ---- bank arrays ----
B_rows = [(g,) + r for g in (0, 1) for r in bg[g]]
bank_fold = np.array([r[0] for r in B_rows], np.int64); bank_stat = np.array([r[1] for r in B_rows]); bank_ai = np.array([r[2] for r in B_rows], np.int64)
bank_iH = np.array([r[3] for r in B_rows], np.int64); bank_bi = np.array([r[4] for r in B_rows], np.int64); bank_iL = np.array([r[5] for r in B_rows], np.int64)
bank_l1bin = np.array([L1key(r[4], r[5])[1] for r in B_rows], np.int64)
bank_s3 = np.array([_s3b.get((segs[r[2]]["name"], int(r[3]), segs[r[4]]["name"], int(r[5])), np.nan) for r in B_rows])
n_s3_missing = int(np.isnan(bank_s3).sum())
assert (bank_fold == np.array([fold[a] for a in bank_ai])).all(), "bank row fold != fold of its H1 segment"
# ---- net-sigma families + June CNN from the shard cache (merge lines 260-271; all fams with net >= the floor were in need_pairs) ----
cnn = {}
for _f in sorted(glob.glob(f"{M.OUT}/cnn_shard_*.npz")):
    _d = np.load(_f); _P = _d["pairs"]; _H = _d["hm"]; _L = _d["lm"]
    for j in range(len(_P)): cnn[(int(_P[j, 0]), int(_P[j, 1]), int(_P[j, 2]), int(_P[j, 3]))] = (float(_H[j]), float(_L[j]))
F_rows = sorted(famN.items(), key=lambda kv: -kv[1][0])
fam_fold = np.array([k[0] for k, v in F_rows], np.int64); fam_l1bin = np.array([k[1][1] for k, v in F_rows], np.int64)
fam_net = np.array([v[0] for k, v in F_rows]); fam_ai = np.array([v[1] for k, v in F_rows], np.int64); fam_iH = np.array([v[2] for k, v in F_rows], np.int64)
fam_bi = np.array([v[3] for k, v in F_rows], np.int64); fam_iL = np.array([v[4] for k, v in F_rows], np.int64)
_hl = [cnn.get((int(a), int(h), int(b), int(l)), (np.nan, np.nan)) for a, h, b, l in zip(fam_ai, fam_iH, fam_bi, fam_iL)]
fam_hm = np.array([x[0] for x in _hl]); fam_lm = np.array([x[1] for x in _hl])
assert (fam_bi == np.array([k[1][0] for k, v in F_rows])).all()
print(f"[vt96-export] bank rows {len(bank_stat)} (S3 missing {n_s3_missing}); families {len(fam_net)} (CNN missing {int(np.isnan(fam_hm).sum())}, "
      f"of which net>={float(os.environ.get('SM_NETSIG_FLOOR','0'))}: {int((np.isnan(fam_hm) & (fam_net >= float(os.environ.get('SM_NETSIG_FLOOR','0')))).sum())})", flush=True)
out = f"{OUTD}/far_tables_{RUN}.npz"
np.savez(out, run=RUN, built=time.strftime("%F %T"), seg_name=np.array([s["name"] for s in segs]), seg_fold=np.array(fold, np.int64),
         seg_gps0=np.array([float(s["gps"][0]) for s in segs]), seg_gps1=np.array([float(s["gps"][-1]) for s in segs]), seg_n=np.array([int(s["n"]) for s in segs], np.int64),
         bank_fold=bank_fold, bank_stat=bank_stat, bank_ai=bank_ai, bank_iH=bank_iH, bank_bi=bank_bi, bank_iL=bank_iL, bank_l1bin=bank_l1bin, bank_s3=bank_s3,
         fam_fold=fam_fold, fam_net=fam_net, fam_ai=fam_ai, fam_iH=fam_iH, fam_bi=fam_bi, fam_iL=fam_iL, fam_l1bin=fam_l1bin, fam_hm=fam_hm, fam_lm=fam_lm,
         far_live0=far_live[0], far_live1=far_live[1], floor=float(floor), s3_thr=S3_THR, netsig_floor=float(os.environ.get("SM_NETSIG_FLOOR", "0")),
         netsig_lm_min=LM_MIN, glitch_thresh=C.GLITCH_THRESH, net_cut=C.NET_CUT, zl_gates=int(C.ZL_GATES), arm_rule=RULE, merge_s=M.MERGE_S,
         stat_hook=os.environ["SM_STAT_HOOK"], s3_hook=os.environ["SM_S3_HOOK"], shard_dir=SHARD_DIR, lr_model=os.environ.get("SM_LR_MODEL", ""))
print(f"[vt96-export] wrote {out} ({(time.time()-t0)/60:.1f} min)", flush=True)

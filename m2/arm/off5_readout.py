"""Read-out of the O4b arm _off5 validation pass (2026-09-29). Partial-safe: prints whatever merge outputs exist so it can run the
moment detections.json (max:-2.49) lands, before the -2.05 and none merges. Sections: (1) availability + job states, (2) counts per merge,
(3) NEW/LOST vs production max:-2.05 and vs production _s3rank, (4) the O4b windows recovered by the per-event check (incl. the 3 S3-passing
ones) traced through the _off5 blindscan, (5) FAR shift on common detections + livetime/bank size, (6) non-catalogue tail with the LM>0.5 rule,
(7) S3 trigger-table lookup for the _off5 detections. Usage: off5_readout.py [--mail] [--out DIR]. Writes <out>/READOUT_OFF5.txt."""
import os, sys, json, glob, re, subprocess
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH; ROOT = MADGRAV_ROOT; M2 = MADGRAV_ROOT
RUN = os.environ.get("OFF5_RUN", "o4b")   # 2026-09-30: run-generic (o4a pass); o4b keeps the original behaviour
PROD = f"{SC}/madgrav2/search_out_{RUN}_m2_gx"; OUT = f"{SC}/madgrav2/search_out_{RUN}_m2_gx_off5"
if "--out" in sys.argv: OUT = sys.argv[sys.argv.index("--out") + 1]
JIDS = f"{ROOT}/launchers/logs_m2/{RUN}_off5_m2_gx_jids.env"; CAT = f"{ROOT}/search_mode/events_gwtc_all.json"
S3NPZ = f"{MADGRAV_SCRATCH}/madgrav2_spec/s3/s3_{RUN}.npz"; S3THR = json.load(open(f"{M2}/m2/rank/s3_threshold.json"))["threshold"]
EVC = f"{SC}/madgrav2/arm/models/event_check_off5.txt"; S3PASS = {"o4b": ["GW241006_015333", "GW240924_000316", "GW241116_151753"]}.get(RUN, [])
FARK = ["far_lr_perarm", "far_net_perarm", "far_cnn_per_yr", "far_net_per_yr", "far_lr_hm", "far_lr_lm", "far_net_hm", "far_net_lm"]
def load(p):
    try: return json.load(open(p))
    except Exception: return None
def fl(x, d=float("nan")):
    try: return float(x)
    except Exception: return d
def key(x): return x.get("matches_known") or f"gps{fl(x['gps']):.0f}"
def dets(p):
    d = load(p); return None if d is None else {key(x): x for x in d}
def bestfar(t):
    if t.get("best_far") not in (None, ""): return fl(t["best_far"])
    v = [fl(t[k]) for k in FARK if k in t and t[k] not in (None, "")]; return min(v) if v else float("nan")
def fmt(f): return "n/a" if f != f else f"{f:.3g}/yr"
L = []; P = L.append
cat = load(CAT) or {}
# ---------- (1) availability ----------
P(f"MADGRAV 2.1 arm _off5 {RUN.upper()} validation pass: READ-OUT ({os.popen('date +%F_%H:%M').read().strip()})"); P(f"prod {PROD}"); P(f"off5 {OUT}"); P("")
files = {"off5 max:-2.49": "detections.json", "off5 max:-2.05": "detections_p205.json", "off5 none": "detections_none.json"}
bs_files = {"off5 max:-2.49": "blindscan.json", "off5 max:-2.05": "blindscan_p205.json", "off5 none": "blindscan_none.json"}
if RUN != "o4b": files = {"off5 max:-2.05": "detections.json"}; bs_files = {"off5 max:-2.05": "blindscan.json"}   # o4a pass: single merge max:-2.05 -> detections.json
have = {k: os.path.exists(f"{OUT}/{v}") for k, v in files.items()}
P("available: " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in have.items()))
if os.path.exists(JIDS):
    J = dict(l.strip().split("=", 1) for l in open(JIDS) if "=" in l and not l.startswith(("#", "EXP")))
    for st in ("RESCORE", "INJ", "LR", "PRUNE", "TOPUP", "MERGE_249", "MERGE_205", "MERGE_NONE", "MERGE", "SUMMARY"):
        if st in J: P(f"  {st:10s} {J[st]}: " + os.popen(f"sacct -j {J[st]} -X -n -o State 2>/dev/null | tr -d ' ' | sort | uniq -c | tr -s ' ' | tr '\\n' ';'").read().strip())
P("")
R = {"prod max:-2.05": dets(f"{PROD}/detections.json"), "prod _s3rank": dets(f"{PROD}/detections_s3rank.json"), "prod none": dets(f"{PROD}/detections_none.json")}
N = {k: dets(f"{OUT}/{v}") for k, v in files.items()}
# ---------- (2) counts ----------
P("== (2) counts per merge: total | catalogue | non-cat || catalogue at FAR<1 / <0.1 / <0.01 per yr")
for name, D in list(R.items()) + list(N.items()):
    if D is None: P(f"  {name:16s}: MISSING"); continue
    c = [k for k in D if k.startswith("GW")]
    P(f"  {name:16s}: {len(D):3d} | {len(c):3d} | {len(D)-len(c):2d} || " + " / ".join(str(sum(1 for k in c if bestfar(D[k]) < t)) for t in (1, 0.1, 0.01)))
P("")
# ---------- (3) new / lost ----------
P("== (3) NEW / LOST per _off5 merge")
for ref in ("prod max:-2.05", "prod _s3rank"):
    B = R[ref]
    if B is None: P(f"  reference {ref} MISSING"); continue
    for name, D in N.items():
        if D is None: continue
        new_ = sorted(set(D) - set(B)); lost = sorted(set(B) - set(D))
        P(f"  [{name}] vs {ref}: NEW {len(new_)}" + (": " + ", ".join(f"{k} ({fmt(bestfar(D[k]))}, {D[k].get('channel','')}, {D[k].get('kept_by','')})" for k in new_) if new_ else ""))
        P(f"  [{name}] vs {ref}: LOST {len(lost)}" + (": " + ", ".join(f"{k} (ref {fmt(bestfar(B[k]))})" for k in lost) if lost else ""))
P("")
# ---------- (4) recovered windows traced through the blindscan ----------
rec = []
try:
    line = [l for l in open(EVC) if l.startswith("RECOVERED")][0]; rec = re.findall(r"(GW\d{6}_\d{6})\(" + RUN, line)
except Exception: pass
rec = sorted(set(rec) | set(S3PASS)); P(f"== (4) {RUN.upper()} windows recovered by the per-event check ({len(rec)}; * = S3-passing" + (", expected +3 at fixed gate)" if RUN == "o4b" else ")"))
BS = {k: load(f"{OUT}/{v}") for k, v in bs_files.items()}; BSP = load(f"{PROD}/blindscan.json")
def trig_near(bs, gps, win=1.0):
    if not bs: return []
    return sorted([t for t in bs.get("triggers", []) if abs(fl(t.get("gps")) - gps) <= win], key=bestfar)
for ev in rec:
    g = cat.get(ev); tag = "*" if ev in S3PASS else " "
    if g is None: P(f"  {tag}{ev}: not in catalogue file"); continue
    row = [f"  {tag}{ev} gps {g:.1f}:"]
    for name, bs in BS.items():
        if bs is None: row.append(f"[{name}: no file]"); continue
        ts = trig_near(bs, g); D = N.get(name) or {}
        if not ts: row.append(f"[{name}: NO TRIGGER within 1 s]"); continue
        t = ts[0]; row.append(f"[{name}: {'DETECTED' if ev in D else 'trigger only'} net {fl(t.get('net')):.2f} loglr {fl(t.get('loglr')):.1f} hm {fl(t.get('cnn_hm')):.2f} lm {fl(t.get('cnn_lm')):.2f} kept_by {t.get('kept_by','')} minFAR {fmt(bestfar(t))}]")
    tp = trig_near(BSP, g) if BSP else []; row.append(f"[prod: {'trigger, minFAR '+fmt(bestfar(tp[0])) if tp else 'no trigger'}]")
    P(" ".join(row))
P("")
# ---------- (5) FAR shift on common detections + livetime / bank ----------
P("== (5) FAR shift on detections common with prod max:-2.05 (prod -> off5); livetime and bank size")
B = R["prod max:-2.05"]
for name, D in N.items():
    if D is None or B is None: continue
    common = sorted(set(D) & set(B)); rows = [(k, bestfar(B[k]), bestfar(D[k])) for k in common]
    worse = [r for r in rows if r[2] > 2 * r[1] and r[2] > 1e-3]; better = [r for r in rows if r[1] > 2 * r[2] and r[1] > 1e-3]
    ratios = sorted(r[2] / r[1] for r in rows if r[1] > 0 and r[2] > 0)
    P(f"  [{name}] {len(common)} common: median FAR ratio {ratios[len(ratios)//2]:.2f} over {len(ratios)} with both FARs>0; >2x worse {len(worse)}, >2x better {len(better)}")
    for k, a, b in sorted(rows, key=lambda r: -(r[2] - r[1]))[:8]: P(f"     {k:18s} {fmt(a):>10s} -> {fmt(b):>10s}   {D[k].get('channel','')}")
for name, bs in [("prod", BSP)] + list(BS.items()):
    if isinstance(bs, dict): P(f"  [{name}] livetime {bs.get('far_live_yr')} yr, n_triggers {bs.get('n_triggers')}, cnn_vetoed {bs.get('n_cnn_vetoed')}, partial {bs.get('partial')}")
def banklines(pattern):
    for f in sorted(glob.glob(pattern), key=os.path.getmtime)[-1:]:
        return [l.strip()[:160] for l in open(f, errors="ignore") if re.search(r"livetime fold|loud bg survivors|bg L1-families|pass CNN gate|ARM_RULE|arm rule", l)][:6]
    return []
for lab, pat in (("prod merge log", f"{ROOT}/launchers/logs_m2/o4b_m2_far_merge_max_1750934.log" if RUN == "o4b" else f"{ROOT}/launchers/logs_m2/{RUN}_m2_far_merge_max_*.log"), ("off5 -2.49 merge log", f"{ROOT}/launchers/logs_m2/*off5*merge*.log"), ("off5 -2.49 merge log (prep/logs)", f"{M2}/m2/prep/logs/*off5*merge*.log")):
    for l in banklines(pat) or []: P(f"  [{lab}] {l}")
P("")
# ---------- (6) non-catalogue tail ----------
P("== (6) non-catalogue tail per _off5 merge (top 8 by min FAR; LM>0.5 = existing net-sigma LM rule)")
for name, bs in BS.items():
    if bs is None: continue
    nc = sorted([t for t in bs.get("triggers", []) if not t.get("matches_known")], key=bestfar); D = N.get(name) or {}
    P(f"  [{name}] {len(nc)} non-cat triggers, {sum(1 for t in nc if bestfar(t) < 1)} at FAR<1/yr, {sum(1 for k in D if not k.startswith('GW'))} in detections")
    for t in nc[:8]: P(f"     gps {fl(t.get('gps')):.1f} seg {t.get('seg','')} net {fl(t.get('net')):.2f} loglr {fl(t.get('loglr')):.1f} lm {fl(t.get('cnn_lm')):.2f} ({'LM keeps' if fl(t.get('cnn_lm')) > 0.5 else 'LM rule removes'}) kept_by {t.get('kept_by','')} minFAR {fmt(bestfar(t))}")
P("")
# ---------- (7) S3 lookup ----------
P(f"== (7) S3 trigger-table lookup for _off5 detections (threshold {S3THR:.4f}; 'n/a' = trigger not in the production S3 table, needs scoring)")
try:
    import numpy as np; z = np.load(S3NPZ, allow_pickle=True)
    s3t = {(str(a), round(float(g), 1)): float(v) for a, g, v, k in zip(z["H1_seg"], z["gps_H"], z["s3"], z["kind"]) if k == "trig"}
    for name, D in N.items():
        if D is None: continue
        vals = [(k, s3t.get((D[k].get("seg"), round(fl(D[k].get("gps")), 1)))) for k in sorted(D)]
        P(f"  [{name}] S3 pass {sum(1 for _, v in vals if v is not None and v >= S3THR)}, fail {sum(1 for _, v in vals if v is not None and v < S3THR)}, n/a {sum(1 for _, v in vals if v is None)}")
        for k, v in vals:
            if v is None or v < S3THR: P(f"     {k:18s} S3 {'n/a' if v is None else f'{v:.3f} FAIL'}  ({D[k].get('channel','')})")
except Exception as e: P(f"  S3 lookup failed: {e}")
txt = "\n".join(L); print(txt)
try: open(f"{OUT}/READOUT_OFF5.txt", "w").write(txt + "\n")
except Exception as e: print("write failed:", e)
if "--mail" in sys.argv:
    tmp = f"{OUT}/READOUT_OFF5.txt"; mailer = MADGRAV_SCRATCH + "/MADGRAV1/search_mode/pastro_final/campaign/mailer.py"
    r = subprocess.run([sys.executable, mailer, f"[MADGRAV] {RUN.upper()} arm _off5 read-out: " + ", ".join(k for k, v in have.items() if v), tmp], capture_output=True, text=True); print("mail rc", r.returncode, r.stdout[-300:], r.stderr[-300:])

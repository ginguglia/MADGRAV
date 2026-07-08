#!/usr/bin/env python
"""Generic MADGRAV chain progress mailer — run-agnostic clone of the proven o4b_mailer_loop.py.
Direct-MX delivery to the configured recipient. Emails immediately, every PERIOD, and a FINAL email with the
FAR/detections table when the foreground merge completes. Parameterized entirely by env:

  CH_RUN        run label (e.g. o4b) -> shown in every subject/body
  CH_OUT        output dir (default search_out_<run>_far)
  CH_FAR_ARRAY  the FAR cnn-shard array job id (banked-shard progress)
  CH_MERGE      the FAR-merge job id
  CH_FG         the job whose COMPLETED = final (pass the merge jid)
  CH_PRUNE      the background time-slide prune array (the real long-pole)
  CH_SHARD_DIR  _<run>_far_shards (prune writes shard_k.npz here)
  CH_NSHARD     shard count

Detached:  CH_RUN=o4b CH_FAR_ARRAY=.. CH_MERGE=.. CH_FG=.. nohup python chain_mailer.py &
"""
import os, sys, time, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_notify as notify

RUN    = os.environ.get("CH_RUN", "run")
LBL    = RUN.upper()
OUT    = os.environ.get("CH_OUT", os.path.join(os.environ.get("SCRATCH", "."), f"search_out_{RUN}_far"))
FAR    = os.environ.get("CH_FAR_ARRAY", "")
MERGE  = os.environ.get("CH_MERGE", "")
FG     = os.environ.get("CH_FG", "")
PRUNE  = os.environ.get("CH_PRUNE", "")
SHARDD = os.environ.get("CH_SHARD_DIR", "")
NSHARD = int(os.environ.get("CH_NSHARD", "64"))
PERIOD = 1800          # 30 min
MAX_HOURS = 48         # multi-day-capable; safety cap only
LOAD_HI = 15.0         # login 1-min load that flags [load] in the subject (clog early-warning)

def sh(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception: return ""

def loadavg():
    try: return os.getloadavg()[0]
    except Exception: return 0.0

def jstate(jid):
    if not jid: return "?"
    q = sh(["squeue", "-j", jid, "-h", "-o", "%T"]).strip()
    if q: return q.split()[0]
    return (sh(["sacct", "-j", jid, "--format=State", "-X", "-n"]).strip().split() or ["finished"])[0]

def done(jid):
    return bool(jid) and "COMPLETED" in sh(["sacct", "-j", jid, "--format=State", "-X", "-n"])

def status():
    banked = len(glob.glob(f"{OUT}/cnn_shard_*.npz"))
    q = sh(["squeue", "-j", FAR, "-h", "-o", "%t"]).split() if FAR else []
    run, pend = q.count("R"), q.count("PD")
    sac = sh(["sacct", "-j", FAR, "--format=State", "-X", "-n"]) if FAR else ""
    return banked, run, pend, sac.count("COMPLETED"), sac.count("OUT_OF_ME"), sac.count("FAILED"), sac.count("CANCELLED")

def prune_status():
    banked = len(glob.glob(f"{SHARDD}/shard_*.npz")) if SHARDD else 0
    q = sh(["squeue", "-j", PRUNE, "-h", "-o", "%t"]).split() if PRUNE else []
    sac = sh(["sacct", "-j", PRUNE, "--format=State", "-X", "-n"]) if PRUNE else ""
    return banked, q.count("R"), q.count("PD"), sac.count("COMPLETED"), sac.count("OUT_OF_ME"), sac.count("FAILED")

def body():
    b, run, pend, comp, oom, fail, canc = status()
    pb, prun, ppend, pcomp, poom, pfail = prune_status()
    return (f"MADGRAV {LBL} auto-pipeline\n{time.strftime('%F %H:%M')}\n\n"
            f"  login load (1m)   : {loadavg():.1f}\n"
            f"  FAR prune (bg)    : banked {pb}/{NSHARD}  running {prun}  pending {ppend}  completed {pcomp}  OOM {poom}  fail {pfail}  [{jstate(PRUNE)}]\n"
            f"  CNN shards banked : {b}/{NSHARD}\n"
            f"  CNN array         : running {run}  pending {pend}  completed {comp}  OOM {oom}  fail {fail}  cancelled {canc}  [{jstate(FAR)}]\n"
            f"  FAR merge         : {jstate(MERGE)}\n"
            f"  Foreground        : {jstate(FG)}\n")

def alert_prefix():
    b, run, pend, comp, oom, fail, canc = status()
    pre = ""
    if loadavg() > LOAD_HI: pre += "[load] "
    started = (comp + oom + fail + canc) > 0 or run > 0
    far_terminal = started and run == 0 and pend == 0
    if not done(FG) and far_terminal and b < NSHARD: pre += "[STALL] "
    return pre

def far_table():
    import json
    try: d = json.load(open(f"{OUT}/detections.json"))
    except Exception as e: return f"(detections.json not readable yet: {e})"
    try: live = json.load(open(f"{OUT}/blindscan.json")).get("far_live_yr")
    except Exception: live = "?"
    f = lambda x, p=".3g": ("n/a" if x is None else format(x, p))
    L = [f"{LBL} — {len(d)} detection(s)  (bg livetime yr/fold: {live})", "",
         f"{'gps':>12} {'net':>5} {'loglr':>6} {'bestFAR/yr':>11} {'UL90':>8} {'chan':>5}  matches_known", "-"*74]
    for t in sorted(d, key=lambda x: (x.get("best_far") if x.get("best_far") is not None else 9e9)):
        L.append(f"{t.get('gps',0):>12.0f} {t.get('net',0):>5.2f} {t.get('loglr',0):>6.2f} "
                 f"{f(t.get('best_far')):>11} {f(t.get('best_ul90')):>8} {str(t.get('channel','')):>5}  {t.get('matches_known','')}")
    return "\n".join(L)

# status() -> body() text is ALWAYS written to <out>/STATUS.txt; email only if NOTIFY_EMAIL is set.
def emit(subj, txt):
    notify.deliver(OUT, "STATUS.txt", subj, txt)

b, *_ = status()
emit(f"{alert_prefix()}[MADGRAV] {LBL} pipeline STARTED — {b}/{NSHARD} FAR shards banked", body())
t0 = time.time()
while time.time() - t0 < MAX_HOURS * 3600:
    time.sleep(PERIOD)
    if done(FG):
        emit(f"[MADGRAV] {LBL} COMPLETE — foreground done",
             body() + "\n" + "="*40 + "\n" + far_table() + f"\n\n(full JSON in {OUT})")
        break
    emit(f"{alert_prefix()}[MADGRAV] {LBL} — {len(glob.glob(f'{OUT}/cnn_shard_*.npz'))}/{NSHARD} banked", body())
print("[mailer] loop exit", flush=True)

#!/usr/bin/env python
"""Generic MADGRAV chain finalizer (detached). Waits for the FAR merge (+ injection campaign), then:
pastro_search.py (provisional p_astro, only if the run's injection dir exists) -> make_deck.py
(FAR/UL90 plots, NO LLM narrative) -> ALWAYS writes <out>/SUMMARY.txt (the results table) and, if
NOTIFY_EMAIL is configured (site.conf), emails the same content + the HTML deck. So results are visible
with zero config (SUMMARY.txt + deck.html + job log); email is an optional add-on.
Paths come from the environment (site.conf via run_chain.sh). Run label from env CH_RUN.
Usage: CH_RUN=o4b python chain_finalize.py <merge_jid> <inject_jid> <out_dir>
"""
import os, sys, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chain_notify as notify

JMRG, JINJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
RUN = os.environ.get("CH_RUN", "run"); LBL = RUN.upper()
MR = os.environ["MADGRAV_ROOT"]; SM = f"{MR}/search_mode"; L = f"{MR}/launchers"
PY = os.environ["PYTHON"]; SC = os.environ["SCRATCH"]
FROZEN = os.environ.get("FROZEN_MODEL", f"{MR}/data/o3a_frozen_lr_off200.npz")
DECK = f"{OUT}/deck.html"

def sh(c):
    try: return subprocess.run(c, capture_output=True, text=True, timeout=40).stdout
    except Exception: return ""
def state(j):
    q = sh(["squeue", "-j", j, "-h", "-o", "%T"]).strip()
    if q: return q.split()[0]
    return (sh(["sacct", "-j", j, "--format=State", "-X", "-n"]).strip().split() or ["UNKNOWN"])[0]

def table(path):
    try: d = json.load(open(path))
    except Exception as e: return f"(detections not readable: {e})", 0
    f = lambda x, p=".3g": "n/a" if x is None else format(x, p)
    hp = any(x.get("p_astro") is not None for x in d)
    L2 = [f"{len(d)} detection(s)" + (" (p_astro PROVISIONAL)" if hp else "") + ":", ""]
    for t in sorted(d, key=lambda x: (x.get('best_far') if x.get('best_far') is not None else 9e9)):
        row = (f"  {str(t.get('matches_known') or t.get('seg')):20} net={t.get('net',0):5.1f} "
               f"loglr={t.get('loglr',0):6.2f} bestFAR={f(t.get('best_far'))}/yr")
        if hp: row += f" p_astro={f(t.get('p_astro'),'.3f')}"
        L2.append(row)
    return "\n".join(L2), len(d)

print(f"[finalize] {LBL}: waiting on merge {JMRG} (+ inject {JINJ})", flush=True)
t0 = time.time()
while time.time() - t0 < 3 * 24 * 3600:
    m = state(JMRG)
    if m in ("", "PENDING", "RUNNING", "REQUEUED", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING"):
        time.sleep(120); continue
    if m != "COMPLETED":
        notify.deliver(OUT, "SUMMARY.txt", f"[MADGRAV] {LBL} chain FAILED at merge: {m} (job {JMRG})",
                       f"FAR merge terminal state {m}. See launchers/{RUN}_far_merge_{JMRG}.log")
        sys.exit(1)
    if state(JINJ) != "COMPLETED" and time.time() - t0 < 4 * 3600:
        time.sleep(120); continue
    break

# p_astro (provisional) only if the injection campaign completed AND its output dir exists
if state(JINJ) == "COMPLETED" and os.path.isdir(f"{SM}/inj_out_{RUN}"):
    env = dict(os.environ, MADGRAV_ROOT=MR, SM_OUT=OUT, SM_INJ=f"{SM}/inj_out_{RUN}", SM_LR_MODEL=FROZEN)
    r = subprocess.run([PY, f"{SM}/pastro_search.py"], env=env, capture_output=True, text=True)
    print("[pastro]", r.stdout[-1500:], r.stderr[-800:], flush=True)
else:
    print(f"[finalize] inject not usable ({state(JINJ)}, dir={os.path.isdir(f'{SM}/inj_out_{RUN}')}) -> deck WITHOUT p_astro", flush=True)

# deck: deterministic plots + table, NO LLM narrative (DECK_LLM=0)
env = dict(os.environ, DECK_OUT=OUT, DECK_FILE=DECK, DECK_LLM="0", DECK_TITLE=f"MADGRAV {LBL} blind search")
r = subprocess.run([PY, f"{L}/make_deck.py"], env=env, capture_output=True, text=True)
print("[deck]", r.stdout, r.stderr, flush=True)

# results: ALWAYS write SUMMARY.txt; email the same content (+ HTML deck) only if NOTIFY_EMAIL is set
src = f"{OUT}/detections_pastro.json" if os.path.exists(f"{OUT}/detections_pastro.json") else f"{OUT}/detections.json"
body_tbl, n = table(src)
body = (f"MADGRAV {LBL} blind search COMPLETE.\n\n{body_tbl}\n\n"
        f"Outputs in {OUT}\n  detections: {os.path.basename(src)}\n  full report (plots): {DECK}\n")
html = open(DECK, encoding="utf-8").read() if os.path.exists(DECK) else None
notify.deliver(OUT, "SUMMARY.txt", f"[MADGRAV] {LBL} blind search COMPLETE ({n} dets)", body, html=html)
print("[finalize] DONE", flush=True)

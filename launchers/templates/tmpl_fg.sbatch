#!/usr/bin/env bash
# BLIND foreground PREVIEW — SM_FG_ONLY. Runs the zero-lag scan ONLY: writes recovery.json (candidate
# list + CNN glitch-gate) then EXITS before the time-slide background. Runs in PARALLEL with the prune
# (own gpu:1), delays the FAR chain by nothing. Emails the candidate table INLINE. Candidate-selection
# knobs are PINNED to the FAR chain (via _science) so the preview list == the merge's.
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
source "$MADGRAV_ROOT/launchers/templates/_science.sh"
unset SM_COH_CEIL SM_NET_GATE                          # background-only knobs; irrelevant to FG candidate finding
export SM_OUT="$SC/search_out_${SM_RUN}_fg"; mkdir -p "$SM_OUT"   # SEPARATE dir; no collide with the FAR merge's _far
export SM_FG_ONLY=1 SM_QT_WORKERS="${SM_QT_WORKERS:-20}"
echo "[$SM_RUN-fg] START $(date '+%F %T')  cache=$SM_SERIESCACHE"
BLIND_DEV=cuda:0 "$PY" "$SM/driver_blindscan.py"; rc=$?
echo "[$SM_RUN-fg] END rc=$rc $(date '+%F %T')"; ls -la "$SM_OUT/recovery.json" 2>/dev/null
# ---- disconnect-proof candidate list: ALWAYS write FG_PREVIEW.txt; email only if NOTIFY_EMAIL is set ----
"$PY" - "$rc" "$SM_OUT" "$SM_RUN" <<'PYEOF' || true
import sys, json, os
sys.path.insert(0, os.path.join(os.environ["MADGRAV_ROOT"], "launchers"))
import chain_notify as notify
rc, out, run = sys.argv[1], sys.argv[2], sys.argv[3]; RUN = run.upper()
try:
    d = json.load(open(f"{out}/recovery.json")); recs = d.get("recovered", [])
    hdr = (f"{RUN} FOREGROUND PREVIEW (rc={rc}) — candidate list, FAR PENDING background.\n"
           f"segments={d.get('n_segments')}  triggers={d.get('n_triggers')}  "
           f"candidates={d.get('n_candidates')}  pass_CNN_glitch_gate={d.get('n_pass_cnn')}\n"
           f"floor loglr>={d.get('floor')}  net-sigma floor>={d.get('netsig_floor')}  glitch_thresh={d.get('glitch_thresh')}\n\n"
           f"{'gps':>12} {'net':>5} {'loglr':>6} {'cnn_hm':>6} {'cnn_lm':>6} {'gate':>6} {'kept_by':>8}  seg / matches_known\n" + "-"*92 + "\n")
    f = lambda x: ("n/a" if x is None else format(x, ".3f"))
    kept = [r for r in recs if not r.get("is_glitch")]
    lines = []
    for r in kept + [r for r in recs if r.get("is_glitch")]:
        lines.append(f"{r.get('gps',0):>12.1f} {r.get('net',0):>5.1f} {r.get('loglr',0):>6.2f} "
                     f"{f(r.get('cnn_hm'))} {f(r.get('cnn_lm'))} "
                     f"{'PASS' if not r.get('is_glitch') else 'glitch':>6} {str(r.get('kept_by','')):>8}  "
                     f"{r.get('seg','')} {('<'+r['matches_known']+'>') if r.get('matches_known') else ''}")
    body = hdr + "\n".join(lines[:200]) + f"\n\n({len(kept)} pass the CNN glitch-gate; full JSON in {out}/recovery.json)"
    subj = f"[MADGRAV] {RUN} FOREGROUND preview — {len(kept)}/{len(recs)} candidates pass CNN gate (FAR pending)"
except Exception as e:
    body = f"{RUN} foreground preview finished rc={rc} but recovery.json unreadable: {e}"
    subj = f"[MADGRAV] {RUN} FOREGROUND preview — ERROR (rc={rc})"
notify.deliver(out, "FG_PREVIEW.txt", subj, body)
PYEOF
exit $rc

#!/usr/bin/env bash
# FAR MERGE — combines the cnn_shard_*.npz -> blindscan.json / detections.json / survivors_bg.json
# (+ per-trigger FAR, ASD veto ON). Guards fail-closed (coverage check; FAR Gate 6). Emails the
# detections summary INLINE on completion. gpu:2 for merge headroom. SM_NSHARD injected at submit.
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
source "$MADGRAV_ROOT/launchers/templates/_science.sh"
: "${SM_NSHARD:?}"
mkdir -p "$SM_OUT"
export SM_PRECOMPUTE_MAXGB="${SM_PRECOMPUTE_MAXGB:-160}" SM_FG_MAXNET="${SM_FG_MAXNET:-400}" SM_MERGE_OK="${SM_MERGE_OK:-1}"
export SM_QT_WORKERS="${SM_QT_WORKERS:-40}" SM_CNN_CACHE="${SM_CNN_CACHE:-1}"
echo "[$SM_RUN-far-merge] START $(date '+%F %T')"
BLIND_DEV=cuda:0 "$PY" "$SM/driver_blindscan_merge.py"; rc=$?
echo "[$SM_RUN-far-merge] END rc=$rc $(date '+%F %T')"; ls -la "$SM_OUT"
# ---- disconnect-proof result: ALWAYS write MERGE_DONE.txt; email only if NOTIFY_EMAIL is set ----
"$PY" - "$rc" "$SM_OUT" "$SM_RUN" <<'PYEOF' || true
import sys, json, os
sys.path.insert(0, os.path.join(os.environ["MADGRAV_ROOT"], "launchers"))
import chain_notify as notify
rc, out, run = sys.argv[1], sys.argv[2], sys.argv[3]; RUN = run.upper()
try: d = json.load(open(f"{out}/detections.json"))
except Exception as e: d = [{"error": str(e)}]
body = f"Full-{RUN} FAR merge complete (rc={rc}). {len(d)} detection(s).\n\n" + json.dumps(d, indent=1)[:6000]
notify.deliver(out, "MERGE_DONE.txt", f"[MADGRAV] {RUN} FAR merge COMPLETE ({len(d)} dets)", body)
PYEOF
exit $rc

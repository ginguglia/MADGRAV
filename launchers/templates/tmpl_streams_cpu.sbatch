#!/usr/bin/env bash
# STREAMS build, CPU-partition copy of tmpl_streams.sbatch (2026-09-26; ~335 of 546 GPU-h of a 2.1 pass were GPU-idle CPU work) — each shard does seg[idx::NSHARD]. Resumable (skips existing _meta.npz).
# Needs the reference ASD (prep). STREAMS_NSHARD + array injected at submit by run_chain.sh.
#SBATCH --cpus-per-task=32 --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
: "${STREAMS_NSHARD:?}"
mkdir -p "$SM_STREAMS"
export SM_DEV=cpu SM_ALLOW_CPU=1 SM_QT_WORKERS=${SM_QT_WORKERS:-24} OMP_NUM_THREADS=8   # CPU lane (2026-09-26): the QT is the cost, the CAE/arm forward is small
export NSHARD="$STREAMS_NSHARD" SHARD="${SLURM_ARRAY_TASK_ID}"
echo "[$SM_RUN-str] shard $SHARD/$NSHARD host=$(hostname) start=$(date '+%F %T')"
t0=$SECONDS
"$PY" "$SM/driver_streams_bg_shard.py"
rc=$?
echo "[$SM_RUN-str] shard $SHARD rc=$rc WALL=$((SECONDS-t0))s end=$(date '+%F %T')"
exit $rc

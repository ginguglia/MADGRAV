#!/usr/bin/env bash
# STREAMS build — each shard does seg[idx::NSHARD]. Resumable (skips existing _meta.npz).
# Needs the reference ASD (prep). STREAMS_NSHARD + array injected at submit by run_chain.sh.
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
: "${STREAMS_NSHARD:?}"
mkdir -p "$SM_STREAMS"
export SM_DEV=cuda:0
export NSHARD="$STREAMS_NSHARD" SHARD="${SLURM_ARRAY_TASK_ID}"
echo "[$SM_RUN-str] shard $SHARD/$NSHARD host=$(hostname) start=$(date '+%F %T')"
t0=$SECONDS
"$PY" "$SM/driver_streams_bg_shard.py"
echo "[$SM_RUN-str] shard $SHARD rc=$? WALL=$((SECONDS-t0))s end=$(date '+%F %T')"

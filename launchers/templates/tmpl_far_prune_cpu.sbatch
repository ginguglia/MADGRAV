#!/usr/bin/env bash
# FAR background PRUNE, CPU-partition copy of tmpl_far_prune.sbatch (2026-09-26; single-thread CPU-bound in 2.1) — shards over all coincident segments. Writes shard_k.npz (partial background
# time-slide survivors) to _<run>_far_shards — the input cnn_shard + far_merge CONSUME. THE ~12-18h
# long-pole. Resumable (skips shard_k.npz already written). SM_NSHARD + array injected at submit.
#SBATCH --cpus-per-task=8 --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
source "$MADGRAV_ROOT/launchers/templates/_science.sh"
: "${SM_NSHARD:?}"
export SM_SHARD="${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SM_SHARD_DIR" "$SM_OUT"
if [ -f "$SM_SHARD_DIR/shard_${SM_SHARD}.npz" ]; then echo "[$SM_RUN-far] shard $SM_SHARD done -> skip"; exit 0; fi
export SM_SURV_ABORT="${SM_SURV_ABORT:-25000000}" SM_HOST_MEM_GB="${SM_HOST_MEM_GB:-40}"
export SM_PRUNE_LOG_EVERY="${SM_PRUNE_LOG_EVERY:-200}"
echo "[$SM_RUN-far] shard $SM_SHARD/$SM_NSHARD host=$(hostname) maxoff=$SM_MAX_OFFSETS netgate=$SM_NET_GATE ceil=$SM_COH_CEIL start=$(date '+%F %T')"
t0=$SECONDS
BLIND_DEV=cpu SM_ALLOW_CPU=1 OMP_NUM_THREADS=8 "$PY" "$SM/driver_blindscan.py"
echo "[$SM_RUN-far] shard $SM_SHARD rc=$? WALL_S=$((SECONDS-t0)) end=$(date '+%F %T')"

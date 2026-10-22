#!/usr/bin/env bash
# Series cache (whitened coh-series) — CPU QOS, 2h/shard, resumable (exists-skip). Needs the reference
# ASD (prep). Independent of streams; both feed the FAR cnn-shard stage. CACHE_NSHARD + array injected.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
: "${CACHE_NSHARD:?}"
mkdir -p "$SM_SERIESCACHE"
export SM_DEV=cpu SM_ALLOW_CPU=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MALLOC_ARENA_MAX=2
export SM_CACHE_NSHARD="$CACHE_NSHARD" SM_CACHE_SHARD="${SLURM_ARRAY_TASK_ID}"
echo "[$SM_RUN-cache] shard $SM_CACHE_SHARD/$CACHE_NSHARD host=$(hostname) start=$(date '+%F %T')"
"$PY" "$SM/build_series_cache_shard.py"
echo "[$SM_RUN-cache] shard $SM_CACHE_SHARD rc=$? end=$(date '+%F %T')"

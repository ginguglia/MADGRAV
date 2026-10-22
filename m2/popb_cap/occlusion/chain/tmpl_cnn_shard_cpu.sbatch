#!/usr/bin/env bash
# CNN-shard array on the CPU partition (crop re-pass, 2026-10-10): driver_blindscan_merge.py with SM_CNN_SHARD=k:NSHARD -> cnn_shard_k.npz in SM_OUT.
# Same worker code as launchers/templates/tmpl_cnn_shard.sbatch; BLIND_DEV=cpu SM_ALLOW_CPU=1 (whole cache on one device); SM_CROP from EXP.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=16 --mem=96G --time=06:00:00 --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?}"; source "$MADGRAV_ROOT/launchers/templates/_common.sh"; source "$MADGRAV_ROOT/launchers/templates/_science.sh"
: "${SM_NSHARD:?}" "${SM_CROP:?}"; mkdir -p "$SM_OUT"
export SM_PRECOMPUTE_MAXGB="${SM_PRECOMPUTE_MAXGB:-40}" SM_FG_MAXNET="${SM_FG_MAXNET:-400}" SM_MERGE_OK=1 SM_QT_WORKERS="${SM_QT_WORKERS:-14}" SM_BLOCK_CKPT=1
export SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4
export SM_CNN_SHARD="${SLURM_ARRAY_TASK_ID}:${SM_NSHARD}"
echo "[$SM_RUN-cnn-shard-cpu] START $(date '+%F %T') shard=${SLURM_ARRAY_TASK_ID}/$SM_NSHARD crop=$SM_CROP out=$SM_OUT shards=$SM_SHARD_DIR host=$(hostname)"
cd "$MADGRAV_ROOT"; "$PY" "$SM/driver_blindscan_merge.py"; rc=$?
echo "[$SM_RUN-cnn-shard-cpu] END rc=$rc $(date '+%F %T')"; ls -la "$SM_OUT"/cnn_shard_${SLURM_ARRAY_TASK_ID}.npz 2>/dev/null; exit $rc

#!/usr/bin/env bash
# FAR cnn-shard array — driver_blindscan_merge.py with SM_CNN_SHARD=k:NSHARD -> cnn_shard_k.npz (sharded
# CNN precompute + FAR). gpu:2 (half-node ~372GB, the OOM fix), block-ckpt on (resumes the shard on
# requeue). CONSUMES _<run>_far_shards from the prune. SM_NSHARD + array injected at submit.
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --requeue
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
source "$MADGRAV_ROOT/launchers/templates/_science.sh"
: "${SM_NSHARD:?}"
mkdir -p "$SM_OUT"
export SM_PRECOMPUTE_MAXGB="${SM_PRECOMPUTE_MAXGB:-48}" SM_FG_MAXNET="${SM_FG_MAXNET:-400}" SM_MERGE_OK="${SM_MERGE_OK:-1}"
export SM_QT_WORKERS="${SM_QT_WORKERS:-44}" SM_BLOCK_CKPT="${SM_BLOCK_CKPT:-1}"
echo "[$SM_RUN-cnn-shard] START $(date '+%F %T') shard=${SLURM_ARRAY_TASK_ID}/$SM_NSHARD"
export SM_CNN_SHARD="${SLURM_ARRAY_TASK_ID}:${SM_NSHARD}"   # k:NSHARD — matches array size + SM_NSHARD
BLIND_DEV=cuda:0 "$PY" "$SM/driver_blindscan_merge.py"; rc=$?
echo "[$SM_RUN-cnn-shard] END rc=$rc $(date '+%F %T')"; ls -la "$SM_OUT"/cnn_shard_${SLURM_ARRAY_TASK_ID}.npz 2>/dev/null
exit $rc

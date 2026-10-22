#!/usr/bin/env bash
# CNN top-up on the CPU partition (crop re-pass): launchers/cnn_topup.py with the chain's EXP (SM_RUN_SEL=<run>_off5_crop), SM_TOPUP_IDX=902.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=16 --mem=120G --time=06:00:00 --requeue
set -uo pipefail; ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL:?}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_PRECOMPUTE_MAXGB=40 SM_QT_WORKERS=14 SM_ARM_TOPUP=${SM_ARM_TOPUP:--1e9} SM_TOPUP_IDX=${SM_TOPUP_IDX:-902} SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4
: "${SM_CROP:?SM_CROP missing from EXP}"
n=$(ls $SM_OUT/cnn_shard_[0-9]*.npz 2>/dev/null | wc -l); [ "$n" -ge "$SM_NSHARD" ] || { echo "FATAL only $n/$SM_NSHARD cnn shards in $SM_OUT"; exit 1; }
cd $ROOT; echo "[topup-cpu] START $(date '+%F %T') run=$SM_RUN_SEL crop=$SM_CROP out=$SM_OUT host=$(hostname)"
"$PY" launchers/cnn_topup.py; rc=$?; echo "[topup-cpu] END rc=$rc $(date '+%F %T')"; ls -la $SM_OUT/cnn_shard_902.npz 2>/dev/null; exit $rc

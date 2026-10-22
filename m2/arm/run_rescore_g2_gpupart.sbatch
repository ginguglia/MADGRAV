#!/usr/bin/env bash
# Rescore g2 (arm v2) of the O4b production streams with the _off5 arm on the CPU partition (0 GPU-h). Array = shards over segments.
# GPU-PARTITION VARIANT (2026-10-01): identical to run_rescore_g2.sbatch except partition/qos/gres (1 H100 slice = 22 cores, quarter-node memory; the submit filter forbids CPU-only jobs) and thread counts. Used only when the CPU partition is blocked; costs 1 GPU-h per slice-hour with the GPU idle.
# Env in (sbatch --export): SRC DST NSHARD SM_ARM_V2_TAG [SMOKE] ; resumable (skips existing DST metas).
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:1 --time=10:00:00 --requeue
#SBATCH --job-name=m2-o4b-rescore-g2 --output=${MADGRAV_ROOT}/m2/prep/logs/rescore_g2_%A_%a.log
set -uo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT} SC=${MADGRAV_SCRATCH}
export SM_RUN=${SM_RUN:-o4b} SM_DEV=cpu SM_ALLOW_CPU=1 BLIND_DEV=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-22} SM_QT_WORKERS=${SM_QT_WORKERS:-20} MALLOC_ARENA_MAX=2
export SM_STRAIN=$SC/strain_${SM_RUN}_full SM_BGJSON=$SC/${SM_RUN}_full_coincident.json SM_PREP=${MADGRAV_ROOT}/data/${SM_RUN}_search_prep
export SM_ARM_V2=1 SM_ARM_V2_DIR=$SC/madgrav2/arm/models SM_ARM_V2_TAG=${SM_ARM_V2_TAG:-_off5} SRC_ARM_TAG=_def
export SHARD=${SLURM_ARRAY_TASK_ID:-0} NSHARD=${NSHARD:?} SRC=${SRC:?} DST=${DST:?} SMIN=${SMIN:-0} SMOKE=${SMOKE:-0}
echo "[rescore] shard $SHARD/$NSHARD host=$(hostname) start=$(date '+%F %T') tag=$SM_ARM_V2_TAG SRC=$SRC DST=$DST SMOKE=$SMOKE"
t0=$SECONDS; ${PY:-python} ${MADGRAV_ROOT}/m2/arm/rescore_g2.py; rc=$?
echo "[rescore] shard $SHARD rc=$rc WALL=$((SECONDS-t0))s end=$(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
# S3 (hm 20-300 Hz ensemble) scores for the zero-lag triggers of the _off5 O4a pass (lnL>=3 plus every catalogue detection), CPU partition.
# Bank rows: 16 campaign rows, tripwire only. Output: ${MADGRAV_SCRATCH}/madgrav2/s3_off5/s3_o4a_t0.npz (NOT the production s3_o4a.npz).
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=16 --mem=120G --time=03:00:00 --job-name=m2-s3-trig-off5-o4a
#SBATCH --output=${MADGRAV_ROOT}/m2/arm/logs/s3trig_%x_%j.log
set -uo pipefail
ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/o4a_off5_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_DEV=cpu BLIND_DEV=cpu SM_QT_WORKERS=14 OMP_NUM_THREADS=16 MALLOC_ARENA_MAX=2
export R_TASK=0 R_NTASK=1 R_MAX=16 R_OUT=${MADGRAV_SCRATCH}/madgrav2/s3_off5
echo "[s3-trig-off5] START $(date '+%F %T') run=$SM_RUN OUT=$SM_OUT host=$(hostname)"
"$PY" ${MADGRAV_ROOT}/m2/arm/s3off5/score_trig_s3_off5.py; rc=$?
echo "[s3-trig-off5] END rc=$rc $(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
# f60 mains veto on detections_s3rank.json of one run (2026-10-05). CPU partition. Usage: sbatch --export=ALL,SM_RUN_SEL=o3b f60.sbatch
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=16 --mem=120G --time=01:00:00
#SBATCH --output=${MADGRAV_ROOT}/m2/arm/s3off5/logs/%x_%A.log
set -uo pipefail; : "${SM_RUN_SEL:?}"
ROOT=${MADGRAV_ROOT}; W=${MADGRAV_ROOT}/m2/arm/s3off5
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_off5_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 OMP_NUM_THREADS=8
[ "$SM_RUN" = "$SM_RUN_SEL" ] || { echo "FATAL run mismatch $SM_RUN vs $SM_RUN_SEL"; exit 1; }
case "$SM_OUT" in *_off5) ;; *) echo "FATAL SM_OUT is not an _off5 dir: $SM_OUT"; exit 1;; esac
echo "[f60] START $(date '+%F %T') run=$SM_RUN OUT=$SM_OUT host=$(hostname)"
cd $ROOT; "$PY" $W/f60_s3rank.py; rc=$?
echo "[f60] END rc=$rc $(date '+%F %T')"; exit $rc

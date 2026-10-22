#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:1 --mem=120G
#SBATCH --time=06:00:00 --job-name=m2-spec-pos
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/pos_%x_%j.log
# env in: SM_RUN_SEL (o3a|o3b|o4a|o4b), R_TASK, R_NTASK [, R_MAXCHUNKS, R_OUT for smoke]
set -uo pipefail
ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_QT_WORKERS=20 BLIND_DEV=cuda:0 MALLOC_ARENA_MAX=2
echo "[spec-pos] START $(date '+%F %T') run=$SM_RUN task=${R_TASK:-0}/${R_NTASK:-4} host=$(hostname)"
"$PY" ${MADGRAV_ROOT}/m2/spec/render_positives.py; rc=$?
echo "[spec-pos] END rc=$rc $(date '+%F %T')"; exit $rc

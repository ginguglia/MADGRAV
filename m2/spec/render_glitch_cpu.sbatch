#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=32 --mem=120G
#SBATCH --time=06:00:00 --job-name=m2-spec-glitch-cpu
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/glitch_%x_%j.log
# env in: SM_RUN_SEL (o3a|o3b|o4a|o4b), R_MODE (gneg|gpos), R_TASK, R_NTASK [, R_NDRAW, R_OUT, R_MAXCHUNKS, R_GG_CLASSES]
set -uo pipefail
ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_QT_WORKERS=30 BLIND_DEV=cpu SM_DEV=cpu SM_ALLOW_CPU=1 OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2   # CPU variant (2026-09-30): the render is QT-bound, GPU idle
echo "[spec-glitch] START $(date '+%F %T') run=$SM_RUN mode=${R_MODE:-gneg} task=${R_TASK:-0}/${R_NTASK:-2} ndraw=${R_NDRAW:-5000} host=$(hostname)"
"$PY" ${MADGRAV_ROOT}/m2/spec/render_glitch.py; rc=$?
echo "[spec-glitch] END rc=$rc $(date '+%F %T')"; exit $rc

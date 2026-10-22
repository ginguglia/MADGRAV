#!/usr/bin/env bash
#SBATCH --account=${SLURM_ACCOUNT} --partition=${PARTITION_CPU} --qos=${QOS_CPU} --cpus-per-task=16 --mem=240G --time=05:00:00
#SBATCH --job-name=m2-rank-bg
#SBATCH --output=${MADGRAV_ROOT}/m2/rank/logs/bg_%x_%j.log
# env in: SM_RUN_SEL (o3a|o3b|o4a|o4b) [, R_NCAP, R_OUT]; campaign env taken from the gx run's jids file (as render_negatives.sbatch)
set -uo pipefail
ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_DEV=cpu OMP_NUM_THREADS=8
echo "[rank-feats] START $(date '+%F %T') run=$SM_RUN out=$SM_OUT gx=${SM_GX:-} lr=${SM_LR_MODEL:-} host=$(hostname)"
"$PY" ${MADGRAV_ROOT}/m2/rank/build_bg_feats.py; rc=$?
echo "[rank-feats] END rc=$rc $(date '+%F %T')"; exit $rc

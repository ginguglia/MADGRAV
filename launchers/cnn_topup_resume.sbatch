#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:2
#SBATCH --time=24:00:00 --job-name=m2-cnntopup-resume
#SBATCH --output=${MADGRAV_ROOT}/launchers/logs_m2/cnntopup_%x_%j.log
set -uo pipefail
ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_PRECOMPUTE_MAXGB=48 SM_QT_WORKERS=40 SM_ARM_TOPUP=${SM_ARM_TOPUP:--3.00}
export SM_TOPUP_CKPT=${SM_TOPUP_CKPT:-$SM_OUT/topup_ckpt_${SM_TOPUP_IDX:-900}}   # resume-safe: finished H1 blocks are reloaded on restart
cd $ROOT; echo "[topup] START $(date '+%F %T') run=$SM_RUN_SEL arm=$SM_ARM_TOPUP host=$(hostname)"
"$PY" launchers/cnn_topup.py; rc=$?
echo "[topup] END rc=$rc $(date '+%F %T')"; exit $rc

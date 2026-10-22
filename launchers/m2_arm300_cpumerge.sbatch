#!/usr/bin/env bash
# Re-merge at the RELAXED arm rule max:-3.00 WITH the net-sigma LM rule on, CPU partition.
# Tagged _arm300 -> detections_arm300.json; the campaign and _lm products are untouched.
# Single-variable vs the _lm merges (same LM rule, arm -2.05), so the diff isolates the arm change.
# Recovers 9 of the 14 catalogue events the -2.05 gate vetoed, for ~8% more background survivors
# (arm_threshold_scan.py, 2026-09-22). Depends on cnn_topup.py having filled the CNN cache.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU}
#SBATCH --cpus-per-task=44 --mem=300G --time=36:00:00
#SBATCH --job-name=m2-arm300
#SBATCH --output=${MADGRAV_ROOT}/launchers/logs_m2/arm300_%x_%j.log
set -uo pipefail
ROOT=${MADGRAV_ROOT}
: "${SM_RUN_SEL:?set SM_RUN_SEL}"
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_CNN_CACHE=1 SM_MERGE_TAG=_arm300 SM_MERGE_OK=1 MALLOC_ARENA_MAX=2 SM_QT_WORKERS=40
export SM_DEV=cpu BLIND_DEV=cpu
export SM_ARM_RULE=max:-3.00
export SM_NETSIG_LM_MIN=0.0382
cd $ROOT
echo "[arm300] START $(date '+%F %T') run=$SM_RUN_SEL host=$(hostname) ARM=$SM_ARM_RULE LM_MIN=$SM_NETSIG_LM_MIN OUT=$SM_OUT"
"$PY" search_mode/driver_blindscan_merge.py; rc=$?
echo "[arm300] END rc=$rc $(date '+%F %T')"; exit $rc

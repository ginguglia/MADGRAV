#!/usr/bin/env bash
# CPU-partition FAR merge for the _off5 arm validation pass (2026-09-29). Env in: JIDS (jids env file), SM_MERGE_TAG, SM_ARM_RULE.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=44 --mem=300G --time=08:00:00
#SBATCH --output=${MADGRAV_ROOT}/m2/prep/logs/off5_merge_%x_%j.log
set -uo pipefail
ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" ${JIDS:?} | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_CNN_CACHE=1 SM_MERGE_TAG=${SM_MERGE_TAG?} SM_ARM_RULE=${SM_ARM_RULE:?} SM_MERGE_OK=1 MALLOC_ARENA_MAX=2 SM_QT_WORKERS=40 SM_DEV=cpu BLIND_DEV=cpu
cd $ROOT; echo "[cpumerge] START $(date '+%F %T') host=$(hostname) OUT=$SM_OUT TAG=$SM_MERGE_TAG RULE=$SM_ARM_RULE STREAMS=$SM_STREAMS LR=$SM_LR_MODEL"
"$PY" search_mode/driver_blindscan_merge.py; rc=$?
echo "[cpumerge] END rc=$rc $(date '+%F %T')"; ls -la "$SM_OUT" | grep -E "detections|blindscan"
if [ $rc -eq 0 ]; then "$PY" ${MADGRAV_ROOT}/m2/arm/off5_readout.py --mail --out "$SM_OUT" > "$SM_OUT/READOUT_OFF5_${SM_MERGE_TAG:-249}.log" 2>&1 || echo "[cpumerge] read-out failed (non-fatal)"; fi
exit $rc

#!/usr/bin/env bash
# Re-merge with the NET-SIGMA LM RULE active (SM_NETSIG_LM_MIN, 2026-09-22), on the CPU partition.
# Tagged _lm, so the campaign's detections.json / blindscan.json are NOT touched -- the rule-on
# products land as detections_lm.json alongside them and the two can be diffed.
# Run is selected by SM_RUN_SEL; shards are already banked, so this is a pure re-reduction.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU}
#SBATCH --cpus-per-task=44 --mem=300G --time=12:00:00
#SBATCH --job-name=m2-lmrule-merge
#SBATCH --output=${MADGRAV_ROOT}/launchers/logs_m2/lmrule_merge_%x_%j.log
set -uo pipefail
ROOT=${MADGRAV_ROOT}
: "${SM_RUN_SEL:?set SM_RUN_SEL to o3a|o3b|o4a|o4b}"
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_CNN_CACHE=1 SM_MERGE_TAG=_lm SM_ARM_RULE=max:-2.05 SM_MERGE_OK=1 MALLOC_ARENA_MAX=2 SM_QT_WORKERS=40
export SM_DEV=cpu BLIND_DEV=cpu
export SM_NETSIG_LM_MIN=0.0382      # p1 of the net-sigma-channel injection LM distribution
cd $ROOT
echo "[lmrule-merge] START $(date '+%F %T') run=$SM_RUN_SEL host=$(hostname) OUT=$SM_OUT SHARDS=$SM_SHARD_DIR NSHARD=$SM_NSHARD LM_MIN=$SM_NETSIG_LM_MIN"
"$PY" search_mode/driver_blindscan_merge.py; rc=$?
echo "[lmrule-merge] END rc=$rc $(date '+%F %T')"
exit $rc

#!/usr/bin/env bash
# PREREG_RANKER_20260924 §3.3: unchanged merge code path + STAT HOOK. Same env/rules as the _lm merges (arm max:-2.05, LM rule 0.0382).
# env in: SM_RUN_SEL, HOOK_FILE (stat_<run>.npz or stat_identity_<run>.npz), TAG (_rank or _rankid)
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=44 --mem=300G --time=16:00:00 --job-name=m2-rank-merge
#SBATCH --output=${MADGRAV_ROOT}/m2/rank/logs/merge_%x_%j.log
set -uo pipefail
ROOT=${MADGRAV_ROOT}
: "${SM_RUN_SEL:?}"; : "${HOOK_FILE:?}"; : "${TAG:?}"
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_CNN_CACHE=1 SM_MERGE_TAG=$TAG SM_ARM_RULE=max:-2.05 SM_MERGE_OK=1 MALLOC_ARENA_MAX=2 SM_QT_WORKERS=40
export SM_DEV=cpu BLIND_DEV=cpu SM_NETSIG_LM_MIN=0.0382 SM_STAT_HOOK=$HOOK_FILE
cd $ROOT
echo "[rank-merge] START $(date '+%F %T') run=$SM_RUN_SEL tag=$TAG hook=$HOOK_FILE host=$(hostname) OUT=$SM_OUT SHARDS=$SM_SHARD_DIR NSHARD=$SM_NSHARD"
"$PY" ${MADGRAV_ROOT}/m2/rank/driver_blindscan_merge_hook.py; rc=$?
echo "[rank-merge] END rc=$rc $(date '+%F %T')"; exit $rc

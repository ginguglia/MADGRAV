#!/usr/bin/env bash
# PREREG_SPECIALISTS A6 option (c): unchanged merge code path + S3 hook (loglr channel) [+ ranker STAT hook]. Same env/rules as the _lm merges.
# env in: SM_RUN_SEL, TAG (_s3 | _s3rank), RANK_HOOK (optional stat_<run>.npz)
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=44 --mem=300G --time=16:00:00 --job-name=m2-s3-merge
#SBATCH --output=${MADGRAV_ROOT}/m2/rank/logs/merge_%x_%j.log
set -uo pipefail
ROOT=${MADGRAV_ROOT}; RK=${MADGRAV_ROOT}/m2/rank
: "${SM_RUN_SEL:?}"; : "${TAG:?}"
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_CNN_CACHE=1 SM_MERGE_TAG=$TAG SM_ARM_RULE=max:-2.05 SM_MERGE_OK=1 MALLOC_ARENA_MAX=2 SM_QT_WORKERS=40
export SM_DEV=cpu BLIND_DEV=cpu SM_NETSIG_LM_MIN=0.0382
export SM_S3_HOOK=${MADGRAV_SCRATCH}/madgrav2_spec/s3/s3_${SM_RUN_SEL}.npz SM_S3_THR=$("$PY" -c "import json;print(json.load(open('$RK/s3_threshold.json'))['threshold'])")
if [ -n "${RANK_HOOK:-}" ]; then export SM_STAT_HOOK=$RANK_HOOK; fi
cd $ROOT
echo "[s3-merge] START $(date '+%F %T') run=$SM_RUN_SEL tag=$TAG s3thr=$SM_S3_THR rank=${RANK_HOOK:-none} host=$(hostname) OUT=$SM_OUT"
"$PY" $RK/driver_blindscan_merge_hook.py; rc=$?
echo "[s3-merge] END rc=$rc $(date '+%F %T')"; exit $rc

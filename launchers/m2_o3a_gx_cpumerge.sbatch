#!/usr/bin/env bash
# CPU-partition merge of the O3b gx run (arm rule max:-2.05), tagged _cpu so the queued GPU merge 1750928 is untouched.
# Same pattern as m2_o4b_coh_zl3_cpumerge.sbatch (2026-09-20, 33 min for 96 shards); o3a is 64 shards / larger candidate list.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU}
#SBATCH --cpus-per-task=44 --mem=300G --time=12:00:00
set -uo pipefail
ROOT=${MADGRAV_ROOT}
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/o3a_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//')
IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_NSHARD=64 SM_CNN_CACHE=1 SM_MERGE_TAG=_cpu SM_ARM_RULE=max:-2.05 SM_MERGE_OK=1 MALLOC_ARENA_MAX=2 SM_QT_WORKERS=40
export SM_DEV=cpu BLIND_DEV=cpu
cd $ROOT
echo "[cpumerge-o3a] START $(date '+%F %T') host=$(hostname) GX=${SM_GX:-unset} ZL_NET_CUT=$SM_ZL_NET_CUT GATES=$SM_ZL_GATES OUT=$SM_OUT SHARDS=$SM_SHARD_DIR TAG=$SM_MERGE_TAG RULE=$SM_ARM_RULE"
"$PY" search_mode/driver_blindscan_merge.py; rc=$?
echo "[cpumerge-o3a] END rc=$rc $(date '+%F %T')"; ls -la "$SM_OUT" | grep -E "_cpu|detections|blindscan"
exit $rc

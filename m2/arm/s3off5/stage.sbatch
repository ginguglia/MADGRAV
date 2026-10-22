#!/usr/bin/env bash
# _off5 -> _s3rank chain for one run (2026-10-02), CPU partition only. One stage per submission, STAGE and SM_RUN_SEL via --export;
# resources via sbatch command-line flags (see logs/chain_<run>_*.txt for the exact submission lines).
#   bgfeats   : bank features of the _off5 survivors_bg.json            -> $O/rank/bgfeats_<run>.npz
#   missbank  : + far-shard bank rows the merge never CNN-scored          -> (same file, appended)
#   feats     : noise/inj/trigger features (trigger block used)           -> $O/rank/feats_<run>.npz
#   s3score   : S3 ensemble scores, 16 array tasks                        -> $O/s3bank/s3_<run>_t<k>.npz
#   concat    : one S3 hook file                                          -> $O/s3bank/s3_<run>.npz
#   rankscore : production ranker models on the _off5 bank + triggers    -> $O/rank/stat_<run>.npz
#   merge     : campaign merge code + S3 hook + STAT hook, TAG=_s3rank    -> $SM_OUT/detections_s3rank.json
# Production tables (madgrav2_spec/rank, madgrav2_spec/s3) are read-only inputs here (models only); every output is under $O.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU}
#SBATCH --output=${MADGRAV_ROOT}/m2/arm/s3off5/logs/%x_%A_%a.log
set -uo pipefail
: "${STAGE:?}"; : "${SM_RUN_SEL:?}"
ROOT=${MADGRAV_ROOT}; W=${MADGRAV_ROOT}/m2/arm/s3off5; O=${MADGRAV_SCRATCH}/madgrav2/s3_off5
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_off5_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_DEV=cpu BLIND_DEV=cpu MALLOC_ARENA_MAX=2 R_RANK_DIR=$O/rank
[ "$SM_RUN" = "$SM_RUN_SEL" ] || { echo "FATAL run mismatch $SM_RUN vs $SM_RUN_SEL"; exit 1; }
case "$SM_OUT" in *_off5) ;; *) echo "FATAL SM_OUT is not an _off5 dir: $SM_OUT"; exit 1;; esac
echo "[off5s3-$STAGE] START $(date '+%F %T') run=$SM_RUN OUT=$SM_OUT shards=$SM_SHARD_DIR lr=$SM_LR_MODEL host=$(hostname) cpus=${SLURM_CPUS_ON_NODE:-?} task=${SLURM_ARRAY_TASK_ID:-}"
case "$STAGE" in
  bgfeats)   export OMP_NUM_THREADS=8 R_OUT=$O/rank; "$PY" $W/build_bg_feats_off5.py; rc=$? ;;
  missbank)  export OMP_NUM_THREADS=8; "$PY" $W/add_missing_bank_off5.py; rc=$? ;;
  feats)     export OMP_NUM_THREADS=8 R_OUT=$O/rank; "$PY" $W/build_feats_off5.py; rc=$? ;;
  s3score)   export SM_QT_WORKERS=28 OMP_NUM_THREADS=4 R_TASK=${SLURM_ARRAY_TASK_ID:?} R_NTASK=16 R_OUT=$O/s3bank; "$PY" $W/score_bank_s3_off5.py; rc=$? ;;
  concat)    export R_OUT=$O/s3bank; "$PY" $W/concat_s3_off5.py "$SM_RUN"; rc=$? ;;
  rankscore) export OMP_NUM_THREADS=8; "$PY" $W/score_rank_off5.py; rc=$? ;;
  merge)
    export SM_CNN_CACHE=1 SM_MERGE_TAG=_s3rank SM_ARM_RULE=max:-2.05 SM_MERGE_OK=1 SM_QT_WORKERS=40 SM_NETSIG_LM_MIN=0.0382
    export SM_S3_HOOK=$O/s3bank/s3_${SM_RUN}.npz SM_STAT_HOOK=$O/rank/stat_${SM_RUN}.npz
    export SM_S3_THR=$("$PY" -c "import json;print(json.load(open('${MADGRAV_ROOT}/m2/rank/s3_threshold.json'))['threshold'])")
    for f in "$SM_S3_HOOK" "$SM_STAT_HOOK"; do [ -s "$f" ] || { echo "FATAL missing $f"; exit 1; }; done
    DRV=${MADGRAV_ROOT}/m2/rank/driver_blindscan_merge_hook.py
    echo "[off5s3-merge] tag=_s3rank s3thr=$SM_S3_THR s3=$SM_S3_HOOK rank=$SM_STAT_HOOK driver_sha256=$(sha256sum $DRV | cut -c1-16)"
    cd $ROOT; "$PY" $DRV; rc=$?
    ls -la "$SM_OUT" | grep -E "detections|blindscan" ;;
  *) echo "FATAL unknown STAGE=$STAGE"; rc=2 ;;
esac
echo "[off5s3-$STAGE] END rc=$rc $(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
# Stage B of the floor test, run-generic (2026-10-09; O4b original: m2/notrig/stageB/stageB.sbatch): S3 + ranker hooks on the NEW bank, all
# outputs in test dirs (madgrav2_spec/rank_<FT>, s3_<FT>; never the production rank/s3 dirs). Production S3 rows reused by key from the
# off5 bank (s3_off5/s3bank/s3_<run>.npz = the bank the production _v2 merge used), ranker applied from models_<run>.joblib, never retrained.
# env in: STEP = bgfeats | missing | feats | s3 (array) | concat | stat | merge | readout ; RUN, FT (nf275|nf250), FLOOR, JIDS, PRODV2
#SBATCH --account=${SLURM_ACCOUNT} --output=${MADGRAV_ROOT}/m2/popb_cap/occlusion/chain/logs/%x_%A_%a.log
set -uo pipefail; : "${STEP:?}" "${RUN:?}" "${FT:?}" "${FLOOR:?}" "${JIDS:?}" "${PRODV2:?}"
ROOT=${MADGRAV_ROOT}; W=${MADGRAV_ROOT}/m2; F=$W/notrig/floors; M2SC=${MADGRAV_SCRATCH}/madgrav2
R=${MADGRAV_SCRATCH}/madgrav2_spec/rank_${FT}; S=${MADGRAV_SCRATCH}/madgrav2_spec/s3_${FT}; O=$M2SC/s3_off5
EXP=$(grep "^EXP=" $JIDS | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
[ "$SM_RUN" = "$RUN" ] || { echo "FATAL SM_RUN=$SM_RUN != RUN=$RUN"; exit 1; }
export SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} R_OUT=$R R_DIR=$R SM_LR_NETMIN=$FLOOR S3_PROD=${MADGRAV_SCRATCH}/madgrav2/popb_occl/s3_empty_${RUN}.npz PRODV2=$PRODV2
cd $ROOT; echo "[stageB:$STEP] START $(date '+%F %T') host=$(hostname) run=$RUN ft=$FT OUT=$SM_OUT shards=$SM_SHARD_DIR streams=$SM_STREAMS veto=${SM_VETO:-none} net_cut=$SM_NET_CUT zl=$SM_ZL_NET_CUT s3_prod=$S3_PROD"
case "$STEP" in
  bgfeats) "$PY" $F/build_bg_feats_nf.py; rc=$? ;;
  missing) "$PY" $F/add_missing_bank_nf.py; rc=$? ;;
  feats)   "$PY" $F/build_feats_nf.py; rc=$? ;;
  s3)      export BLIND_DEV=cpu SM_DEV=cpu SM_ALLOW_CPU=1 SM_QT_WORKERS=14 R_TASK=${SLURM_ARRAY_TASK_ID:-0} R_NTASK=${R_NTASK:-8} R_OUT=$S; "$PY" $F/score_bank_s3_nf.py; rc=$? ;;
  concat)  export R_OUT=$S; "$PY" $F/concat_s3_nf.py ${R_NTASK:-8}; rc=$? ;;
  stat)    "$PY" $F/apply_stat_nf.py; rc=$? ;;
  merge)   export SM_CNN_CACHE=1 SM_MERGE_TAG=_v2 SM_ARM_RULE=max:-2.05 SM_MERGE_OK=1 SM_QT_WORKERS=40 SM_NETSIG_LM_MIN=0.0382
           export SM_ASD_VETO=1 SM_ASD_CNN_CLAUSE=auto SM_ASD_PLATEAU_THR=0.04 SM_ASD_PLATEAU_JSON=$W/asdveto/plateau_${RUN}.json R_RANK_DIR=$R
           export SM_S3_HOOK=$S/s3_${RUN}.npz SM_STAT_HOOK=$R/stat_${RUN}.npz SM_S3_THR=$("$PY" -c "import json;print(json.load(open('$W/rank/s3_threshold.json'))['threshold'])")
           for f in "$SM_S3_HOOK" "$SM_STAT_HOOK" "$SM_ASD_PLATEAU_JSON"; do [ -s "$f" ] || { echo "FATAL missing $f"; exit 1; }; done
           n=$(ls $SM_SHARD_DIR/shard_*.npz | wc -l); [ "$n" -eq "$SM_NSHARD" ] || { echo "FATAL $n/$SM_NSHARD shards"; exit 1; }
           DRV=$W/rank/driver_blindscan_merge_hook.py; echo "[stageB:merge] hooks s3=$SM_S3_HOOK stat=$SM_STAT_HOOK thr=$SM_S3_THR plateau=$("$PY" -c "import json;print(json.load(open('$SM_ASD_PLATEAU_JSON'))['max_frac'])") driver=$(sha256sum $DRV | cut -c1-16)"
           "$PY" $DRV; rc=$?; ls -la $SM_OUT | grep -E "detections_v2|blindscan_v2"; [ $rc -eq 0 ] && { cd $W/dq && "$PY" audit_merge.py $RUN _v2 $SM_OUT > $SM_OUT/audit_v2.txt 2>&1; grep -E "^\[audit\]|NON-CAT" $SM_OUT/audit_v2.txt | head; } ;;
  readout) cd $F && "$PY" -I $F/stageB_readout_nf.py | tee ${MADGRAV_ROOT}/m2/popb_cap/occlusion/chain/STAGEB_READOUT_${FT^^}_${RUN}.txt; rc=${PIPESTATUS[0]} ;;
  *) echo "unknown STEP"; rc=2 ;;
esac
echo "[stageB:$STEP] END rc=$rc $(date '+%F %T')"; exit $rc

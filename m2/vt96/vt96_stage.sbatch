#!/usr/bin/env bash
# vt96 (2026-10-19): injection-efficiency stages of the frozen version-2 recipe, CPU partition only. STAGE and SM_RUN_SEL via --export.
#   export   : far_tables_<run>.npz from the production _v2 merge inputs of the crop re-pass (export_far_tables96.py)
#   validate : far96.py validate -> must print PASS (per-trigger FARs of blindscan_v2.json, kept/removed of detections_v2_nsguard.json, 96 set)
#   score    : far96.py score on VT96_INJ_DIRS -> VT96_OUT/inj_scored_<run>_m96<VT96_SUF>.npz
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU}
#SBATCH --output=${MADGRAV_ROOT}/m2/vt96/logs/%x_%A.log
set -uo pipefail
: "${STAGE:?}"; : "${SM_RUN_SEL:?}"
ROOT=${MADGRAV_ROOT}; W=${MADGRAV_ROOT}/m2/vt96; SPEC=${MADGRAV_SCRATCH}/madgrav2_spec; export VT96_OUT=${MADGRAV_SCRATCH}/madgrav2/vt96
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_off5_crop_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
[ "$SM_RUN" = "$SM_RUN_SEL" ] || { echo "FATAL run mismatch $SM_RUN vs $SM_RUN_SEL"; exit 1; }
case "$SM_OUT" in *_off5_crop) ;; *) echo "FATAL SM_OUT is not an _off5_crop dir: $SM_OUT"; exit 1;; esac
[ "$SM_CROP" = "coinc" ] && [ "$SM_NET_CUT" = "2.75" ] && [ "$SM_ZL_NET_CUT" = "2.75" ] && [ "$SM_ZL_GATES" = "1" ] || { echo "FATAL frozen env check: SM_CROP=$SM_CROP SM_NET_CUT=$SM_NET_CUT SM_ZL_NET_CUT=$SM_ZL_NET_CUT SM_ZL_GATES=$SM_ZL_GATES"; exit 1; }
# the production _v2 merge settings (m2/popb_cap/occlusion/chain/stageB_crop.sbatch, STEP=merge)
export SM_ARM_RULE=max:-2.05 SM_NETSIG_LM_MIN=0.0382 SM_S3_HOOK=$SPEC/s3_crop/s3_${SM_RUN}.npz SM_STAT_HOOK=$SPEC/rank_crop/stat_${SM_RUN}.npz
export SM_ASD_PLATEAU_JSON=${MADGRAV_ROOT}/m2/asdveto/plateau_${SM_RUN}.json SM_ASD_PLATEAU_THR=0.04
export SM_S3_THR=$("$PY" -c "import json;print(json.load(open('${MADGRAV_ROOT}/m2/rank/s3_threshold.json'))['threshold'])")
for f in "$SM_S3_HOOK" "$SM_STAT_HOOK" "$SM_ASD_PLATEAU_JSON"; do [ -s "$f" ] || { echo "FATAL missing $f"; exit 1; }; done
mkdir -p $VT96_OUT; echo "[vt96-$STAGE] START $(date '+%F %T') run=$SM_RUN OUT=$SM_OUT shards=$SM_SHARD_DIR lr=$SM_LR_MODEL s3=$SM_S3_HOOK stat=$SM_STAT_HOOK thr=$SM_S3_THR host=$(hostname) cpus=${SLURM_CPUS_ON_NODE:-?}"
case "$STAGE" in
  export)   n=$(ls $SM_SHARD_DIR/shard_*.npz | wc -l); [ "$n" -eq "$SM_NSHARD" ] || { echo "FATAL $n/$SM_NSHARD shards"; exit 1; }; cd $ROOT; "$PY" $W/export_far_tables96.py; rc=$? ;;
  validate) cd $ROOT; "$PY" $W/far96.py validate --run $SM_RUN --tables $VT96_OUT/far_tables_${SM_RUN}.npz --outdir $SM_OUT | tee $W/logs/validate_${SM_RUN}_$(date +%Y%m%d_%H%M).txt; rc=${PIPESTATUS[0]} ;;
  score)    : "${VT96_INJ_DIRS:?}"; cd $ROOT; "$PY" $W/far96.py score --run $SM_RUN --tables $VT96_OUT/far_tables_${SM_RUN}.npz --inj-dirs $VT96_INJ_DIRS --out $VT96_OUT/inj_scored_${SM_RUN}_m96${VT96_SUF:-}.npz ${VT96_SCORE_FLAGS:-}; rc=$? ;;
  env)      echo "[vt96-env] ok: PY=$PY SM_PREP=$SM_PREP SM_OUT=$SM_OUT nshard=$SM_NSHARD lm_min=$SM_NETSIG_LM_MIN arm=$SM_ARM_RULE"; rc=0 ;;
  *) echo "FATAL unknown STAGE=$STAGE"; rc=2 ;;
esac
echo "[vt96-$STAGE] END rc=$rc $(date '+%F %T')"; exit $rc

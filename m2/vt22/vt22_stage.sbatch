#!/usr/bin/env bash
# MADGRAV 2.2 injection efficiency stages (2026-10-04), CPU partition only. STAGE and SM_RUN_SEL via --export.
#   export   : far_tables_<run>.npz from the _off5 + S3 + ranker merge inputs (export_far_tables.py)
#   validate : reproduce the merge's per-trigger FARs from the tables (far22.py validate) -> must print PASS
#   score    : per-injection detection under 2.2 (far22.py score) from VT22_INJ_DIRS -> VT22_OUT/inj_scored_<run>_m22<VT22_SUF>.npz
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU}
#SBATCH --output=${MADGRAV_ROOT}/m2/vt22/logs/%x_%A.log
set -uo pipefail
: "${STAGE:?}"; : "${SM_RUN_SEL:?}"
ROOT=${MADGRAV_ROOT}; W=${MADGRAV_ROOT}/m2/vt22; O=${MADGRAV_SCRATCH}/madgrav2/s3_off5; export VT22_OUT=${MADGRAV_SCRATCH}/madgrav2/vt22
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${SM_RUN_SEL}_off5_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_DEV=cpu BLIND_DEV=cpu MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
[ "$SM_RUN" = "$SM_RUN_SEL" ] || { echo "FATAL run mismatch $SM_RUN vs $SM_RUN_SEL"; exit 1; }
case "$SM_OUT" in *_off5) ;; *) echo "FATAL SM_OUT is not an _off5 dir: $SM_OUT"; exit 1;; esac
export SM_ARM_RULE=max:-2.05 SM_NETSIG_LM_MIN=0.0382 SM_S3_HOOK=$O/s3bank/s3_${SM_RUN}.npz SM_STAT_HOOK=$O/rank/stat_${SM_RUN}.npz
export SM_S3_THR=$("$PY" -c "import json;print(json.load(open('${MADGRAV_ROOT}/m2/rank/s3_threshold.json'))['threshold'])")
mkdir -p $VT22_OUT; echo "[vt22-$STAGE] START $(date '+%F %T') run=$SM_RUN OUT=$SM_OUT shards=$SM_SHARD_DIR lr=$SM_LR_MODEL host=$(hostname) cpus=${SLURM_CPUS_ON_NODE:-?}"
case "$STAGE" in
  export)   for f in "$SM_S3_HOOK" "$SM_STAT_HOOK"; do [ -s "$f" ] || { echo "FATAL missing $f"; exit 1; }; done; cd $ROOT; "$PY" $W/export_far_tables.py; rc=$? ;;
  export_validate) for f in "$SM_S3_HOOK" "$SM_STAT_HOOK"; do [ -s "$f" ] || { echo "FATAL missing $f"; exit 1; }; done; cd $ROOT; "$PY" $W/export_far_tables.py; rc=$?
            [ $rc -eq 0 ] && { "$PY" $W/far22.py validate --run $SM_RUN --tables $VT22_OUT/far_tables_${SM_RUN}.npz --blindscan $SM_OUT/blindscan_s3rank.json | tee $W/logs/validate_${SM_RUN}_$(date +%Y%m%d).txt; rc=${PIPESTATUS[0]}; } ;;
  figure)   export MG_FIELD_DIR=$VT22_OUT MG_FIELD_TAG=m22 MG_PENDING_RUNS="" MG_FIG_LABEL="MADGRAV 2.2" MG_FIG_LEGLOC="upper left" MG_FIG_YTOP=85
            export MG_HL_JSON=${MADGRAV_SCRATCH}/madgrav2/figures_m2/hl_snr_o3o4_m2_off5s3rank.json MG_FIG_OUT=${MADGRAV_SCRATCH}/madgrav2/figures_m2/mtot_snr_smooth_m22_off5s3rank
            export MG_FIG_CBLABEL=$'injection recovery probability at FAR < 1/yr\n(MADGRAV 2.2 campaign, seed 20261004)'
            export MG_FIG_FOOT="MADGRAV 2.2 detections (80 at FAR < 1/yr) on the MADGRAV 2.2 injection recovery field (fresh-seed campaign, same recipe; no local-ASD veto, no null-calibration factor)"
            cd ${MADGRAV_ROOT}/figures/catalog_o3o4; "$PY" plot_mtot_snr_smooth_m2.py 2>&1 | tee $W/logs/figure22_$(date +%Y%m%d_%H%M).txt; rc=${PIPESTATUS[0]}
            if [ $rc -eq 0 ]; then { echo "Four-panel recovery plane on the MADGRAV 2.2 injection field (automatic SLURM chain, $(date '+%F %T'))."; echo; cat $W/logs/figure22_*.txt | tail -6; echo; echo "Files: $MG_FIG_OUT.{pdf,png}"; } > $W/logs/mail_figure22.txt
              python3 ${MADGRAV_ROOT}/search_mode/pastro_final/campaign/mailer.py "MADGRAV 2.2: recovery plane on the 2.2 injection field" $W/logs/mail_figure22.txt $MG_FIG_OUT.pdf $MG_FIG_OUT.png || echo "[vt22-figure] mail failed (compute node SMTP?) -- files are on scratch"; fi ;;
  validate) "$PY" $W/far22.py validate --run $SM_RUN --tables $VT22_OUT/far_tables_${SM_RUN}.npz --blindscan $SM_OUT/blindscan_s3rank.json; rc=$? ;;
  score)    : "${VT22_INJ_DIRS:?}"; cd $ROOT; "$PY" $W/far22.py score --run $SM_RUN --tables $VT22_OUT/far_tables_${SM_RUN}.npz --inj-dirs $VT22_INJ_DIRS --out $VT22_OUT/inj_scored_${SM_RUN}_m22${VT22_SUF:-}.npz ${VT22_SCORE_FLAGS:-}; rc=$? ;;
  *) echo "FATAL unknown STAGE=$STAGE"; rc=2 ;;
esac
echo "[vt22-$STAGE] END rc=$rc $(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
# 2026-10-11: resume the 4 crop chains after the concat failure (concat_s3_nf.py built a float64 empty mask from the zero-row empty
# S3_PROD bank -> IndexError; fixed with dtype=bool, verified on the o4b task files). S3 task files (16/16 per run) and stat files are
# intact, so only concat -> merge_v2 -> readout are resubmitted. Old merge_v2/readout jobs (DependencyNeverSatisfied) are cancelled.
# Usage: DRY=1|0 m2/popb_cap/occlusion/chain/resume_after_concat_fix.sh
set -uo pipefail
DRY="${DRY:-1}"; FT=crop; FLOOR=2.75
ROOT=${MADGRAV_ROOT}; M2SC=${MADGRAV_SCRATCH}/madgrav2; PY=${PY:-python}
W=${MADGRAV_ROOT}/m2; C=$W/popb_cap/occlusion/chain; L=$ROOT/launchers/logs_m2; S=${MADGRAV_SCRATCH}/madgrav2_spec/s3_${FT}; R=${MADGRAV_SCRATCH}/madgrav2_spec/rank_${FT}
CPU=(--partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --requeue)
grep -q 'dtype=bool)' $W/notrig/floors/concat_s3_nf.py || { echo "concat fix not in place"; exit 1; }
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@" | cut -d';' -f1; fi; }
for RUN in o3a o3b o4a o4b; do
  JIDS=$L/${RUN}_off5_${FT}_m2_gx_jids.env; PRODV2=$M2SC/search_out_${RUN}_m2_gx_off5_nf275
  [ "$(ls $S/s3_${RUN}_t*.npz | wc -l)" = 16 ] || { echo "$RUN: S3 task files incomplete"; exit 1; }
  [ -s $R/stat_${RUN}.npz ] || { echo "$RUN: stat file missing"; exit 1; }
  OLD_MV=$(grep "^MERGE_V2=" $JIDS | tail -1 | cut -d= -f2); OLD_RO=$(grep "^READOUT=" $JIDS | tail -1 | cut -d= -f2)
  log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
  if [ "$DRY" = 0 ]; then scancel $OLD_MV $OLD_RO; echo "# $(date '+%F %T') $RUN resume after concat fix (resume_after_concat_fix.sh): cancelled stale merge_v2 $OLD_MV / readout $OLD_RO; concat->merge_v2->readout resubmitted" >> $JIDS; else echo "  scancel $OLD_MV $OLD_RO"; fi
  J_CC=$(submit crop-${RUN}-concat "${CPU[@]}" --cpus-per-task=4 --mem=30G --time=00:30:00 --export=ALL,STEP=concat,R_NTASK=16,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "CONCAT=$J_CC"
  J_MV=$(submit crop-${RUN}-merge-v2 "${CPU[@]}" --cpus-per-task=44 --mem=300G --time=14:00:00 --dependency=afterok:$J_CC --export=ALL,STEP=merge,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "MERGE_V2=$J_MV"
  J_RO=$(submit crop-${RUN}-readout "${CPU[@]}" --cpus-per-task=4 --mem=16G --time=00:20:00 --dependency=afterok:$J_MV --export=ALL,STEP=readout,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "READOUT=$J_RO"
done
if [ "$DRY" = 0 ]; then
  echo "[ctl] $(date '+%F %T') resume after concat fix (resume_after_concat_fix.sh)" >> $C/logs/crop_controller.log
  nohup $PY -I $C/crop_mailer.py > $C/logs/crop_mailer.out 2>&1 & echo $! > $C/logs/crop_mailer.pid; echo "mailer pid $(cat $C/logs/crop_mailer.pid)"
  squeue -u USER -o "%.10i %.28j %.9T %R" | grep crop
fi

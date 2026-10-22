#!/usr/bin/env bash
# Crop re-pass chain (2026-10-10, DECISIONS_PLACEMENT.md): the floor-2.75 production definition (nf275 EXP) with SM_CROP=coinc, own out dir,
# the nf275 far shards REUSED (no prune), full CNN cache regenerated on the CPU partition, top-up, merge _p205, Stage A, Stage B with the
# FULL S3 bank re-scored (empty S3_PROD), ranker applied (not retrained), merge _v2 with hooks + ASD veto, read-out vs the nf275 _v2.
# Usage: DRY=1|0 RUN=o4b m2/popb_cap/occlusion/chain/crop_chain.sh
set -uo pipefail
DRY="${DRY:-1}"; RUN="${RUN:?}"; FT=crop; FLOOR=2.75
ROOT=${MADGRAV_ROOT}; M2SC=${MADGRAV_SCRATCH}/madgrav2; PYV=${PY:-python}
W=${MADGRAV_ROOT}/m2; F=$W/notrig/floors; C=$W/popb_cap/occlusion/chain; LOGS=$C/logs; L=$ROOT/launchers/logs_m2
SEL=${RUN}_off5_${FT}; JIDS=$L/${SEL}_m2_gx_jids.env
R=${MADGRAV_SCRATCH}/madgrav2_spec/rank_${FT}; S=${MADGRAV_SCRATCH}/madgrav2_spec/s3_${FT}; mkdir -p $R $S $LOGS
PRODV2=$M2SC/search_out_${RUN}_m2_gx_off5_nf275; OUT=$M2SC/search_out_${RUN}_m2_gx_off5_${FT}; SHD=$M2SC/_${RUN}_m2_gx_off5_nf275_far_shards
EXP=$(grep "^EXP=" $L/${RUN}_off5_nf275_m2_gx_jids.env | tail -1); EXP=${EXP#EXP=}
EXP=$(echo "$EXP" | sed -e "s|SM_OUT=[^,]*|SM_OUT=$OUT|"); EXP="$EXP,SM_CROP=coinc"
for k in SM_RUN SM_STREAMS SM_VETO SM_OUT SM_SHARD_DIR SM_NET_CUT SM_ZL_NET_CUT SM_LR_MODEL SM_NSHARD SM_ARM_V2_TAG SM_CROP; do echo "  $k = $(echo "$EXP" | tr ',' '\n' | grep "^$k=" | cut -d= -f2-)"; done
NSHARD=$(echo "$EXP" | tr ',' '\n' | grep "^SM_NSHARD=" | cut -d= -f2)
[ "$(echo "$EXP" | tr ',' '\n' | grep -c "^SM_OUT=$OUT$")" = 1 ] && [ "$(echo "$EXP" | tr ',' '\n' | grep -c "^SM_SHARD_DIR=$SHD$")" = 1 ] && [ "$(echo "$EXP" | tr ',' '\n' | grep -c "^SM_ZL_NET_CUT=$FLOOR$")" = 1 ] || { echo "EXP check failed (OUT/SHARD_DIR/floor)"; exit 1; }
[ "$(ls $SHD/shard_*.npz | wc -l)" = "$NSHARD" ] || { echo "shards incomplete in $SHD"; exit 1; }
for f in $PRODV2/detections_v2.json $PRODV2/blindscan_v2.json $M2SC/s3_off5/rank/stat_${RUN}.npz ${MADGRAV_SCRATCH}/madgrav2_spec/rank/models_${RUN}.joblib $W/asdveto/plateau_${RUN}.json $M2SC/popb_occl/s3_empty_${RUN}.npz; do [ -s $f ] || { echo "MISSING $f"; exit 1; }; done
[ -e $OUT ] && [ -n "$(ls $OUT/cnn_shard_*.npz 2>/dev/null)" ] && { echo "$OUT already has cnn shards; refusing (fresh dir required)"; exit 1; }
CPU=(--partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --requeue)
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" | sed "s|$EXP|\$EXP|" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@" | cut -d';' -f1; fi; }
log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
if [ "$DRY" = 0 ]; then
  echo "# $(date '+%F %T') $RUN crop re-pass (m2/popb_cap/occlusion/chain/crop_chain.sh): nf275 EXP + SM_CROP=coinc, own OUT, nf275 far shards reused, full CNN cache + full S3 bank re-scored on CPU, LR model reused, ranker applied" >> $JIDS; echo "EXP=$EXP" >> $JIDS; mkdir -p $OUT
fi
J_SH=$(submit m2-${RUN}-crop-cnnshard "${CPU[@]}" --output=$LOGS/${FT}_${RUN}_cnnshard_%A_%a.log --array=0-$((NSHARD-1))%40 --export="$EXP" $C/tmpl_cnn_shard_cpu.sbatch); log "CNN_SHARDS=$J_SH"
J_TU=$(submit m2-${RUN}-crop-cnntopup "${CPU[@]}" --output=$LOGS/${FT}_${RUN}_cnntopup_%j.log --dependency=afterok:$J_SH --export="ALL,SM_RUN_SEL=$SEL,SM_ARM_TOPUP=-1e9,SM_TOPUP_IDX=902" $C/cnn_topup_cpu.sbatch); log "TOPUP=$J_TU"
J_M=$(submit m2-${RUN}-crop-merge-205 "${CPU[@]}" --time=12:00:00 --output=$LOGS/${FT}_${RUN}_merge205_%j.log --dependency=afterok:$J_TU --export="ALL,JIDS=$JIDS,SM_MERGE_TAG=_p205,SM_ARM_RULE=max:-2.05,OFF5_RUN=$RUN" $ROOT/launchers/m2_off5_cpumerge.sbatch); log "MERGE_205=$J_M"
J_A=$(submit crop-${RUN}-stageA "${CPU[@]}" --cpus-per-task=4 --mem=16G --time=00:20:00 --dependency=afterok:$J_M --output=$LOGS/stageA_${FT}_${RUN}_%j.log --export=ALL,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,PRODV2=$PRODV2 --wrap="cd $F && $PYV -I stageA_readout_nf.py | tee $C/STAGEA_READOUT_${FT^^}_${RUN}.txt"); log "STAGEA_READOUT=$J_A"
J_BG=$(submit crop-${RUN}-bgfeats "${CPU[@]}" --cpus-per-task=32 --mem=240G --time=04:00:00 --dependency=afterok:$J_M --export=ALL,STEP=bgfeats,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "BGFEATS=$J_BG"
J_FE=$(submit crop-${RUN}-feats "${CPU[@]}" --cpus-per-task=16 --mem=120G --time=03:00:00 --dependency=afterok:$J_M --export=ALL,STEP=feats,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "FEATS=$J_FE"
J_MI=$(submit crop-${RUN}-missing "${CPU[@]}" --cpus-per-task=16 --mem=120G --time=02:00:00 --dependency=afterok:$J_BG --export=ALL,STEP=missing,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "MISSING=$J_MI"
J_S3=$(submit crop-${RUN}-s3 "${CPU[@]}" --cpus-per-task=16 --mem=120G --time=08:00:00 --array=0-15 --dependency=afterok:$J_MI --export=ALL,STEP=s3,R_NTASK=16,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "S3=$J_S3"
J_CC=$(submit crop-${RUN}-concat "${CPU[@]}" --cpus-per-task=4 --mem=30G --time=00:30:00 --dependency=afterok:$J_S3 --export=ALL,STEP=concat,R_NTASK=16,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "CONCAT=$J_CC"
J_ST=$(submit crop-${RUN}-stat "${CPU[@]}" --cpus-per-task=16 --mem=120G --time=01:00:00 --dependency=afterok:$J_FE:$J_MI --export=ALL,STEP=stat,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "STAT=$J_ST"
J_MV=$(submit crop-${RUN}-merge-v2 "${CPU[@]}" --cpus-per-task=44 --mem=300G --time=14:00:00 --dependency=afterok:$J_CC:$J_ST --export=ALL,STEP=merge,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "MERGE_V2=$J_MV"
J_RO=$(submit crop-${RUN}-readout "${CPU[@]}" --cpus-per-task=4 --mem=16G --time=00:20:00 --dependency=afterok:$J_MV --export=ALL,STEP=readout,RUN=$RUN,FT=$FT,FLOOR=$FLOOR,JIDS=$JIDS,PRODV2=$PRODV2 $C/stageB_crop.sbatch); log "READOUT=$J_RO"
echo "[crop-chain] done RUN=$RUN DRY=$DRY jids=$JIDS"

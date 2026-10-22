#!/usr/bin/env bash
# MADGRAV 2.1 arm _off5 O4b validation pass (2026-09-29): production o4b gx EXP with the rescored streams (g2 = _off5), re-injected
# injections, own LR / bank / out dir. Stages: LR (CPU) -> prune (CPU template, 96 shards) -> CNN top-up of bank pairs missing from the
# production CNN cache (GPU, the only GPU stage) -> merges max:-2.49 / max:-2.05 / none (CPU) -> summary mail (CPU).
# Usage: DRY=1|0 RESCORE=<array jid> INJ=<array jid> launchers/m2_o4b_off5_chain.sh
set -uo pipefail
DRY="${DRY:-1}"; ROOT=${MADGRAV_ROOT}; SC=${MADGRAV_SCRATCH}; PYV=${PY:-python}
L=$ROOT/launchers/logs_m2; TPL=$ROOT/launchers/templates; LOGS=${MADGRAV_ROOT}/m2/prep/logs; JIDS=$L/o4b_off5_m2_gx_jids.env
source ${MADGRAV_ROOT}/site.conf
EXP=$(grep "^EXP=" $L/o4b_m2_gx_jids.env | tail -1)
EXP=$(echo "$EXP" | sed -e "s|SM_STREAMS=[^,]*|SM_STREAMS=$SC/madgrav2/streams_o4b_full_cat2_off5|" -e "s|SM_ARM_V2_TAG=_def|SM_ARM_V2_TAG=_off5|" \
  -e "s|SM_OUT=[^,]*|SM_OUT=$SC/madgrav2/search_out_o4b_m2_gx_off5|" -e "s|SM_SHARD_DIR=[^,]*|SM_SHARD_DIR=$SC/madgrav2/_o4b_m2_gx_off5_far_shards|" \
  -e "s|SM_LR_INJ_GLOB=[^,]*|SM_LR_INJ_GLOB=$SC/madgrav2/inj_o4b_m2_off5/*/*_inj.npz|" -e "s|SM_LR_OUT=[^,]*|SM_LR_OUT=$SC/madgrav2/lr_o4b_m2_gx_off5.npz|" -e "s|SM_LR_MODEL=[^,]*|SM_LR_MODEL=$SC/madgrav2/lr_o4b_m2_gx_off5.npz|")
EXP=${EXP#EXP=}
for k in SM_STREAMS SM_ARM_V2_TAG SM_OUT SM_SHARD_DIR SM_LR_INJ_GLOB SM_LR_MODEL SM_NSHARD SM_ZL_NET_CUT SM_ZL_GATES; do echo "  $k = $(echo "$EXP" | tr ',' '\n' | grep "^$k=" | cut -d= -f2-)"; done
OUT=$SC/madgrav2/search_out_o4b_m2_gx_off5; mkdir -p $OUT
CPU=(--partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --requeue); GPU=(--partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --requeue)
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" | sed "s|$EXP|\$EXP|" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@"; fi; }
log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
[ "$DRY" = 0 ] && { echo "# $(date '+%F %T') o4b _off5 validation chain from $ROOT ($(git -C $ROOT log --oneline -1 | cut -c1-60))" >> $JIDS; echo "EXP=$EXP" >> $JIDS; log "RESCORE=${RESCORE:?}"; log "INJ=${INJ:?}"; }
# production CNN cache reused (pair keys = segment index + window index, identical grids; CNN scores are tile-only, arm-independent)
if [ "$DRY" = 0 ]; then for f in $SC/madgrav2/search_out_o4b_m2_gx/cnn_shard_*.npz; do ln -sfn $f $OUT/$(basename $f); done; echo "  linked $(ls $OUT | grep -c cnn_shard) production cnn shards into $OUT"; fi
J_LR=$(submit m2-o4b-off5-lr "${CPU[@]}" --dependency=afterok:${RESCORE:?}:${INJ:?} --output=$LOGS/off5_lr_%j.log --export="$EXP" $ROOT/launchers/m2_refit_lr.sbatch); log "LR=$J_LR"
J_PR=$(submit m2-o4b-off5-far-prune "${CPU[@]}" --output=$LOGS/off5_far_%A_%a.log --array=0-95%32 --dependency=afterok:$J_LR --export="$EXP" $TPL/tmpl_far_prune_cpu.sbatch); log "PRUNE=$J_PR"
J_TU=$(submit m2-o4b-off5-cnntopup "${GPU[@]}" --gres=gpu:2 --time=10:00:00 --output=$LOGS/off5_cnntopup_%j.log --dependency=afterok:$J_PR --export="ALL,SM_RUN_SEL=o4b_off5,SM_ARM_TOPUP=-1e9,SM_TOPUP_IDX=901" $ROOT/launchers/cnn_topup.sbatch); log "TOPUP=$J_TU"
J_M1=$(submit m2-o4b-off5-merge-249 "${CPU[@]}" --dependency=afterok:$J_TU --export="ALL,JIDS=$JIDS,SM_MERGE_TAG=,SM_ARM_RULE=max:-2.49" $ROOT/launchers/m2_off5_cpumerge.sbatch); log "MERGE_249=$J_M1"
J_M2=$(submit m2-o4b-off5-merge-205 "${CPU[@]}" --dependency=afterok:$J_TU --export="ALL,JIDS=$JIDS,SM_MERGE_TAG=_p205,SM_ARM_RULE=max:-2.05" $ROOT/launchers/m2_off5_cpumerge.sbatch); log "MERGE_205=$J_M2"
J_M3=$(submit m2-o4b-off5-merge-none "${CPU[@]}" --dependency=afterok:$J_TU --export="ALL,JIDS=$JIDS,SM_MERGE_TAG=_none,SM_ARM_RULE=none" $ROOT/launchers/m2_off5_cpumerge.sbatch); log "MERGE_NONE=$J_M3"
J_SUM=$(submit m2-o4b-off5-summary "${CPU[@]}" --cpus-per-task=2 --mem=16G --time=00:30:00 --dependency=afterany:$J_M1:$J_M2:$J_M3 --output=$LOGS/off5_summary_%j.log --export="ALL,JIDS=$JIDS" --wrap="cd $ROOT && $PYV ${MADGRAV_ROOT}/m2/arm/off5_summary.py"); log "SUMMARY=$J_SUM"
echo "[off5-chain] done DRY=$DRY"

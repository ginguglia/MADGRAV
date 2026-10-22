#!/usr/bin/env bash
# MADGRAV 2.1 arm _off5 O4a validation pass (2026-09-30): the O4b chain (m2_o4b_off5_chain.sh) made run-generic (RUN=o4a). Production o4a gx EXP
# with the rescored streams (g2 = _off5), re-injected injections, own LR / bank / out dir. Stages: LR (CPU) -> prune (CPU, SM_NSHARD shards) ->
# CNN top-up of bank pairs missing from the production CNN cache (GPU, the only GPU stage) -> ONE merge max:-2.05 -> detections.json (CPU; the
# -2.49 gate buys nothing on O4b and 8/9 recovered O4a windows already pass at -2.05) -> summary mail (CPU). Watchdogs mail on stage failure.
# Usage: DRY=1|0 RESCORE=<array jid> INJ=<array jid> RUN=o4a launchers/m2_o4a_off5_chain.sh
set -uo pipefail
DRY="${DRY:-1}"; RUN="${RUN:-o4a}"; ROOT=${MADGRAV_ROOT}; SC=${MADGRAV_SCRATCH}; PYV=${PY:-python}
L=$ROOT/launchers/logs_m2; TPL=$ROOT/launchers/templates; LOGS=${MADGRAV_ROOT}/m2/prep/logs; JIDS=$L/${RUN}_off5_m2_gx_jids.env
source ${MADGRAV_ROOT}/site.conf
export OFF5_RUN=$RUN
EXP=$(grep "^EXP=" $L/${RUN}_m2_gx_jids.env | tail -1)
STREAMS_PROD=$(echo "$EXP" | tr ',' '\n' | grep "^SM_STREAMS=" | cut -d= -f2-)
EXP=$(echo "$EXP" | sed -e "s|SM_STREAMS=[^,]*|SM_STREAMS=${STREAMS_PROD}_off5|" -e "s|SM_ARM_V2_TAG=_def|SM_ARM_V2_TAG=_off5|" \
  -e "s|SM_OUT=[^,]*|SM_OUT=$SC/madgrav2/search_out_${RUN}_m2_gx_off5|" -e "s|SM_SHARD_DIR=[^,]*|SM_SHARD_DIR=$SC/madgrav2/_${RUN}_m2_gx_off5_far_shards|" \
  -e "s|SM_LR_INJ_GLOB=[^,]*|SM_LR_INJ_GLOB=$SC/madgrav2/inj_${RUN}_m2_off5/*/*_inj.npz|" -e "s|SM_LR_OUT=[^,]*|SM_LR_OUT=$SC/madgrav2/lr_${RUN}_m2_gx_off5.npz|" -e "s|SM_LR_MODEL=[^,]*|SM_LR_MODEL=$SC/madgrav2/lr_${RUN}_m2_gx_off5.npz|")
EXP=${EXP#EXP=}
NSHARD=$(echo "$EXP" | tr ',' '\n' | grep "^SM_NSHARD=" | cut -d= -f2)
for k in SM_RUN SM_STREAMS SM_ARM_V2_TAG SM_OUT SM_SHARD_DIR SM_LR_INJ_GLOB SM_LR_MODEL SM_NSHARD SM_ZL_NET_CUT SM_ZL_GATES SM_PREP; do echo "  $k = $(echo "$EXP" | tr ',' '\n' | grep "^$k=" | cut -d= -f2-)"; done
[ "$(echo "$EXP" | tr ',' '\n' | grep -c "^SM_ARM_V2_TAG=_off5")" = 1 ] || { echo "SM_ARM_V2_TAG substitution failed"; exit 1; }
OUT=$SC/madgrav2/search_out_${RUN}_m2_gx_off5; mkdir -p $OUT
CPU=(--partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --requeue); GPU=(--partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --requeue)
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" | sed "s|$EXP|\$EXP|" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@"; fi; }
log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
watch(){ local st=$1 jid=$2; submit m2-off5-${RUN}-watch-$st "${CPU[@]}" --cpus-per-task=1 --mem=2G --time=00:10:00 --dependency=afternotok:$jid --output=$LOGS/off5_${RUN}_watch_%x_%j.log \
  --wrap="cd $ROOT/launchers && $PYV -c \"import chain_notify as n, os; s=os.popen('sacct -j $jid -X -n -o JobID,State,Elapsed').read(); n.deliver('$OUT','FAILED_$st.txt','[MADGRAV] ${RUN^^} _off5 pass: stage $st (job $jid) FAILED', 'Stage $st job $jid ended in a failed state; downstream afterok jobs will stay pending (cancel them or fix and resubmit).\\n'+s)\""; }
[ "$DRY" = 0 ] && { echo "# $(date '+%F %T') $RUN _off5 validation chain from $ROOT ($(git -C $ROOT log --oneline -1 | cut -c1-60))" >> $JIDS; echo "EXP=$EXP" >> $JIDS; log "RESCORE=${RESCORE:?}"; log "INJ=${INJ:?}"; }
# production CNN cache reused (pair keys = segment index + window index, identical grids; CNN scores are tile-only, arm-independent); incl. cnn_shard_900 (arm -3.00 top-up)
if [ "$DRY" = 0 ]; then for f in $SC/madgrav2/search_out_${RUN}_m2_gx/cnn_shard_*.npz; do ln -sfn $f $OUT/$(basename $f); done; echo "  linked $(ls $OUT | grep -c cnn_shard) production cnn shards into $OUT"; fi
J_LR=$(submit m2-${RUN}-off5-lr "${CPU[@]}" --dependency=afterok:${RESCORE:?}:${INJ:?} --output=$LOGS/off5_${RUN}_lr_%j.log --export="$EXP,OFF5_RUN=$RUN" $ROOT/launchers/m2_refit_lr.sbatch); log "LR=$J_LR"
J_PR=$(submit m2-${RUN}-off5-far-prune "${CPU[@]}" --output=$LOGS/off5_${RUN}_far_%A_%a.log --array=0-$((NSHARD-1))%32 --dependency=afterok:$J_LR --export="$EXP,OFF5_RUN=$RUN" $TPL/tmpl_far_prune_cpu.sbatch); log "PRUNE=$J_PR"
J_TU=$(submit m2-${RUN}-off5-cnntopup "${GPU[@]}" --gres=gpu:2 --time=10:00:00 --output=$LOGS/off5_${RUN}_cnntopup_%j.log --dependency=afterok:$J_PR --export="ALL,SM_RUN_SEL=${RUN}_off5,SM_ARM_TOPUP=-1e9,SM_TOPUP_IDX=901,OFF5_RUN=$RUN" $ROOT/launchers/cnn_topup.sbatch); log "TOPUP=$J_TU"
J_M=$(submit m2-${RUN}-off5-merge-205 "${CPU[@]}" --dependency=afterok:$J_TU --export="ALL,JIDS=$JIDS,SM_MERGE_TAG=,SM_ARM_RULE=max:-2.05,OFF5_RUN=$RUN" $ROOT/launchers/m2_off5_cpumerge.sbatch); log "MERGE=$J_M"
J_SUM=$(submit m2-${RUN}-off5-summary "${CPU[@]}" --cpus-per-task=2 --mem=16G --time=00:30:00 --dependency=afterany:$J_M --output=$LOGS/off5_${RUN}_summary_%j.log --export="ALL,JIDS=$JIDS,OFF5_RUN=$RUN" --wrap="cd $ROOT && $PYV ${MADGRAV_ROOT}/m2/arm/off5_summary.py"); log "SUMMARY=$J_SUM"
for st_j in RESCORE:${RESCORE:?} INJ:${INJ:?} LR:$J_LR PRUNE:$J_PR TOPUP:$J_TU MERGE:$J_M; do st=${st_j%%:*}; jid=${st_j#*:}; w=$(watch $st $jid); log "WATCH_$st=$w"; done
echo "[off5-chain] done RUN=$RUN DRY=$DRY"

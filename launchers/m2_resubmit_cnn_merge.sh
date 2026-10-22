#!/usr/bin/env bash
# Resubmit the CNN-shard + far-merge (+ merge-none) stages of a pending MADGRAV2 chain with extra EXP variables
# (2026-09-20: zero-lag-3 adoption -> SM_ZL_NET_CUT=3.0,SM_ZL_GATES=1). Mirrors launchers/m2_run_chain.sh lines 56-58 exactly.
#   DRY=1 (default) prints; DRY=0 scancels the old CNN/MERGE/MERGE_NONE jobs and submits, appending the new ids to the jids file.
# usage: [DRY=0] m2_resubmit_cnn_merge.sh <run> <jids.env> <EXP line no|last> <PRUNE> <oldCNN> <oldMERGE> <oldMERGE_NONE> <extra EXP vars>
set -uo pipefail
RUN=$1; JIDS=$2; EL=$3; PRUNE=$4; OCNN=$5; OMRG=$6; ONONE=$7; EXTRA=$8; DRY="${DRY:-1}"
if [ "$EL" = last ]; then EXP=$(grep "^EXP=" "$JIDS" | tail -1 | sed 's/^EXP=//'); else EXP=$(sed -n "${EL}p" "$JIDS" | sed 's/^EXP=//'); fi
[[ "$EXP" == ALL,SM_RUN=$RUN,* ]] || { echo "FATAL: EXP line does not start with ALL,SM_RUN=$RUN: ${EXP:0:60}" >&2; exit 1; }
ROOT=$(echo "$EXP" | tr ',' '\n' | grep '^MADGRAV_ROOT=' | cut -d= -f2); NSHARD=$(echo "$EXP" | tr ',' '\n' | grep '^SM_NSHARD=' | cut -d= -f2)
[ -n "$ROOT" ] && [ -n "$NSHARD" ] || { echo "FATAL: ROOT/NSHARD missing in EXP" >&2; exit 1; }
grep -q "ZL_GATES" "$ROOT/search_mode/_pb_shard_common.py" || { echo "FATAL: $ROOT lacks the SM_ZL_GATES patch" >&2; exit 1; }
for j in $OCNN $OMRG $ONONE; do st=$(squeue -h -j $j -o %T 2>/dev/null | head -1); [ "$st" = PENDING ] || { echo "FATAL: old job $j is '$st', not PENDING -> refusing" >&2; exit 1; }; done
EXP="$EXP,$EXTRA"; L="$ROOT/launchers/logs_m2"; TPL="$ROOT/launchers/templates"
GPU=(--partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --requeue)
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" | sed "s|$EXP|\$EXP|" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@"; fi; }
log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
echo "[resubmit] run=$RUN ROOT=$ROOT NSHARD=$NSHARD PRUNE=$PRUNE old=($OCNN $OMRG $ONONE) extra=$EXTRA DRY=$DRY jids=$JIDS"
if [ "$DRY" = 0 ]; then scancel $OCNN $OMRG $ONONE && echo "[resubmit] cancelled $OCNN $OMRG $ONONE"; echo "# $(date '+%F %T') RESUBMIT cnn/merge with $EXTRA (zero-lag-3 adoption; old $OCNN/$OMRG/$ONONE cancelled) from $ROOT ($(git -C "$ROOT" log --oneline -1 | cut -c1-60))" >> "$JIDS"; echo "EXP=$EXP" >> "$JIDS"; fi
J_CNN=$(submit m2-${RUN}-cnn-shard "${GPU[@]}" --output="$L/${RUN}_m2_cnnshard_%A_%a.log" --array=0-$((NSHARD-1))%16 --dependency=afterany:$PRUNE --export="$EXP" "$TPL/tmpl_cnn_shard.sbatch"); log "CNN=$J_CNN"
J_MRG=$(submit m2-${RUN}-far-merge "${GPU[@]}" --output="$L/${RUN}_m2_far_merge_max_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_ARM_RULE=max:-2.05" "$TPL/tmpl_far_merge.sbatch"); log "MERGE=$J_MRG"
J_NONE=$(submit m2-${RUN}-merge-none "${GPU[@]}" --output="$L/${RUN}_m2_far_merge_none_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_MERGE_TAG=_none,SM_ARM_RULE=none" "$TPL/tmpl_far_merge.sbatch"); log "MERGE_NONE=$J_NONE"

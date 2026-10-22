#!/usr/bin/env bash
# MADGRAV2 per-run chain (2026-09-19 recipe): streams (centroid v2 + arm v2 + CAT2 union veto) -> inject (re-whitened coh) -> LR refit
#   -> far-prune -> cnn-shard -> far-merge (arm rule max:-2.05). Re-whitened coherence + 50 Hz guard, full GWTC excision from the
#   background and the refit noise class, livetime fix, series-cache view (cache hits). Outputs under $SC/madgrav2/<run>_m2/...
# Usage: DRY=1 launchers/m2_run_chain.sh <run>            (prints)
#        DRY=0 STAGES=prep launchers/m2_run_chain.sh <run>  (submits streams + inject [+ o3b cohref after the cache job])
#        DRY=0 STAGES=far  STR=<jid> INJ=<jid> [COHREF=<jid>] launchers/m2_run_chain.sh <run>   (LR -> prune -> cnn -> merges, after the assessment)
#        DRY=0 STAGES=inj  launchers/m2_run_chain.sh <run>   (inject only, e.g. to add a feature column: MADGRAV2.1 gx 2026-09-19)
#        DRY=0 STAGES=all launchers/m2_run_chain.sh <run>   (everything with dependencies)
set -uo pipefail
RUN="${1:?run (o3a|o3b|o4a|o4b)}"; DRY="${DRY:-1}"; STAGES="${STAGES:-all}"; TAG="${TAG:-}"   # TAG suffixes the far-stage outputs/LR/jids (e.g. TAG=_nogx for a reference run); injections are untagged
ROOT="${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; SC=${MADGRAV_SCRATCH}; PYV=${PY:-python}
L="$ROOT/launchers/logs_m2"; TPL="$ROOT/launchers/templates"; JIDS="$L/${RUN}_m2${TAG}_jids.env"; mkdir -p "$L"
N=$($PYV -c "import json;print(json.load(open('$SC/${RUN}_full_coincident.json'))['n_segments'])")
ceil_div(){ echo $(( ($1 + $2 - 1) / $2 )); }; round16(){ echo $(( ( ($1 + 8) / 16 ) * 16 )); }
STREAMS_NSHARD=$(ceil_div $N 3); NSHARD=$(round16 $(ceil_div $N 12)); [ $NSHARD -lt 16 ] && NSHARD=16
NINJ=$(wc -l < "$L/injfull_manifest_${RUN}.txt")
CACHE="$SC/madgrav2/${RUN}-fast_full/_seriescache"; REF="$CACHE/_cohref"; VETO="${SM_VETO:-$SC/madgrav2/dq/veto_mask_${RUN}_hw.json}"   # 2026-10-05: CAT2 union + hardware-injection flags (MADGRAV2/m2/dq/build_veto_mask_hw.py); v1 = veto_mask_${RUN}.json
[ -f "$VETO" ] || { echo "FATAL: veto mask $VETO missing" >&2; exit 1; }
EXP="ALL,SM_RUN=$RUN,MADGRAV_ROOT=$ROOT,SCRATCH=$SC,PYTHON=$PYV"
EXP="$EXP,SM_STREAMS=$SC/madgrav2/streams_${RUN}_full,SM_PREP=${MADGRAV_ROOT}/data/${RUN}_search_prep,SM_STRAIN=$SC/strain_${RUN}_full,SM_BGJSON=$SC/${RUN}_full_coincident.json"
EXP="$EXP,SM_VETO=$VETO,SM_SERIESCACHE=$CACHE,SM_COH_REWHITEN=1,SM_COHREF_DIR=$REF,SM_COH_GUARD_HZ=50,SM_COH_GUARD_FLOOR=0.115"
EXP="$EXP,SM_GX=${SM_GX:-1},SM_GX_CAP=${SM_GX_CAP:-15}"   # MADGRAV2.1 guided second-leg excess LR feature (search_mode/guided_excess.py); SM_GX=0 -> 7-feature LR
EXP="$EXP,SM_ARM_V2=1,SM_ARM_V2_DIR=$SC/madgrav2/arm/models,SM_ARM_V2_TAG=_def"
EXP="$EXP,SM_OUT=$SC/madgrav2/search_out_${RUN}_m2${TAG},SM_SHARD_DIR=$SC/madgrav2/_${RUN}_m2${TAG}_far_shards"
EXP="$EXP,SM_LR_INJ_GLOB=$SC/madgrav2/inj_${RUN}_m2/*/*_inj.npz,SM_LR_OUT=$SC/madgrav2/lr_${RUN}_m2${TAG}.npz,SM_LR_MODEL=$SC/madgrav2/lr_${RUN}_m2${TAG}.npz"
if [ "$RUN" = o4b ]; then   # o4b: pilot streams + CAT2 veto applied post hoc (apply_veto_to_streams.py, bit-identical to a rebuild); re-whitened injections from the 2026-09-19 coh chain (inject does not use veto/guard)
  EXP=$(echo "$EXP" | sed "s|SM_STREAMS=[^,]*|SM_STREAMS=$SC/madgrav2/streams_o4b_full_cat2|")   # o4b: pilot streams + CAT2 veto; injections = inj_o4b_m2 (re-injected with gx, STAGES=inj)
fi
EXP="$EXP,SM_EVENTSJSON=$ROOT/search_mode/events_gwtc_all.json"
EXP="$EXP,SM_CAND_FLOOR=3.0,SM_NETSIG_FLOOR=3.0,SM_NET_CUT=3.0,SM_N_CNN_CANDIDATES=1000000,SM_FG_MAXNET=1000000,SM_BLIND_FLOOR=1,SM_SURV_ABORT=60000000,SM_NSHARD=$NSHARD"
GPU=(--partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --requeue); CPU=(--partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --requeue)
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" | sed "s|$EXP|\$EXP|" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@"; fi; }
log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
echo "[m2-chain] run=$RUN N=$N streams_shards=$STREAMS_NSHARD far_shards=$NSHARD inj_tasks=$NINJ DRY=$DRY STAGES=$STAGES"; echo "[m2-chain] EXP=$EXP"
[ "$DRY" = 0 ] && { echo "# $(date '+%F %T') m2 chain $RUN STAGES=$STAGES from $ROOT ($(git -C "$ROOT" log --oneline -1))" >> "$JIDS"; echo "EXP=$EXP" >> "$JIDS"; }
J_STR="${STR:-}"; J_INJ="${INJ:-}"; J_REF="${COHREF:-}"
if [ "$RUN" = o4b ] && { [ "$STAGES" = prep ] || [ "$STAGES" = all ]; }; then echo "[m2-chain] o4b prep = existing streams_o4b_full_cat2 (no streams to submit); injections via STAGES=inj, then STAGES=far STR=1747187 INJ=<inj jid>"; J_STR=1747187; STAGES=$([ "$STAGES" = all ] && echo inj_far || echo inj); fi
if [ "$STAGES" = inj ] || [ "$STAGES" = inj_far ]; then
  J_INJ=$(submit m2-${RUN}-inject "${GPU[@]}" --output="$L/${RUN}_m2_inject_%A_%a.log" --array=0-$((NINJ-1))%6 --export="$EXP" "$ROOT/launchers/m2_inject_run.sbatch"); log "INJ=$J_INJ"
  [ "$STAGES" = inj_far ] && STAGES=far
fi
if [ "$STAGES" = prep ] || [ "$STAGES" = all ]; then
  J_STR=$(submit m2-${RUN}-str "${GPU[@]}" --output="$L/${RUN}_m2_str_%A_%a.log" --array=0-$((STREAMS_NSHARD-1))%8 --export="$EXP,STREAMS_NSHARD=$STREAMS_NSHARD" "$TPL/tmpl_streams.sbatch"); log "STR=$J_STR"
  J_INJ=$(submit m2-${RUN}-inject "${GPU[@]}" --output="$L/${RUN}_m2_inject_%A_%a.log" --array=0-$((NINJ-1))%6 --export="$EXP" "$ROOT/launchers/m2_inject_run.sbatch"); log "INJ=$J_INJ"
  if [ "$RUN" = o3b ] && [ -n "${CACHE_JOB:-}" ]; then   # o3b: cache rebuilt under o3b prep (job $CACHE_JOB) -> cohref after it; inject/prune need the refs
    J_REF=$(submit m2-o3b-cohref "${CPU[@]}" --output="$L/o3b_m2_cohref_%j.log" --dependency=afterok:$CACHE_JOB --time=01:00:00 --cpus-per-task=6 --export="$EXP" --wrap="cd $ROOT && SM_SERIESCACHE=$CACHE SM_COHREF_DIR=$REF $PYV search_mode/build_coh_ref.py --procs 6"); log "COHREF=$J_REF"
  fi
fi
if [ "$STAGES" = far ] || [ "$STAGES" = all ]; then
  if [ "$RUN" = o4b ]; then DEP="${EXTRA_DEP:?o4b far stage needs EXTRA_DEP (its streams/injections already exist; e.g. afterany:<coh merges>)}"   # prerequisites finished long ago -> not valid Slurm dependencies
  else [ -n "$J_STR" ] && [ -n "$J_INJ" ] || { echo "FATAL: STAGES=far needs STR=<jid> INJ=<jid>" >&2; exit 1; }
    DEP="afterok:$J_STR:$J_INJ"; [ -n "$J_REF" ] && DEP="$DEP:$J_REF"; [ -n "${EXTRA_DEP:-}" ] && DEP="$DEP,$EXTRA_DEP"; fi   # EXTRA_DEP e.g. afterany:<coh merges> (o4b: start only after the coh-run assessment material exists)
  J_LR=$(submit m2-${RUN}-lr --dependency=$DEP --output="$L/${RUN}_m2_lr_%j.log" --export="$EXP" "$ROOT/launchers/m2_refit_lr.sbatch"); log "LR=$J_LR"
  J_PR=$(submit m2-${RUN}-far-prune "${GPU[@]}" --output="$L/${RUN}_m2_far_%A_%a.log" --array=0-$((NSHARD-1))%16 --dependency=afterok:$J_LR --export="$EXP" "$TPL/tmpl_far_prune.sbatch"); log "PRUNE=$J_PR"
  J_CNN=$(submit m2-${RUN}-cnn-shard "${GPU[@]}" --output="$L/${RUN}_m2_cnnshard_%A_%a.log" --array=0-$((NSHARD-1))%16 --dependency=afterany:$J_PR --export="$EXP" "$TPL/tmpl_cnn_shard.sbatch"); log "CNN=$J_CNN"
  J_MRG=$(submit m2-${RUN}-far-merge "${GPU[@]}" --output="$L/${RUN}_m2_far_merge_max_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_ARM_RULE=max:-2.05" "$TPL/tmpl_far_merge.sbatch"); log "MERGE=$J_MRG"
  J_NONE=$(submit m2-${RUN}-merge-none "${GPU[@]}" --output="$L/${RUN}_m2_far_merge_none_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_MERGE_TAG=_none,SM_ARM_RULE=none" "$TPL/tmpl_far_merge.sbatch"); log "MERGE_NONE=$J_NONE"
fi
echo "[m2-chain] done (DRY=$DRY)"

#!/usr/bin/env bash
# ============================================================================================
# MADGRAV generic blind-search launcher — one command, pick the run. PORTABLE (site.conf driven).
#
#   bash run_chain.sh <run> [--dry-run] [--far-only]
#     <run>       run label, e.g. o3a o3b o4a o4b (needs $SCRATCH/<run>_full_coincident.json + strain)
#     --dry-run   print the full plan (paths, shard counts, every sbatch command) and submit NOTHING
#     --far-only  skip the injection stage even if its JSONs exist (deck without p_astro)
#
# ALL machine-specific values (paths, SLURM account/partitions, notification) live in site.conf
# (git-ignored; copy site.conf.example). Nothing author-specific is baked into the code.
# Convention-only per run: paths derived from SCRATCH + <run>; segment count from the manifest;
# shard counts derived from it. Correct stage order, all 8 blind-run lessons baked in:
#   prep -> (streams || cache || inject) -> PRUNE -> cnn_shard -> far_merge   (+ parallel fg preview)
#   afterany throughout, --requeue on every array/GPU job, results ALWAYS written to <out>/SUMMARY.txt.
# ============================================================================================
set -uo pipefail
RUN=""; DRY=0; FARONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --far-only) FARONLY=1 ;;
    -*) echo "unknown flag $a"; exit 2 ;;
    *) RUN="$a" ;;
  esac
done
[ -n "$RUN" ] || { echo "usage: bash run_chain.sh <run> [--dry-run] [--far-only]"; exit 2; }

# --- site config (paths / SLURM / notifications) ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # .../launchers
SITE="$HERE/../site.conf"
if [ ! -f "$SITE" ]; then
  echo "!! no site.conf at $SITE"; echo "   cp $HERE/../site.conf.example $HERE/../site.conf  and edit it for your system."; exit 1
fi
set -a; source "$SITE"; set +a
for v in MADGRAV_ROOT SCRATCH PYTHON SLURM_ACCOUNT PARTITION_GPU QOS_GPU PARTITION_CPU QOS_CPU; do
  [ -n "${!v:-}" ] || { echo "!! site.conf is missing required value: $v"; exit 1; }
done

MR="$MADGRAV_ROOT"; L="$MR/launchers"; TPL="$L/templates"; PY="$PYTHON"; SC="$SCRATCH"; SM="$MR/search_mode"
OUT=$SC/search_out_${RUN}_far
MANIFEST=$SC/${RUN}_full_coincident.json
log(){ echo "[$(date '+%F %T')] $*"; }
ceil_div(){ echo $(( ($1 + $2 - 1) / $2 )); }
round16(){ echo $(( ( ($1 + 8) / 16 ) * 16 )); }
# SLURM site args spliced into each submit (GPU vs CPU stages)
GPU_SB=(--partition="$PARTITION_GPU" --account="$SLURM_ACCOUNT" --qos="$QOS_GPU")
CPU_SB=(--partition="$PARTITION_CPU" --account="$SLURM_ACCOUNT" --qos="$QOS_CPU")

# --- derive segment count + shard geometry from the manifest ---
[ -f "$MANIFEST" ] || { echo "!! manifest $MANIFEST not found — is <run> correct and downloaded to $SCRATCH?"; exit 1; }
N=$("$PY" -c "import json;print(json.load(open('$MANIFEST'))['n_segments'])")
NFILES=$(( 2 * N ))
STREAMS_NSHARD=$(ceil_div $N 3)                 # ~3 segs/shard   (o4a 821->274, o4b 1086->362)
CACHE_NSHARD=$(ceil_div $(( 2 * N )) 11)         # ~11 (seg,det)/shard
NSHARD=$(round16 $(ceil_div $N 12))              # FAR shards, rounded to 16 (o4a 821->64, o4b 1086->96)

INJECT=0
if [ "$FARONLY" != 1 ] && [ -f "$SM/${RUN}_events_inj.json" ] && [ -f "$SM/${RUN}_seg_event_inj.json" ]; then INJECT=1; fi
NOTIFY_STATE="$([ -n "${NOTIFY_EMAIL:-}" ] && echo "email -> $NOTIFY_EMAIL (+ file)" || echo 'FILE ONLY (SUMMARY.txt; no email)')"

log "==== MADGRAV chain plan: $RUN ===="
echo "  site.conf       : $SITE"
echo "  manifest        : $MANIFEST  (n_segments=$N)"
echo "  strain expected : $NFILES npz in $SC/strain_${RUN}_full"
echo "  out dir         : $OUT"
echo "  SLURM           : acct=$SLURM_ACCOUNT  gpu=$PARTITION_GPU/$QOS_GPU  cpu=$PARTITION_CPU/$QOS_CPU"
echo "  shard geometry  : streams=$STREAMS_NSHARD  cache=$CACHE_NSHARD  FAR(prune/cnn/merge)=$NSHARD"
echo "  inject/p_astro  : $([ $INJECT = 1 ] && echo 'YES' || echo 'NO (FAR-only; deck without p_astro)')"
echo "  notifications   : $NOTIFY_STATE"
echo "  mode            : $([ $DRY = 1 ] && echo 'DRY-RUN (submit nothing)' || echo 'LIVE (submit)')"
echo

if [ "$DRY" != 1 ]; then
  n=$(ls $SC/strain_${RUN}_full/*.npz 2>/dev/null | wc -l)
  log "$RUN strain: $n/$NFILES npz on disk"
  if [ "$n" -lt "$NFILES" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "!! strain incomplete ($n/$NFILES). Re-run when complete, or FORCE=1 bash run_chain.sh $RUN"; exit 1
  fi
fi

submit(){ local name=$1; shift
  if [ "$DRY" = 1 ]; then echo "  sbatch --job-name=$name $*" >&2; echo "DRY_${name}"
  else sbatch --parsable --job-name="$name" "$@"; fi
}

EXP="ALL,SM_RUN=$RUN"                 # --export=ALL carries every site.conf var to the job
J_PREP=$(submit  ${RUN}-prep    "${CPU_SB[@]}" --output=$L/${RUN}_prep_%j.log      --export=$EXP $TPL/tmpl_prep.sbatch)
log "prep      $J_PREP"
J_STR=$(submit   ${RUN}-str-gpu "${GPU_SB[@]}" --output=$L/${RUN}_str_gpu_%A_%a.log --array=0-$((STREAMS_NSHARD-1))%8  --dependency=afterany:$J_PREP --export=$EXP,STREAMS_NSHARD=$STREAMS_NSHARD $TPL/tmpl_streams.sbatch)
log "streams   $J_STR   (afterany prep, $STREAMS_NSHARD shards)"
J_CACHE=$(submit ${RUN}-cache   "${CPU_SB[@]}" --output=$L/${RUN}_cache_%A_%a.log --array=0-$((CACHE_NSHARD-1))%32 --dependency=afterany:$J_PREP --export=$EXP,CACHE_NSHARD=$CACHE_NSHARD $TPL/tmpl_cache.sbatch)
log "cache     $J_CACHE (afterany prep, $CACHE_NSHARD shards)"

if [ $INJECT = 1 ]; then
  J_INJ=$(submit ${RUN}-inject "${GPU_SB[@]}" --output=$L/${RUN}_inject_%A_%a.log --array=0-11%6 --dependency=afterany:$J_PREP --export=$EXP $TPL/tmpl_inject.sbatch)
  log "inject    $J_INJ   (afterany prep, parallel)"
else
  J_INJ=""; log "inject    SKIPPED (FAR-only; finalizer renders deck without p_astro)"
fi

J_PRUNE=$(submit ${RUN}-far-prune "${GPU_SB[@]}" --output=$L/${RUN}_far_%A_%a.log --array=0-$((NSHARD-1))%16 --dependency=afterany:$J_STR:$J_CACHE --export=$EXP,SM_NSHARD=$NSHARD $TPL/tmpl_far_prune.sbatch)
log "PRUNE     $J_PRUNE (afterany streams+cache, $NSHARD shards) -- ~12-18h long-pole, writes _${RUN}_far_shards"
J_FG=$(submit    ${RUN}-fg        "${GPU_SB[@]}" --output=$L/${RUN}_fg_%j.log      --dependency=afterany:$J_STR:$J_CACHE --export=$EXP $TPL/tmpl_fg.sbatch)
log "fg-preview $J_FG   (afterany streams+cache, parallel; candidate list ~1h in)"
J_CNN=$(submit   ${RUN}-cnn-shard "${GPU_SB[@]}" --output=$L/${RUN}_cnnshard_%A_%a.log --array=0-$((NSHARD-1))%16 --dependency=afterany:$J_PRUNE --export=$EXP,SM_NSHARD=$NSHARD $TPL/tmpl_cnn_shard.sbatch)
log "cnn_shard $J_CNN   (afterany PRUNE, $NSHARD shards)"
J_MRG=$(submit   ${RUN}-far-merge "${GPU_SB[@]}" --output=$L/${RUN}_far_merge_%j.log --dependency=afterany:$J_CNN --export=$EXP,SM_NSHARD=$NSHARD $TPL/tmpl_far_merge.sbatch)
log "far_merge $J_MRG   (afterany cnn_shard)"

if [ "$DRY" = 1 ]; then
  echo; log "==== DRY-RUN complete — nothing submitted ===="
  echo "JIDS(dry)  prep=$J_PREP streams=$J_STR cache=$J_CACHE inject=${J_INJ:-skip} prune=$J_PRUNE fg=$J_FG cnn=$J_CNN merge=$J_MRG"
  exit 0
fi

# --- progress mailer (detached): writes <out>/STATUS.txt every cycle; emails only if configured ---
CH_RUN=$RUN CH_OUT=$OUT CH_FAR_ARRAY=$J_CNN CH_MERGE=$J_MRG CH_FG=$J_MRG CH_PRUNE=$J_PRUNE \
  CH_SHARD_DIR=$SC/_${RUN}_far_shards CH_NSHARD=$NSHARD \
  setsid nohup "$PY" "$L/chain_mailer.py" >"$L/${RUN}_mailer.log" 2>&1 &
log "progress mailer watching (detached)"

# --- finalizer (detached): merge -> pastro_search -> deck -> writes <out>/SUMMARY.txt (+ email if configured) ---
JINJ_ARG=${J_INJ:-$J_MRG}
CH_RUN=$RUN setsid nohup "$PY" "$L/chain_finalize.py" "$J_MRG" "$JINJ_ARG" "$OUT" >"$L/${RUN}_finalize.log" 2>&1 &
log "finalizer watching (detached)"

log "==== $RUN CHAIN LAUNCHED ===="
echo "JIDS  prep=$J_PREP streams=$J_STR cache=$J_CACHE inject=${J_INJ:-skip} prune=$J_PRUNE fg=$J_FG cnn=$J_CNN merge=$J_MRG"
echo "$J_PREP $J_STR $J_CACHE ${J_INJ:-skip} $J_PRUNE $J_FG $J_CNN $J_MRG" > "$L/.${RUN}_chain_jids"
echo "Results land in $OUT/SUMMARY.txt (+ deck.html, + email if NOTIFY_EMAIL set). fg preview ~1h."

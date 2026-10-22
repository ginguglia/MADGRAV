#!/usr/bin/env bash
# MADGRAV2 O4b "coh" rerun: the 2026-09-17 pilot composition (frozen CAE + centroid v2 + arm v2 + floor 3, caps off)
# with the RE-WHITENED coherence series (SM_COH_REWHITEN=1, search_mode/coh_rewhiten.py) and the MADGRAV2 series-cache
# view (sidecars point at MADGRAV2/data/o4b_search_prep -> cache HITS; the pilot recomputed whitening in every flush).
# Chain: inject (24 tasks, new coh) -> LR refit -> far-prune (96 shards) -> cnn-shard (96) -> far-merge max + none/mean/both/ramp/max3/max4.
# All outputs go to NEW paths (inj_o4b_coh, lr_o4b_coh.npz, _o4b_coh_far_shards, search_out_o4b_coh_far): the pilot is untouched.
# Usage: DRY=1 launchers/m2_o4b_coh_chain.sh   (prints)   |   DRY=0 launchers/m2_o4b_coh_chain.sh   (submits)
# Options: EXCISE_ALL=1 (excise all GWTC events; default 0 = pilot-identical background), REUSE_INJ=<jobid> (skip the inject stage, depend on that job)
# Do NOT start while the pilot's cnn/merge jobs are still queued/running if GPU budget is tight (both use %16 arrays).
set -uo pipefail
DRY="${DRY:-1}"
ROOT="${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; SC=${MADGRAV_SCRATCH}; PYV=${PY:-python}
L="$ROOT/launchers/logs_m2"; TPL="$ROOT/launchers/templates"; JIDS="$L/o4b_coh_jids.env"; mkdir -p "$L"
CACHE="$SC/madgrav2/o4b-fast_full/_seriescache"; REF="$CACHE/_cohref"
[ -f "$REF/_manifest.json" ] || { echo "FATAL: $REF/_manifest.json missing -- run: SM_SERIESCACHE=$CACHE $PYV $ROOT/search_mode/build_coh_ref.py" >&2; exit 1; }
[ "$(ls "$CACHE"/*_series.json 2>/dev/null | wc -l)" -ge 2172 ] || { echo "FATAL: series-cache view $CACHE incomplete" >&2; exit 1; }
grep -q 'MADGRAV2/data/o4b_search_prep' "$CACHE/o4b_1396417050_H1_series.json" || { echo "FATAL: cache sidecars do not point at MADGRAV2 prep (would recompute)" >&2; exit 1; }
EXP="ALL,SM_RUN=o4b,MADGRAV_ROOT=$ROOT,SCRATCH=$SC,PYTHON=$PYV"
EXP="$EXP,SM_STREAMS=$SC/madgrav2/streams_o4b_full,SM_PREP=${MADGRAV_ROOT}/data/o4b_search_prep"   # MUST equal the cache sidecar prep string (cache hits); ASD files identical in both trees; guard accepts it under any root
EXP="$EXP,SM_SERIESCACHE=$CACHE,SM_COH_REWHITEN=1,SM_COHREF_DIR=$REF"
EXP="$EXP,SM_ARM_V2=1,SM_ARM_V2_DIR=$SC/madgrav2/arm/models,SM_ARM_V2_TAG=_def"
EXP="$EXP,SM_OUT=$SC/madgrav2/search_out_o4b_coh_far,SM_SHARD_DIR=$SC/madgrav2/_o4b_coh_far_shards"
EXP="$EXP,SM_LR_INJ_GLOB=$SC/madgrav2/inj_o4b_coh/*/*_inj.npz,SM_LR_OUT=$SC/madgrav2/lr_o4b_coh.npz,SM_LR_MODEL=$SC/madgrav2/lr_o4b_coh.npz"
[ "${EXCISE_ALL:-0}" = 1 ] && EXP="$EXP,SM_EVENTSJSON=$ROOT/search_mode/events_gwtc_all.json"   # EXCISE_ALL=1 only; default OFF for the 1-to-1 coherence comparison with the pilot (2026-09-19). 2026-09-19: excise ALL 391 GWTC events (161 in O4b) from background + LR-refit noise; the pilot excised none in O4b (default list = 2 O4a events)
EXP="$EXP,SM_CAND_FLOOR=3.0,SM_NETSIG_FLOOR=3.0,SM_NET_CUT=3.0,SM_N_CNN_CANDIDATES=1000000,SM_FG_MAXNET=1000000,SM_BLIND_FLOOR=1,SM_SURV_ABORT=60000000,SM_NSHARD=96"
SB=(--partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --requeue)
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" | sed "s|$EXP|\$EXP|" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@"; fi; }
log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
[ "$DRY" = 0 ] && { echo "# $(date '+%F %T') coh chain from $ROOT ($(git -C "$ROOT" log --oneline -1))" >> "$JIDS"; echo "EXP=$EXP" >> "$JIDS"; }
echo "[coh-chain] DRY=$DRY root=$ROOT"; echo "[coh-chain] EXP=$EXP"
if [ -n "${REUSE_INJ:-}" ]; then J_INJ="$REUSE_INJ"; echo "[coh-chain] reusing running inject job $J_INJ"; else J_INJ=$(submit m2-o4b-coh-inject --array=0-23%8 --export="$EXP" "$ROOT/launchers/m2_inject_o4b_coh.sbatch"); fi;                                                        log "INJ=$J_INJ"
J_LR=$(submit  m2-o4b-coh-lr  --dependency=afterok:$J_INJ --output="$L/o4b_coh_lr_%j.log" --export="$EXP" "$ROOT/launchers/m2_refit_lr.sbatch");                     log "LR=$J_LR"
J_PR=$(submit  m2-o4b-coh-far-prune "${SB[@]}" --output="$L/o4b_coh_far_%A_%a.log" --array=0-95%16 --dependency=afterok:$J_LR --export="$EXP" "$TPL/tmpl_far_prune.sbatch"); log "PRUNE=$J_PR"
J_CNN=$(submit m2-o4b-coh-cnn-shard "${SB[@]}" --output="$L/o4b_coh_cnnshard_%A_%a.log" --array=0-95%16 --dependency=afterany:$J_PR --export="$EXP" "$TPL/tmpl_cnn_shard.sbatch"); log "CNN=$J_CNN"
J_MRG=$(submit m2-o4b-coh-far-merge "${SB[@]}" --output="$L/o4b_coh_far_merge_max_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_ARM_RULE=max:-2.05" "$TPL/tmpl_far_merge.sbatch");     log "MERGE=$J_MRG"
OTH=""; declare -A RULES=([none]=none [mean]=mean:-1.29 [both]=both:-2.76 [ramp]=ramp:-2.47 [max3]=max:-3.0 [max4]=max:-4.0)   # 2026-09-19: rules were never exported before (all tagged merges would have run as none); max3/max4 = looser hard thresholds for the arm-rule decision
for tag in none mean both ramp max3 max4; do J=$(submit m2-o4b-coh-merge-$tag "${SB[@]}" --output="$L/o4b_coh_far_merge_${tag}_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_MERGE_TAG=_$tag,SM_ARM_RULE=${RULES[$tag]}" "$TPL/tmpl_far_merge.sbatch"); OTH="$OTH $tag=$J"; done; log "MERGES_OTHER=$OTH"
echo "[coh-chain] done (DRY=$DRY). Compare: $SC/madgrav2/search_out_o4b_coh_far/detections.json vs pilot $SC/madgrav2/search_out_o4b_far/detections.json"

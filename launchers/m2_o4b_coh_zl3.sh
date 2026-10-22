#!/usr/bin/env bash
# MADGRAV2 O4b coherence-run ZERO-LAG-3 test (2026-09-20): re-run ONLY the cnn-shard + merge stages of the finished coh chain
# (jids logs_m2/o4b_coh_jids.env, prune bank _o4b_coh_far_shards reused, same LR model) with
#   SM_ZL_NET_CUT=3.0  zero-lag candidate windows at net>3 (pilot/coh: 4; the prune background is already banked at net>3), and
#   SM_ZL_GATES=1      centroid-v2 + arm gates on the zero-lag windows (== the prune's foreground definition; the pilot/coh
#                      shard setup took ungated windows).
# Purpose: does the 'b NO-TRIGGER with a net 3-4 pair' miss class yield catalogue detections, and is any of the 202 net>4
# candidates lost by cluster bridging? FARs of existing candidates cannot change (per-trigger vs the same background bank).
# Outputs to NEW paths: search_out_o4b_coh_zl3 (detections.json = max:-2.05 primary, detections_max3.json). Nothing else touched.
# Usage: DRY=1 launchers/m2_o4b_coh_zl3.sh (prints) | DRY=0 launchers/m2_o4b_coh_zl3.sh (submits)
set -uo pipefail
DRY="${DRY:-1}"
ROOT="${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; SC=${MADGRAV_SCRATCH}; PYV=${PY:-python}
L="$ROOT/launchers/logs_m2"; TPL="$ROOT/launchers/templates"; JIDS="$L/o4b_coh_zl3_jids.env"; mkdir -p "$L"
CACHE="$SC/madgrav2/o4b-fast_full/_seriescache"; REF="$CACHE/_cohref"
[ "$(ls "$SC/madgrav2/_o4b_coh_far_shards"/shard_*.npz 2>/dev/null | wc -l)" -eq 96 ] || { echo "FATAL: coh prune bank incomplete" >&2; exit 1; }
[ -f "$SC/madgrav2/lr_o4b_coh.npz" ] || { echo "FATAL: coh LR model missing" >&2; exit 1; }
# EXP = verbatim the coh chain's export (o4b_coh_jids.env, 2026-09-19 09:09) + SM_OUT redirected + the two test switches
EXP="ALL,SM_RUN=o4b,MADGRAV_ROOT=$ROOT,SCRATCH=$SC,PYTHON=$PYV"
EXP="$EXP,SM_STREAMS=$SC/madgrav2/streams_o4b_full,SM_PREP=${MADGRAV_ROOT}/data/o4b_search_prep"
EXP="$EXP,SM_SERIESCACHE=$CACHE,SM_COH_REWHITEN=1,SM_COHREF_DIR=$REF"
EXP="$EXP,SM_ARM_V2=1,SM_ARM_V2_DIR=$SC/madgrav2/arm/models,SM_ARM_V2_TAG=_def"
EXP="$EXP,SM_OUT=$SC/madgrav2/search_out_o4b_coh_zl3,SM_SHARD_DIR=$SC/madgrav2/_o4b_coh_far_shards"
EXP="$EXP,SM_LR_INJ_GLOB=$SC/madgrav2/inj_o4b_coh/*/*_inj.npz,SM_LR_OUT=$SC/madgrav2/lr_o4b_coh.npz,SM_LR_MODEL=$SC/madgrav2/lr_o4b_coh.npz"
EXP="$EXP,SM_CAND_FLOOR=3.0,SM_NETSIG_FLOOR=3.0,SM_NET_CUT=3.0,SM_N_CNN_CANDIDATES=1000000,SM_FG_MAXNET=1000000,SM_BLIND_FLOOR=1,SM_SURV_ABORT=60000000,SM_NSHARD=96"
EXP="$EXP,SM_ZL_NET_CUT=3.0,SM_ZL_GATES=1"
SB=(--partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --requeue)
submit(){ local name=$1; shift; if [ "$DRY" != 0 ]; then echo "  sbatch --job-name=$name $*" | sed "s|$EXP|\$EXP|" >&2; echo "DRY_$name"; else sbatch --parsable --job-name="$name" "$@"; fi; }
log(){ echo "$*"; [ "$DRY" = 0 ] && echo "$*" >> "$JIDS"; }
[ "$DRY" = 0 ] && { echo "# $(date '+%F %T') zl3 test from $ROOT ($(git -C "$ROOT" log --oneline -1))" >> "$JIDS"; echo "EXP=$EXP" >> "$JIDS"; }
echo "[zl3] DRY=$DRY root=$ROOT"; echo "[zl3] EXP=$EXP"
J_CNN=$(submit m2-o4b-zl3-cnn-shard "${SB[@]}" --output="$L/o4b_zl3_cnnshard_%A_%a.log" --array=0-95%16 --export="$EXP" "$TPL/tmpl_cnn_shard.sbatch"); log "CNN=$J_CNN"
J_MRG=$(submit m2-o4b-zl3-far-merge "${SB[@]}" --output="$L/o4b_zl3_far_merge_max_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_ARM_RULE=max:-2.05" "$TPL/tmpl_far_merge.sbatch"); log "MERGE=$J_MRG"
J_M3=$(submit m2-o4b-zl3-merge-max3 "${SB[@]}" --output="$L/o4b_zl3_far_merge_max3_%j.log" --dependency=afterany:$J_CNN --export="$EXP,SM_MERGE_TAG=_max3,SM_ARM_RULE=max:-3.0" "$TPL/tmpl_far_merge.sbatch"); log "MERGES_OTHER= max3=$J_M3"
echo "[zl3] done (DRY=$DRY). Compare: $SC/madgrav2/search_out_o4b_coh_zl3/detections.json vs coh $SC/madgrav2/search_out_o4b_coh_far/detections.json"

#!/usr/bin/env bash
# Random-injection campaign (p_astro p_s ingredient) — inject.py over 12 noise segments, mass-stratified.
# Reads RAW strain, whitens on the fly with the run reference ASD -> does NOT depend on streams/cache/FAR,
# so runs IN PARALLEL with the FAR stage. GATED on data/<run>_search_prep AND <run>_events_inj.json.
# array injected at submit (0-11%6).
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
export SM_DEV=cuda:0
export SM_INJ="$SM/inj_out_${SM_RUN}"
export SM_EVENTSJSON="$SM/${SM_RUN}_events_inj.json"
export SM_SEGJSON_EV="$SM/${SM_RUN}_seg_event_inj.json"
export SM_BANK_UM="$MADGRAV_ROOT/data/ultramassive_bank"
export SM_BANK_SIG="$MADGRAV_ROOT/data/o1_o3_signal_bank_projected_2s_x10"
mkdir -p "$SM_INJ"
if [ ! -e "$SM_PREP" ]; then echo "[$SM_RUN-inject] SM_PREP $SM_PREP missing -> abort"; exit 1; fi
if [ ! -f "$SM_EVENTSJSON" ]; then echo "[$SM_RUN-inject] $SM_EVENTSJSON missing -> abort"; exit 1; fi
NAME=$("$PY" -c "import json;print(list(json.load(open('$SM_EVENTSJSON')))[${SLURM_ARRAY_TASK_ID}])")
echo "[$SM_RUN-inject] task ${SLURM_ARRAY_TASK_ID} seg=$NAME host=$(hostname) start=$(date '+%F %T')"
t0=$SECONDS
"$PY" "$SM/inject.py" --event "$NAME"; rc=$?
echo "[$SM_RUN-inject] $NAME rc=$rc WALL=$((SECONDS-t0))s end=$(date '+%F %T')"
exit $rc

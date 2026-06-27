#!/usr/bin/env bash
# MADGRAV merge -- portable (NO systemd / NO RAM-watchdog wrapper).
# Merges the SM_NSHARD shard_*.npz into blindscan.json/detections.json/survivors_bg.json.
# Science env BYTE-IDENTICAL to run_o3a_56_merge_capped.sh (floor 4.5 / net-sigma 4.0 / per-arm / COH_CEIL 0.85).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MADGRAV_ROOT="${MADGRAV_ROOT:-$(cd "$HERE/.." && pwd)}"
cd "$MADGRAV_ROOT"
PY="${MADGRAV_PY:-python}"
SM=search_mode

OUT="${SM_OUT:-$SM/search_out_o3a_56_perarm}"
SHARD_DIR="${SM_SHARD_DIR:-$SM/_o3a_56_perarm_shards}"
FAST="${MADGRAV_FAST:-$SM/o3a-fast}"
N="${SM_NSHARD:-16}"
DEV="${BLIND_DEV:-${BLIND_GPU:-cuda:1}}"
mkdir -p "$OUT"

# ---- O3a 56-seg data env (MUST match the shards' config: build_setup re-reads it) ----
export SM_PREP="${SM_PREP:-$MADGRAV_ROOT/data/o3a_search_prep}"
export SM_STRAIN="${SM_STRAIN:-$SM/strain_o3a}"
export SM_STREAMS="${SM_STREAMS:-$SM/streams_o3a}"
export SM_INJ="${SM_INJ:-$SM/inj_out_o3a}"
export SM_SERIESCACHE="${SM_SERIESCACHE:-$FAST/_seriescache}"
export SM_BGJSON="${SM_BGJSON:-$SM/o3a_bg_segments_56.json}"
export SM_SEGJSON_EV="${SM_SEGJSON_EV:-$SM/o3a_segments_event.json}"
export SM_EVENTSJSON="${SM_EVENTSJSON:-$SM/o3a_events.json}"
export SM_VETO="${SM_VETO:-$SM/veto_mask_o3a_56.json}"
export SM_OUT="$OUT"

# ---- science knobs: BYTE-IDENTICAL ----
export SM_CAND_FLOOR="${SM_CAND_FLOOR:-4.5}"
export SM_NETSIG_FLOOR="${SM_NETSIG_FLOOR:-4.0}"
export SM_PERARM="${SM_PERARM:-1}"
export SM_BLIND_FLOOR="${SM_BLIND_FLOOR:-1}"
export DET_FAR="${DET_FAR:-1.0}"
export SM_COH_CEIL="${SM_COH_CEIL:-0.85}"
export SM_NSHARD="$N"
export SM_SHARD_DIR="$SHARD_DIR"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export SM_PRECOMPUTE_MAXGB="${SM_PRECOMPUTE_MAXGB:-120}"   # bounded CNN precompute
export SM_MERGE_OK="${SM_MERGE_OK:-1}"                     # lift the pre-registration HOLD

echo "[madgrav-merge] START $(date '+%F %T') on $DEV  cap=${SM_PRECOMPUTE_MAXGB}GB -> $OUT"
BLIND_DEV="$DEV" "$PY" "$SM/driver_blindscan_merge.py"
rc=$?
echo "[madgrav-merge] END rc=$rc $(date '+%F %T')"
ls -la "$OUT"
exit $rc

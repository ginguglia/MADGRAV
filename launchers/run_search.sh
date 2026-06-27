#!/usr/bin/env bash
# MADGRAV blind GW search -- portable single-node runner (NO systemd dependency).
# Runs SM_NSHARD sequential shards of driver_blindscan.py, then the merge.
# Science knobs are BYTE-IDENTICAL to the accepted O3a 56-seg per-arm run
# (run_o3a_56_perarm_1gpu.sh): floor 4.5 / net-sigma 4.0 / per-arm / blind-floor / COH_CEIL 0.85.
# Resumable: a shard whose shard_${k}.npz exists is skipped.
set -uo pipefail

# ---- package root (one root for everything) ----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MADGRAV_ROOT="${MADGRAV_ROOT:-$(cd "$HERE/.." && pwd)}"
cd "$MADGRAV_ROOT"
PY="${MADGRAV_PY:-python}"
SM=search_mode

# ---- output / shard dirs ----
OUT="${SM_OUT:-$SM/search_out_o3a_56_perarm}"
SHARD_DIR="${SM_SHARD_DIR:-$SM/_o3a_56_perarm_shards}"
FAST="${MADGRAV_FAST:-$SM/o3a-fast}"
N="${SM_NSHARD:-16}"
DEV="${BLIND_DEV:-${BLIND_GPU:-cuda:1}}"
mkdir -p "$OUT" "$SHARD_DIR" "$FAST"

# ---- O3a 56-seg data env (MADGRAV_ROOT-relative) ----
export SM_PREP="${SM_PREP:-$MADGRAV_ROOT/data/o3a_search_prep}"
export SM_STRAIN="${SM_STRAIN:-$SM/strain_o3a}"
export SM_STREAMS="${SM_STREAMS:-$SM/streams_o3a}"
export SM_INJ="${SM_INJ:-$SM/inj_out_o3a}"
export SM_SERIESCACHE="${SM_SERIESCACHE:-$FAST/_seriescache}"   # reclaimable page-cache (~130G, built once)
export SM_BGJSON="${SM_BGJSON:-$SM/o3a_bg_segments_56.json}"
export SM_SEGJSON_EV="${SM_SEGJSON_EV:-$SM/o3a_segments_event.json}"
export SM_EVENTSJSON="${SM_EVENTSJSON:-$SM/o3a_events.json}"
export SM_VETO="${SM_VETO:-$SM/veto_mask_o3a_56.json}"
export SM_OUT="$OUT"

# ---- science knobs: BYTE-IDENTICAL to the accepted O3a run -- DO NOT CHANGE ----
export SM_CAND_FLOOR="${SM_CAND_FLOOR:-4.5}"
export SM_NETSIG_FLOOR="${SM_NETSIG_FLOOR:-4.0}"
export SM_PERARM="${SM_PERARM:-1}"
export SM_BLIND_FLOOR="${SM_BLIND_FLOOR:-1}"
export DET_FAR="${DET_FAR:-1.0}"
export SM_COH_CEIL="${SM_COH_CEIL:-0.85}"
# NOTE: SM_NO_EXCISION UNSET -> excision ON; SM_MAX_OFFSETS UNSET -> full 400 offsets.

# ---- host-memory budget (ONE knob; do NOT set SM_BLOCK_SURV -- let _budget derive it) ----
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export SM_HOST_MEM_GB="${SM_HOST_MEM_GB:-120}"
export SM_NSHARD="$N"
export SM_SHARD_DIR="$SHARD_DIR"

echo "[madgrav] MADGRAV_ROOT=$MADGRAV_ROOT"
echo "[madgrav] $N sequential shards on $DEV -> $SHARD_DIR  (floor $SM_CAND_FLOOR / per-arm / COH_CEIL $SM_COH_CEIL)"
for k in $(seq 0 $((N-1))); do
  if [ -f "$SHARD_DIR/shard_${k}.npz" ]; then
    echo "[madgrav] shard $k already done -> skip"; continue
  fi
  echo "[madgrav] === shard $k/$N START $(date '+%H:%M:%S') ==="
  SM_SHARD=$k BLIND_DEV="$DEV" "$PY" "$SM/driver_blindscan.py" > "$SHARD_DIR/shard_${k}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$SHARD_DIR/shard_${k}.npz" ]; then
    echo "[madgrav] shard $k FAILED (rc=$rc) -- see $SHARD_DIR/shard_${k}.log ; fix, then re-run this script to resume"; exit 1
  fi
  echo "[madgrav] shard $k DONE $(date '+%H:%M:%S')"
done

echo "[madgrav] all $N shards done -> merge"
SM_NSHARD=$N BLIND_DEV="$DEV" bash "$HERE/run_merge.sh"
echo "[madgrav] merge complete -> $OUT (blindscan.json / detections.json / survivors_bg.json)"

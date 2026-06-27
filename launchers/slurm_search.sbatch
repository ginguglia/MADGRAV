#!/usr/bin/env bash
# MADGRAV blind GW search -- SLURM ARRAY job: ONE shard per array task.
#   SM_SHARD   = $SLURM_ARRAY_TASK_ID
#   SM_NSHARD  = array size (keep --array max in sync; 0-15 -> 16 shards)
#
# Submit the search array, then the dependent merge:
#   jid=$(sbatch --parsable launchers/slurm_search.sbatch)
#   sbatch --dependency=afterok:${jid} launchers/run_merge.sh
# (run_merge.sh is plain bash; wrap it in your own one-task #SBATCH header, or
#  submit:  sbatch --dependency=afterok:${jid} --wrap="bash launchers/run_merge.sh")
#
# ---- EDIT THESE PLACEHOLDERS for your cluster ----
#SBATCH --job-name=madgrav-search
#SBATCH --array=0-15                 # 16 shards; must equal SM_NSHARD below
#SBATCH --partition=PARTITION        # <-- set
#SBATCH --account=ACCOUNT            # <-- set
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm_madgrav_%A_%a.log
set -uo pipefail

# ---- package root: assumes submission from the package root (sbatch CWD) ----
export MADGRAV_ROOT="${MADGRAV_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$MADGRAV_ROOT"
PY="${MADGRAV_PY:-python}"
SM=search_mode

export SM_NSHARD="${SM_NSHARD:-16}"           # MUST equal the --array span size
export SM_SHARD="${SLURM_ARRAY_TASK_ID}"
SHARD_DIR="${SM_SHARD_DIR:-$SM/_o3a_56_perarm_shards}"
export SM_SHARD_DIR="$SHARD_DIR"
FAST="${MADGRAV_FAST:-$SM/o3a-fast}"
DEV="${BLIND_DEV:-cuda:0}"                     # one GPU per task; cuda:0 within the task's allocation
mkdir -p "$SHARD_DIR" "$FAST"

if [ -f "$SHARD_DIR/shard_${SM_SHARD}.npz" ]; then
  echo "[madgrav-slurm] shard $SM_SHARD already done -> skip"; exit 0
fi

# ---- O3a 56-seg data env (MADGRAV_ROOT-relative) ----
export SM_PREP="${SM_PREP:-$MADGRAV_ROOT/data/o3a_search_prep}"
export SM_STRAIN="${SM_STRAIN:-$SM/strain_o3a}"
export SM_STREAMS="${SM_STREAMS:-$SM/streams_o3a}"
export SM_INJ="${SM_INJ:-$SM/inj_out_o3a}"
export SM_SERIESCACHE="${SM_SERIESCACHE:-$FAST/_seriescache}"
export SM_BGJSON="${SM_BGJSON:-$SM/o3a_bg_segments_56.json}"
export SM_SEGJSON_EV="${SM_SEGJSON_EV:-$SM/o3a_segments_event.json}"
export SM_EVENTSJSON="${SM_EVENTSJSON:-$SM/o3a_events.json}"
export SM_VETO="${SM_VETO:-$SM/veto_mask_o3a_56.json}"
export SM_OUT="${SM_OUT:-$SM/search_out_o3a_56_perarm}"
mkdir -p "$SM_OUT"

# ---- science knobs: BYTE-IDENTICAL to the accepted O3a run -- DO NOT CHANGE ----
export SM_CAND_FLOOR="${SM_CAND_FLOOR:-4.5}"
export SM_NETSIG_FLOOR="${SM_NETSIG_FLOOR:-4.0}"
export SM_PERARM="${SM_PERARM:-1}"
export SM_BLIND_FLOOR="${SM_BLIND_FLOOR:-1}"
export DET_FAR="${DET_FAR:-1.0}"
export SM_COH_CEIL="${SM_COH_CEIL:-0.85}"

# ---- host-memory budget (ONE knob; do NOT set SM_BLOCK_SURV) ----
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export SM_HOST_MEM_GB="${SM_HOST_MEM_GB:-120}"

echo "[madgrav-slurm] shard $SM_SHARD/$SM_NSHARD on $DEV (floor $SM_CAND_FLOOR / per-arm / COH_CEIL $SM_COH_CEIL)"
SM_SHARD="$SM_SHARD" BLIND_DEV="$DEV" "$PY" "$SM/driver_blindscan.py"
rc=$?
if [ $rc -ne 0 ] || [ ! -f "$SHARD_DIR/shard_${SM_SHARD}.npz" ]; then
  echo "[madgrav-slurm] shard $SM_SHARD FAILED (rc=$rc)"; exit 1
fi
echo "[madgrav-slurm] shard $SM_SHARD DONE"

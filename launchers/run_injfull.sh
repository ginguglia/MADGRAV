#!/usr/bin/env bash
# SINGLE UNIFIED CAMPAIGN inj_full: one campaign covering source-frame Mtot 10-400 and
# network SNR 5-80, replacing the inj_fixed2 + inj_hi patchwork with one tag and one
# directory. Three mass strata at 1/3 each (low-mass / stellar / ultramassive) with
# SM_INJ_NPER=450, so each stratum gets ~150 per SNR level per segment -- the same
# per-stratum density as the adopted 2-stratum campaign, not a dilution of it.
#
# The injected SNR label is the REAL full-signal network SNR (SM_INJ_SNR_FULL=1), not the
# SNR retained by the stored 2 s crop. Whatever the 1 s analysis tile then fails to recover
# shows up as reduced recovery rather than being absorbed into the axis (design decision).
# At Mtot 10-22 the crop holds ~90% of the signal; above 20 the two agree to 0.3%.
#
# Pre-flight checks passed 2026-09-04 on a 2-level, 40-per-level smoke run:
#   strata 30/34/36%; is_um never set on a low-mass draw; no NaNs; all 24 columns present;
#   simulated relabel G3: 56 primary + 24 lm-fallback, 0 unmatched -> would PASS.
#SBATCH --job-name=injfull
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=launchers/injfull_%A_%a.log
set -uo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
cd "$MADGRAV_ROOT"
PY=${PY:-python}
MAN=launchers/injfull_manifest.txt

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
IFS='|' read -r NAME EVJ SGJ STRAIN PREP GRID OUT REF <<< "$LINE"
mkdir -p "$OUT"

export SM_EVENTSJSON="$EVJ" SM_SEGJSON_EV="$SGJ" SM_STRAIN="$STRAIN" SM_PREP="$MADGRAV_ROOT/$PREP"
export SM_SNR_GRID="$GRID" SM_INJ="$OUT"
export SM_INJ_CNN=1 SM_INJ_ASDVETO=1 SM_INJ_NORM_FIXED=1 SM_VERIFY_DRAWS_ONLY=1 SM_ASD_TAG=fixed2
export SM_INJ_NPER=450
export SM_DEV=cuda:0
export SM_ASD_HALF=64.0
export SM_BANK_UM="$MADGRAV_ROOT/data/ultramassive_bank"
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
export SM_LM_FRAC=0.3333 SM_UM_FRAC=0.5 SM_INJ_SNR_FULL=1

echo "[injfull] task ${SLURM_ARRAY_TASK_ID} name=$NAME grid=$GRID nper=$SM_INJ_NPER out=$OUT host=$(hostname) start=$(date '+%F %T')"
T0=$SECONDS
$PY search_mode/inject.py --event "$NAME"
RC=$?
echo "[injfull] $NAME rc=$RC WALL=$((SECONDS-T0))s end=$(date '+%F %T')"
exit $RC

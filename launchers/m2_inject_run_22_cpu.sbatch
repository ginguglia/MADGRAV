#!/usr/bin/env bash
# 2026-10-04: MADGRAV 2.2 injection-efficiency campaign (fresh seed, June CNN + S3 scored per injection), CPU partition.
# Copy of m2_inject_run_off5_cpu.sbatch with: INJ_TAG fixed to _22 (override INJ22_TAG), SM_INJ_CNN=1 SM_INJ_S3=1,
# SM_INJ_SEED=20261004 (independent of the LR-calibration draw 20260614), BLIND_DEV=cpu SM_ALLOW_CPU=1 (driver_blindscan on CPU),
# SM_INJ_SAVE_SERIES=0 (the coherence series are not needed for the efficiency), manifest selectable (INJ22_MANIFEST) for the
# low-SNR tail. Arm = _off5 (the 2.2 recipe). Everything else identical to the _off5 injection stage.
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --job-name=m2-inject-22
#SBATCH --cpus-per-task=16
#SBATCH --output=${MADGRAV_ROOT}/m2/vt22/logs/inject22_%A_%a.log
set -uo pipefail
export MADGRAV_ROOT="${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$MADGRAV_ROOT"
PY=${PY:-python}; SC=${MADGRAV_SCRATCH}
: "${SM_RUN:?}"; MAN=${INJ22_MANIFEST:-launchers/logs_m2/injfull_manifest_${SM_RUN}.txt}; LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
[ -n "$LINE" ] || { echo "[inj22] no manifest line $((SLURM_ARRAY_TASK_ID+1)) in $MAN"; exit 1; }
IFS='|' read -r NAME EVJ SGJ STRAIN PREP GRID OUTDIR REF <<< "$LINE"
TAG=${INJ22_TAG:-_22}; OUT=$SC/madgrav2/inj_${SM_RUN}_m2${TAG}/$(basename "$OUTDIR"); mkdir -p "$OUT"
if [ -s "$OUT/${NAME}_inj.npz" ]; then echo "[inj22] $NAME already done"; exit 0; fi
export SM_EVENTSJSON="$EVJ" SM_SEGJSON_EV="$SGJ" SM_STRAIN="$STRAIN" SM_PREP="$MADGRAV_ROOT/$PREP"
export SM_SNR_GRID="$GRID" SM_INJ="$OUT"
export SM_INJ_CNN=1 SM_INJ_S3=1 SM_INJ_SEED=${SM_INJ_SEED:-20261004} BLIND_DEV=cpu SM_ALLOW_CPU=1
export SM_INJ_NORM_FIXED=1 SM_ASD_TAG=fixed2 SM_INJ_NPER=${SM_INJ_NPER:-450} SM_DEV=cpu SM_QT_WORKERS=14 OMP_NUM_THREADS=16 SM_ASD_HALF=64.0
unset SM_INJ_ASDVETO
export SM_BANK_UM="$MADGRAV_ROOT/data/ultramassive_bank"
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
export SM_LM_FRAC=0.3333 SM_UM_FRAC=0.5 SM_INJ_SNR_FULL=1
export SM_ARM_V2=1 SM_ARM_V2_DIR=$SC/madgrav2/arm/models SM_ARM_V2_TAG=${SM_ARM_V2_TAG:-_off5}
export SM_INJ_SAVE_SERIES=0
export SM_COH_REWHITEN=1 SM_SERIESCACHE="${SM_SERIESCACHE:-$SC/madgrav2/${SM_RUN}-fast_full/_seriescache}" SM_COHREF_DIR="${SM_COHREF_DIR:-$SC/madgrav2/${SM_RUN}-fast_full/_seriescache/_cohref}"
echo "[inj22] task ${SLURM_ARRAY_TASK_ID} name=$NAME grid=$GRID nper=$SM_INJ_NPER seed=$SM_INJ_SEED out=$OUT host=$(hostname) start=$(date '+%F %T')"
T0=$SECONDS; $PY search_mode/inject.py --event "$NAME"; RC=$?
echo "[inj22] $NAME rc=$RC WALL=$((SECONDS-T0))s end=$(date '+%F %T')"; exit $RC

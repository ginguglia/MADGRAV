#!/usr/bin/env bash
#SBATCH --account=${SLURM_ACCOUNT} --partition=${PARTITION_CPU} --qos=${QOS_CPU}
#SBATCH --cpus-per-task=16 --mem=64G --time=06:00:00 --requeue
#SBATCH --job-name=m2-o4b-inject-cnn
#SBATCH --output=${MADGRAV_ROOT}/launchers/logs_m2/inject_cnn_%A_%a.log
# 2026-09-20 weak-arm CNN rule study (request): re-inject the O4b campaign banks with SM_INJ_CNN=1 so every injection carries the
# HM/LM CNN arm scores at its peak window (cnn_hm, cnn_lm). CPU partition (SM_ALLOW_CPU=1: CNN forward on CPU, non-calibration).
# Output goes to inj_o4b_m2_cnn/ -- the campaign banks inj_o4b_m2/ are NOT touched. Otherwise identical to m2_inject_run.sbatch.
set -uo pipefail
export MADGRAV_ROOT="${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$MADGRAV_ROOT"
PY=${PY:-python}; SC=${MADGRAV_SCRATCH}; SM_RUN=o4b
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" launchers/logs_m2/injfull_manifest_${SM_RUN}.txt)
IFS='|' read -r NAME EVJ SGJ STRAIN PREP GRID OUTDIR REF <<< "$LINE"
OUT=$SC/madgrav2/inj_${SM_RUN}_m2_cnn/$(basename "$OUTDIR"); mkdir -p "$OUT"
if [ -s "$OUT/${NAME}_inj.npz" ]; then echo "[m2-inj-cnn] $NAME already done"; exit 0; fi
export SM_RUN SM_EVENTSJSON="$EVJ" SM_SEGJSON_EV="$SGJ" SM_STRAIN="$STRAIN" SM_PREP="$MADGRAV_ROOT/$PREP"
export SM_SNR_GRID="$GRID" SM_INJ="$OUT"
export SM_INJ_CNN=1 SM_INJ_NORM_FIXED=1 SM_ASD_TAG=fixed2 SM_INJ_NPER=450 SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 SM_ASD_HALF=64.0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
unset SM_INJ_ASDVETO
export SM_BANK_UM="$MADGRAV_ROOT/data/ultramassive_bank"
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
export SM_LM_FRAC=0.3333 SM_UM_FRAC=0.5 SM_INJ_SNR_FULL=1
export SM_ARM_V2=1 SM_ARM_V2_DIR=$SC/madgrav2/arm/models SM_ARM_V2_TAG=_def
export SM_INJ_SAVE_SERIES=0
export SM_COH_REWHITEN=1 SM_SERIESCACHE="$SC/madgrav2/${SM_RUN}-fast_full/_seriescache" SM_COHREF_DIR="$SC/madgrav2/${SM_RUN}-fast_full/_seriescache/_cohref"
echo "[m2-inj-cnn] task ${SLURM_ARRAY_TASK_ID} name=$NAME grid=$GRID out=$OUT host=$(hostname) start=$(date '+%F %T')"
T0=$SECONDS; $PY search_mode/inject.py --event "$NAME"; RC=$?
echo "[m2-inj-cnn] $NAME rc=$RC WALL=$((SECONDS-T0))s end=$(date '+%F %T')"; exit $RC

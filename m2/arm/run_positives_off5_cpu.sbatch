#!/usr/bin/env bash
# 2.1 arm retrain, step 1 (2026-09-28): placement-robust positives (merger anywhere within +-0.5 s of the tile centre, M2_OFF=0.5, the arm-22
# round-3 recipe) under the FROZEN whitening of run SM_RUN, generated on the CPU partition (QT-bound; CAE sigma on CPU). Env: SM_RUN, M2_N,
# M2_TAG, M2_OUT, optional M2_SNR_LO/HI (loud set), M2_SEED.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=32 --mem=96G --time=08:00:00
#SBATCH --job-name=m2-arm-pos-off5 --output=${MADGRAV_ROOT}/m2/prep/logs/arm_pos_off5_%x_%j.log
set -uo pipefail
export MADGRAV_ROOT=${MADGRAV_SCRATCH}/MADGRAV1 SCRATCH=${MADGRAV_SCRATCH} PYTHON=${PY:-python}
export SM_RUN=${SM_RUN:?}; export SM_PREP=$MADGRAV_ROOT/data/${SM_RUN}_search_prep
source $MADGRAV_ROOT/launchers/templates/_common.sh; source $MADGRAV_ROOT/launchers/templates/_science.sh
export M2_OFF=0.5 M2_N=${M2_N:-20000} M2_TAG=${M2_TAG:-_off5} M2_OUT=${M2_OUT:-${MADGRAV_SCRATCH}/madgrav2/arm/positives_off5} SM_QT_WORKERS=30 OMP_NUM_THREADS=4
echo "[m2-pos-off5] START $(date '+%F %T') node=$(hostname) run=$SM_RUN N=$M2_N tag=$M2_TAG off=$M2_OFF snr=${M2_SNR_LO:-5}-${M2_SNR_HI:-30} prep=$SM_PREP out=$M2_OUT"; T0=$SECONDS
BLIND_DEV=cpu SM_ALLOW_CPU=1 $PY ${MADGRAV_ROOT}/m2/positives/gen_positives_m2.py; rc=$?
echo "[m2-pos-off5] END rc=$rc WALL=$((SECONDS-T0))s $(date '+%F %T')"; exit $rc

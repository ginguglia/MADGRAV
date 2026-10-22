#!/usr/bin/env bash
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=m2-o4b-lr
#SBATCH --output=${MADGRAV_ROOT}/launchers/logs_m2/o4b_lr_%j.log
set -uo pipefail
: "${MADGRAV_ROOT:?}"; source "$MADGRAV_ROOT/launchers/templates/_common.sh"; source "$MADGRAV_ROOT/launchers/templates/_science.sh"
export SM_DEV=cpu OMP_NUM_THREADS=8
echo "[m2-lr] $SM_RUN streams=$SM_STREAMS series=$SM_SERIESCACHE start=$(date '+%F %T')"; t0=$SECONDS
"$PY" "$SM/refit_lr_m2.py"; rc=$?
echo "[m2-lr] rc=$rc WALL=$((SECONDS-t0))s end=$(date '+%F %T')"; exit $rc

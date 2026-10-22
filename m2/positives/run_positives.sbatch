#!/bin/bash
#SBATCH -J m2-pos
#SBATCH -p ${PARTITION_GPU}
#SBATCH --qos=${QOS_GPU}
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -t 03:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/positives/run_pos_%j.log
export MADGRAV_ROOT=${MADGRAV_SCRATCH}/MADGRAV1          # MADGRAV1: data, model, prep, strain (read-only)
export SCRATCH=${MADGRAV_SCRATCH} PYTHON=${PY:-python}
export SM_RUN=${SM_RUN:-o3a}; export SM_PREP=$MADGRAV_ROOT/data/${SM_RUN}_search_prep
source $MADGRAV_ROOT/launchers/templates/_common.sh; source $MADGRAV_ROOT/launchers/templates/_science.sh
echo "[m2-pos] START $(date '+%F %T') node=$(hostname) run=$SM_RUN N=${M2_N:-5000}"
BLIND_DEV=cuda:0 $PY ${MADGRAV_ROOT}/m2/positives/gen_positives_m2.py; rc=$?
echo "[m2-pos] END rc=$rc $(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=16 --mem=64G --time=03:00:00
#SBATCH --job-name=m2-arm-evcheck-off5 --output=${MADGRAV_ROOT}/m2/prep/logs/arm_evcheck_off5_%j.log
export OMP_NUM_THREADS=16 SM_QT_WORKERS=14 SM_ALLOW_CPU=1 BLIND_DEV=cpu
cd ${MADGRAV_ROOT}/m2/arm && ${PY:-python} event_check_off5.py; echo "END rc=$? $(date '+%F %T')"

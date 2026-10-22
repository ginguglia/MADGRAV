#!/usr/bin/env bash
#SBATCH --account=${SLURM_ACCOUNT} --partition=${PARTITION_CPU} --qos=${QOS_CPU} --cpus-per-task=16 --mem=48G --time=00:40:00 --job-name=m2-rank-sideeval
#SBATCH --output=${MADGRAV_ROOT}/m2/rank/logs/sideeval_%j.log
export OMP_NUM_THREADS=16; cd ${MADGRAV_ROOT}/m2/rank; ${PY:-python} -W ignore eval_sideband.py

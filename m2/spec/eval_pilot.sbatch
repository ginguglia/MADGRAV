#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=16G --time=00:20:00 --job-name=m2-spec-eval
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/eval_%x_%j.log
cd ${MADGRAV_ROOT}/m2/spec; ${PY:-python} -W ignore eval_pilot.py

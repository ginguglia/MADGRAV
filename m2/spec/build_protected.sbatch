#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=120G --time=01:00:00 --job-name=m2-spec-protsel
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/protsel_%j.log
cd ${MADGRAV_ROOT}/m2/spec; ${PY:-python} build_protected.py

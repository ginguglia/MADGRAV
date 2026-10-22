#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=120G --time=04:00:00
#SBATCH --job-name=m2-spec-negsel --output=${MADGRAV_ROOT}/m2/spec/logs/negsel_%j.log
${PY:-python} ${MADGRAV_ROOT}/m2/spec/select_negatives.py

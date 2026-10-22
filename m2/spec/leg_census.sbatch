#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=120G --time=03:00:00
#SBATCH --job-name=m2-spec-legs --output=${MADGRAV_ROOT}/m2/spec/logs/legcensus_%j.log
${PY:-python} ${MADGRAV_ROOT}/m2/spec/leg_census.py

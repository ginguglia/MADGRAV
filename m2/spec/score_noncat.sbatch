#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:1 --time=00:20:00 --job-name=m2-spec-noncatscore
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/noncatscore_%j.log
cd ${MADGRAV_ROOT}/m2/spec; ${PY:-python} -W ignore score_noncat.py

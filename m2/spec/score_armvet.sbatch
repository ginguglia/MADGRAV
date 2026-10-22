#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:1 --time=00:20:00 --job-name=m2-spec-armvetscore
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/armvetscore_%j.log
cd ${MADGRAV_ROOT}/m2/spec; ${PY:-python} -W ignore score_armvet.py

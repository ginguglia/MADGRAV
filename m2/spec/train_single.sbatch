#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:1 --cpus-per-task=16 --mem=180G
#SBATCH --time=02:00:00 --job-name=m2-spec-single
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/single_%x_%j.log
export SPEC_FOLD=${SPEC_FOLD:-0} SPEC_SEED=${SPEC_SEED:-0}
echo "[spec-train] START $(date '+%F %T') host=$(hostname) cfg=$SPEC_CFG n=$SPEC_N fold=$SPEC_FOLD seed=$SPEC_SEED smoke=${SPEC_SMOKE:-0} data=${SPEC_DATA:-default}"
cd ${MADGRAV_ROOT}/m2/spec; ${PY:-python} train_single.py; rc=$?
echo "[spec-train] END $(date '+%F %T') rc=$rc"; exit $rc

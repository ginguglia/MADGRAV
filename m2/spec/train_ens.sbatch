#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:1 --cpus-per-task=16 --mem=180G
#SBATCH --time=06:00:00 --job-name=m2-spec-ens
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/train_%x_%A_%a.log
# 2x5 production-recipe ensemble (train_spec.py, SPEC_CFG hm, SPEC_N 0 = full): array index i -> fold i%2, seed i//2. Env: SPEC_DATA, SPEC_EXTRA, SPEC_OUT.
export SPEC_CFG=${SPEC_CFG:-hm} SPEC_N=0 SPEC_FOLD=$((SLURM_ARRAY_TASK_ID % 2)) SPEC_SEED=$((SLURM_ARRAY_TASK_ID / 2))
echo "[spec-ens] START $(date '+%F %T') host=$(hostname) cfg=$SPEC_CFG fold=$SPEC_FOLD seed=$SPEC_SEED data=${SPEC_DATA:-default} extra=${SPEC_EXTRA:-none} out=${SPEC_OUT:-default}"
cd ${MADGRAV_ROOT}/m2/spec; ${PY:-python} train_spec.py; rc=$?
echo "[spec-ens] END $(date '+%F %T') rc=$rc"; exit $rc

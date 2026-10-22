#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --gres=gpu:1 --cpus-per-task=16 --mem=180G
#SBATCH --time=06:00:00 --job-name=m2-spec-train
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/train_%x_%A_%a.log
# PILOT array (prereg §7.3 learning curve, fold 0, seed 0): index i -> cfg = (hm,lm,s2)[i%3], N = (25k,50k,100k,full)[i//3]
# Override with SPEC_CFG/SPEC_N/SPEC_FOLD/SPEC_SEED in the env for single runs.
CFGS=(hm lm s2); NS=(25000 50000 100000 0)
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then export SPEC_CFG=${CFGS[$((SLURM_ARRAY_TASK_ID % 3))]} SPEC_N=${NS[$((SLURM_ARRAY_TASK_ID / 3))]}; fi
export SPEC_FOLD=${SPEC_FOLD:-0} SPEC_SEED=${SPEC_SEED:-0}
echo "[spec-train] START $(date '+%F %T') host=$(hostname) cfg=$SPEC_CFG n=$SPEC_N fold=$SPEC_FOLD seed=$SPEC_SEED smoke=${SPEC_SMOKE:-0} data=${SPEC_DATA:-default}"
cd ${MADGRAV_ROOT}/m2/spec; ${PY:-python} train_spec.py; rc=$?
echo "[spec-train] END $(date '+%F %T') rc=$rc"; exit $rc

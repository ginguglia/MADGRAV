#!/bin/bash
#SBATCH -J m2-arm
#SBATCH -p ${PARTITION_GPU}
#SBATCH --qos=${QOS_GPU}
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/arm/train_%A_%a.log
# array index i -> fold i%2, seed i//2
export ARM_FOLD=$((SLURM_ARRAY_TASK_ID % 2)) ARM_SEED=$((SLURM_ARRAY_TASK_ID / 2))
cd ${MADGRAV_ROOT}/m2/arm; echo "start $(date) fold=$ARM_FOLD seed=$ARM_SEED smoke=${ARM_SMOKE:-0}"
${PY:-python} train_arm.py; echo "end $(date) rc=$?"

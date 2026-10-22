#!/usr/bin/env bash
#SBATCH --job-name=statmask-census --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=96G --time=03:00:00
#SBATCH --array=0-3 --output=${MADGRAV_ROOT}/m2/dq/logs/%x_%A_%a.log
RUNS=(o3a o3b o4a o4b); RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
cd ${MADGRAV_ROOT}/m2/dq && echo "START $(date '+%F %T') $(hostname) $RUN" && ${PY:-python} -I stat_mask_census.py $RUN; echo "END rc=$? $(date '+%F %T')"

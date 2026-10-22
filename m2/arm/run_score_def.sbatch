#!/bin/bash
#SBATCH -J m2-arm-score-def
#SBATCH -p ${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/arm/score_def_%j.log
cd ${MADGRAV_ROOT}/m2/arm; export ARM_DATA=${MADGRAV_SCRATCH}/madgrav2/arm/data_v3
${PY:-python} score_families.py _def 2>&1 | grep -vE 'arn' | tee ${MADGRAV_SCRATCH}/madgrav2/arm/models/score_families_def.md

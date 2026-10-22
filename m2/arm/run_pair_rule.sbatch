#!/bin/bash
#SBATCH -J m2-arm-pairrule
#SBATCH -p ${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/arm/pair_rule_%j.log
cd ${MADGRAV_ROOT}/m2/arm; ${PY:-python} pair_rule.py 2>&1 | grep -vE 'arn' | tee ${MADGRAV_SCRATCH}/madgrav2/arm/models/pair_rule_def.md

#!/bin/bash
#SBATCH -J m2-arm-assemble
#SBATCH -p ${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/arm/assemble_%j.log
cd ${MADGRAV_ROOT}/m2/arm; echo "start $(date)"; ${PY:-python} assemble.py; echo "end $(date) rc=$?"

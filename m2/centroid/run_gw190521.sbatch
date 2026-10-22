#!/bin/bash
#SBATCH -J m2-gw190521
#SBATCH -p ${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/centroid/run_gw190521_%j.log
cd ${MADGRAV_ROOT}/m2/centroid
${PY:-python} gw190521_families.py GW190521
echo "end rc=$?"; ${PY:-python} gw190521_families.py GW190521_074359; echo "end2 rc=$?"

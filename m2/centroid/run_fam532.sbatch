#!/bin/bash
#SBATCH -J m2-fam532
#SBATCH -p ${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/centroid/run_fam532_%j.log
cd ${MADGRAV_ROOT}/m2/centroid
echo "start $(date) host $(hostname)"
${PY:-python} render_families.py
echo "end $(date) rc=$?"

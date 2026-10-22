#!/bin/bash
#SBATCH -J m2-centroid
#SBATCH -p ${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -o ${MADGRAV_SCRATCH}/madgrav2/centroid/run_full_%j.log
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd ${MADGRAV_ROOT}/m2/centroid
echo "start $(date) host $(hostname)"
${PY:-python} run_pilot.py det47 top50 pos glitch murky
echo "features done $(date)"
${PY:-python} analyze.py > ${MADGRAV_SCRATCH}/madgrav2/centroid/analysis_stdout.txt 2>&1
echo "end $(date) rc=$?"

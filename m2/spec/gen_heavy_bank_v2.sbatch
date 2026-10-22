#!/usr/bin/env bash
#SBATCH --time=00:30:00 --mem=8G --cpus-per-task=1 --job-name=m2-heavy-v2
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/heavy_v2_%A_%a.log
# one chunk per array task (40-399), 250 waveforms each -> heavy_bank_v2 (90k new rows)
export OMP_NUM_THREADS=1
OUT=${MADGRAV_SCRATCH}/madgrav2/spec/heavy_bank_v2/projectedbank__mtot_150.0_800.0__qmax_4.0__fs_4096__minFreq_10.0__apx_IMRPhenomXPHM
cd ${MADGRAV_ROOT}/m2/spec
${PY:-python} gen_heavy_bank_v2.py --out "$OUT" --chunk "$SLURM_ARRAY_TASK_ID" --n 250; rc=$?
echo "[heavy-v2] END rc=$rc $(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
#SBATCH --time=00:40:00 --mem=8G --cpus-per-task=1 --job-name=m2-bank-v2 --requeue
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/bank_v2_%A_%a.log
# one chunk per array task (0-479), 500 waveforms each -> unified bank v2 (240k rows, Mtot 10-800, physical H1-L1 delay)
export OMP_NUM_THREADS=1
OUT=${MADGRAV_SCRATCH}/madgrav2_spec/bank_v2
cd ${MADGRAV_ROOT}/m2/spec
[ -s "$OUT/signals_$(printf %04d $SLURM_ARRAY_TASK_ID).csv" ] && { echo "[bank-v2] chunk exists, skip"; exit 0; }
${PY:-python} gen_bank_v2.py --out "$OUT" --chunk "$SLURM_ARRAY_TASK_ID" --n 500 2>&1 | grep -v -i warn; rc=${PIPESTATUS[0]}
echo "[bank-v2] END rc=$rc $(date '+%F %T')"; exit $rc

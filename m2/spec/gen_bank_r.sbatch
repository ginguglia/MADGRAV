#!/usr/bin/env bash
# Restricted-population banks for the CAE family test (2026-09-25, DECISIONS_WHITENING 16:10): Mtot log-flat 25-200, q uniform 1-2.5,
# |chi| uniform 0-0.3 (precessing), otherwise gen_bank_v2 as is. Four banks, disjoint seeds: training XPHM / PhenomPv2 (80 chunks x 500),
# held-out test XPHM / PhenomPv2 (12 chunks x 500). Tasks 0-79 / 80-159 / 160-171 / 172-183.
#SBATCH --time=00:40:00 --mem=8G --cpus-per-task=1 --job-name=m2-bank-r --requeue --array=0-183
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/bank_r_%A_%a.log
export OMP_NUM_THREADS=1; i=$SLURM_ARRAY_TASK_ID; ROOT=${MADGRAV_SCRATCH}/madgrav2_spec
if   [ $i -lt 80 ];  then B=bank_r_xphm;      APX=IMRPhenomXPHM; SEED=20260925; K=$i
elif [ $i -lt 160 ]; then B=bank_r_pv2;       APX=IMRPhenomPv2;  SEED=20260926; K=$((i-80))
elif [ $i -lt 172 ]; then B=bank_r_test_xphm; APX=IMRPhenomXPHM; SEED=20260927; K=$((i-160))
else                      B=bank_r_test_pv2;  APX=IMRPhenomPv2;  SEED=20260928; K=$((i-172)); fi
OUT=$ROOT/$B; cd ${MADGRAV_ROOT}/m2/spec
[ -s "$OUT/signals_$(printf %04d $K).csv" ] && { echo "[bank-r] chunk exists, skip"; exit 0; }
${PY:-python} gen_bank_v2.py --out "$OUT" --chunk "$K" --n 500 --seed $SEED --mtot-lo 25 --mtot-hi 200 --qmax 2.5 --chi-max 0.3 --apx $APX 2>&1 | grep -v -i warn; rc=${PIPESTATUS[0]}
echo "[bank-r] $B chunk $K END rc=$rc $(date '+%F %T')"; exit $rc

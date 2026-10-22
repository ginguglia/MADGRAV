#!/usr/bin/env bash
# vt96 validation gate (CPU): far96.py validate on all four runs (via vt96_stage.sbatch STAGE=validate); exits non-zero on any FAIL.
# Runs after the four export jobs (afterok). Writes logs/VALIDATION_<date>.txt with the PASS/FAIL line of every run.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=8 --mem=64G --time=01:30:00
#SBATCH --job-name=vt96-validate --output=${MADGRAV_ROOT}/m2/vt96/logs/%x_%A.log
set -uo pipefail; W=${MADGRAV_ROOT}/m2/vt96; OUTF=$W/logs/VALIDATION_$(date +%Y%m%d_%H%M).txt; rc=0
echo "vt96 validation $(date '+%F %T') host $(hostname)" > $OUTF
for r in o3a o3b o4a o4b; do
  echo "[vt96-validate-all] $r START $(date '+%F %T')"
  STAGE=validate SM_RUN_SEL=$r bash $W/vt96_stage.sbatch; rr=$?; rc=$((rc+rr))
  f=$(ls -t $W/logs/validate_${r}_*.txt | head -1); grep "^\[vt96-validate\] $r: \(PASS\|FAIL\)" $f >> $OUTF || echo "[vt96-validate] $r: FAIL (no verdict line, rc=$rr)" >> $OUTF
done
ln -sf $OUTF $W/logs/VALIDATION_latest.txt; cat $OUTF; echo "[vt96-validate-all] END rc=$rc $(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
# 2026-10-22 vt96p smoke tests on the CPU partition (never on the login node): manifest line 1 of o4b (o4b_1396417050), grid 8,15, nper 20.
#   task 0 idA : patched inject.py, SM_INJ_PLACE=grid   -> $SC/madgrav2/inj_o4b_m2_96pidA/o4b
#   task 1 idB : git-HEAD inject.py (pre-patch), via inject96_head.py -> $SC/madgrav2/inj_o4b_m2_96pidB/o4b   (A == B byte-identical is the gate)
#   task 2 prod: patched inject.py, SM_INJ_PLACE=prod   -> $SC/madgrav2/inj_o4b_m2_96psmoke/o4b
#SBATCH --account=${SLURM_ACCOUNT} --partition=${PARTITION_CPU} --qos=${QOS_CPU}
#SBATCH --cpus-per-task=16 --mem=100G --time=01:30:00 --array=0-2
#SBATCH --job-name=m2-smoke-96p --output=${MADGRAV_ROOT}/m2/vt96p/logs/smoke96p_%A_%a.log
W=${MADGRAV_ROOT}/m2/vt96p
case "$SLURM_ARRAY_TASK_ID" in
  0) export INJ96_TAG=_96pidA INJ96P_PLACE=grid ;;
  1) export INJ96_TAG=_96pidB INJ96P_PLACE=grid INJ96_ENTRY=$W/inject96_head.py ;;
  2) export INJ96_TAG=_96psmoke INJ96P_PLACE=prod ;;
esac
rm -f ${MADGRAV_SCRATCH}/madgrav2/inj_o4b_m2${INJ96_TAG}/o4b/o4b_1396417050_inj.npz
export SM_RUN=o4b SM_INJ_NPER=20 INJ96_SNR_GRID=8,15 SLURM_ARRAY_TASK_ID=0
exec bash $W/m2_inject_run_96p_cpu.sbatch

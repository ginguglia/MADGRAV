#!/usr/bin/env bash
# Source-frame relabel of the vt96 injection field (m2/vt22/vt_relabel_comoving22.py, unchanged; SM tag _m96), runs selectable via SM_RUNS.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=48G --time=02:00:00
#SBATCH --job-name=vt96-relabel --output=${MADGRAV_ROOT}/m2/vt96/logs/%x_%A.log
set -uo pipefail
PY=${PY:-python}; SC=${MADGRAV_SCRATCH}/madgrav2; W=${MADGRAV_ROOT}/m2/vt22
export SM_RELABEL_DIR=$SC/vt96 SM_VT_SUF=_m96 SM_VT_MASS_EDGES=${SM_VT_MASS_EDGES:-10,15,20,30,40,60,80,100,130,160,200,260,330,400}
export SM_BANK_LM=${MADGRAV_ROOT}/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2
for r in o3a o3b o4a o4b; do export SM_INJ_DIRS_$r=$SC/inj_${r}_m2_96/$r,$SC/inj_${r}_m2_96/${r}_lowsnr; done
export SM_RUNS=${SM_RUNS:-O3a,O3b,O4a,O4b} SM_ASD_TAG=fixed2 SM_VT_FIXED_ASD=${SM_VT_FIXED_ASD:-1}
echo "[vt96-relabel] START $(date '+%F %T') runs=$SM_RUNS dir=$SM_RELABEL_DIR"; cd $W; OMP_NUM_THREADS=4 $PY vt_relabel_comoving22.py; rc=$?
echo "[vt96-relabel] END rc=$rc $(date '+%F %T')"; ls -la $SC/vt96/relabel_inj_*_m96.npz $SC/vt96/vt_relabel_comoving_m96.json 2>/dev/null; exit $rc

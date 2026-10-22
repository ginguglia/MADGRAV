#!/usr/bin/env bash
# Background side of the local-ASD veto check (2026-10-07): re-score the K loudest CNN-passing slid pairs under local ASDs. CPU only.
# Usage: sbatch --export=ALL,RUN=o4b asdveto_bg.sbatch
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --time=08:00:00 --mem=150G --cpus-per-task=8
#SBATCH --output=${MADGRAV_ROOT}/m2/asdveto/logs/%x_%j.log
set -uo pipefail; ROOT=${MADGRAV_ROOT}; M2SC=${MADGRAV_SCRATCH}/madgrav2; : "${RUN:?}"
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${RUN}_off5_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export M2SC SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2 AV_K=${AV_K:-400} AV_OUT=$M2SC/asdveto SM_NETSIG_LM_MIN=0.0382
echo "[asdveto-bg] START $(date '+%F %T') run=$RUN strain=$SM_STRAIN prep=$SM_PREP host=$(hostname)"
cd $ROOT; "$PY" ${MADGRAV_ROOT}/m2/asdveto/asdveto_bg.py; rc=$?; echo "[asdveto-bg] END rc=$rc $(date '+%F %T')"; exit $rc

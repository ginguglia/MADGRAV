#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --time=01:30:00 --mem=60G --cpus-per-task=8
#SBATCH --output=${MADGRAV_ROOT}/m2/asdveto/logs/%x_%j.log
set -uo pipefail; ROOT=${MADGRAV_ROOT}; : "${RUN:?}"; W=${MADGRAV_ROOT}/m2/asdveto
EXP=$(grep "^EXP=" $ROOT/launchers/logs_m2/${RUN}_off5_m2_gx_jids.env | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
source $ROOT/launchers/templates/_common.sh; source $ROOT/launchers/templates/_science.sh
export SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1 OMP_NUM_THREADS=8 MALLOC_ARENA_MAX=2 SM_QT_WORKERS=1
echo "[hybrid] START $(date '+%F %T') run=$RUN strain=$SM_STRAIN prep=$SM_PREP host=$(hostname)"
cd $ROOT; "$PY" $W/check_asd_hybrid.py $RUN $W/hybrid_${RUN}.json; rc=$?; echo "[hybrid] END rc=$rc $(date '+%F %T')"; exit $rc

#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=100G --time=02:00:00
#SBATCH --job-name=popb-capcensus --output=${MADGRAV_ROOT}/m2/popb_cap/logs/census_%j.log
set -uo pipefail; mkdir -p ${MADGRAV_ROOT}/m2/popb_cap/logs; cd ${MADGRAV_ROOT}/m2/popb_cap
echo "[census] $(hostname) $(date '+%F %T')"; ${PY:-python} -I cap_census.py o4b o4a o3a o3b 2>&1 | tee CAP_CENSUS_FINE_20261008.txt; echo "[census] rc=${PIPESTATUS[0]} $(date '+%F %T')"

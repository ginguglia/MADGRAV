#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=32 --mem=64G
#SBATCH --time=04:00:00 --job-name=m2-spec-asm
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/asm_%x_%j.log
# env in (optional): SPEC_DATA (default madgrav2_spec/train), SPEC_MAXFILES (smoke)
echo "[spec-asm] START $(date '+%F %T') host=$(hostname) data=${SPEC_DATA:-default} maxfiles=${SPEC_MAXFILES:-all}"
cd ${MADGRAV_ROOT}/m2/spec; SPEC_WORKERS=16 ${PY:-python} assemble_spec.py; rc=$?
echo "[spec-asm] END $(date '+%F %T') rc=$rc"; exit $rc

#!/usr/bin/env bash
# Generate the LOW-MASS projected signal bank (source-frame Mtot 10-22), 10 shards x 1024
# sources, so the injection campaign can cover Mtot 10-400 instead of 20-400.
# Conventions reproduced from the existing stellar bank and validated 2026-09-04:
#   antenna factors exact (max|diff| 0.0000), npz/CSV row alignment correct,
#   waveform overlap with regenerated existing entries 0.81-0.996.
# NOT reproduced: the existing bank's per-source absolute amplitude (ratio 0.72-2.62 at
# high overlap). Immaterial for injections -- inject.py renormalises every injection to a
# target SNR -- but it means this bank must not be mixed with the existing one for VT,
# whose horizons are built from the bank amplitude/distance relation.
#SBATCH --job-name=lmbank
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --qos=<your-qos>
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=launchers/lmbank_%A_%a.log
set -uo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
cd "$MADGRAV_ROOT"
OUT=$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2
mkdir -p "$OUT"
echo "[lmbank] shard ${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date '+%F %T')"
${WFGEN_PY:-python} tools/make_lowmass_bank.py \
  --out "$OUT" --n 1024 --shard 1024 --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --m1-lo 5 --m1-hi 19 --m2-min 3 --mtot-lo 10 --mtot-hi 22 --seed 20260904
RC=$?
echo "[lmbank] shard ${SLURM_ARRAY_TASK_ID} rc=$RC end=$(date '+%F %T')"
exit $RC

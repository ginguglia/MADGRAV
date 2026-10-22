#!/usr/bin/env bash
# Local-ASD veto on injections (2026-10-07): the 2.2 injection field (seed 20261004, same manifest, nper 450, CNN + S3) re-drawn
# byte-identically with SM_INJ_ASDVETO=1 SM_INJ_LOCAL_ALL=1 -> every injection carries net_loc / hm_loc / lm_loc / veto_keep / s3_loc.
# Output: madgrav2/inj_<run>_m2_22veto/. CPU partition only. Usage: sbatch --array=0-$((N-1)) --export=ALL,SM_RUN=<run> inject_veto.sbatch
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --time=10:00:00 --mem=60G --cpus-per-task=16
#SBATCH --output=${MADGRAV_ROOT}/m2/asdveto/logs/%x_%A_%a.log
set -uo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT}; cd "$MADGRAV_ROOT"
PY=${PY:-python}; SC=${MADGRAV_SCRATCH}
: "${SM_RUN:?}"; MAN=launchers/logs_m2/injfull_manifest_${SM_RUN}.txt; LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
[ -n "$LINE" ] || { echo "[injveto] no manifest line $((SLURM_ARRAY_TASK_ID+1)) in $MAN"; exit 1; }
IFS='|' read -r NAME EVJ SGJ STRAIN PREP GRID OUTDIR REF <<< "$LINE"
OUT=$SC/madgrav2/inj_${SM_RUN}_m2_22veto/$(basename "$OUTDIR"); mkdir -p "$OUT"
if [ -s "$OUT/${NAME}_inj.npz" ]; then echo "[injveto] $NAME already done"; exit 0; fi
export SM_EVENTSJSON="$EVJ" SM_SEGJSON_EV="$SGJ" SM_STRAIN="$STRAIN" SM_PREP="$MADGRAV_ROOT/$PREP"
export SM_SNR_GRID="$GRID" SM_INJ="$OUT"
export SM_INJ_CNN=1 SM_INJ_S3=1 SM_INJ_ASDVETO=1 SM_INJ_LOCAL_ALL=1 SM_INJ_SEED=20261004 BLIND_DEV=cpu SM_ALLOW_CPU=1
export SM_INJ_NORM_FIXED=1 SM_ASD_TAG=fixed2 SM_INJ_NPER=450 SM_DEV=cpu SM_QT_WORKERS=14 OMP_NUM_THREADS=16 SM_ASD_HALF=64.0
export SM_BANK_UM="$MADGRAV_ROOT/data/ultramassive_bank"
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
export SM_LM_FRAC=0.3333 SM_UM_FRAC=0.5 SM_INJ_SNR_FULL=1
export SM_ARM_V2=1 SM_ARM_V2_DIR=$SC/madgrav2/arm/models SM_ARM_V2_TAG=_off5
export SM_INJ_SAVE_SERIES=0
export SM_COH_REWHITEN=1 SM_SERIESCACHE=$SC/madgrav2/${SM_RUN}-fast_full/_seriescache SM_COHREF_DIR=$SC/madgrav2/${SM_RUN}-fast_full/_seriescache/_cohref
echo "[injveto] task ${SLURM_ARRAY_TASK_ID} name=$NAME grid=$GRID nper=$SM_INJ_NPER seed=$SM_INJ_SEED out=$OUT inject_sha256=$(sha256sum search_mode/inject.py | cut -c1-16) host=$(hostname) start=$(date '+%F %T')"
T0=$SECONDS; $PY search_mode/inject.py --event "$NAME"; RC=$?
echo "[injveto] $NAME rc=$RC WALL=$((SECONDS-T0))s end=$(date '+%F %T')"; exit $RC

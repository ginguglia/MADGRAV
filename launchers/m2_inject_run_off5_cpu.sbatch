#!/usr/bin/env bash
# 2026-09-29: _off5 arm re-injection on the CPU partition (INJ_TAG, SM_ARM_V2_TAG from --export); everything else identical to m2_inject_run_cpu.sbatch
# CPU-partition copy of m2_inject_run.sbatch (SM_DEV_OVERRIDE=cpu), 2026-09-19: GPU queue saturated; same code path, torch on CPU.
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --job-name=m2-o4b-inject-off5
#SBATCH --cpus-per-task=16
#SBATCH --output=${MADGRAV_ROOT}/m2/prep/logs/inject_off5_%A_%a.log
# MADGRAV2 O4b pilot: injection features (sigma, coh, centroid v1+v2, arm v1+v2) for the LR refit. Frozen CAE calibration.
set -uo pipefail
export MADGRAV_ROOT="${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$MADGRAV_ROOT"   # guided-excess worktree (2026-09-19) until merged into MADGRAV2
PY=${PY:-python}; SC=${MADGRAV_SCRATCH}
: "${SM_RUN:?}"; LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" launchers/logs_m2/injfull_manifest_${SM_RUN}.txt)
IFS='|' read -r NAME EVJ SGJ STRAIN PREP GRID OUTDIR REF <<< "$LINE"
OUT=$SC/madgrav2/inj_${SM_RUN}_m2${INJ_TAG:?}/$(basename "$OUTDIR"); mkdir -p "$OUT"
if [ -s "$OUT/${NAME}_inj.npz" ]; then echo "[m2-inj] $NAME already done"; exit 0; fi
export SM_EVENTSJSON="$EVJ" SM_SEGJSON_EV="$SGJ" SM_STRAIN="$STRAIN" SM_PREP="$MADGRAV_ROOT/$PREP"
export SM_SNR_GRID="$GRID" SM_INJ="$OUT"
export SM_INJ_CNN=0 SM_INJ_NORM_FIXED=1 SM_ASD_TAG=fixed2 SM_INJ_NPER=450 SM_DEV=cpu SM_QT_WORKERS=14 OMP_NUM_THREADS=16 SM_ASD_HALF=64.0   # SM_DEV_OVERRIDE=cpu -> CPU-partition copy of the array (2026-09-19, GPU queue saturated)
unset SM_INJ_ASDVETO
export SM_BANK_UM="$MADGRAV_ROOT/data/ultramassive_bank"
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
export SM_LM_FRAC=0.3333 SM_UM_FRAC=0.5 SM_INJ_SNR_FULL=1
export SM_ARM_V2=1 SM_ARM_V2_DIR=$SC/madgrav2/arm/models SM_ARM_V2_TAG=${SM_ARM_V2_TAG:?}
export SM_INJ_SAVE_SERIES=1   # MADGRAV2.1: keep the peak-window re-whitened series (CPU pilots of pair statistics without re-injecting)
export SM_COH_REWHITEN=1 SM_SERIESCACHE="${SM_SERIESCACHE:-$SC/madgrav2/${SM_RUN}-fast_full/_seriescache}" SM_COHREF_DIR="${SM_COHREF_DIR:-$SC/madgrav2/${SM_RUN}-fast_full/_seriescache/_cohref}"   # re-whitened coherence (coh_rewhiten.py), same refs as the prune
echo "[m2-inj] task ${SLURM_ARRAY_TASK_ID} name=$NAME grid=$GRID out=$OUT host=$(hostname) start=$(date '+%F %T')"
T0=$SECONDS; $PY search_mode/inject.py --event "$NAME"; RC=$?
echo "[m2-inj] $NAME rc=$RC WALL=$((SECONDS-T0))s end=$(date '+%F %T')"; exit $RC

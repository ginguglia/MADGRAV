#!/usr/bin/env bash
# 2026-10-19: vt96 -- injection-efficiency campaign of the FROZEN MADGRAV 2 version-2 recipe (96 catalogue detections), CPU partition only.
# Copy of MADGRAV2-gx/launchers/m2_inject_run_22_cpu.sbatch (the 2.2 campaign of 2026-10-04) with:
#   INJ tag _96 (out: $SC/madgrav2/inj_<run>_m2_96/<run>[_lowsnr]); SAME seed SM_INJ_SEED=20261004 and nper 450 -> the draw is
#   paired injection-by-injection with the 2.2 field (identical sources, SNRs, placements); SM_CROP=coinc via m2/vt96/inject96.py
#   (crop_start() override of inject._cnn_s3_from_wh, see that file); SM_INJ_ASDVETO=1 (local +-64 s ASD columns net_loc/hm_loc/lm_loc/
#   veto_keep/s3_loc for the merge's local-ASD clauses), SM_INJ_LOCAL_ALL=1 on o3a/o3b (the O3 S3-stability clause acts on loglr-channel
#   rows that may fail the June CNN gate, so every injection needs the local columns) and 0 on o4a/o4b (only the net clause applies there,
#   on net-sigma-channel rows which always pass the June gate; INJ96_LOCAL_ALL overrides); the frozen production env of the run is
#   sourced from launchers/logs_m2/<run>_off5_crop_m2_gx_jids.env (SM_PREP, SM_GX=1, SM_GX_CAP=15, SM_ARM_V2_TAG=_off5, SM_NET_CUT=
#   SM_ZL_NET_CUT=2.75, SM_ZL_GATES=1, SM_COH_GUARD_*, SM_CROP=coinc) BEFORE the injection switches, so nothing is re-typed.
#   inject.py applies NO net floor (peak window = argmax net over the 3-window grid; the 2.75 floor acts at scoring time in far96.py).
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${PARTITION_CPU}
#SBATCH --qos=${QOS_CPU}
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --job-name=m2-inject-96
#SBATCH --cpus-per-task=16
#SBATCH --output=${MADGRAV_ROOT}/m2/vt96/logs/inject96_%A_%a.log
set -uo pipefail
export MADGRAV_ROOT="${MADGRAV_ROOT:-${MADGRAV_ROOT}}"; cd "$MADGRAV_ROOT"
PY=${PY:-python}; SC=${MADGRAV_SCRATCH}; W=${MADGRAV_ROOT}/m2/vt96
: "${SM_RUN:?}"
# ---- frozen production env of this run (last EXP= line of the crop re-pass env file) ----
ENVF=launchers/logs_m2/${SM_RUN}_off5_crop_m2_gx_jids.env; [ -s "$ENVF" ] || { echo "[inj96] FATAL no env file $ENVF"; exit 1; }
EXP=$(grep "^EXP=" $ENVF | tail -1 | sed 's/^EXP=ALL,//'); IFS=',' read -ra KV <<< "$EXP"; for kv in "${KV[@]}"; do export "$kv"; done
[ "$SM_CROP" = "coinc" ] && [ "$SM_ARM_V2_TAG" = "_off5" ] && [ "$SM_GX" = "1" ] && [ "$SM_NET_CUT" = "2.75" ] || { echo "[inj96] FATAL frozen env check failed: SM_CROP=$SM_CROP SM_ARM_V2_TAG=$SM_ARM_V2_TAG SM_GX=$SM_GX SM_NET_CUT=$SM_NET_CUT"; exit 1; }
MAN=${INJ96_MANIFEST:-launchers/logs_m2/injfull_manifest_${SM_RUN}.txt}; LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAN")
[ -n "$LINE" ] || { echo "[inj96] no manifest line $((SLURM_ARRAY_TASK_ID+1)) in $MAN"; exit 1; }
IFS='|' read -r NAME EVJ SGJ STRAIN PREP GRID OUTDIR REF <<< "$LINE"
TAG=${INJ96_TAG:-_96}; OUT=$SC/madgrav2/inj_${SM_RUN}_m2${TAG}/$(basename "$OUTDIR"); mkdir -p "$OUT"
if [ -s "$OUT/${NAME}_inj.npz" ]; then echo "[inj96] $NAME already done"; exit 0; fi
# manifest prep must be the frozen per-run prep of the production env (same files; MADGRAV2/data and MADGRAV2-gx/data are identical)
[ "$(basename "$PREP")" = "$(basename "$SM_PREP")" ] || { echo "[inj96] FATAL manifest prep $PREP != frozen SM_PREP $SM_PREP"; exit 1; }
export SM_EVENTSJSON="$EVJ" SM_SEGJSON_EV="$SGJ" SM_STRAIN="$STRAIN"
export SM_SNR_GRID="${INJ96_SNR_GRID:-$GRID}" SM_INJ="$OUT"
export SM_INJ_CNN=1 SM_INJ_S3=1 SM_INJ_SEED=${SM_INJ_SEED:-20261004} BLIND_DEV=cpu SM_ALLOW_CPU=1
export SM_INJ_ASDVETO=1 SM_ASD_HALF=64.0
case "$SM_RUN" in o3a|o3b) export SM_INJ_LOCAL_ALL=${INJ96_LOCAL_ALL:-1};; *) export SM_INJ_LOCAL_ALL=${INJ96_LOCAL_ALL:-0};; esac
export SM_INJ_NORM_FIXED=1 SM_ASD_TAG=fixed2 SM_INJ_NPER=${SM_INJ_NPER:-450} SM_DEV=cpu SM_QT_WORKERS=${SM_QT_WORKERS:-14} OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export SM_BANK_UM="$MADGRAV_ROOT/data/ultramassive_bank"
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
export SM_LM_FRAC=0.3333 SM_UM_FRAC=0.5 SM_INJ_SNR_FULL=1
export SM_ARM_V2=1 SM_ARM_V2_DIR=$SC/madgrav2/arm/models
export SM_INJ_SAVE_SERIES=0
echo "[inj96] task ${SLURM_ARRAY_TASK_ID:-smoke} name=$NAME grid=$SM_SNR_GRID nper=$SM_INJ_NPER seed=$SM_INJ_SEED out=$OUT host=$(hostname) start=$(date '+%F %T')"
echo "[inj96] env: SM_CROP=$SM_CROP SM_PREP=$SM_PREP SM_ARM_V2_TAG=$SM_ARM_V2_TAG SM_GX=$SM_GX SM_GX_CAP=$SM_GX_CAP SM_NET_CUT=$SM_NET_CUT SM_ZL_NET_CUT=$SM_ZL_NET_CUT SM_ZL_GATES=$SM_ZL_GATES SM_COH_REWHITEN=$SM_COH_REWHITEN SM_COH_GUARD_HZ=$SM_COH_GUARD_HZ SM_SERIESCACHE=$SM_SERIESCACHE SM_INJ_ASDVETO=$SM_INJ_ASDVETO SM_INJ_LOCAL_ALL=$SM_INJ_LOCAL_ALL inject_sha256=$(sha256sum search_mode/inject.py | cut -c1-16) inject96_sha256=$(sha256sum $W/inject96.py | cut -c1-16)"
T0=$SECONDS; $PY $W/inject96.py --event "$NAME"; RC=$?
echo "[inj96] $NAME rc=$RC WALL=$((SECONDS-T0))s end=$(date '+%F %T')"; exit $RC

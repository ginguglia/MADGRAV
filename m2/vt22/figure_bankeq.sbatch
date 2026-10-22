#!/usr/bin/env bash
# bank tags + equal-weighted recovery plane (2026-10-05). CPU. Output figures_m2/mtot_snr_smooth_m22_off5s3rank_bankeq.{pdf,png}
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --time=00:40:00 --mem=24G --cpus-per-task=2
#SBATCH --output=${MADGRAV_ROOT}/m2/vt22/logs/%x_%j.log
set -uo pipefail; PY=${PY:-python}; W=${MADGRAV_ROOT}/m2/vt22; SC=${MADGRAV_SCRATCH}/madgrav2
export MPLCONFIGDIR=$SC/vt22/mpl; mkdir -p $MPLCONFIGDIR
echo "[figure-bankeq] START $(date '+%F %T')"; cd $W && $PY bank_tags22.py; rc=$?; [ $rc -eq 0 ] || { echo "[figure-bankeq] bank tags FAILED rc=$rc"; exit $rc; }
export MG_FIELD_DIR=$SC/vt22 MG_FIELD_TAG=m22 MG_PENDING_RUNS="" MG_FIG_LABEL="MADGRAV 2.2" MG_FIG_LEGLOC="upper left" MG_FIG_YTOP=85 MG_BANK_EQUAL=1
export MG_HL_JSON=$SC/figures_m2/hl_snr_o3o4_m2_off5s3rank.json MG_FIG_OUT=$SC/figures_m2/mtot_snr_smooth_m22_off5s3rank_bankeq
export MG_FIG_CBLABEL="injection recovery probability at FAR < 1/yr
(MADGRAV 2.2 campaign, seed 20261004; banks equal-weighted)"
export MG_FIG_FOOT="MADGRAV 2.2 detections (80 at FAR < 1/yr) on the MADGRAV 2.2 injection recovery field; the two injection banks are equal-weighted per cell and their seam (22 Msun detector frame) is marked; no local-ASD veto, no null-calibration factor"
cd ${MADGRAV_ROOT}/figures/catalog_o3o4 && $PY plot_mtot_snr_smooth_m2.py; rc=$?
echo "[figure-bankeq] END rc=$rc $(date '+%F %T')"; ls -la $MG_FIG_OUT.* 2>/dev/null; exit $rc

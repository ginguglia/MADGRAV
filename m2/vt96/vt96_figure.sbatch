#!/usr/bin/env bash
# vt96 figures (CPU): four-panel recovery plane with the 96 stars on the 96 field (plot_mtot_snr_smooth_m2_v2_96.py, copy of the dossier
# script; MG_FIELD_TAG=m96 MG_FIELD_DIR=vt96) and the paper-layout <VT> figure (fig_vt96_paper.py).
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=32G --time=00:40:00
#SBATCH --job-name=vt96-figure --output=${MADGRAV_ROOT}/m2/vt96/logs/%x_%A.log
set -uo pipefail
PY=${PY:-python}; W=${MADGRAV_ROOT}/m2/vt96; SC=${MADGRAV_SCRATCH}/madgrav2
export MG_FIELD_DIR=$SC/vt96 MG_FIELD_TAG=m96 MG_PENDING_RUNS="" MG_FIG_LABEL="MADGRAV 2 version 2 (96)" MG_FIG_LEGLOC="upper left" MG_FIG_YTOP=85
export MG_DET_CSV=$W/detections_v2_96.csv MG_HL_JSON=$SC/figures_m2/hl_snr_o3o4_m2_crop96.json MG_FIG_OUT=$SC/figures_m2/mtot_snr_smooth_m2_v2_96_vt96
export MG_FIG_CBLABEL=$'injection recovery probability at FAR < 1/yr\n(frozen version-2 recipe, vt96 campaign, seed 20261004)'
export MG_FIG_FOOT="MADGRAV 2 version 2 detections (96 at FAR < 1/yr) on the version-2 injection recovery field (vt96: same seed as the 2.2 field, full recipe incl. local-ASD veto, guard and cap bypass; no null-calibration factor)"
export MPLCONFIGDIR=$SC/vt96/mpl; mkdir -p $MPLCONFIGDIR
echo "[vt96-figure] START $(date '+%F %T')"; cd $W
OMP_NUM_THREADS=4 $PY plot_mtot_snr_smooth_m2_v2_96.py 2>&1 | tee $W/logs/figure96_$(date +%Y%m%d_%H%M).txt; rc=${PIPESTATUS[0]}
OMP_NUM_THREADS=4 $PY fig_vt96_paper.py 2>&1 | tee -a $W/logs/figure96_$(date +%Y%m%d_%H%M).txt; rc2=${PIPESTATUS[0]}
echo "[vt96-figure] END rc=$rc/$rc2 $(date '+%F %T')"; ls -la $MG_FIG_OUT.* $SC/figures_m2/vt96_paper.* 2>/dev/null; exit $((rc+rc2))

#!/usr/bin/env bash
# vt96 final products (CPU), after the relabel: VT table (vt_table96.py), paired comparison (compare96.py), four-panel recovery plane with
# the 96 stars on the 96 field (plot_mtot_snr_smooth_m2_v2_96.py, MG_FIELD_TAG=m96) and the paper-layout <VT> figure (fig_vt96_paper.py).
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --job-name=vt96-final --output=${MADGRAV_ROOT}/m2/vt96/logs/%x_%A.log
set -uo pipefail
PY=${PY:-python}; W=${MADGRAV_ROOT}/m2/vt96; SC=${MADGRAV_SCRATCH}/madgrav2; D=$(date +%Y%m%d)
export MPLCONFIGDIR=$SC/vt96/mpl OMP_NUM_THREADS=4; mkdir -p $MPLCONFIGDIR; cd $W; rc=0
for f in $SC/vt96/vt_relabel_comoving_m96.json $SC/vt96/inj_scored_{o3a,o3b,o4a,o4b}_m96.npz $SC/vt96/relabel_inj_{o3a,o3b,o4a,o4b}_m96.npz; do [ -s $f ] || { echo "FATAL missing $f"; exit 1; }; done
echo "[vt96-final] START $(date '+%F %T')"
export VT96_TABLE=$W/VT_TABLE_M96_$D.md VT96_COMPARE=$W/COMPARE_M96_vs_M22_$D.md
$PY -I vt_table96.py; rc=$((rc+$?))
$PY -I compare96.py > $W/logs/compare96_$D.txt 2>&1; rc=$((rc+$?)); tail -3 $W/logs/compare96_$D.txt
export MG_FIELD_DIR=$SC/vt96 MG_FIELD_TAG=m96 MG_PENDING_RUNS="" MG_FIG_LABEL="MADGRAV 2 version 2 (96)" MG_FIG_LEGLOC="upper left" MG_FIG_YTOP=85
export MG_DET_CSV=$W/detections_v2_96.csv MG_HL_JSON=$SC/figures_m2/hl_snr_o3o4_m2_crop96.json MG_FIG_OUT=$SC/figures_m2/mtot_snr_smooth_m2_v2_96_vt96
export MG_FIG_CBLABEL=$'injection recovery probability at FAR < 1/yr\n(frozen version-2 recipe, vt96 campaign, seed 20261004)'
export MG_FIG_FOOT="MADGRAV 2 version 2 detections (96 at FAR < 1/yr) on the version-2 injection recovery field (vt96: same seed as the 2.2 field, full recipe incl. local-ASD veto, guard and cap bypass; no null-calibration factor)"
$PY plot_mtot_snr_smooth_m2_v2_96.py 2>&1 | tee $W/logs/figure96_$D.txt; rc=$((rc+${PIPESTATUS[0]}))
$PY fig_vt96_paper.py 2>&1 | tee -a $W/logs/figure96_$D.txt; rc=$((rc+${PIPESTATUS[0]}))
ls -la $VT96_TABLE $VT96_COMPARE $MG_FIG_OUT.* $SC/figures_m2/vt96_paper.* 2>/dev/null
echo "[vt96-final] END rc=$rc $(date '+%F %T')"; exit $rc

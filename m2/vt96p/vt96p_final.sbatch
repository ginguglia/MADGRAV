#!/usr/bin/env bash
# vt96p final products (CPU), after the relabel: VT table (vt_table96p.py), paired comparison with vt96 (compare96p.py), recovery plane with the
# 96 stars on the production-placement field (m2/vt96/plot_mtot_snr_smooth_m2_v2_96.py, MG_FIELD_TAG=m96p) and the paper-layout <VT> figure
# (m2/vt96/fig_vt96_paper.py with VT96_JSON = the m96p relabel, written to figures_m2/vt96p/ so the vt96 figure is not overwritten).
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=32G --time=01:00:00
#SBATCH --job-name=vt96p-final --output=${MADGRAV_ROOT}/m2/vt96p/logs/%x_%A.log
set -uo pipefail
PY=${PY:-python}; P=${MADGRAV_ROOT}/m2/vt96p; W=${MADGRAV_ROOT}/m2/vt96; SC=${MADGRAV_SCRATCH}/madgrav2; D=$(date +%Y%m%d)
export MPLCONFIGDIR=$SC/vt96/mpl OMP_NUM_THREADS=4; mkdir -p $MPLCONFIGDIR $SC/figures_m2/vt96p; rc=0
for f in $SC/vt96/vt_relabel_comoving_m96p.json $SC/vt96/inj_scored_{o3a,o3b,o4a,o4b}_m96p.npz $SC/vt96/relabel_inj_{o3a,o3b,o4a,o4b}_m96p.npz $SC/vt96/vt_relabel_comoving_m96.json; do [ -s $f ] || { echo "FATAL missing $f"; exit 1; }; done
echo "[vt96p-final] START $(date '+%F %T')"
export VT96P_TABLE=$P/VT_TABLE_M96P_$D.md VT96P_COMPARE=$P/COMPARE_M96P_vs_M96_$D.md
cd $P; $PY -I vt_table96p.py; rc=$((rc+$?))
$PY -I compare96p.py > $P/logs/compare96p_$D.txt 2>&1; rc=$((rc+$?)); tail -4 $P/logs/compare96p_$D.txt
cd $W
export MG_FIELD_DIR=$SC/vt96 MG_FIELD_TAG=m96p MG_PENDING_RUNS="" MG_FIG_LABEL="MADGRAV 2 version 2 (96)" MG_FIG_LEGLOC="upper left" MG_FIG_YTOP=85
export MG_DET_CSV=$W/detections_v2_96.csv MG_HL_JSON=$SC/figures_m2/hl_snr_o3o4_m2_crop96.json MG_FIG_OUT=$SC/figures_m2/mtot_snr_smooth_m2_v2_96_vt96p
export MG_FIG_CBLABEL=$'injection recovery probability at FAR < 1/yr\n(frozen version-2 recipe, vt96p campaign: search placement, seed 20261004)'
export MG_FIG_FOOT="MADGRAV 2 version 2 detections (96 at FAR < 1/yr) on the version-2 injection recovery field at the search's own 1 s placement (vt96p: same injections as vt96 and the 2.2 field; full recipe incl. local-ASD veto, guard and cap bypass; no null-calibration factor)"
$PY plot_mtot_snr_smooth_m2_v2_96.py 2>&1 | tee $P/logs/figure96p_$D.txt; rc=$((rc+${PIPESTATUS[0]}))
VT96_JSON=$SC/vt96/vt_relabel_comoving_m96p.json MG_FIGDIR=$SC/figures_m2/vt96p $PY fig_vt96_paper.py 2>&1 | tee -a $P/logs/figure96p_$D.txt; rc=$((rc+${PIPESTATUS[0]}))
ls -la $VT96P_TABLE $VT96P_COMPARE $MG_FIG_OUT.* $SC/figures_m2/vt96p/vt96_paper.* 2>/dev/null
echo "[vt96p-final] END rc=$rc $(date '+%F %T')"; exit $rc

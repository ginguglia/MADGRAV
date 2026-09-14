#!/usr/bin/env bash
# Glitch-arm gate: inj_full scoring chain (recovery figure, efficiency tables, source-frame VT), mirror of
# injfull_score.sh + the comoving-prior m20 VT products used by the paper, with the gate. No mail.
set -uo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
PY=${PY:-python}; PF=search_mode/pastro_final
D=${MADGRAV_SCRATCH:-$MADGRAV_ROOT/scratch}/mtot_snr_sim_experiment
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
DET=$MADGRAV_ROOT/details/successor_statistic
KE=$DET/ke_gnet.json; THR=$($PY -c "import json;print(json.load(open('$DET/gnet_threshold.json'))['threshold'])")
FARCSV=$MADGRAV_ROOT/figures/catalog_o3o4/far_lronly_g106_gnet.csv
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
TAG=x1cnnfullgveto
fail () { echo "[fullg] FAILED at: $1"; exit 1; }
echo "[fullg] start $(date '+%F %T') gate g_net>=$THR KE=$(cat $KE)"
echo "[fullg] 1/7 pastro_final on inj_full -> _x1cnnfullg"
SM_SNR_GRID_EXT=30,40,60,80 \
SM_TRIALS_OFF=1 SM_INJ_CNN_GATE=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_INJ_BG_NETMAX=1 SM_GNET=$THR SM_KE=$KE SM_FAR_LR_CSV=$FARCSV \
SM_INJ_DIR=inj_full SM_SUF_EXTRA=fullg \
SM_ADOPTED_AXIS=1 SM_AXIS_COL=far_excl SM_ADMIT=all SM_DET_RULE=far \
  nice -n 5 $PY $PF/pastro_final.py || fail "pastro_final"
echo "[fullg] 2/7 build_inj_veto"
SM_VETO_SRC=inj_full SM_VETO_SUF=_x1cnnfullg SM_VETO_NOREF=1 nice -n 5 $PY $PF/build_inj_veto.py || fail "build_inj_veto"
echo "[fullg] 3/7 vt_relabel_comoving (Euclidean prior; feeds the recovery figure via relabel_inj_*_${TAG}.npz)"
SM_VT_SUF=_${TAG} SM_VT_COMOVING_PRIOR=1 SM_INJ_CAMPAIGN=inj_full SM_VT_FIXED_ASD=1 SM_ASD_TAG=fixed2 nice -n 5 $PY $PF/vt_relabel_comoving.py || fail "vt_relabel_comoving"
echo "[fullg] 4/7 mark the weighted side outputs INVALID (SM_SNR_GRID_EXT set), as injfull_score.sh does"
for f in $PF/pastro_final_x1cnnfullg.json $PF/pastro_final_x1cnnfullg.csv $PF/vt_relabel_comoving_${TAG}.json; do
  [ -f "$f" ] && mv "$f" "${f%.*}_INVALID_popweights_figonly.${f##*.}" && echo "  marked $(basename $f)"
done
echo "[fullg] 5/7 comoving-prior m20 products (the paper's VT: vt_relabel_comoving_${TAG}_m20.json)"
for r in o3a o3b o4a o4b; do ln -sfn inj_scored_${r}_${TAG}.npz $PF/inj_scored_${r}_${TAG}_m20.npz; done
SM_VT_SUF=_${TAG} SM_VT_OUT_SUF=_${TAG}_m20 SM_VT_COMOVING_PRIOR=1 SM_INJ_CAMPAIGN=inj_full SM_VT_FIXED_ASD=1 SM_ASD_TAG=fixed2 \
  nice -n 5 $PY $PF/vt_relabel_comoving.py > $PF/vt_relabel_comoving_${TAG}_m20.log || fail "vt_relabel_comoving m20"
SM_VT_SUF=_${TAG}_m20 SM_INJ_CAMPAIGN=inj_full nice -n 5 $PY $PF/build_neff.py || fail "build_neff"
SM_VT_SUF=_${TAG}_m20 SM_INJ_CAMPAIGN=inj_full nice -n 5 $PY $PF/build_eff_srcframe.py || fail "build_eff_srcframe"
echo "[fullg] 6/7 figures: VT frames, four-epoch ratio, recovery plane"
SM_VT_SUF=_${TAG}_m20 SM_INJ_CAMPAIGN=inj_full nice -n 5 $PY $PF/fig_vt_frames.py || fail "fig_vt_frames"
SM_VT_SUF=_${TAG}_m20 SM_INJ_CAMPAIGN=inj_full SM_FIG_CLEARED=1 nice -n 5 $PY $PF/fig_fourepoch_ratio.py || fail "fig_fourepoch_ratio"
MG_FIG_TAG=$TAG MG_FIG_OUT=$D/mtot_snr_smooth_gnet nice -n 5 $PY $D/plot_mtot_snr_smooth.py || fail "recovery figure"
echo "[fullg] 7/7 efficiency / comparison tables"
MG_FIG_TAG=$TAG $PY $D/make_body_snrext.py > "$D/body_fullg.txt" || fail "body"
echo "[fullg] DONE $(date '+%F %T')"

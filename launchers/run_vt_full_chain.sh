#!/usr/bin/env bash
# Sensitive volume-time and p_astro on the UNIFIED inj_full campaign (tag _x1cnnfullveto),
# with the two 2026-09-05 corrections switched on:
#   SM_VT_COMOVING_PRIOR=1  rho^-4 population weights put in the same (comoving, (1+z)-dilated)
#                           measure as the reference volume; J(z) median 0.51-0.58; VT x1.35-1.37
#   inj_full (13 SNR levels, rho 5-80, three mass strata) instead of the rho<=25 inj_fixed2 grid,
#                           whose midpoint-width renormalisation truncated the high-SNR tail
# Both switches default OFF: the adopted _x1cnnadopt48f* products rebuild byte-identically.
#
# This file re-states, as one script, the sequence of commands used interactively on 2026-09-05
# to produce the *_x1cnnfullveto{,_m20}.json, vt_pipelines_target_zc_lm10.json and
# pastro_final_x1cnnfullcp/sw products committed beside it. Prerequisites: the inj_full campaign
# in $MADGRAV_SCRATCH/inj_full (launchers/run_injfull.sh) and steps 1-2 of injfull_score.sh
# (pastro_final -> _x1cnnfull, build_inj_veto -> _x1cnnfullveto).
set -euo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
PY=${PY:-python}
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
PF=search_mode/pastro_final
KE=$MADGRAV_ROOT/details/successor_statistic/ke_adopted.json
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
LM_EDGES=10,15,20,40,60,80,100,130,160,200,260,330,400      # extended grid (MADGRAV-only frames figure)
COMMON="SM_INJ_CAMPAIGN=inj_full SM_VT_FIXED_ASD=1 SM_ASD_TAG=fixed2 SM_VT_COMOVING_PRIOR=1"

echo "[full-vt] 1/5 relabel, adopted 20-400 grid -> _x1cnnfullveto_m20 (ratio-figure numerator)"
env $COMMON SM_VT_SUF=_x1cnnfullveto SM_VT_OUT_SUF=_x1cnnfullveto_m20 $PY $PF/vt_relabel_comoving.py
# build_neff / build_eff_srcframe / vt_compare_pipelines pair relabel_inj_<run><SUF> with
# inj_scored_<run><SUF>; the scored file does not depend on the mass grid, so expose it under
# the output suffix as well.
for r in o3a o3b o4a o4b; do cp -n $PF/inj_scored_${r}_x1cnnfullveto.npz $PF/inj_scored_${r}_x1cnnfullveto_m20.npz; done

echo "[full-vt] 2/5 relabel, extended 10-400 grid -> _x1cnnfullveto"
env $COMMON SM_VT_MASS_EDGES=$LM_EDGES SM_VT_SUF=_x1cnnfullveto $PY $PF/vt_relabel_comoving.py

echo "[full-vt] 3/5 N_eff, efficiency, LVK comparator, cross-pipeline VT"
for SUF in _x1cnnfullveto_m20 _x1cnnfullveto; do
  SM_VT_SUF=$SUF SM_INJ_CAMPAIGN=inj_full $PY $PF/build_neff.py
  SM_VT_SUF=$SUF SM_INJ_CAMPAIGN=inj_full $PY $PF/build_eff_srcframe.py
done
SM_VT_MASS_EDGES=$LM_EDGES SM_VT_TGT_SUF=_lm10 $PY $PF/vt_pipelines_target_zc.py        # comparator on the 13-edge grid
SM_VT_SUF=_x1cnnfullveto_m20 SM_INJ_CAMPAIGN=inj_full $PY $PF/vt_compare_pipelines.py
SM_VT_MASS_EDGES=$LM_EDGES SM_VT_SUF=_x1cnnfullveto SM_VT_TGT_SUF=_lm10 SM_INJ_CAMPAIGN=inj_full $PY $PF/vt_compare_pipelines.py

echo "[full-vt] 4/5 figures: frames (extended grid) and four-epoch ratio (20-400 grid)"
SM_VT_SUF=_x1cnnfullveto SM_INJ_CAMPAIGN=inj_full $PY $PF/fig_vt_frames.py
SM_VT_SUF=_x1cnnfullveto_m20 SM_INJ_CAMPAIGN=inj_full SM_FIG_CLEARED=1 $PY $PF/fig_fourepoch_ratio.py

echo "[full-vt] 5/5 p_astro with the comoving-prior per-injection weights -> _x1cnnfullcp (and FAR sweep -> _x1cnnfullsw)"
BASE="SM_SNR_GRID_EXT=30,40,60,80 SM_TRIALS_OFF=1 SM_INJ_CNN_GATE=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_KE=$KE SM_INJ_DIR=inj_full SM_ADOPTED_AXIS=1 SM_AXIS_COL=far_excl SM_ADMIT=all SM_DET_RULE=far SM_PASTRO_W0_FROM=_x1cnnfullveto_m20"
env $BASE SM_SUF_EXTRA=fullcp $PY $PF/pastro_final.py
env $BASE SM_SUF_EXTRA=fullsw SM_FAR_SWEEP=0.005,0.01,0.02,0.05,0.1,0.2,0.5,1 $PY $PF/pastro_final.py
echo "[full-vt] DONE $(date '+%F %T')"

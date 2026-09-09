#!/usr/bin/env bash
# VT + p_astro chain for the 47-detection criterion (calibrated FAR point estimate < 1/yr).
# Mirrors run_vt_adopted_chain.sh EXACTLY except for SM_DET_RULE=far (injection admission) and
# the candidate-axis flags. Suffix _x1cnnadopt48f -> _x1cnnadopt48fveto; no accepted file touched.
set -euo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
cd "$MADGRAV_ROOT"
PY=${PY:-python}
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
KE=$MADGRAV_ROOT/details/successor_statistic/ke_adopted.json

echo "[47] 1/5 pastro_final on inj_fixed2 -> _x1cnnadopt48f"
SM_TRIALS_OFF=1 SM_INJ_CNN_GATE=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_KE=$KE \
SM_INJ_DIR=inj_fixed2 SM_SUF_EXTRA=adopt48f \
SM_ADOPTED_AXIS=1 SM_AXIS_COL=far_excl SM_ADMIT=all SM_DET_RULE=far \
  nice -n 5 $PY search_mode/pastro_final/pastro_final.py

echo "[47] 2/5 build_inj_veto -> _x1cnnadopt48fveto"
SM_VETO_SRC=inj_fixed2 SM_VETO_SUF=_x1cnnadopt48f SM_VETO_REF=inj_cnn \
  nice -n 5 $PY search_mode/pastro_final/build_inj_veto.py

echo "[47] 3/5 vt_relabel_comoving (corrected as-run ASD)"
SM_VT_SUF=_x1cnnadopt48fveto SM_INJ_CAMPAIGN=inj_fixed2 SM_VT_FIXED_ASD=1 SM_ASD_TAG=fixed2 \
  nice -n 5 $PY search_mode/pastro_final/vt_relabel_comoving.py

echo "[47] 4/5 N_eff + efficiency"
SM_VT_SUF=_x1cnnadopt48fveto SM_INJ_CAMPAIGN=inj_fixed2 nice -n 5 $PY search_mode/pastro_final/build_neff.py
SM_VT_SUF=_x1cnnadopt48fveto SM_INJ_CAMPAIGN=inj_fixed2 nice -n 5 $PY search_mode/pastro_final/build_eff_srcframe.py

echo "[47] 5/5 figures 3 and 4"
SM_VT_SUF=_x1cnnadopt48fveto SM_INJ_CAMPAIGN=inj_fixed2 nice -n 5 $PY search_mode/pastro_final/fig_vt_frames.py
SM_VT_SUF=_x1cnnadopt48fveto SM_INJ_CAMPAIGN=inj_fixed2 SM_FIG_CLEARED=1 \
  nice -n 5 $PY search_mode/pastro_final/fig_fourepoch_ratio.py
echo "[47] DONE $(date '+%F %T')"

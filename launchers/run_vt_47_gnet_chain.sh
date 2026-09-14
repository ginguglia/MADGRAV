#!/usr/bin/env bash
# Glitch-arm gate: adopted-campaign (inj_fixed2) p_astro chain, mirror of run_vt_47_chain.sh with the gate.
set -euo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
PY=${PY:-python}
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
DET=$MADGRAV_ROOT/details/successor_statistic
KE=$DET/ke_gnet.json; THR=$($PY -c "import json;print(json.load(open('$DET/gnet_threshold.json'))['threshold'])")
FARCSV=$MADGRAV_ROOT/figures/catalog_o3o4/far_lronly_g106_gnet.csv
echo "[47g] start $(date '+%F %T') gate g_net>=$THR KE=$(cat $KE)"
echo "[47g] 1/2 pastro_final on inj_fixed2 -> _x1cnnadopt48fg"
SM_TRIALS_OFF=1 SM_INJ_CNN_GATE=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_INJ_BG_NETMAX=1 SM_GNET=$THR SM_KE=$KE SM_FAR_LR_CSV=$FARCSV \
SM_INJ_DIR=inj_fixed2 SM_SUF_EXTRA=adopt48fg \
SM_ADOPTED_AXIS=1 SM_AXIS_COL=far_excl SM_ADMIT=all SM_DET_RULE=far \
  nice -n 5 $PY search_mode/pastro_final/pastro_final.py
echo "[47g] 2/2 build_inj_veto -> _x1cnnadopt48fgveto"
SM_VETO_SRC=inj_fixed2 SM_VETO_SUF=_x1cnnadopt48fg SM_VETO_REF=inj_cnn nice -n 5 $PY search_mode/pastro_final/build_inj_veto.py
echo "[47g] DONE $(date '+%F %T')"

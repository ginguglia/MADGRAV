#!/usr/bin/env bash
# Glitch-arm gate: null calibration in the exact ke_adopted configuration (excldet, netmax 10.6, lronly) + SM_GNET
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
PY=${PY:-python}
SM_EXCL_DET=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_GNET=-4.02 OMP_NUM_THREADS=8 nice -n 5 $PY search_mode/perarm_nullcal.py 100000000

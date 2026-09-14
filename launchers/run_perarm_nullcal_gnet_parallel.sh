#!/usr/bin/env bash
# Glitch-arm gate null calibration, one process per run (ke_adopted configuration: excldet, netmax 10.6, lronly, full population)
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
PY=${PY:-python}
for R in O3a O3b O4a O4b; do
  r=$(echo $R | tr A-Z a-z)
  SM_RUNS=$R SM_TAG_EXTRA=_$r SM_EXCL_DET=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_GNET=-4.02 OMP_NUM_THREADS=4 \
    nohup nice -n 5 $PY search_mode/perarm_nullcal.py 100000000 > launchers/perarm_nullcal_gnet_$R.log 2>&1 &
  echo "launched $R pid $!"
done

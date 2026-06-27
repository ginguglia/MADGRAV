#!/usr/bin/env bash
# MADGRAV quick demo: recover GW190521 (the IMBH) from the bundled ~256 s segment using the
# vendored weights. No GWOSC fetch; ~2 min on a GPU.
#
# DEV can be any free GPU (e.g. cuda:0, cuda:2, ...) or CPU. CPU works with SM_ALLOW_CPU=1 but is
# slower and NOT byte-identical to the frozen GPU calibration. Default device is cuda:0.
#   DEV=cuda:2 bash demo/run_demo.sh        # pick another GPU
#   DEV=cpu SM_ALLOW_CPU=1 bash demo/run_demo.sh   # CPU fallback
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MADGRAV_ROOT="$(cd "$HERE/.." && pwd)"
cd "$MADGRAV_ROOT"
export DEV="${DEV:-cuda:0}"
export SM_ALLOW_CPU="${SM_ALLOW_CPU:-1}"
PY="${PYTHON:-python}"
echo "[run_demo] MADGRAV_ROOT=$MADGRAV_ROOT DEV=$DEV"
exec "$PY" demo/recover_event.py "$@"

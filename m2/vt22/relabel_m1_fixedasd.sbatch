#!/usr/bin/env bash
# Like-for-like VT reference (2026-10-05): relabel the MADGRAV1 adopted field (inj_scored_<run>_x1cnnadopt48fveto, inj_cnn campaign)
# with the SAME horizon convention as the 2.2 field (SM_VT_FIXED_ASD=1, fixed2 ASD), into scratch (MADGRAV1 tree untouched).
# Pass 1: frozen-ASD rebuild (SM_VT_FIXED_ASD=0) -> must reproduce pastro_final/vt_relabel_comoving_x1cnnadopt48fveto.json if that file
# was built frozen (tells us which convention it carries). Pass 2: fixed ASD -> the number to compare with the 2.2 table.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=48G --time=02:00:00
#SBATCH --job-name=vt-m1-fixedasd --output=${MADGRAV_ROOT}/m2/vt22/logs/%x_%j.log
set -uo pipefail; PY=${PY:-python}; W=${MADGRAV_ROOT}/m2/vt22; D=${MADGRAV_SCRATCH}/madgrav2/vt_m1fixed
export SM_RELABEL_DIR=$D SM_VT_SUF=_x1cnnadopt48fveto SM_RUNS=O3a,O3b,O4a,O4b OMP_NUM_THREADS=4
cd $W
echo "[m1-relabel] PASS 1 frozen ASD $(date '+%F %T')"; SM_VT_FIXED_ASD=0 SM_VT_OUT_SUF=_x1cnnadopt48fveto_frozenasd $PY vt_relabel_comoving22.py > $D/pass1_frozen.log 2>&1; echo "rc=$?"; grep -E "^\[O|VT|done" $D/pass1_frozen.log | tail -8
$PY - <<'PYEOF'
import json, numpy as np
a=json.load(open("${MADGRAV_SCRATCH}/MADGRAV1/search_mode/pastro_final/vt_relabel_comoving_x1cnnadopt48fveto.json"))
b=json.load(open("${MADGRAV_SCRATCH}/madgrav2/vt_m1fixed/vt_relabel_comoving_x1cnnadopt48fveto_frozenasd.json"))
K="vt_comoving_srcframe_gpc3yr"
for r in a["runs"]:
    x=np.array(a["runs"][r][K],float); y=np.array(b["runs"][r][K],float); print(f"[m1-relabel] {r}: original vs frozen rebuild max rel diff {np.nanmax(np.abs(x-y)/np.maximum(np.abs(x),1e-12)):.3g}")
PYEOF
echo "[m1-relabel] PASS 2 fixed ASD $(date '+%F %T')"; SM_VT_FIXED_ASD=1 SM_ASD_TAG=fixed2 SM_VT_OUT_SUF=_x1cnnadopt48fveto_fixedasd $PY vt_relabel_comoving22.py > $D/pass2_fixed.log 2>&1; echo "rc=$?"; grep -E "^\[O|VT|done" $D/pass2_fixed.log | tail -8
echo "[m1-relabel] END $(date '+%F %T')"; ls -la $D

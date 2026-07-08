#!/usr/bin/env bash
# Reference-ASD build -> data/<run>_search_prep (reference_psd_{H1,L1}.npz) via build_o3b_prep.py (same
# Welch procedure, any run). CPU, single task. PREREQ for streams / cache / inject.
# job-name / output injected at submit by run_chain.sh.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -uo pipefail
: "${MADGRAV_ROOT:?submit via run_chain.sh — MADGRAV_ROOT unset}"
source "$MADGRAV_ROOT/launchers/templates/_common.sh"
export SM_PREP_OUT="$SM_PREP"
export OMP_NUM_THREADS=8
echo "[$SM_RUN-prep] START $(date '+%F %T')  strain=$SM_STRAIN -> $SM_PREP_OUT"
"$PY" "$SM/build_o3b_prep.py" 4000; rc=$?
echo "[$SM_RUN-prep] END rc=$rc $(date '+%F %T')"; ls -la "$SM_PREP_OUT" 2>/dev/null
exit $rc

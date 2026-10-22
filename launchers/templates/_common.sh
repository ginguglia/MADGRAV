#!/usr/bin/env bash
# Shared setup for every MADGRAV chain template. Sourced at the top of each tmpl_*.sbatch.
# Site vars (MADGRAV_ROOT, SCRATCH, PYTHON) arrive via the environment: run_chain.sh sources site.conf
# and submits with --export=ALL. SM_RUN identifies the run. Per-run paths are derived from SCRATCH +
# SM_RUN by convention; anything pre-set in the env WINS (override for a run that breaks the convention).
: "${SM_RUN:?SM_RUN not set — submit via run_chain.sh <run>}"
: "${MADGRAV_ROOT:?MADGRAV_ROOT not set — is site.conf sourced? (submit via run_chain.sh)}"
: "${SCRATCH:?SCRATCH not set — cp site.conf.example site.conf and edit}"
: "${PYTHON:?PYTHON not set — set it in site.conf}"
export SC="$SCRATCH"
export PY="$PYTHON"
export SM="${SM:-search_mode}"
cd "$MADGRAV_ROOT"
# --- convention-derived per-run paths (override by pre-setting the var) ---
export SM_STRAIN="${SM_STRAIN:-$SC/strain_${SM_RUN}_full}"
export SM_STREAMS="${SM_STREAMS:-$SC/streams_${SM_RUN}_full}"
export SM_SERIESCACHE="${SM_SERIESCACHE:-$SC/${SM_RUN}-fast_full/_seriescache}"
export SM_BGJSON="${SM_BGJSON:-$SC/${SM_RUN}_full_coincident.json}"
# SM_PREP is ENFORCED, never name-defaulted (decision 2026-08-22, after the same wrong-prep
# defect landed twice in one day). The guard refuses an unset SM_PREP on runs whose frozen
# convention is not the name-derived one (o3b -> o3a_search_prep per V2_FREEZE_20260821.md),
# refuses any SM_PREP that contradicts the convention, self-tests against the authority doc,
# and echoes the resolved value to the job log.
if ! SM_PREP="$("$PY" "$MADGRAV_ROOT/launchers/templates/_prep_guard.py" "$SM_RUN" "${SM_PREP:-}")"; then
  echo "[prep] FATAL — refusing to run under an unverified whitening prep (see above)" >&2
  exit 1
fi
export SM_PREP
export SM_OUT="${SM_OUT:-$SC/search_out_${SM_RUN}_far}"
export SM_SHARD_DIR="${SM_SHARD_DIR:-$SC/_${SM_RUN}_far_shards}"

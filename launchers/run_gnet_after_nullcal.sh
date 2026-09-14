#!/usr/bin/env bash
# Orchestrator: wait for the four per-run null-calibration jsons, merge, derive ke_gnet.json, then run the two
# gated injection chains. Logs to launchers/gnet_after_nullcal.log.
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"; PY=${PY:-python}; DET=details/successor_statistic
while :; do n=0; for r in o3a o3b o4a o4b; do [ -f $DET/perarm_nullcal_excldet_netmax_lronly_gnet_$r.json ] && n=$((n+1)); done; [ $n -eq 4 ] && break; sleep 60; done
echo "[orch] all four null-calibration files present $(date '+%F %T')"
$PY search_mode/merge_nullcal_runs.py excldet_netmax_lronly_gnet || exit 1
$PY search_mode/make_ke_gnet.py || exit 1
echo "[orch] chain 1: adopted campaign p_astro"; launchers/run_vt_47_gnet_chain.sh > launchers/gnet_chain47.log 2>&1 || { tail -20 launchers/gnet_chain47.log; exit 1; }
echo "[orch] chain 2: inj_full recovery + VT"; launchers/injfull_score_gnet.sh > launchers/gnet_injfull.log 2>&1 || tail -20 launchers/gnet_injfull.log
echo "[orch] DONE $(date '+%F %T')"

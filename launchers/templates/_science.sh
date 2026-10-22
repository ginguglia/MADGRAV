#!/usr/bin/env bash
# Frozen SHARED science config for the FAR-stat stages (prune / cnn_shard / far_merge / fg).
# Blind-test config, IDENTICAL across O3a..O4b (frozen held-out model, o3a blind evloc, floor 4.0,
# coh-ceiling 0.999, per-arm, net-gate). Every value overridable by pre-setting it in the environment.
export SM_INJ="${SM_INJ:-$SM/inj_out_o3a}"                     # unused Xs loader under the frozen model (numerically irrelevant)
export SM_EVENTSJSON="${SM_EVENTSJSON:-$SM/o3a_events.json}"   # O3a events not inside these segs -> evloc empty -> fully blind
export SM_LR_MODEL="${SM_LR_MODEL:-${FROZEN_MODEL:-$MADGRAV_ROOT/data/o3a_frozen_lr_off200.npz}}"   # frozen held-out model (site.conf FROZEN_MODEL)
export SM_INJ_BOTHFOLDS="${SM_INJ_BOTHFOLDS:-1}"
export SM_COH_CEIL="${SM_COH_CEIL:-0.999}"
export SM_COH_REWHITEN="${SM_COH_REWHITEN:-0}"               # 1 -> re-whitened coherence series (search_mode/coh_rewhiten.py; 2026-09-18 study: signal-blind coherence fixed)
export SM_COHREF_DIR="${SM_COHREF_DIR:-$SM_SERIESCACHE/_cohref}"  # per-(seg,det) noise reference spectra (build_coh_ref.py)                     # tripwire-guarded lossless prefilter; net-gate does the filtering
export SM_MAX_OFFSETS="${SM_MAX_OFFSETS:-50}"
export SM_CAND_FLOOR="${SM_CAND_FLOOR:-4.0}"
export SM_N_CNN_CANDIDATES="${SM_N_CNN_CANDIDATES:-500}"       # 2026-08-14 amendment: corrected whitening -> ~312 loglr>=4 floor-crossers (~2.6x the 120-cap design assumption); cap must not truncate CNN admission. Shared by cnn_shard + far_merge (same above[:N] selection). Detection floor/statistic/FAR UNCHANGED.
export SM_NETSIG_FLOOR="${SM_NETSIG_FLOOR:-4.0}"
export SM_PERARM="${SM_PERARM:-1}"
export SM_BLIND_FLOOR="${SM_BLIND_FLOOR:-1}"
export DET_FAR="${DET_FAR:-1.0}"
export SM_NET_GATE="${SM_NET_GATE:-1}"
export SM_NO_SHUFFLE="${SM_NO_SHUFFLE:-1}"
export SM_SHARD_MODE="${SM_SHARD_MODE:-block}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
unset SM_VETO

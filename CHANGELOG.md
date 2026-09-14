# Changelog

## 2026-09-14 — final version: injection-set glitch-arm gate, consistent injection-side background

Detection statistic (Sec. 4 of the README)
- Glitch-arm gate `g_net = (gH + gL)/sqrt(2) >= -4.02` after the sigma_net < 10.6 veto. The
  threshold is the 1st percentile of `g_net` over the gate-passing, triggering injections of the
  adopted campaign (`search_mode/gnet_threshold.py` -> `details/successor_statistic/gnet_threshold.json`);
  candidates, detections and background are not read when it is set. Applied identically to
  candidates, background pairs and injections (`SM_GNET`; default off).
- `search_mode/build_bg_g.py`: looks up the cached per-window glitch-arm logits for every
  background pair (`bg_g_<run>.npz`, not committed: rebuilt from the stream caches).
- `successor_stat.py` (`Background.gpass`), `inclusive_exclusive_far.py` (candidate `g_net`,
  `gnet_pass` column), `perarm_nullcal.py` (`SM_RUNS`, `SM_TAG_EXTRA` for per-run jobs, gate on
  background and pseudo-candidates), `merge_nullcal_runs.py`, `make_ke_gnet.py`.
- Null calibration with the gate: `K = 5.60, 4.59, 1.43, 3.62` (`ke_gnet.json`; per-run and merged
  outputs committed). Same 47 detections; summed background count over them 843 -> 508.
- `figures/catalog_o3o4/adopted_set.py` now defaults to the gated table and calibration
  (`far_lronly_g106_gnet.csv`, `ke_gnet.json`); `SM_FAR_LR_CSV`, `SM_KE_JSON` select others.
  The pre-gate files are kept.

p_astro and sensitivity (`search_mode/pastro_final/`)
- `pastro_final.py`: `SM_GNET` on injections and on the background they are scored against;
  `SM_INJ_BG_NETMAX=1` applies the sigma_net < 10.6 veto to that background as well, so injections
  and candidates are counted against the same background (previously the veto acted on injections
  and candidates only). `SM_FAR_LR_CSV` selects the candidate FAR table. Fix: the scoring block
  was nested under the veto branch; production always set the veto, so accepted products are
  unaffected.
- Products: `pastro_final_x1cnnadopt48fg.{json,csv}` (all 47 with p_astro > 0.90, 42 >= 0.99),
  `vt_relabel_comoving_x1cnnfullgveto_m20.json`, `eff_srcframe_x1cnnfullgveto_m20.json`,
  `neff_srcframe_x1cnnfullgveto_m20.json`, `figures/vt_fourepoch/vt_fourepoch_ratio_x1cnnfullgveto_m20.json`
  (+ caption).
- Helpers: `vt_text_numbers.py`, `pastro_text_numbers.py`, `comparison_text_numbers.py`,
  `body_to_tables.py`; `figures/catalog_o3o4/plot_far_final_adopt.py` (`SM_FIG_OUT`).

Launchers
- `run_perarm_nullcal_gnet.sh`, `run_perarm_nullcal_gnet_parallel.sh`, `run_gnet_after_nullcal.sh`,
  `run_vt_47_gnet_chain.sh`, `injfull_score_gnet.sh` (site settings as placeholders).

Inputs previously missing from the repository
- `search_mode/o3a_events.json` (the five O3a seed events) and
  `figures/catalog_o3o4/hl_snr_o3o4.json` (catalogue H1/L1 SNR per event), read by the table
  builder, the demo and the miss diagnostics.

## 2026-09-09 — corrected sensitive volume, unified injection campaign, pipeline fixes

Code (`search_mode/pastro_final/`)
- `vt_relabel_comoving.py`: `SM_VT_COMOVING_PRIOR` (Jacobian J(z) on the rho^-4 weights;
  effective `w0` saved in the relabel file), `SM_VT_MASS_EDGES`, `SM_VT_OUT_SUF`, `SM_RHO_TH`,
  third low-mass bank stratum via `SM_BANK_LM`.
- `vt_pipelines_gwtc.py`: shares `SM_VT_MASS_EDGES` with the numerator.
- `vt_compare_pipelines.py`: support mask now read from the same suffixed campaign as the VT
  (previously an unsuffixed, older campaign); effective `w0` preferred; x-axis follows the grid.
- `build_neff.py`, `build_eff_srcframe.py`: prefer the relabel layer's effective `w0`.
- `pastro_final.py`: `SM_SNR_GRID_EXT`, `SM_FAR_SWEEP` (`det_frac_sweep`), `SM_PASTRO_W0_FROM`.
- `build_inj_veto.py`: `SM_VETO_NOREF` for merged campaigns; sweep column vetoed alongside.
- `fig_fourepoch_ratio.py`, `vt_pipelines_target_zc.py`: `SM_VT_TGT_SUF`; title states whether
  the two sides are matched; `SM_FIG_NOTITLE`.
- New `vt_pipelines_target_rho.py` (comparator with a matched H1+L1 optimal-SNR restriction).

Code (search and training)
- `search_mode/inject.py`: low-mass stratum (`SM_BANK_LM`, `SM_LM_FRAC`), full-signal SNR labelling
  (`SM_INJ_SNR_FULL`), `SM_UM_FRAC`, `SM_CALIB` calibration override.
- `improved/improved_pipeline.py`: Q-transform tile cache written to a private partial file and
  published atomically, plus an all-zero-tile guard on load (a concurrent job could previously
  read a preallocated, still-empty cache); `SM_SELECT_AXIS` for checkpoint selection.
- Added `spectrogram_cascade/` (deployed scoring cascade + frozen BA calibration) and `tools/`
  (waveform-bank builders), both imported by the injection engine.

Products
- `*_x1cnnfullveto{,_m20}.json`, `vt_pipelines_target_zc_lm10.json`,
  `pastro_final_x1cnnfullcp.*`, `pastro_final_x1cnnfullsw.*` (see README Sec. 6).

Launchers
- `run_injfull.sh` + `injfull_manifest.txt`, `injfull_score.sh`, `run_vt_47_chain.sh`,
  `run_vt_full_chain.sh`, `run_lowmass_bank.sh`, with site-specific SLURM settings as placeholders.

Fixes to the 2026-09-01 snapshot
- In 26 scripts the `MADGRAV_ROOT`/`MADGRAV_SCRATCH` resolution block had been inserted after the
  first use of those names, so the scripts raised `NameError` on import. The block now precedes
  the first use in every file (checked by AST for the whole tree).

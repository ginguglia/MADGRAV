# Changelog

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

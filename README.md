<p align="center">
  <img src="assets/madgrav_logo.png" alt="MADGRAV" width="620">
</p>

<p align="center">
  <em>A deep-learning search for high-mass compact binary coalescences in LIGO data.</em>
</p>

This repository contains the pipeline configuration that produced **48 candidates above the
detection threshold, of which 47 have a calibrated false-alarm rate below 1 yr⁻¹**, in a search
over O3a, O3b, O4a and O4b Hanford–Livingston data. This is the **final version** of the
pipeline: the detection statistic of Sec. 4 (with the injection-set glitch-arm gate) and the
products it selects are the ones the paper reports.

MADGRAV is a cascade of convolutional networks — anomaly detection, glitch classification,
coherence and signal ranking — operating on 
Q-transform tiles. **No waveform template bank is
matched-filtered against the data.** Waveform models enter only through the training distribution
and through the injection campaigns used for calibration and sensitivity estimation. 

>[!Important]
>**MADGRAV currently operates on a 2-detector configuration including LIGO detectors (MADGRAV-2D), a single-detector (MADGRAV) version and a three-detector configuration including LIGO and Virgo (MADGRAV-3D) are in preparation.**

---

## 1. Where your data lives

Nothing is hardcoded. Two environment variables control every path:

```bash
export MADGRAV_ROOT=/path/to/this/repo      # code, weights, small reference products
export MADGRAV_SCRATCH=/path/to/big/data    # strain, injection sets, background caches
```

A third variable, `MADGRAV_EXTDATA`, points at external releases that are not part of this
repository — the GWOSC event-API cache and the LVK sensitivity-injection releases used for the
pipeline comparison:

```bash
export MADGRAV_EXTDATA=/path/to/external    # expects gwosc_eventapi/ and gwtc5_sensitivity/
```

`MADGRAV_ROOT` defaults to the repository root, so **after a clone it already works** and you only
need to set it if you run scripts from elsewhere. `MADGRAV_SCRATCH` defaults to
`$MADGRAV_ROOT/scratch` and `MADGRAV_EXTDATA` to the parent of `MADGRAV_ROOT`; set them to
filesystems with room for the strain and background products, which run to hundreds of GB for a
full observing run.

Copy `site.conf.example` to `site.conf` and edit it for anything site-specific.

## 2. Install

```bash
python -m venv madgrav-venv && source madgrav-venv/bin/activate
pip install numpy scipy matplotlib torch gwpy astropy
```

A GPU is needed for the search itself; the post-processing runs on CPU.

## 3. What is here

```
search_mode/                 the search: triggers, streams, background, ranking
  driver_search_multi.py       foreground search over a segment list
  driver_streams_bg.py         time-slide background construction
  driver_blindscan.py          blind scan driver
  successor_stat.py            the adopted ranking statistic
  inclusive_exclusive_far.py   FAR against inclusive / foreground-excluded background
  perarm_nullcal.py            null calibration: measures the K factors
  build_bg_g.py                glitch-arm scores of every cached background pair (bg_g_<run>.npz)
  gnet_threshold.py            glitch-arm gate threshold from the injection population
  merge_nullcal_runs.py, make_ke_gnet.py   per-run null calibration -> ke_gnet.json
  apply_asd_veto.py            local-spectrum consistency veto
search_mode/pastro_final/    calibration, p_astro, sensitivity
  pastro_final.py              FGMC p_astro on the FAR axis
  vt_relabel_comoving.py       comoving sensitive volume-time (source-frame rebin)
  build_inj_veto.py            folds the veto into injection scoring
  build_neff.py, build_eff_srcframe.py   N_eff and efficiency per source-frame mass bin
  vt_pipelines_gwtc.py, vt_pipelines_target_zc.py   LVK sensitivity releases reweighted to our population
  vt_compare_pipelines.py, fig_fourepoch_ratio.py   cross-pipeline VT and the four-epoch ratio figure
  *_x1cnnadopt48fg*.{json,csv} final p_astro products (inj_fixed2 campaign, gated; Sec. 5)
  *_x1cnnfullgveto_m20.json    final sensitive-volume products (inj_full campaign, gated; Sec. 6)
  *_x1cnnadopt48f*.json, *_x1cnnfullveto*.json   the same products before the gate (superseded)
  *_text_numbers.py, body_to_tables.py   digests of the quoted numbers and LaTeX tables
figures/catalog_o3o4/          detection list: adopted_set.py, far_lronly_g106_gnet.csv (final FAR table),
                               plot_far_final_adopt.py (FAR vs mass figure)
details/successor_statistic/   gnet_threshold.json, ke_gnet.json, null-calibration outputs
search_mode/inject.py          injection engine (SNR grid, mass strata, CNN gate, ASD veto)
spectrogram_cascade/           the deployed scoring cascade and its frozen BA calibration
improved/improved_pipeline.py  the CAE training script and Q-transform tile cache
tools/                         waveform-bank builders (stellar / ultramassive / low-mass strata; needs pycbc)
launchers/                     shell and SLURM chains that produced the committed products
                               (account/partition are placeholders; paths resolve from MADGRAV_ROOT)
assets/models/, lr_cascade/    deployed network weights
```

## 4. The adopted detection criterion

A candidate is a detection when its **calibrated** false-alarm rate is below 1 yr⁻¹:

    sigma_net > 4 trigger  ->  CNN glitch gate  ->  sigma_net < 10.6 veto  ->  glitch-arm gate
      ->  lnLambda-channel per-arm FAR against the foreground-excluded time-slide background
      ->  multiplied by the per-run null-calibration factor K

Both selections after the CNN gate are set on the injection population alone and applied
identically to candidates and to the time-slide background:

* **sigma_net < 10.6**: the 99.9th percentile of gate-passing injections.
* **glitch-arm gate** `g_net = (gH + gL)/sqrt(2) >= -4.02`: the 1st percentile of `g_net` over the
  57,641 gate-passing, triggering injections of the `inj_fixed2` campaign, pooled over the four
  runs (`gnet_threshold.py` -> `gnet_threshold.json`). `g` is the deploy-arm glitch logit already
  cached per 1-s window per detector; it enters lnLambda as a ramped feature and here also as a
  gate. The gate removes 37% of the 532 background families that set the detection FARs and
  keeps all 47 detections.

`K = 5.60, 4.59, 1.43, 3.62` for O3a, O3b, O4a, O4b, measured by scoring every gate-passing
background family as a pseudo-foreground candidate (`perarm_nullcal.py` with `SM_GNET`;
`ke_gnet.json`). The pre-gate values `K = 5.54, 4.54, 2.34, 5.46` (`ke_adopted.json`,
`far_lronly_g106.csv`) select the same 47 events and are kept for reference.

The 90% upper limit is reported per event but is not part of the criterion.

## 5. Reproducing the detection list

```bash
export MADGRAV_ROOT=$PWD
cd figures/catalog_o3o4
python -c "import adopted_set; print(len(adopted_set.load()))"    # -> 47
```

`adopted_set.py` reads `far_lronly_g106_gnet.csv` and `ke_gnet.json` by default; `SM_FAR_LR_CSV`
and `SM_KE_JSON` select other tables (e.g. the pre-gate ones).

The full chain, given the background caches, stream caches and injection campaigns in
`$MADGRAV_SCRATCH`:

```bash
DET=details/successor_statistic; SM=search_mode/pastro_final
python search_mode/build_bg_g.py                       # glitch-arm scores of the background pairs
python search_mode/gnet_threshold.py                   # -> $DET/gnet_threshold.json  (-4.02)
launchers/run_perarm_nullcal_gnet_parallel.sh          # null calibration per run, with the gate
python search_mode/merge_nullcal_runs.py excldet_netmax_lronly_gnet
python search_mode/make_ke_gnet.py                     # -> $DET/ke_gnet.json
SM_GNET=-4.02 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_EXCL_DET=1 \
  python search_mode/inclusive_exclusive_far.py        # -> figures/catalog_o3o4/far_lronly_g106_gnet.csv
launchers/run_vt_47_gnet_chain.sh                      # p_astro on inj_fixed2 -> *_x1cnnadopt48fg
launchers/injfull_score_gnet.sh                        # recovery, efficiency, VT on inj_full -> *_x1cnnfullgveto_m20
```

The p_astro step inside the chains is

```bash
SM_TRIALS_OFF=1 SM_INJ_CNN_GATE=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_INJ_BG_NETMAX=1 \
SM_GNET=-4.02 SM_KE=$DET/ke_gnet.json SM_FAR_LR_CSV=figures/catalog_o3o4/far_lronly_g106_gnet.csv \
SM_INJ_DIR=inj_fixed2 SM_SUF_EXTRA=adopt48fg \
SM_ADOPTED_AXIS=1 SM_AXIS_COL=far_excl SM_ADMIT=all SM_DET_RULE=far \
  python $SM/pastro_final.py
```

Key switches: `SM_LR_ONLY` ranks on the lnLambda channel alone; `SM_NETMAX` applies the sigma_net
veto to candidates and injections alike, and `SM_INJ_BG_NETMAX=1` applies it also to the
background the injections are counted against (the same background the candidates see);
`SM_GNET` is the glitch-arm gate threshold; `SM_KE` supplies the calibration factors;
`SM_DET_RULE` sets the injection admission rule so the signal model matches the detection list.
All gate switches default **off**, so the pre-gate products rebuild unchanged.

## 6. Sensitive volume on the unified injection campaign

The adopted chain above draws its injections from `inj_fixed2`, a two-stratum campaign on a
network-SNR grid capped at rho = 25. Two corrections were identified and implemented on
2026-09-05; both are switches that default **off**, so every adopted product rebuilds
byte-identically, and the corrected products are committed under their own tag:

* **`inj_full` campaign** (`launchers/run_injfull.sh`): one campaign, 321,750 injections, 13 SNR
  levels over rho 5-80 and three mass strata (low-mass 10-22, stellar, ultramassive). The rho^-4
  population weights are built with midpoint bin widths and renormalised over the grid, so on
  the capped grid the dropped high-SNR tail was redistributed onto low-SNR levels where the
  efficiency is ~0, biasing VT low. `SM_SNR_GRID_EXT` extends the grid; `SM_BANK_LM`,
  `SM_LM_FRAC`, `SM_INJ_SNR_FULL` add the third stratum and label it by full-signal SNR.
* **Comoving prior** (`SM_VT_COMOVING_PRIOR=1` in `vt_relabel_comoving.py`): the grid weights are
  uniform in Euclidean volume while the reference volume is comoving with (1+z) dilation; the
  Jacobian J(z_i) puts both in the same measure. The effective per-injection weight is stored in
  `relabel_inj_<run><suffix>.npz["w0"]` and read back by `build_neff.py`, `build_eff_srcframe.py`
  and `vt_compare_pipelines.py`. `SM_PASTRO_W0_FROM=<suffix>` routes the same weights into
  `pastro_final.py` (product `pastro_final_x1cnnfullcp.json`; effect on p_astro is negligible).
* **Mass grid**: `SM_VT_MASS_EDGES` sets the source-frame bin edges in both the numerator
  (`vt_relabel_comoving.py`) and the comparator (`vt_pipelines_gwtc.py`) from one switch;
  `SM_VT_OUT_SUF` / `SM_VT_TGT_SUF` name the outputs so nothing adopted is overwritten.

Products: `vt_relabel_comoving_x1cnnfullveto_m20.json` (20-400 grid, the cross-pipeline ratio
numerator), `vt_relabel_comoving_x1cnnfullveto.json` (10-400 grid, MADGRAV-only frames figure),
the matching `eff_srcframe_*`, `neff_srcframe_*`, `vt_compare_pipelines_*` files, the 13-edge
comparator `vt_pipelines_target_zc_lm10.json`, and `pastro_final_x1cnnfullcp.{json,csv}` /
`pastro_final_x1cnnfullsw.{json,csv}` (the latter adds `det_frac_sweep`, detected fraction per
calibrated-FAR threshold via `SM_FAR_SWEEP`). Bins below 20 Msun are not usable for the
cross-pipeline ratio: the comparator releases give N_eff below the 300 cut there, and the
source-frame rebin and the release's native source-frame binning treat the bin boundary
differently.

`launchers/run_vt_full_chain.sh` re-states the full sequence with these switches;
`launchers/injfull_score.sh` is the scoring step that precedes it.

**Final products (2026-09-09).** `launchers/injfull_score_gnet.sh` runs the same chain with the
glitch-arm gate and `SM_INJ_BG_NETMAX=1`, producing `vt_relabel_comoving_x1cnnfullgveto_m20.json`,
`eff_srcframe_x1cnnfullgveto_m20.json`, `neff_srcframe_x1cnnfullgveto_m20.json` and
`figures/vt_fourepoch/vt_fourepoch_ratio_x1cnnfullgveto_m20.json`; these are the numbers the paper
quotes.

## 7. Data not in this repository

Training sets, injection campaigns, strain and background caches are hundreds of GB and live on
`$MADGRAV_SCRATCH`. Strain is public via GWOSC (https://gwosc.org). The LVK sensitivity-injection
releases used for the pipeline comparison are on Zenodo (GWTC-3: 10.5281/zenodo.5546676).

## 8. Citation

See `CITATION.cff`.

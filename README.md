# MADGRAV

**M**ultilevel **A**nomaly **D**etection for **GRAV**itational-wave science — a portable, cluster-deployable
(SLURM) blind gravitational-wave search.

## Overview

MADGRAV is a multilevel anomaly-detection search: a frozen convolutional autoencoder (CAE) front end →
LR / glitch-arm cascade → CNN (high-mass + low-mass specialists) + coherence → time-slide-calibrated FAR.
This package is configured for the **O3a 56-segment validation** run (the accepted config:
candidate floor 4.5 / net-σ floor 4.0 / per-arm CNN ranking / blind floor / coherence ceiling 0.85).
It **scales to the full O3a (485-segment) analysis** by swapping the segment list (`SM_BGJSON` /
`SM_SEGJSON_EV` / `SM_VETO`) and growing the shard count / SLURM array size — no code changes.

All science constants, thresholds, the device-resolution logic, and the ranking statistic are unchanged
from the dev pipeline; only paths were made portable via a single `MADGRAV_ROOT` root.

## Layout

```
madgrav/
  README.md  environment.yml  .gitignore  check_install.py
  search_mode/          # core drivers + small vendored .pt/.json assets
  spectrogram_cascade/  # massive_pipeline.py + massive_calibration_BA.json
  improved/             # improved_pipeline.py, prepare_o1_data.py, utilities.py (vendored)
  lr_cascade/p1v42/     # arm_deploy_seed0..4.pt (5-seed glitch arm)
  assets/models/        # baseline_cae_weaksup_best.pt (frozen CAE)
  data/o3a_search_prep/ # reference_psd_H1.npz, reference_psd_L1.npz (run-matched ASD)
  launchers/            # run_search.sh, slurm_search.sbatch, run_merge.sh
```

## Install

```bash
conda env create -f environment.yml
conda activate madgrav
export MADGRAV_ROOT=$(pwd)        # the package root
```

Pin `torch` + CUDA in `environment.yml` to the build the frozen weights were trained/calibrated with.
Do **not** add `ml4gw` (its whitening changes the coherence statistic and the results).

## Data provisioning

The cluster has **no strain data shipped with this package** (~262 GB). Build it once, in order, from
within `$MADGRAV_ROOT` (set `MADGRAV_ROOT` first):

1. `python search_mode/prep_o3a_56.py` — build the segment / veto JSONs (already vendored for the 56-seg
   config; re-run to regenerate or to build a different segment set).
2. `python search_mode/fetch_locks.py` then `python search_mode/fetch_bg.py` — pull ~262 GB of O3a strain
   from GWOSC into `search_mode/strain_o3a/`. (`fetch_bg_par.py` / `fetch_bg_resilient.py` are
   parallel / resume-safe variants.)
3. `python search_mode/inject.py --event NAME` — build injection priors (efficiency / recovery).
4. `python search_mode/driver_streams.py` — build the per-detector streams.
5. `python search_mode/build_series_cache_o3a.py` — build the ~130 GB whitened coherence-series cache
   (reclaimable page-cache; reused across all shards).

## Run

Single node (bash, no systemd):
```bash
export MADGRAV_ROOT=$(pwd)
bash launchers/run_search.sh          # runs SM_NSHARD sequential shards, then the merge
```

SLURM (one shard per array task):
```bash
export MADGRAV_ROOT=$(pwd)
jid=$(sbatch --parsable launchers/slurm_search.sbatch)
sbatch --dependency=afterok:${jid} --wrap="bash launchers/run_merge.sh"
```
Edit the `#SBATCH` placeholders (`--partition`, `--account`, `--time`, `--mem`, `--cpus-per-task`) and keep
`--array=0-15` in sync with `SM_NSHARD`. Results land in `search_mode/search_out_o3a_56_perarm/`
(`blindscan.json`, `detections.json`, `survivors_bg.json`).

Knobs (env): `MADGRAV_PY` (python), `BLIND_DEV` (GPU, default `cuda:1`; the device logic degrades gracefully
and allows CPU only with `SM_ALLOW_CPU=1`), `SM_NSHARD`, `SM_HOST_MEM_GB`.

## Verify

```bash
cd $MADGRAV_ROOT
MADGRAV_ROOT=$(pwd) SM_ALLOW_CPU=1 python check_install.py
```
Checks every vendored asset is present and the core module closure imports. It does **not** run the
pipeline, fetch data, or require strain.

## Notes

- The frozen weights (CAE, glitch arm, HM/LM CNNs) are **calibration-locked**. The **GPU forward pass is the
  calibrated path**; CPU forward is **not** byte-identical, so a production / FAR run must run on GPU
  (`SM_ALLOW_CPU=1` is for install checks only).
- Do **not** add `ml4gw` — it changes the coherence statistic and the results.
- Whitening uses the run-matched reference ASD in `data/o3a_search_prep/`.

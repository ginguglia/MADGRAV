# MADGRAV 2

Version 2 of MADGRAV, a deep-learning search for compact binary coalescences in LIGO Hanford–Livingston data. This
repository holds the search code, the SLURM chain, the model weights and calibration files that define the version,
and the scripts that trained the version-2 models. The run over O3a, O3b, O4a and O4b with this configuration gives
96 catalogue detections at a false-alarm rate below 1 yr⁻¹ (O3a 16, O3b 8, O4a 32, O4b 40); the detection table is
`m2/vt96/detections_v2_96.csv` and the merge outputs are under `scratch/madgrav2/search_out_<run>_m2_gx_off5_crop/`.

MADGRAV is a cascade of convolutional networks on Q-transform tiles of whitened strain: an anomaly-detecting
autoencoder trigger, a glitch-classification arm, a guided second-leg excess and coherence as ranking features, and
high-mass / low-mass specialist CNNs. No template bank is matched-filtered against the data.

## 1. Paths

```bash
cp site.conf.example site.conf        # edit
export MADGRAV_ROOT=/path/to/this/repo
export MADGRAV_SCRATCH=/path/to/big/storage   # default $MADGRAV_ROOT/scratch; the shipped models live there
```

Every script resolves `MADGRAV_ROOT`, `MADGRAV_SCRATCH` and `MADGRAV_EXTDATA` at the top of the file.

## 2. Install

```bash
conda env create -f environment.yml
MADGRAV_ROOT=$(pwd) SM_ALLOW_CPU=1 python check_install.py
```

CNN inference runs on GPU; streams, pruning, merges, scoring and injections run on CPU.

## 3. Layout

| path | content |
|---|---|
| `search_mode/` | streams, injections, LR refit, FAR pruning and merge, local-ASD consistency veto, guided excess, coherence; CNN gate weights `hm_native_seed0.pt`, `lm_native_seed0.pt`; catalogue list `events_gwtc_all.json`; segment lists |
| `improved/`, `spectrogram_cascade/` | the autoencoder front end and its calibration `massive_calibration_BA.json` |
| `assets/models/`, `lr_cascade/p1v42/` | the frozen autoencoder and the first-generation glitch arm (inherited from MADGRAV 1, `main` branch) |
| `launchers/` | per-run SLURM chain: `m2_run_chain.sh`, `m2_o4a_off5_chain.sh`, `m2_o4b_off5_chain.sh`, `m2_off5_cpumerge.sbatch`, `m2_inject_*`, `m2_refit_lr.sbatch`, `templates/` |
| `m2/arm/` | 20-model glitch arm: `train_arm.py`, `run_train_arm.sbatch`, assembly and pair rule |
| `m2/positives/` | training positives and glitch bank generation for the arm |
| `m2/centroid/` | centroid gate |
| `m2/spec/` | S3 specialist CNNs: bank generation (`gen_bank_v2.py`), rendering, `train_spec.py`, scoring |
| `m2/rank/` | ranker: feature building, `train_rank.py`, `train_prod.py`, `s3_threshold.py`, merge hook `driver_blindscan_merge_hook.py` |
| `m2/gx/` | guided second-leg excess |
| `m2/asdveto/` | local-ASD veto scripts and plateau metric |
| `m2/dq/` | data-quality and hardware-injection flag tooling, veto mask builder |
| `m2/popb_cap/` | loglr cap bypass and net-sigma guard (`occlusion/chain/netsig_guard.py`) |
| `m2/vt96/`, `m2/vt96p/`, `m2/vt22/` | injection scoring (`far22.py`), VT tables and figures |
| `m2/pastro22/` | p_astro |
| `m2/freeze/v2_96/` | hash manifest of every model, calibration file and product of version 2 |
| `scratch/madgrav2/`, `scratch/madgrav2_spec/` | shipped weights and products: arm models, LR models, veto masks, S3 specialists, rankers, merge outputs |
| `data/<run>_search_prep/` | per-run reference PSD and whitening |

## 4. Detection criterion

Coincident H1–L1 streams on CAT1 data with the CAT2 ∪ hardware-injection veto mask; autoencoder sigma per leg with the
220 Hz centroid gate; glitch-arm pair rule max g2 ≥ −2.05; net floor 2.75; loglr channel with the S3 specialist gate
(threshold 0.5357) and the ranker statistic; net-sigma channel with the CNN gate (coincident-argmax crop), LM rule
cnn_lm > 0.0382 and the net-sigma guard; per-arm FAR with point FAR < 1 yr⁻¹ and Poisson UL90 < 1 yr⁻¹; local-ASD
consistency veto (net clause always, CNN clause where the plateau metric exceeds 0.04); loglr cap bypass at L = 22.99.

## 5. Running

```bash
source site.conf
DRY=1 launchers/m2_run_chain.sh o4a          # print the chain
DRY=0 STAGES=all launchers/m2_run_chain.sh o4a
```

Production environment: `SM_CROP=coinc`, `SM_ARM_V2=1`, `SM_ARM_V2_TAG=_off5`, `SM_GX=1`, `SM_NET_CUT=2.75`,
`SM_PREP=$MADGRAV_ROOT/data/<run>_search_prep`. Merges: `launchers/m2_off5_cpumerge.sbatch`. Injection scoring and VT:
`m2/vt96/`, `m2/vt96p/`.

## 6. Training (version-2 models)

- Glitch arm: `m2/positives/` (positives, glitch bank) then `m2/arm/run_train_arm.sbatch`.
- S3 specialists: `m2/spec/gen_bank_v2.sbatch`, `render_*.py`, `train_spec.py`, `assemble_spec.sbatch`.
- Ranker: `m2/rank/build_feats.sbatch`, `train_rank.py`, `train_prod.py`, `s3_threshold.py`.
- LR: `search_mode/refit_lr_m2.py` via `launchers/m2_refit_lr.sbatch`.
- The autoencoder and the CNN gate are the frozen MADGRAV 1 models; their training is on the `main` branch of this repository (MADGRAV 1).

## 7. Not in this repository

| group | path (site-relative) | size | sha256 |
|---|---|---|---|
| s3_bank_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/s3_crop/s3_o3a.npz` | 111 MB | `e908ebcc1c133dbb…` |
| s3_bank_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/s3_crop/s3_o3b.npz` | 37 MB | `25ecd8e8814e41b0…` |
| s3_bank_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/s3_crop/s3_o4a.npz` | 130 MB | `4e532241b90300ca…` |
| s3_bank_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/s3_crop/s3_o4b.npz` | 45 MB | `3d0686086eb559ff…` |
| stat_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/rank_crop/stat_o3a.npz` | 77 MB | `361b2eb790a3f918…` |
| stat_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/rank_crop/stat_o3b.npz` | 26 MB | `1eb4b2ac0222ef56…` |
| stat_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/rank_crop/stat_o4a.npz` | 88 MB | `f66654854181a1d5…` |
| stat_crop | `${MADGRAV_SCRATCH}/madgrav2_spec/rank_crop/stat_o4b.npz` | 31 MB | `b86e610002fa3c61…` |
| far_shards | `${MADGRAV_SCRATCH}/madgrav2/_o3a_m2_gx_off5_nf275_far_shards` | 5381 MB | `?…` |
| far_shards | `${MADGRAV_SCRATCH}/madgrav2/_o3b_m2_gx_off5_nf275_far_shards` | 3097 MB | `?…` |
| far_shards | `${MADGRAV_SCRATCH}/madgrav2/_o4a_m2_gx_off5_nf275_far_shards` | 7104 MB | `?…` |
| far_shards | `${MADGRAV_SCRATCH}/madgrav2/_o4b_m2_gx_off5_nf275_far_shards` | 4490 MB | `?…` |
| cnn_cache | `${MADGRAV_SCRATCH}/madgrav2/search_out_o3a_m2_gx_off5_crop` | 135 MB | `?…` |
| cnn_cache | `${MADGRAV_SCRATCH}/madgrav2/search_out_o3b_m2_gx_off5_crop` | 106 MB | `?…` |
| cnn_cache | `${MADGRAV_SCRATCH}/madgrav2/search_out_o4a_m2_gx_off5_crop` | 161 MB | `?…` |
| cnn_cache | `${MADGRAV_SCRATCH}/madgrav2/search_out_o4b_m2_gx_off5_crop` | 126 MB | `?…` |

Strain is public via GWOSC (https://gwosc.org). Training sets and injection campaigns live on `$MADGRAV_SCRATCH`.

## 8. Citation

See `CITATION.cff`.

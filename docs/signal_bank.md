# Projected-waveform signal banks

The optional injection / `p_astro` stage draws simulated signals from two **projected-waveform template
banks**. These are a **user-provided input** — deliberately *not* shipped: they are multi-GB, and the
source population behind them is a scientific choice you control. This document specifies their on-disk
format and the population the published analysis assumed, so you can supply your own bank (or reproduce
the paper's).

Skip all of this by running the FAR search with `run_chain.sh <run> --far-only` — the search and its
time-slide FAR are fully reproducible **without** the banks; only `p_astro` / efficiency need them.

**The astrophysics is applied downstream, not in the bank.** The bank is just a pool of projected
waveforms spanning the mass range. The astrophysical population is imposed as **importance weights** in
the FGMC stage (`lr_cascade/pastro_fgmc.py`): a **GWTC-3 Power-Law+Peak mass model with a flat-in-log
high-mass extension**, with SNR drawn ∝ ρ⁻⁴ (the uniform-in-comoving-volume leading behaviour,
`PASTRO_SPEC §4`). So a valid bank only has to **cover** the mass range densely enough for that
reweighting — you do not bake a population into the waveform pool.

## The two banks

| bank dir (under `data/`, or `SM_BANK_*`)      | contents                                  |
|-----------------------------------------------|-------------------------------------------|
| `o1_o3_signal_bank_projected_2s_x10/`         | stellar-mass CBC waveforms (`SM_BANK_SIG`) |
| `ultramassive_bank/`                          | high-mass / intermediate-mass BH waveforms (`SM_BANK_UM`) |

## On-disk format (as read by `improved_pipeline.load_o1_signal_bank`)

```
<bank_dir>/
  **/signals_*.npz        # globbed recursively, sorted
  **/signals_*.csv        # optional sidecar, same basename as the .npz
```

Each `signals_*.npz` holds one chunk of sources:

- key `"H1"` — `float32` array, shape `(n_sources, N)`: the waveform **projected onto H1**
- key `"L1"` — `float32` array, shape `(n_sources, N)`: the same source projected onto L1

`N` is the sample length of a **2 s** window at **4096 Hz** (i.e. `N = 8192`), time-aligned so the merger
sits at the window centre (injections are added at a random sub-stride offset by `inject.py`).

The optional per-chunk `.csv` (same basename) carries source metadata, one row per source, columns:

- `source_id` — provenance label
- `total_mass` — source-frame total mass (M☉); used for the mass-stratified draw
- `distance_mpc` — luminosity distance the waveform was scaled to
- (`projection_mode` — optional; recorded in the loader's `projection_modes` set)

If the `.csv` is absent the loader still works but `total_mass` is `NaN` (the mass-stratified 50/50
stellar/ultramassive split then loses its mass labelling).

## Providing / generating a bank

Any waveform set in the format above works. Build one with your preferred toolchain (`pycbc` / `bilby` /
`lalsuite`); the format — not a specific generator — is the contract:

1. Draw a source list covering the mass range. The stellar bank should span the O1–O3 CBC range; the
   ultramassive bank should reach high total mass (Mtot into the IMBH range, so GW231028 / GW231123-like
   sources have support). Coverage/density matters, **not** the shape — the population shape is applied by
   the FGMC reweighting (above). (The reference bank name `..._2s_x10` denotes 2 s windows and a 10×
   augmentation of the base source list.)
2. Generate each waveform with an approximant of your choice (the reference analysis used a precessing
   IMR model), at `fs = 4096 Hz`, cropped/windowed to 2 s centred on merger.
3. Project onto H1 and L1 (antenna response for the drawn sky position / polarization), scale to
   `distance_mpc`.
4. Stack per chunk and write `signals_<k>.npz` with `H1` / `L1` `float32` arrays; write the matching
   `signals_<k>.csv` with `source_id, total_mass, distance_mpc` rows.

Point the pipeline at your bank via `SM_BANK_SIG` (stellar) and `SM_BANK_UM` (ultramassive), or place it
at `data/o1_o3_signal_bank_projected_2s_x10/` and `data/ultramassive_bank/`.

**Reproducing the published `p_astro` / VT** requires the *same* bank + population the paper used; a
different bank changes the (already `PROVISIONAL`) `p_astro` and sensitive-volume numbers. To just run the
search, use `--far-only`.

Validate any bank against the loader before use:

```python
from improved.improved_pipeline import load_o1_signal_bank
b = load_o1_signal_bank("data/o1_o3_signal_bank_projected_2s_x10")
print(len(b["H1"]), b["H1"][0].shape, b.get("total_mass")[:3])
```

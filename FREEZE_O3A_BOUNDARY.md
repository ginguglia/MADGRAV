# MADGRAV configuration freeze — O3a→O3b boundary

**Frozen:** 2026-07-04 (Europe/Vienna)
**Purpose:** This document + the commit that carries it are the pre-registration point for the
MADGRAV search-mode pipeline. O3a was the **development/validation** set (the pipeline was tuned and
understood there). Everything below is now **fixed** and MUST NOT be altered through the O3b blind test
or the O4a/O4b blind search. Any change after this point invalidates the blind claim for that run.

---

## 1. Detection statistic — FROZEN: HM+LM, per-arm

- **`SM_PERARM=1`** — per-arm statistic.
  `best_far = 2 × min(far_lr_perarm, far_net_perarm)`, where
  `far_{lr,net}_perarm = 2 × min(far_hm, far_lm)`.
- **Two CNN specialist arms, OR-combined** (`kept_by = "HM"+"+LM"`):
  - **HM** — high-mass / IMBH specialist (native-res, ~20–140 Hz band).
  - **LM** — low-mass / chirp specialist (~50–500 Hz band).
- **Two ranking channels**: `loglr` and `net-sigma`; the ×2 channel-trials factor is retained as run.

### Why HM+LM and not HM-only (decided on background/injection grounds, NOT on GW190521)
Of the 9 O3a foreground candidates, the **LM arm is the deciding (lower-FAR) arm for 6 of 9**; only 3
are HM-driven. Dropping the LM arm (HM-only) would lose GW190727_060333 outright (HM FAR 5.7/yr vs
LM 0.065/yr) and degrade GW190408/GW190602. The LM arm carries the majority of detection sensitivity,
so HM+LM is the principled freeze. **GW190521-IMBH remaining a near-miss (~1.2/yr) is a consequence of
this choice, not a reason for it** — HM-only would rescue it, which is exactly why HM-only is disallowed
(that would be tuning the statistic on a single target event).

---

## 2. Frozen numerical config (source: `launchers/o3a_cnn_shard_f40.sbatch`)

| Parameter | Value | Meaning |
|---|---|---|
| `SM_PERARM` | `1` | per-arm HM+LM statistic (§1) |
| `SM_CAND_FLOOR` | `4.0` | loglr candidate floor |
| `SM_NETSIG_FLOOR` | `4.0` | net-sigma candidate floor |
| `SM_BLIND_FLOOR` | `1` | blind-floor mode on |
| `DET_FAR` | `1.0` | detection threshold, /yr |
| `SM_NET_GATE` | `1` | background gated at net>4.0, pinned to the foreground trigger cut |
| `SM_COH_CEIL` | `0.999` | coherence prefilter ceiling (tripwire-guarded lossless superset) |
| `SM_NO_SHUFFLE` | `1` | deterministic pairing order |
| `SM_SHARD_MODE` | `block` | locality-preserving sharding |
| `SM_MAX_OFFSETS` | `50` | time-slides → ~370 yr/fold (fold0) + ~405 yr/fold (fold1) |
| `SM_FG_MAXNET` | `400` | cap on CNN-scored net candidates |
| `SM_VETO` | unset | CAT2 applied as later veto mask (per coincident JSON) |
| `SM_LR_MODEL` | `o3a_frozen_lr_off200.npz` | frozen held-out LR model |
| injection model | `search_mode/o3a_events.json` | 5-event cross-fit seed set |
| whitening | averaged-ASD Welch, streams as built (`o3a_streams_gpu.sbatch`) | 1 s-stride bg math |
| sample rate | 4096 Hz | |

Cross-fit: 2 GPS-grouped folds, candidate in fold *i* ranked against fold *1−i* background
(no train-on-test). No `SM_INJ_BOTHFOLDS` in production (that was a preview-only optimism).

---

## 3. Frozen code state

The statistic above lives in these files; the freeze commit captures them exactly as the O3a FAR run
(SLURM job 1521406) executed them:

- `search_mode/driver_blindscan.py`
- `search_mode/driver_blindscan_merge.py`
- `search_mode/_pb_cnn_precompute.py`
- `search_mode/_pb_shard_common.py`
- `search_mode/driver_search_multi.py`

Identified by annotated tag `madgrav-freeze-o3a-20260704`
(resolve to the exact commit with `git rev-list -n1 madgrav-freeze-o3a-20260704`).

### Model artifact integrity (sha256)

The statistic depends on three trained-weight files, frozen at these exact hashes:

| artifact | role | sha256 |
|---|---|---|
| `o3a_frozen_lr_off200.npz` | held-out LR ranking model | `fe817cd8ec80353fad641ad9b8102095fadc0cf02c088e6d9508270b2f8ff0fe` |
| `search_mode/hm_native_seed0.pt` | HM (high-mass) CNN arm | `37a57c50bb1f92ccdf9e507285ed5a56932c21635b3274512b39d634abe966f5` |
| `search_mode/lm_native_seed0.pt` | LM (low-mass) CNN arm | `91996c07503f88c18f7ea92f7db154b3bd70be358994f47c4cb79b9859354ea1` |

Any O3b/O4 run must load weights matching these hashes or the freeze is broken.

---

## 4. O3a disposition recorded at freeze (development set, non-quotable point FAR)

93.75%-background lo/hi snapshot; the 16/16 merge was still finalizing at freeze time. 9 foreground
candidates, all GPS-confirmed catalog events, zero false positives. 4/5 seeded targets recovered;
GW190521-IMBH is the near-miss (#10, just outside). Final point FAR pending merge completion — the
freeze fixes the *method*, not these interim numbers.

---

## 5. Applies unchanged to

- **O3b** — blind TEST on knowns (ignore all event GPS/positions; full coincident-livetime scan).
- **O4a, O4b** — the blind SEARCH, each with its own same-run separated FAR.

Full-coverage scans only (all coincident livetime, not near-event segments), matching the O3a standard.

---

## AMENDMENT 1 — 2026-07-04 (O3b): candidate-scoring caps under-covered the glitch background

**Discovered by the O3b blind test.** The FG candidate-scoring caps — `N_CNN_CANDIDATES=120` (loglr channel,
was hardcoded) and `SM_FG_MAXNET=400` (net channel) — are *compute* shortcuts that assume "real events sit at
the top of the ranking." True on O3a's clean background; **false on O3b**, whose ~36 loud-glitch segments (L1 σ
up to 46) crowded 4 of 5 real above-floor heavy events (GW200129, GW200224, GW200311, GW191222; loglr 8–12,
net 8–10) out of the CNN-scored set. Only GW191109 (clean segment) surfaced. Verified by direct frozen-model
loglr evaluation at each event's stride.

**Correction:** `N_CNN_CANDIDATES` made env-driven (`SM_N_CNN_CANDIDATES`, default **120 unchanged** →
O3a runs byte-identical); O3b sets it high to score **ALL** loglr≥floor candidates. **Detection floor (4.0),
per-arm statistic, coherence, and FAR are UNCHANGED** — this is a coverage fix, not a threshold change.
Scope: `search_mode/driver_blindscan.py:82`, `launchers/o3b_fg.sbatch`. The two genuine misses (GW191230,
GW200219; loglr 3.05/−2.55, sub-threshold) are unaffected — real low-SNR near-misses, not cap artifacts.

# Pre-registration amendment: net-gated background + coherence-ceiling override

**Date:** 2026-07-01 · **Run:** full-O3a FAR (jobs 1519021 prune / 1519022 merge) · **Status:** committed BEFORE reading the resulting per-event FARs.

This amends the O3a/O4a FAR pre-registration to add two changes to `driver_blindscan.py`. Both are argued as **correctness fixes to a mis-specified background**, not new tuning levers. Reviewed by an 11-agent adversarial panel (wf_a7c56f59): verdict FIX-FIRST, **no numerical/anti-conservative defect found**; the only blocker was that these touch a quoted statistic and must be written down.

## Change 1 — Net-gated loglr-channel background (`SM_NET_GATE`)

**What:** the time-slide background for the loglr channel is restricted to windows with `net = (sH+sL)/√2 > NET_CUT` before the coherence-ceiling prefilter (`driver_blindscan.py`, prune loop). The net-σ channel (`famN`, `NETSIG_FLOOR`) is unchanged.

**Why it is a correctness fix, not a lever:** the foreground forms a trigger **only** at `net > NET_CUT = 4.0` (`driver_blindscan.py:250`, strict `>`). A time-slide window with `net ≤ 4` is by definition **not a trigger** and can never appear in the foreground. Including it in the background is a foreground/background **definition mismatch** that inflates the FAR denominator with non-triggers. Gating the background at net>4 makes the background definition **match the trigger definition** — the standard matched-background requirement.

**Falsifiability / anti-tuning safeguards:**
- The gate value is **code-pinned to `NET_CUT`** (`NET_GATE = NET_CUT if os.environ.get("SM_NET_GATE") else 0.0`) — it is a boolean, not a free float; it **cannot drift** from the foreground cut.
- It applies to **every candidate identically**, not to GW190521 specifically.
- Direction: it **removes** non-trigger background → FAR decreases. This is *correct* de-conservatism (excluding events that were never triggers), not hiding real background.
- **GW190521 is not rescued by this change:** it remains the pre-declared marginal target (net 5.8, loglr 4.53); its FAR is still exactly whatever the net>4 background yields. The gate does not move its loglr or lower its floor.

## Change 2 — Coherence-ceiling override `SM_COH_CEIL=0.95`

**What:** the prefilter coherence ceiling is set to 0.95 instead of the auto Gumbel `pop_bound` (~0.512).

**Why it does not touch the quoted FAR:** `COH_CEIL` is only a per-row **upper bound inside the prefilter**; `flush_block` recomputes the **exact** coherence and keeps survivors on the **exact** loglr. So `COH_CEIL` has **zero degrees of freedom on the FAR**, *provided* `ceiling ≥ every survivor's realized coherence` — a condition the **tripwire enforces** (`driver_blindscan.py` flush_block: abort if any realized coh ≥ ceiling).

**Why the override is required (a correctness fix):** with the net-gate, the rescored population is the *loud* net>4 tail, whose coherence reaches 0.625 (`o3a_far_1518977_0.log` tripwire abort at auto-0.512). The auto pop_bound is fit on the general-population subsample (`sub_max 0.254`, `beta 0.0137`) and is **structurally too tight** for the loud tail → it would have **dropped real net>4 background → biased the FAR low**. `0.95` is a **lossless superset** (`o3a_far_1518984_0.log`: net>4frac=1.0, no trip, kept bg 321 vs the wrongly-dropped 45). It can only *add* correct background (conservative), never subtract.

## What is NOT changed
Candidate floor (loglr ≥ 4.5), per-arm statistic (`2·min(hm,lm)×n_channels`), the 2-fold cross-fit, DET_FAR=1.0, and the livetime accounting are all unchanged.

## Required companion actions (from the review)
1. **Loud-event invariance check:** before quoting the loud events (loglr 8–14.7), read the merged per-fold **max background loglr** and confirm 0 background ≥ each loud event's loglr (near-certain; a one-line check, not an assumption).
2. **O4a mirror:** the O4a matched-config FAR must use the identical net-gate + ceiling, or O3a/O4a are computed under different background definitions.

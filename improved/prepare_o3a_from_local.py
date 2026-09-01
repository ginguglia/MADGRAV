"""Build the CAE training set from LOCAL O3a strain, in the layout load_prepared_o1_arrays() expects.

Replaces the GWOSC-download path of prepare_o1_data.py: the strain is already on scratch, and the
DQ information lives in search_mode/veto_mask_o3a_56.json as `good_segments` intervals
(CBC_CAT2 AND NO_CBC_HW_INJ AND lock) rather than as per-second masks inside the frame files.

HARD GUARD: a strain segment with no veto-mask entry is REFUSED, never silently used. The .npz
strain carries no DQ or injection mask, so an adapter that skipped this would train the CAE on
hardware injections labelled as noise -- simulated signals in the noise class, in a model whose
whole discriminant is "noise reconstructs well, signal reconstructs badly".

Selection matches the original recipe's intent: N blocks of BLOCK_SECONDS spread across the run.
Blocks are drawn STRATIFIED-RANDOM -- the run is split into N equal calendar strata and one block
is drawn per stratum -- so a draw is both random and spread, and different --seed values give
genuinely independent draws for a set-to-set stability test.

Usage:
  prepare_o3a_from_local.py --output-dir data/o3a_train_draw1 --seed 1 [--hours 36]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.signal import welch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


FS = 4096
SEG_SECONDS = 4.0
STRIDE_SECONDS = 2.0
BLOCK_SECONDS = 4096.0          # the original recipe's per-file block size
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2
PSD_SEGMENTS = 2048
DETECTORS = ("H1", "L1")
STRAIN_DIR = MADGRAV_SCRATCH + "/strain_o3a_56"
VETO_JSON = "search_mode/veto_mask_o3a_56.json"
EVENTS_JSON = "search_mode/o3a_events.json"
EVENT_HALF_WINDOW = 16.0
EVENT_STEP = 0.25

SEG_SAMPLES = int(SEG_SECONDS * FS)
STEP_SAMPLES = int(STRIDE_SECONDS * FS)


def load_segments(root):
    """-> [(name, gps_start, gps_end, good_intervals)] for segments with BOTH detectors present."""
    veto = json.load(open(os.path.join(root, VETO_JSON)))
    out = []
    for name, v in veto.items():
        paths = {d: os.path.join(STRAIN_DIR, f"{name}_{d}.npz") for d in DETECTORS}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        good = [(float(a), float(b)) for a, b in v["good_segments"] if b - a >= SEG_SECONDS]
        if not good:
            continue
        out.append((name, min(a for a, _ in good), max(b for _, b in good), good))
    out.sort(key=lambda r: r[1])
    return out


def enumerate_blocks(segs):
    """Chop DQ-clean intervals into BLOCK_SECONDS blocks -> [(seg_name, block_start_gps)]."""
    blocks = []
    for name, _, _, good in segs:
        for a, b in good:
            n = int((b - a) // BLOCK_SECONDS)
            for i in range(n):
                blocks.append((name, a + i * BLOCK_SECONDS))
    return blocks


def subtract_events(segs, ev_gps, pad):
    """Remove [gps-pad, gps+pad] around every known event from the DQ-clean intervals.

    The 56 O3a strain segments are EVENT-CENTRED locks, so `good_segments` -- which is
    CBC_CAT2 AND NO_CBC_HW_INJ AND lock, i.e. clean of HARDWARE injections -- still contains
    every real detection. Drawing 36 h out of 209 h usually missed them; taking the whole pool
    does not. Training the noise class on real signals is exactly the contamination the model's
    discriminant ("noise reconstructs well, signal does not") cannot tolerate, so the windows
    come out here. Fragments shorter than one 4 s segment are dropped.
    """
    out, removed_s, hit = [], 0.0, set()
    for name, g0, g1, good in segs:
        cur = list(good)
        for gps in ev_gps:
            lo, hi = gps - pad, gps + pad
            nxt = []
            for a_, b_ in cur:
                if hi <= a_ or lo >= b_:
                    nxt.append((a_, b_)); continue
                hit.add(gps)
                removed_s += min(b_, hi) - max(a_, lo)
                if a_ < lo:
                    nxt.append((a_, lo))
                if hi < b_:
                    nxt.append((hi, b_))
            cur = nxt
        cur = [(a_, b_) for a_, b_ in cur if b_ - a_ >= SEG_SECONDS]
        if cur:
            out.append((name, min(a_ for a_, _ in cur), max(b_ for _, b_ in cur), cur))
    return out, removed_s, sorted(hit)


def choose_blocks(blocks, n_blocks, seed):
    """Stratified random: split the run calendar into n_blocks strata, draw one block from each."""
    rng = np.random.default_rng(seed)
    blocks = sorted(blocks, key=lambda b: b[1])
    if len(blocks) <= n_blocks:
        return blocks
    edges = np.linspace(0, len(blocks), n_blocks + 1, dtype=int)
    chosen = []
    for i in range(n_blocks):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        chosen.append(blocks[int(rng.integers(lo, hi))])
    return sorted(chosen, key=lambda b: b[1])


def cut_windows(strain, gps_start, good, blk_start, blk_end):
    """4 s windows at 2 s stride inside [blk_start, blk_end) AND inside a good interval."""
    wins, starts = [], []
    for a, b in good:
        lo, hi = max(a, blk_start), min(b, blk_end)
        if hi - lo < SEG_SECONDS:
            continue
        i0 = int(round((lo - gps_start) * FS))
        i1 = int(round((hi - gps_start) * FS))
        i0 = max(0, i0); i1 = min(len(strain), i1)
        for s in range(i0, i1 - SEG_SAMPLES + 1, STEP_SAMPLES):
            w = strain[s:s + SEG_SAMPLES]
            if len(w) != SEG_SAMPLES or not np.all(np.isfinite(w)):
                continue
            wins.append(w)
            starts.append(gps_start + s / FS)
    return wins, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hours", type=float, default=36.0, help="target DQ-clean hours PER DETECTOR")
    ap.add_argument("--all-blocks", action="store_true",
                    help="use EVERY DQ-clean block instead of a stratified draw (--hours ignored)")
    ap.add_argument("--exclude-events", action="store_true",
                    help="carve +/- --event-pad s around every known O3a event out of the noise pool")
    ap.add_argument("--event-pad", type=float, default=64.0,
                    help="half-width of the excluded window around each known event [s]")
    ap.add_argument("--root", default=os.environ.get("MADGRAV_ROOT", "."))
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)

    segs = load_segments(a.root)
    strain_names = {os.path.basename(p)[:-len("_H1.npz")]
                    for p in __import__("glob").glob(os.path.join(STRAIN_DIR, "*_H1.npz"))}
    veto_names = set(json.load(open(os.path.join(a.root, VETO_JSON))))
    unguarded = sorted(strain_names - veto_names)
    if unguarded:
        sys.exit(f"REFUSING: strain segments with no veto-mask entry (DQ/hw-injection status unknown): {unguarded}")

    ev_excl = {}
    if a.exclude_events:
        ev = {}
        for jf in (EVENTS_JSON, "search_mode/o3a_events_full.json"):
            pth = os.path.join(a.root, jf)
            if os.path.exists(pth):
                ev.update(json.load(open(pth)))
        if not ev:
            sys.exit("REFUSING --exclude-events: no event list found")
        h0 = sum(b - a_ for _, _, _, g in segs for a_, b in g) / 3600.0
        segs, rm_s, hit = subtract_events(segs, sorted(ev.values()), a.event_pad)
        h1 = sum(b - a_ for _, _, _, g in segs for a_, b in g) / 3600.0
        ev_excl = dict(n_events_listed=len(ev), n_events_intersecting=len(hit),
                       pad_seconds=a.event_pad, removed_seconds=float(rm_s),
                       clean_hours_before=h0, clean_hours_after=h1)
        print(f"[prep] event exclusion: {len(ev)} known O3a events, {len(hit)} fell inside the "
              f"DQ-clean pool; removed {rm_s / 3600.0:.2f} h (+/-{a.event_pad:.0f}s each), "
              f"{h0:.1f} -> {h1:.1f} h/detector", flush=True)
    else:
        print("[prep] WARNING: event exclusion OFF -- the noise pool may contain real detections",
              flush=True)

    blocks = enumerate_blocks(segs)
    if a.all_blocks:
        n_blocks = len(blocks)
        chosen = sorted(blocks, key=lambda b: b[1])
    else:
        n_blocks = int(round(a.hours * 3600.0 / BLOCK_SECONDS))
        chosen = choose_blocks(blocks, n_blocks, a.seed)
    print(f"[prep] {len(segs)} segments, {len(blocks)} DQ-clean blocks of {BLOCK_SECONDS:.0f}s available", flush=True)
    print(f"[prep] {'ALL blocks' if a.all_blocks else f'target {a.hours} h/detector'} -> "
          f"{n_blocks} blocks, took {len(chosen)} (seed {a.seed})", flush=True)

    by_seg = {}
    for name, bs in chosen:
        by_seg.setdefault(name, []).append(bs)
    seg_lookup = {s[0]: s for s in segs}

    noise, rows = [], []
    for name in sorted(by_seg):
        _, _, _, good = seg_lookup[name]
        for det in DETECTORS:
            d = np.load(os.path.join(STRAIN_DIR, f"{name}_{det}.npz"))
            strain = d["strain"]; gs = float(d["gps_start"])
            keep_idx = 0
            for bs in sorted(by_seg[name]):
                w, st = cut_windows(strain, gs, good, bs, bs + BLOCK_SECONDS)
                for win, s in zip(w, st):
                    noise.append(win)
                    rows.append({"detector": det, "file_gps_start": bs,
                                 "segment_gps_start": s, "segment_index_in_file": keep_idx})
                    keep_idx += 1
            del strain, d
        print(f"[prep] {name}: {len(by_seg[name])} block(s), running total {len(noise)} windows", flush=True)

    if not noise:
        sys.exit("no windows extracted")
    noise_all = np.asarray(noise, dtype=np.float32); del noise
    hours_per_det = len(noise_all) / len(DETECTORS) * STRIDE_SECONDS / 3600.0
    print(f"[prep] {len(noise_all)} windows total = {hours_per_det:.2f} h/detector "
          f"(windows overlap at {STRIDE_SECONDS:.0f}s stride; pooled over {len(DETECTORS)} detectors)", flush=True)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(noise_all))
    noise_all = noise_all[perm]; rows = [rows[i] for i in perm]

    n_test = max(1, int(len(noise_all) * TEST_FRACTION))
    n_val = max(1, int(len(noise_all) * VAL_FRACTION))
    noise_test, noise_val, noise_train = noise_all[:n_test], noise_all[n_test:n_test+n_val], noise_all[n_test+n_val:]
    rows_test, rows_val, rows_train = rows[:n_test], rows[n_test:n_test+n_val], rows[n_test+n_val:]

    O = a.output_dir
    np.save(os.path.join(O, "noise_train.npy"), noise_train)
    np.save(os.path.join(O, "noise_val.npy"), noise_val)
    np.save(os.path.join(O, "noise_test.npy"), noise_test)
    for nm, rr in (("train", rows_train), ("val", rows_val), ("test", rows_test)):
        with open(os.path.join(O, f"noise_{nm}_metadata.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["detector", "file_gps_start", "segment_gps_start", "segment_index_in_file"])
            w.writeheader(); w.writerows(rr)

    freq, psd = welch(noise_train[:min(len(noise_train), PSD_SEGMENTS)].astype(np.float64), fs=FS, nperseg=SEG_SAMPLES)
    psd = psd.mean(axis=0)
    np.savez(os.path.join(O, "reference_psd.npz"), freq=freq, psd=psd)
    np.save(os.path.join(O, "psd_freqs.npy"), freq.astype(np.float64))
    np.save(os.path.join(O, "sample_rate.npy"), np.array(FS, dtype=np.int64))
    for det in DETECTORS:
        idx = [i for i, r in enumerate(rows_train) if r["detector"] == det][:PSD_SEGMENTS]
        f2, p2 = welch(noise_train[np.array(idx, dtype=np.int64)].astype(np.float64), fs=FS, nperseg=SEG_SAMPLES)
        p2 = p2.mean(axis=0)
        np.savez(os.path.join(O, f"reference_psd_{det}.npz"), freq=f2, psd=p2)
        np.save(os.path.join(O, f"psd_{det}.npy"), p2.astype(np.float64))

    # event windows (reporting only; the loader requires a non-empty set)
    events = json.load(open(os.path.join(a.root, EVENTS_JSON)))
    ew, er = [], []
    offs = np.arange(-EVENT_HALF_WINDOW, EVENT_HALF_WINDOW + 1e-9, EVENT_STEP)
    for name, gps in sorted(events.items()):
        for det in DETECTORS:
            p = os.path.join(STRAIN_DIR, f"{name}_{det}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p); strain = d["strain"]; gs = float(d["gps_start"])
            for o in offs:
                c = gps + o
                s = int(round((c - SEG_SECONDS / 2 - gs) * FS))
                if s < 0 or s + SEG_SAMPLES > len(strain):
                    continue
                w = strain[s:s + SEG_SAMPLES]
                if not np.all(np.isfinite(w)):
                    continue
                ew.append(w)
                er.append({"event_name": name, "detector": det, "event_gps": gps,
                           "window_center_gps": c, "offset_seconds": float(o)})
            del strain, d
    np.save(os.path.join(O, "event_windows.npy"), np.asarray(ew, dtype=np.float32))
    with open(os.path.join(O, "event_metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event_name", "detector", "event_gps", "window_center_gps", "offset_seconds"])
        w.writeheader(); w.writerows(er)

    audit = dict(seed=a.seed, target_hours_per_detector=a.hours, block_seconds=BLOCK_SECONDS,
                 n_blocks_requested=n_blocks, n_blocks_drawn=len(chosen),
                 n_blocks_available=len(blocks), n_segments_available=len(segs),
                 segment_seconds=SEG_SECONDS, stride_seconds=STRIDE_SECONDS,
                 windows_total=int(len(noise_all)), n_train=int(len(noise_train)),
                 n_val=int(len(noise_val)), n_test=int(len(noise_test)),
                 hours_per_detector=float(hours_per_det),
                 n_detectors=len(DETECTORS),
                 event_windows=int(len(ew)),
                 dq_basis="CBC_CAT2 AND NO_CBC_HW_INJ AND coincident lock (veto_mask_o3a_56.json)",
                 all_blocks=bool(a.all_blocks), event_exclusion=ev_excl,
                 blocks=[[n, float(b)] for n, b in chosen])
    json.dump(audit, open(os.path.join(O, "draw_audit.json"), "w"), indent=1)
    np.savez(os.path.join(O, "summary.npz"), **{k: v for k, v in audit.items() if not isinstance(v, (list, str, dict))})
    print(f"[prep] wrote {O}: train {len(noise_train)} val {len(noise_val)} test {len(noise_test)} "
          f"event {len(ew)} | {audit['hours_per_detector']:.1f} h/det", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""GATE for the final p_astro: reproduce the frozen 44-row madgrav_far_final.csv
from the compact bg caches, byte-identical in (N_bg, far, channel, livetime).

Counting = verbatim re-implementation of cumulative_far_snapshot.py:220-268:
  loglr channel : louder bg pairs (ll > ev.loglr, same fold, self-seg guarded),
                  per-arm L1-family count where bg arm score >= event arm score;
                  per-arm FAR = 2*min(hm,lm)/livetime.
  net channel   : famN = per-(fold,L1key) MAX-net rep; reps with net >= ev.net,
                  same fold, self-seg guarded, counted (not deduped) per arm.
  best_far      = n_channels * min(channel per-arm FARs)   [n_channels = 2]

Run: python far_repro_check.py
"""
import json, csv
import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


RUNS = {
    "O3a": MADGRAV_SCRATCH + "/search_out_o3a_far_f40",
    "O3b": MADGRAV_SCRATCH + "/search_out_o3b_far_f40",
    "O4a": MADGRAV_SCRATCH + "/search_out_o4a_far",
    "O4b": MADGRAV_SCRATCH + "/search_out_o4b_far",
}
CSV = MADGRAV_ROOT + "/figures/catalog_o3o4/madgrav_far_final.csv"


def perarm_counts(bg, ev_seg_ix, ev_fold, ev_loglr, ev_net, ev_hm, ev_lm):
    """Return (n_lr_hm, n_lr_lm, n_net_hm, n_net_lm) for one event."""
    fold = bg["fold"]; hseg = bg["hseg"]; lseg = bg["lseg"]
    notself = (hseg != ev_seg_ix) & (lseg != ev_seg_ix)
    samefold = fold == ev_fold

    # loglr channel: families by arm among louder pairs
    m = samefold & notself & (bg["loglr"] > ev_loglr)
    fam = bg["fam"][m]; hm = bg["cnn_hm"][m]; lm = bg["cnn_lm"][m]
    n_lr_hm = len(np.unique(fam[hm >= ev_hm]))
    n_lr_lm = len(np.unique(fam[lm >= ev_lm]))

    # net channel: famN reps (max net per (fold,fam)), then count reps >= ev_net by arm
    # (self-seg guard applies to the REP's segments, as in the original loop)
    key = bg["fold"].astype(np.int64) * (1 << 60) + bg["fam"]
    order = np.lexsort((-bg["net"], key))          # per key, best net first
    k_sorted = key[order]
    first = np.ones(len(k_sorted), bool); first[1:] = k_sorted[1:] != k_sorted[:-1]
    rep = order[first]                              # famN representative rows
    rm = (bg["fold"][rep] == ev_fold) & (bg["net"][rep] >= ev_net) \
         & (bg["hseg"][rep] != ev_seg_ix) & (bg["lseg"][rep] != ev_seg_ix)
    rr = rep[rm]
    n_net_hm = int(np.sum(bg["cnn_hm"][rr] >= ev_hm))
    n_net_lm = int(np.sum(bg["cnn_lm"][rr] >= ev_lm))
    return n_lr_hm, n_lr_lm, n_net_hm, n_net_lm


def best_far_from_counts(counts, flt):
    n_lr_hm, n_lr_lm, n_net_hm, n_net_lm = counts
    lr = 2 * min(n_lr_hm, n_lr_lm) / flt
    nt = 2 * min(n_net_hm, n_net_lm) / flt
    ch = "net-sigma" if nt < lr else "loglr"
    return 2 * min(lr, nt), ch, min((n_lr_hm, n_lr_lm) if ch == "loglr" else (n_net_hm, n_net_lm))


def main():
    rows = list(csv.DictReader(open(CSV)))
    n_ok = n_bad = 0
    for run, out_dir in RUNS.items():
        z = np.load(f"{out_dir}/bg_cache_{run.lower()}.npz", allow_pickle=False)
        bg = {k: z[k] for k in ("hseg", "lseg", "fold", "fam", "loglr", "net", "cnn_hm", "cnn_lm")}
        seg_ix = {nm: i for i, nm in enumerate(z["seg_names"])}
        seg_fold = z["seg_fold"]; far_live = z["far_live"]
        dets = json.load(open(f"{out_dir}/detections.json"))
        def match(r):
            # CSV names were assigned later (catalog GPS match); the (loglr, net)
            # floats are copied verbatim from detections.json -> exact float match.
            cand = [d for d in dets if abs(d["loglr"] - float(r["loglr"])) < 1e-9
                    and abs(d["net"] - float(r["net"])) < 1e-9]
            assert len(cand) == 1, f"{run} {r['name']}: {len(cand)} matches"
            return cand[0]
        for r in rows:
            if r["run"] != run:
                continue
            d = match(r)
            six = seg_ix[d["seg"]]; g = int(seg_fold[six]); flt = float(far_live[g])
            counts = perarm_counts(bg, six, g, d["loglr"], d["net"], d["cnn_hm"], d["cnn_lm"])
            far, ch, nbg = best_far_from_counts(counts, flt)
            ok = (abs(far - float(r["far"])) < 1e-9 and ch == r["channel"]
                  and nbg == int(float(r["N_bg"])) and abs(flt - float(r["livetime_yr"])) < 5e-4)
            n_ok += ok; n_bad += (not ok)
            flag = "OK " if ok else "MISMATCH"
            print(f"{flag} {run} {r['name']:24s} far={far:.6g} (csv {float(r['far']):.6g}) "
                  f"ch={ch}/{r['channel']} N={nbg}/{r['N_bg']} lt={flt:.3f}/{r['livetime_yr']}")
    print(f"\n{'PASS' if n_bad == 0 else 'FAIL'}: {n_ok}/{n_ok+n_bad} rows reproduced")

if __name__ == "__main__":
    main()

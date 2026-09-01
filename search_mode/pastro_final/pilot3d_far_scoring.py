#!/usr/bin/env python
"""STEP 3f: FAR-LEVEL escalation of the 3d control (incident entry,
2026-08-12). MEASUREMENT ONLY - no corrections, no preview gate.

Scores the existing 3d random-segment injections (all four runs) at FAR
level through the STANDARD det_frac machinery, imported UNCHANGED from
pastro_final.py: same pooled 47 empirical detection cnn pairs, same frozen
LR fold models with cross-fit (fold-g segments scored by the other fold's
model), same per-segment-excluded background counting curves, same
best_far/UL90 AND-criterion (trials-corrected best_far < 1/yr AND UL90 <
1/yr AND net > 4). The event-hosting side needs no rescoring: its det_frac
is inj_scored_<run>.npz (order-verified against the raw files).

Segment resolution: pilot3d segment names are looked up in the run's
bg_cache directly (they are scanned segments); if absent, the fold is taken
from the GPS-nearest bg segment and the counting excludes no segment
(six = -1, a mode RunBG.curves supports) - counted and reported.

Deliverable per run, mass bin, and sample (random | event-hosting):
  det_frac(FAR)  w0-weighted mean over injections of the 47-pair mean
  eps(trigger)   w0-weighted net > 4 fraction (same trigger def as 3d)
  ratio          det_frac(FAR) / eps(trigger)
plus the O3a-vs-O3b contrast of that ratio (in-sample vs out-of-sample).

Out: pilot3d_far_{run}.npz (per-injection det_frac), pilot3d_far_report.
{json,txt}. Run: madgrav-venv python pilot3d_far_scoring.py
"""
import glob
import json
import os
import sys

import numpy as np

MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
SM = f"{MG}/search_mode"
sys.path.insert(0, HERE)
import pastro_final as PF
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


VT_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
NBIN = len(VT_EDGES) - 1
RUNS = ("O3a", "O3b", "O4a", "O4b")


def w0_of(snr):
    mids = (PF.SNR_GRID[1:] + PF.SNR_GRID[:-1]) / 2
    dr = np.diff(np.concatenate([[PF.SNR_GRID[0]], mids, [PF.SNR_GRID[-1]]]))
    w = PF.SNR_GRID ** -4.0 * dr
    w /= w.sum()
    lut = {float(s): float(v) for s, v in zip(PF.SNR_GRID, w)}
    return np.array([lut[float(s)] for s in snr])


def pooled_pairs():
    pairs = []
    for run in PF.RUNS:
        for d in json.load(open(f"{PF.RUNS[run]['out']}/detections.json")):
            pairs.append((d["cnn_hm"], d["cnn_lm"]))
    return np.array(pairs)


def score_random(run, pairs):
    bg = PF.RunBG(run)
    qs_hm = np.unique(pairs[:, 0])
    qs_lm = np.unique(pairs[:, 1])
    npair = len(pairs)
    starts = np.array([int(n.rsplit("_", 1)[1]) for n in bg.seg_names])
    order = np.argsort(starts)
    files = sorted(glob.glob(f"{SM}/inj_out_pilot3d_{run.lower()}/*_inj.npz"))
    assert files, f"{run}: no pilot3d injections"
    DF, MT, SN, NET, SEG = [], [], [], [], []
    n_fallback = 0
    for f in files:
        seg = os.path.basename(f)[:-8]
        if seg in bg.seg_ix:
            six = bg.seg_ix[seg]
            g = int(bg.seg_fold[six])
        else:
            n_fallback += 1
            t0 = int(seg.rsplit("_", 1)[1])
            near = bg.seg_names[order[np.clip(
                np.searchsorted(starts[order], t0) - 1, 0, len(order) - 1)]]
            g = int(bg.seg_fold[bg.seg_ix[near]])
            six = -1
        z = np.load(f)
        F = PF.feats(z)
        net = z["net"].astype(float)
        x = PF.loglr_of(F, 1 - g)          # cross-fit, unchanged
        cur = bg.curves(six, g, qs_hm, qs_lm)
        flt = float(bg.far_live[g])
        lr_ok = x >= PF.FLOOR
        nt_ok = net >= PF.NETSIG_FLOOR
        trig = net > PF.NET_CUT
        dblk = np.empty((npair, len(x)), bool)
        for pi, (phm, plm) in enumerate(pairs):
            nlrh = PF.RunBG.n_at(cur["lr"][("hm", phm)], x, strict=True)
            nlrl = PF.RunBG.n_at(cur["lr"][("lm", plm)], x, strict=True)
            nnth = PF.RunBG.n_at(cur["net"][("hm", phm)], net, strict=False)
            nntl = PF.RunBG.n_at(cur["net"][("lm", plm)], net, strict=False)
            far, ul = PF.best_far_vec(nlrh, nlrl, nnth, nntl, flt, lr_ok, nt_ok)
            dblk[pi] = trig & np.isfinite(far) & (far < PF.DET_FAR) & (ul < PF.DET_FAR)
        DF.append(dblk.mean(axis=0))
        MT.append(z["mtot"].astype(float))
        SN.append(z["net_snr"].astype(float))
        NET.append(net)
        SEG.append(np.full(len(x), seg))
        print(f"  [{run}] {seg}: {len(x)} inj scored "
              f"(fold {g}, six={'own' if six >= 0 else 'FALLBACK-none'})",
              flush=True)
    out = dict(det_frac=np.concatenate(DF), mtot=np.concatenate(MT),
               net_snr=np.concatenate(SN), net=np.concatenate(NET),
               seg=np.concatenate(SEG))
    np.savez(f"{HERE}/pilot3d_far_{run.lower()}.npz", **out,
             npair=npair, n_fallback=n_fallback)
    return out, n_fallback


def load_event_side(run):
    sc = np.load(f"{HERE}/inj_scored_{run.lower()}.npz")
    nets = []
    for d in PF.RUNS[run]["inj"]:
        for f in sorted(glob.glob(f"{d}/*_inj.npz")):
            nets.append(np.load(f)["net"].astype(float))
    net = np.concatenate(nets)
    assert len(net) == len(sc["mtot"]), f"{run}: event-side order mismatch"
    return dict(det_frac=sc["det_frac"], mtot=sc["mtot"],
                net_snr=sc["net_snr"], net=net)


def bin_table(S):
    w = w0_of(S["net_snr"])
    bins = np.digitize(S["mtot"], VT_EDGES) - 1
    rows = []
    for k in range(NBIN):
        sel = bins == k
        if not sel.sum():
            rows.append(None)
            continue
        ww = w[sel]
        df = float((S["det_frac"][sel] * ww).sum() / ww.sum())
        et = float(((S["net"][sel] > PF.NET_CUT) * ww).sum() / ww.sum())
        rows.append(dict(n=int(sel.sum()), det_frac_far=df, eps_trigger=et,
                         ratio=df / et if et > 0 else None))
    return rows


def main():
    pairs = pooled_pairs()
    print(f"[3f] {len(pairs)} pooled det cnn pairs (unchanged machinery)",
          flush=True)
    rep = {"machinery": "pastro_final det_frac, unchanged (pooled pairs, "
           "cross-fit LR folds, per-seg-excluded curves, AND criterion)",
           "runs": {}}
    lines = ["STEP 3f: FAR-level det_frac vs trigger-level eps "
             "(measurement only; no corrections, no preview)", ""]
    for run in RUNS:
        rnd, nfb = score_random(run, pairs)
        evt = load_event_side(run)
        tr, te = bin_table(rnd), bin_table(evt)
        rep["runs"][run] = dict(random=tr, event=te, n_fallback_segments=nfb)
        lines.append(f"[{run}] det_frac(FAR)/eps(trigger) per bin "
                     f"(random || event-hosting); fallback segs: {nfb}")
        for k in range(NBIN):
            a, b = tr[k], te[k]
            def fmt(x):
                return ("  -  " if x is None or x["ratio"] is None
                        else f"{x['det_frac_far']:.3f}/{x['eps_trigger']:.3f}"
                             f"={x['ratio']:.2f}")
            lines.append(f"  {VT_EDGES[k]:3.0f}-{VT_EDGES[k+1]:3.0f}: "
                         f"rand {fmt(a)}   evt {fmt(b)}")
        lines.append("")
    # O3a vs O3b contrast of the FAR/trigger ratio (in- vs out-of-sample)
    lines.append("O3a-vs-O3b contrast of ratio(det_frac/eps_trigger), "
                 "random sample:")
    for k in range(NBIN):
        a = rep["runs"]["O3a"]["random"][k]
        b = rep["runs"]["O3b"]["random"][k]
        if a and b and a["ratio"] and b["ratio"]:
            lines.append(f"  {VT_EDGES[k]:3.0f}-{VT_EDGES[k+1]:3.0f}: "
                         f"O3a {a['ratio']:.2f} vs O3b {b['ratio']:.2f} "
                         f"-> O3a/O3b = {a['ratio'] / b['ratio']:.2f}")
    json.dump(rep, open(f"{HERE}/pilot3d_far_report.json", "w"), indent=1)
    with open(f"{HERE}/pilot3d_far_report.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

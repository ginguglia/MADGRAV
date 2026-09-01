#!/usr/bin/env python
"""Merge asd_consistency_bg.py shards -> details/successor_statistic/bg_veto_<run>.npz (keyed by bg_cache index) and
append the vetoed-fraction table (per run/fold/channel over the top-M rep pairs) to bg_veto_counts.txt.  (S2 output)"""
import os, sys, glob, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import successor_stat as S

DET = S.DET

def merge(run):
    r = run.lower(); files = sorted(glob.glob(f"{DET}/veto_shards/bg_veto_{r}_shard*.npz"))
    plan = np.load(f"{DET}/veto_pairs_{r}.npz"); want = plan["idx"].astype(np.int64)
    idx = []; hm = []; lm = []; nl = []; tp = []
    for f in files:
        z = np.load(f); idx.append(z["idx"]); hm.append(z["hm_loc"]); lm.append(z["lm_loc"]); nl.append(z["net_loc"]); tp.append(z["t_pair_s"])
    idx = np.concatenate(idx).astype(np.int64); hm = np.concatenate(hm); lm = np.concatenate(lm); nl = np.concatenate(nl); tp = np.concatenate(tp)
    o = np.argsort(idx); idx, hm, lm, nl, tp = idx[o], hm[o], lm[o], nl[o], tp[o]
    missing = np.setdiff1d(want, idx); nfail = int((~np.isfinite(hm)).sum())
    assert len(missing) == 0, f"{run}: {len(missing)} planned pairs missing from shards ({len(files)} files)"
    assert len(np.unique(idx)) == len(idx)
    dst = f"{DET}/bg_veto_{r}.npz"
    np.savez(dst, idx=idx, hm_loc=hm, lm_loc=lm, net_loc=nl, t_pair_s=tp, n_shards=np.array(len(files)), n_failed=np.array(nfail), date=np.array(time.strftime("%F %T")))
    # counts table
    bg = S.Background(run, dst, verbose=False); L = []
    for f in (0, 1):
        F = bg.F[f]; tm = bg.top_M_pairs(f)
        for ch, pairs, vet in (("loglr", tm["lr_pairs"], bg.cnn_vet), ("net-sigma", tm["net_pairs"], bg.net_vet)):
            ev = bg.evaluated[pairs]; nv = int(vet[pairs].sum()); n = len(pairs)
            extra = f", of which net_loc<4 only: {int((bg.net_vet[pairs] & ~bg.cnn_vet[pairs]).sum())}" if ch == "net-sigma" else ""
            L.append(f"{run} fold {f} {ch:9s}: top-M={tm['M']} rep pairs, evaluated {int(ev.sum())}, vetoed {nv} ({100*nv/max(n,1):.1f}%){extra}; "
                     f"surviving families with channel: {int(F['valid_lr'].sum()) if ch=='loglr' else int(F['valid_net'].sum())} of {int(F['has_lr'].sum()) if ch=='loglr' else int(F['has_net'].sum())}")
    L.append(f"{run}: {len(idx)} pairs evaluated in {len(files)} shards, {nfail} failed evaluations (counted as surviving), mean {np.nanmean(tp):.2f} s/pair, wall sum {np.nansum(tp)/3600:.2f} GPU-h")
    txt = "\n".join(L); print(txt)
    with open(f"{DET}/bg_veto_counts.txt", "a") as fh: fh.write(f"# {time.strftime('%F %T')} {run}\n" + txt + "\n")
    return dst

if __name__ == "__main__":
    for run in sys.argv[1:]: merge(run)

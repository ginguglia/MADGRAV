"""Run-level plateau metric for the conditional CNN clause of the local-ASD veto (DECISIONS_ASDVETO.md, 2026-10-07).
Per run and detector: fraction of stream livetime (windows) spent in segments whose sigma stream exceeds 2 in more than 20 % of windows.
Uses no coincidence, no zero-lag candidate and no catalogue. Usage: plateau_metric.py <run> <streams_dir> -> writes plateau_<run>.json here."""
import sys, glob, json, os, numpy as np
run, d = sys.argv[1], sys.argv[2]; THR_SIG, THR_FRAC, MIN_WIN = 2.0, 0.20, 100
out = dict(run=run, streams_dir=d, sigma_thresh=THR_SIG, plateau_frac_thresh=THR_FRAC, min_windows=MIN_WIN)
for det in ("H1", "L1"):
    fs = sorted(glob.glob(f"{d}/*_{det}_meta.npz")); n_tot = n_pl = 0; nseg = 0
    for f in fs:
        s = np.asarray(np.load(f, allow_pickle=True)["sigma"]); 
        if len(s) < MIN_WIN: continue
        nseg += 1; n_tot += len(s); n_pl += len(s) if (s > THR_SIG).mean() > THR_FRAC else 0
    out[det] = dict(n_segments=nseg, n_windows=int(n_tot), plateau_livetime_frac=(n_pl / n_tot if n_tot else None))
out["max_frac"] = max(out["H1"]["plateau_livetime_frac"] or 0, out["L1"]["plateau_livetime_frac"] or 0)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"plateau_{run}.json"), "w"), indent=1)
print(f"{run}: H1 {out['H1']['plateau_livetime_frac']:.4f} L1 {out['L1']['plateau_livetime_frac']:.4f} max {out['max_frac']:.4f} ({out['H1']['n_segments']}/{out['L1']['n_segments']} segs)")

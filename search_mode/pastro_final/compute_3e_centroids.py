#!/usr/bin/env python
"""Exact centroids for the 3e background TRIGGER windows (net > 4), both
variants (whitened with the variant's own prep - as-run vs corrected refs),
so analyze_3e_survival.py --exact replaces the fixed-centroid approximation.
GPU job (step 3g); windows re-read from scratch strain via saved start_idx.
Out: pilot3e_bg_out/cen_<seg>_<variant>.npz {cenH, cenL, idx}
"""
import glob
import os
import sys

import numpy as np

MADGRAV_ROOT = MADGRAV_ROOT
for _p in ("search_mode", "improved", "spectrogram_cascade"):
    sys.path.insert(0, os.path.join(MADGRAV_ROOT, _p))
import driver_streams as DS
from massive_pipeline import MassiveEventPipeline

SC = MADGRAV_SCRATCH
OUT = f"{MADGRAV_ROOT}/search_mode/pilot3e_bg_out"
FS = 4096
WN = 4 * FS
NET_CUT = 4.0
DEV = os.environ.get("SM_DEV", "cuda:0")
PREPS = {"asrun": "{mg}/data/{run}_search_prep",
         "corrected": "{mg}/data/{run}_search_prep_release"}

pipes = {}


def pipe_for(run, variant):
    key = (run, variant)
    if key not in pipes:
        prep = PREPS[variant].format(mg=MADGRAV_ROOT, run=run)
        pipes[key] = MassiveEventPipeline(
            calib_path=f"{DS.SC}/massive_calibration_BA.json",
            prep=prep, device=DEV)
    return pipes[key]


def main():
    for f in sorted(glob.glob(f"{OUT}/*_bg.npz")):
        base = os.path.basename(f)[:-7]
        seg, variant = base.rsplit("_", 1)
        run = seg.split("_")[0]
        cenf = f"{OUT}/cen_{seg}_{variant}.npz"
        if os.path.exists(cenf):
            continue
        z = np.load(f)
        trig = z["net"] > NET_CUT
        idx = z["start_idx"][trig]
        if not len(idx):
            np.savez(cenf, cenH=np.array([]), cenL=np.array([]), idx=idx)
            continue
        pipe = pipe_for(run, variant)
        cen = {}
        for det in ("H1", "L1"):
            raw = np.load(f"{SC}/strain_{run}_full/{seg}_{det}.npz")["strain"]
            outs = []
            for i0 in range(0, len(idx), 256):
                X = np.stack([raw[s:s + WN] for s in idx[i0:i0 + 256]])
                wh = pipe._whiten(X.astype(np.float32), det)
                outs.append(pipe._centroid(wh))
            cen[det] = np.concatenate(outs)
        np.savez(cenf, cenH=cen["H1"], cenL=cen["L1"], idx=idx)
        print(f"[cen] {seg} {variant}: {len(idx)} trigger windows", flush=True)
    print("[cen] done", flush=True)


if __name__ == "__main__":
    main()

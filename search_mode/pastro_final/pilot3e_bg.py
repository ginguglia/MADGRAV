#!/usr/bin/env python
"""STEP 3e pilot, background leg: score background-only stretches through the
FROZEN chain front end under a selectable reference-ASD variant.

Purpose (design decision 2026-08-12): the rescan go/no-go will be decided on
(a) recovery vs physical SNR in the newly-accessible volume and (b) the
background trigger rate under corrected refs vs as-run. This script measures
(b) at the blindscan STAGE-1 trigger definition - zero-lag coincident window
with net sigma = (sigH+sigL)/sqrt(2) >= NET_CUT(=4.0, pinned from
driver_blindscan) - which is the entry gate that drives FAR-floor cost.
The chain is UNCHANGED: whitening/QT/sigma/glitch-arm are the exact
functions inject.py's score_block uses (MassiveEventPipeline + driver_streams
+ p1v42 arms); only SM_PREP selects which reference ASD whitens.

Both variants are scanned on the IDENTICAL window grid (4 s windows, 1 s
stride, zero-lag H1^L1 same-index), so the corrected/as-run rate ratio is
free of grid convention. +-16 s around the segment's anchor GPS is excluded.

Env: SM_RUN (o4a|o4b), SM_PREP (variant prep dir), SM_VARIANT (label),
     SM_STRAIN, SM_SEGS (comma-separated names), SM_OUT, SM_DEV (cuda:0 on
     single-GPU allocations), SM_EVENTSJSON (anchor GPS map).
Out: SM_OUT/<seg>_<variant>_bg.npz (per-window sigH,sigL,net,coh,gH,gL)
     + one summary line per segment on stdout.
"""
import json
import os
import sys
import time

import numpy as np
import torch

MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or MADGRAV_ROOT
for _p in ("search_mode", "improved", "spectrogram_cascade"):
    _ap = os.path.join(MADGRAV_ROOT, _p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)
import driver_streams as DS
from massive_pipeline import MassiveEventPipeline

FS = 4096
WN = 4 * FS
STRIDE_S = 1.0
STEP = int(STRIDE_S * FS)
NET_CUT = 4.0            # pinned to driver_blindscan.py STAGE-1 (net sigma > 4)
EXCL_S = 16.0            # exclusion half-width around the anchor GPS
BATCH = 256

RUN = os.environ["SM_RUN"]
PREP = os.environ["SM_PREP"]
VARIANT = os.environ.get("SM_VARIANT", "unnamed")
STRAIN = os.environ["SM_STRAIN"]
SEGS = os.environ["SM_SEGS"].split(",")
OUT = os.environ["SM_OUT"]
DEV = os.environ.get("SM_DEV", "cuda:0")
EVENTS = json.load(open(os.environ["SM_EVENTSJSON"]))
SEGJ = json.load(open(os.environ["SM_SEGJSON_EV"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    pipe = MassiveEventPipeline(calib_path=f"{DS.SC}/massive_calibration_BA.json",
                                prep=PREP, device=DEV)
    arms = [DS.GlitchArm().to(DEV) for _ in range(5)]
    for i, arm in enumerate(arms):
        arm.load_state_dict(torch.load(f"{DS.LRD}/p1v42/arm_deploy_seed{i}.pt",
                                       map_location=DEV))
        arm.eval()
    print(f"[pilot3e-bg] run={RUN} variant={VARIANT} prep={PREP} "
          f"segs={len(SEGS)} grid=4s/{STRIDE_S}s stride", flush=True)

    for name in SEGS:
        t0 = time.time()
        gps0 = float(EVENTS[name])
        seg_t0 = float(SEGJ[name]["coincident_lock"][0])
        rawH = np.load(f"{STRAIN}/{name}_H1.npz")["strain"]
        rawL = np.load(f"{STRAIN}/{name}_L1.npz")["strain"]
        n = min(len(rawH), len(rawL))
        if n < 2 * WN:
            print(f"  [{name}] too short ({n} samples) -- skipped", flush=True)
            continue
        starts = np.arange(0, n - WN, STEP)
        centers = seg_t0 + (starts + WN // 2) / FS
        starts = starts[np.abs(centers - gps0) > EXCL_S]
        feats = {k: [] for k in ("sigH", "sigL", "net", "coh", "gH", "gL")}
        for i0 in range(0, len(starts), BATCH):
            ss = starts[i0:i0 + BATCH]
            XH = np.stack([rawH[s:s + WN] for s in ss])
            XL = np.stack([rawL[s:s + WN] for s in ss])
            whH = pipe._whiten(XH.astype(np.float32), "H1")
            whL = pipe._whiten(XL.astype(np.float32), "L1")
            qtH = DS.build_qt(pipe, whH)
            qtL = DS.build_qt(pipe, whL)
            sH = DS.sigma_from_qt(pipe, qtH, "H1")
            sL = DS.sigma_from_qt(pipe, qtL, "L1")
            feats["sigH"].append(sH)
            feats["sigL"].append(sL)
            feats["net"].append((sH + sL) / np.sqrt(2.0))
            feats["coh"].append(pipe._coherence(whH, whL))
            feats["gH"].append(DS.g_from_qt(arms, qtH))
            feats["gL"].append(DS.g_from_qt(arms, qtL))
        F = {k: np.concatenate(v) for k, v in feats.items()}
        F["start_idx"] = starts
        np.savez(f"{OUT}/{name}_{VARIANT}_bg.npz", **F)
        hours = len(starts) * STRIDE_S / 3600.0
        n_net = int((F["net"] >= NET_CUT).sum())
        n_coinc4 = int(((F["sigH"] >= 4.0) & (F["sigL"] >= 4.0)).sum())
        print(f"  [{name}] {VARIANT}: windows={len(starts)} ({hours:.2f} h) "
              f"net>={NET_CUT:.0f}: {n_net} ({n_net / hours:.2f}/h) "
              f"sig4-coinc: {n_coinc4} ({n_coinc4 / hours:.2f}/h) "
              f"({(time.time() - t0) / 60:.1f} m)", flush=True)
    print(f"[pilot3e-bg] {RUN}/{VARIANT} done", flush=True)


if __name__ == "__main__":
    main()

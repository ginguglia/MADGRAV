"""Score real O3a events with each trained CAE variant, in ONE fixed configuration.

Every model is evaluated on the SAME cached tiles: the whitened/QT real-event windows
(event_qt.npy) and the SAME held-out noise split (noise_qt_te.npy) used to calibrate sigma.
That removes the two confounds that made the training-metric comparison unreadable -- each run
having its own validation set, and the best-checkpoint argmax.

sigma is the standardized reconstruction error: (err - mu_noise) / sd_noise, calibrated per
detector on held-out noise, exactly as the deployed pipeline defines its anomaly score. An event
counts as flagged at 3 sigma if its PEAK window (over the +/-16 s offsets) reaches it.

Usage: score_real_events.py <qtcache_dir> <model.pt> [<model.pt> ...]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, MADGRAV_ROOT + "/improved")
import importlib.util
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

_spec = importlib.util.spec_from_file_location("ip", MADGRAV_ROOT + "/improved/improved_pipeline.py")
ip = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(ip)
except SystemExit:
    pass

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH = 256


def load_model(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    k = sd["bott_down.weight"].shape[0] if "bott_down.weight" in sd else None
    m = ip.BaselineCAE(latent_channels=k)
    m.load_state_dict(sd, strict=True)
    return m.to(DEV).eval(), k


def recon_err(model, arr):
    """Per-tile mean squared reconstruction error."""
    out = np.empty(len(arr), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(arr), BATCH):
            x = torch.from_numpy(np.asarray(arr[i:i + BATCH])).float().to(DEV)
            if x.dim() == 3:
                x = x.unsqueeze(1)
            y = model(x)
            out[i:i + BATCH] = ((y - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return out


def main(cache, paths):
    ev = np.load(os.path.join(cache, "event_qt.npy"), mmap_mode="r")
    meta = json.load(open(os.path.join(cache, "event_meta.json")))
    noise = np.load(os.path.join(cache, "noise_qt_te.npy"), mmap_mode="r")
    ndet = [r["detector"] for r in json.load(open(os.path.join(cache, "noise_test_meta.json")))]
    n_cal = min(len(noise), 6000)
    noise = noise[:n_cal]; ndet = ndet[:n_cal]
    names = sorted({r["event_name"] for r in meta})
    print(f"cache={cache}  events={len(names)}  event tiles={len(ev)}  noise calib tiles={len(noise)}")
    print(f"{'model':<26}{'k':>5}   " + "".join(f"{n.replace('GW',''):>14}" for n in names) + f"{'>=3sig':>9}")
    for p in paths:
        model, k = load_model(p)
        en = recon_err(model, noise)
        cal = {}
        for d in ("H1", "L1"):
            m = np.array([x == d for x in ndet])
            if m.sum() > 10:
                cal[d] = (en[m].mean(), en[m].std())
        ee = recon_err(model, ev)
        peak = {}
        for i, r in enumerate(meta):
            d = r["detector"]
            if d not in cal:
                continue
            mu, sd = cal[d]
            s = (ee[i] - mu) / sd if sd > 0 else 0.0
            key = r["event_name"]
            peak[key] = max(peak.get(key, -1e9), s)
        nflag = sum(1 for n in names if peak.get(n, -1e9) >= 3.0)
        tag = os.path.basename(os.path.dirname(os.path.dirname(p)))
        print(f"{tag:<26}{str(k):>5}   " + "".join(f"{peak.get(n, float('nan')):>14.2f}" for n in names)
              + f"{nflag:>6}/{len(names)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

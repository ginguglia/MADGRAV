#!/usr/bin/env python
"""STEP 3b: per-mass-bin horizon volumes with the GWTC-5 release run-median
PSDs for O4a/O4b, under BOTH normalization conventions (decision pending):

  A 'release-absolute'  : reference_psd_release_*.npz used as-is (physical
                          one-sided PSD normalization of the release).
  B 'mid-band-matched'  : release SHAPE x scalar s(run,det), s = median of
                          current/release over 160-400 Hz - preserves the
                          pipeline PSD convention where the current refs are
                          closest to physical; caveat: o4b H1 mid-band sits
                          ABOVE o3a (unphysical), so B inherits that bias.

  O3a/O3b use their current references unchanged in both variants (O3b L1
  low-f excess flagged, pending an O3 release PSD source).

Reuses vt_search.horizons() verbatim via a monkeypatched ASD loader, so the
bank/SNR machinery is identical to the validated VT path.
Run: madgrav-venv python step3_horizons_release.py
Output: step3_horizons_release.json + printed D_h summary at 330-400 Msun.
"""
import json
import sys

import numpy as np

MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
sys.path.insert(0, HERE)
sys.path.insert(0, f"{MG}/improved")

import improved_pipeline as ip
from improved_pipeline import FrequencySeries
import vt_search as vs
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MATCH_BAND = (160.0, 400.0)
orig_loader = ip.load_detector_asd_o1


def make_loader(variant):
    def loader(prepared_dir, detector):
        run = prepared_dir.rstrip("/").split("/")[-1].split("_")[0]  # o3a/o3b/o4a/o4b
        if run not in ("o4a", "o4b"):
            return orig_loader(prepared_dir, detector)
        rel = np.load(f"{prepared_dir}/reference_psd_release_{detector}.npz")
        f, psd = rel["freq"].astype(float), rel["psd"].astype(np.float64)
        if variant == "matched":
            cur = np.load(f"{prepared_dir}/reference_psd_{detector}.npz")["psd"]
            band = (f >= MATCH_BAND[0]) & (f <= MATCH_BAND[1])
            s = float(np.median(cur[band] / psd[band]))
            psd = psd * s
            loader.scales[f"{run}_{detector}"] = s
        good = np.isfinite(psd) & (psd > 0)
        psd = np.where(good, psd, np.median(psd[good]) * 1e10)
        return FrequencySeries(np.sqrt(psd), f0=float(f[0]), df=float(f[1] - f[0]))
    loader.scales = {}
    return loader


def main():
    out = {"mass_edges": vs.MASS_EDGES.tolist(), "match_band_hz": MATCH_BAND,
           "variants": {}}
    for variant in ("current", "absolute", "matched"):
        ip.load_detector_asd_o1 = (orig_loader if variant == "current"
                                   else make_loader(variant))
        vv = {}
        for run in vs.RUNS:
            v_m = vs.horizons(run)
            vv[run] = {tag: [None if not np.isfinite(x) else float(x)
                             for x in vm] for tag, vm in v_m.items()}
        if variant != "current":
            vv["_scales"] = dict(ip.load_detector_asd_o1.scales)
        out["variants"][variant] = vv
        # D_h from mean UM volume in the last bin (330-400)
        line = []
        for run in vs.RUNS:
            v = vv[run]["um"][-1]
            dh = (3 * v / (4 * np.pi)) ** (1 / 3) if v else float("nan")
            line.append(f"{run} {dh:8.0f} Mpc")
        print(f"[{variant:9}] D_h(UM, 330-400): " + " | ".join(line))
        if variant != "current":
            print(f"            scales: {vv.get('_scales', {})}")
    ip.load_detector_asd_o1 = orig_loader
    json.dump(out, open(f"{HERE}/step3_horizons_release.json", "w"), indent=1)
    print(f"[done] -> {HERE}/step3_horizons_release.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Reference-ASD sanity gate (specified abort condition, 2026-08-13).

Computes the BNS (1.4+1.4 Msun) sky-averaged SNR-8 inspiral range implied by
each reference_psd_{H1,L1}.npz in a prep dir and checks it against the
detector's published range for that run. A reference ASD biased high at low
frequency (the O4 as-run failure mode) suppresses the implied range and trips
the gate BEFORE the prep is used by any downstream stage.

Usage:  python prep_sanity_gate.py <prep_dir> <run>     # exit 1 on FAIL
        python prep_sanity_gate.py --survey              # report all preps, no abort

Published-range anchors and the accept band live in PUBLISHED below; ranges
are approximate run-median BNS ranges from LVK run summaries (GWOSC/obs-run
papers). The band is deliberately loose: this is an order-of-magnitude abort
gate against estimator poisoning, not a calibration check.
"""
import sys

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


G = 6.674e-11
C = 299792458.0
MSUN = 1.989e30
MPC = 3.0857e22
MC = 1.219 * MSUN            # chirp mass of 1.4+1.4
F_LO, F_HI = 20.0, 1570.0    # to ~ISCO of 2.8 Msun
SNR_THRESH = 8.0
SKY_FACTOR = 2.264           # horizon -> sky/orientation-averaged range

# approximate published run-median BNS ranges [Mpc] (anchor, not calibration)
PUBLISHED = {
    "o3a": {"H1": 108.0, "L1": 135.0},
    "o3b": {"H1": 115.0, "L1": 133.0},
    "o4a": {"H1": 152.0, "L1": 158.0},
    "o4b": {"H1": 160.0, "L1": 165.0},
}
BAND = (0.6, 1.67)           # implied/published must fall inside


def bns_range_mpc(npz_path):
    z = np.load(npz_path)
    f, S = z["freq"].astype(float), z["psd"].astype(float)
    m = (f >= F_LO) & (f <= F_HI) & (S > 0)
    f, S = f[m], S[m]
    A2 = (5.0 / 24.0) * np.pi ** (-4.0 / 3.0) * C ** 2 * (G * MC / C ** 3) ** (5.0 / 3.0)
    integ = np.trapezoid(A2 * f ** (-7.0 / 3.0) / S, f)
    d_horizon = np.sqrt(4.0 * integ) / SNR_THRESH        # metres
    return d_horizon / SKY_FACTOR / MPC


def check(prep_dir, run):
    # normalize rescan-style labels (o4ars -> o4a) to a published anchor
    if run not in PUBLISHED:
        cands = [k for k in PUBLISHED if run.startswith(k)]
        if not cands:
            print(f"[prep-gate] no published anchor for run label '{run}' - "
                  f"add it to PUBLISHED before gating", flush=True)
            return False
        run = max(cands, key=len)
    ok = True
    for det in ("H1", "L1"):
        r = bns_range_mpc(f"{prep_dir}/reference_psd_{det}.npz")
        pub = PUBLISHED[run][det]
        ratio = r / pub
        verdict = "OK" if BAND[0] <= ratio <= BAND[1] else "FAIL"
        if verdict == "FAIL":
            ok = False
        print(f"[prep-gate] {run} {det}: implied BNS range {r:6.1f} Mpc  "
              f"published ~{pub:.0f}  ratio {ratio:.2f}  {verdict}", flush=True)
    return ok


def main():
    if sys.argv[1:2] == ["--survey"]:
        mr = MADGRAV_ROOT
        for run in ("o3a", "o3b", "o4a", "o4b"):
            for tag, d in (("as-run", f"{mr}/data/{run}_search_prep"),
                           ("release", f"{mr}/data/{run}_search_prep_release")):
                try:
                    print(f"--- {run} {tag}")
                    check(d, run)
                except FileNotFoundError:
                    print(f"--- {run} {tag}: (no such prep)")
        return
    prep_dir, run = sys.argv[1], sys.argv[2]
    if not check(prep_dir, run):
        print(f"[prep-gate] ABORT: {prep_dir} implies a BNS range inconsistent "
              f"with the {run} published range - reference ASD suspect "
              f"(estimator poisoning / wrong sample). Do not use this prep.",
              flush=True)
        sys.exit(1)
    print("[prep-gate] PASS", flush=True)


if __name__ == "__main__":
    main()

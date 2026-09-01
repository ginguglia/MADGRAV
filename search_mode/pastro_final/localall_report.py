"""Gate-complement pilot: how many prep-gate FAILURES does local-ASD whitening rescue?

Reads /scratch/.../inj_localall/<band>/*_inj.npz (SM_INJ_LOCAL_ALL=1, so hm_loc/lm_loc are filled
for EVERY injection, not just gate-passers) and reports, per run / SNR / mass:

  flip_to_pass  = P(local gate PASS | prep gate FAIL)   <- the unmeasured complement; the upside
  flip_to_fail  = P(local gate FAIL | prep gate PASS)   <- already known (the ASD veto), for balance
  net gate rate = prep gate rate vs local gate rate

Predictions registered before the run are in PILOT_PREDICTION_localall_20260831.md; the gap between
those and these numbers is the point of the exercise, so both are printed side by side.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


INJ = MADGRAV_SCRATCH + "/inj_localall"
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = 0.5
MASS_EDGES = [20, 60, 100, 160, 240, 400]
PRED = {("o3a", "low"): "10-20%", ("o3a", "high"): "5-12%",
        ("o4b", "low"): "<=3%", ("o4b", "high"): "<=3%"}


def main():
    files = sorted(glob.glob(f"{INJ}/*/*_inj.npz"))
    if not files:
        raise SystemExit(f"no pilot output yet under {INJ}")
    cells = defaultdict(lambda: dict(gp=0, gf=0, f2p=0, f2f=0, n=0))
    by_mass = defaultdict(lambda: dict(gf=0, f2p=0))
    by_snr = defaultdict(lambda: dict(gf=0, f2p=0))
    nan_fail = 0
    for p in files:
        band = os.path.basename(os.path.dirname(p))
        run = band.replace("_lowsnr", "")
        key = (run, "low" if band.endswith("_lowsnr") else "high")
        z = np.load(p)
        prep = np.maximum(z["cnn_hm"], z["cnn_lm"]) > GATE
        loc_raw = np.maximum(z["hm_loc"], z["lm_loc"])
        ok = np.isfinite(loc_raw)
        nan_fail += int((~ok).sum())
        loc = ok & (loc_raw > GATE)
        c = cells[key]
        c["n"] += len(prep); c["gp"] += int(prep.sum()); c["gf"] += int((~prep).sum())
        c["f2p"] += int((~prep & loc).sum()); c["f2f"] += int((prep & ~loc).sum())
        for i in range(len(MASS_EDGES) - 1):
            m = (z["mtot"] >= MASS_EDGES[i]) & (z["mtot"] < MASS_EDGES[i + 1])
            b = by_mass[(key[0], MASS_EDGES[i])]
            b["gf"] += int((~prep & m).sum()); b["f2p"] += int((~prep & loc & m).sum())
        for s in np.unique(z["net_snr"]):
            m = z["net_snr"] == s
            b = by_snr[(key[0], float(s))]
            b["gf"] += int((~prep & m).sum()); b["f2p"] += int((~prep & loc & m).sum())

    print(f"gate-complement pilot: {len(files)} files")
    if nan_fail:
        print(f"  WARNING: {nan_fail} injections still have NaN local columns "
              f"(SM_INJ_LOCAL_ALL may not have been set) -- they are counted as local FAIL")
    print(f"\n{'cell':<12}{'n':>8}{'prep gate':>11}{'local gate':>12}"
          f"{'flip->pass':>12}{'predicted':>12}{'flip->fail':>12}")
    for key in sorted(cells):
        c = cells[key]
        prep_rate = c["gp"] / c["n"]
        local_rate = (c["gp"] - c["f2f"] + c["f2p"]) / c["n"]
        f2p = c["f2p"] / max(c["gf"], 1)
        f2f = c["f2f"] / max(c["gp"], 1)
        print(f"{key[0] + '/' + key[1]:<12}{c['n']:>8}{prep_rate:>11.3f}{local_rate:>12.3f}"
              f"{f2p:>12.3f}{PRED.get(key, '-'):>12}{f2f:>12.3f}")

    print("\nflip->pass by injected SNR (all mass):")
    for run in sorted({k[0] for k in by_snr}):
        row = f"  {run:<5}"
        for s in sorted({k[1] for k in by_snr if k[0] == run}):
            b = by_snr[(run, s)]
            row += f"  snr{s:g}={b['f2p'] / max(b['gf'], 1):.3f}"
        print(row)
    print("\nflip->pass by Mtot (all SNR)  [(A) 'leak hides signal' predicts a rise with mass]:")
    for run in sorted({k[0] for k in by_mass}):
        row = f"  {run:<5}"
        for i in range(len(MASS_EDGES) - 1):
            b = by_mass[(run, MASS_EDGES[i])]
            row += f"  {MASS_EDGES[i]}-{MASS_EDGES[i + 1]}={b['f2p'] / max(b['gf'], 1):.3f}"
        print(row)

    out = {f"{k[0]}_{k[1]}": dict(n=v["n"], prep_gate=v["gp"] / v["n"],
                                  flip_to_pass=v["f2p"] / max(v["gf"], 1),
                                  flip_to_fail=v["f2f"] / max(v["gp"], 1))
           for k, v in cells.items()}
    json.dump(out, open(f"{HERE}/localall_report.json", "w"), indent=1)
    print(f"\n-> {HERE}/localall_report.json")


if __name__ == "__main__":
    main()

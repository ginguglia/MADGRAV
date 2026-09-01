#!/usr/bin/env python
"""STEP 3e pilot report: the two go/no-go numbers for the O4 rescan decision.

(a) RECOVERY vs PHYSICAL SNR in the newly-accessible volume: the pilot
    injection campaign runs inject.py UNCHANGED with SM_PREP pointing at the
    corrected (release run-median) prep, so its nominal SNR grid IS physical
    SNR, at levels 5,6,8,10,12 - almost entirely below 5*c(bin), i.e. inside
    the volume the as-run search never probed (c from
    vt_relabel_release.json). Recovery here = peak-grid net sigma >= 4.0,
    the blindscan STAGE-1 trigger definition (pre-registered proxy; the
    FAR<1/yr-level mapping needs the bg leg, which is measured alongside).
    The as-run injection results for the SAME segment (as-run nominal grid,
    axis marked) are tabulated next to it for orientation.

(b) BACKGROUND TRIGGER RATE corrected vs as-run: net>=4 zero-lag rate per
    hour from pilot3e_bg.py, both variants on the identical window grid;
    ratio with Poisson 90% CI. This is the FAR-floor cost driver.

Inputs (produced by step3e_pilot.sbatch):
  search_mode/inj_out_pilot3e_<run>/<ev>_inj.npz     corrected-prep injections
  search_mode/inj_out_<run>/<ev>_inj.npz             as-run injections (existing)
  <PILOT_OUT>/<seg>_{asrun,corrected}_bg.npz         bg features per segment
Output: pilot3e_report.json + text table on stdout (mailed by the watcher).
"""
import glob
import json
import os

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
SM = f"{MG}/search_mode"
PILOT_OUT = os.environ.get("SM_PILOT_OUT", f"{SM}/pilot3e_bg_out")
NET_CUT = 4.0
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])


def poisson_ci90(n):
    from scipy.stats import chi2
    lo = 0.5 * chi2.ppf(0.05, 2 * n) if n > 0 else 0.0
    hi = 0.5 * chi2.ppf(0.95, 2 * (n + 1))
    return lo, hi


def inj_recovery(npz_path):
    z = np.load(npz_path)
    out = {}
    for pop, sel_pop in (("sig", z["is_um"] == 0), ("um", z["is_um"] == 1),
                         ("all", np.ones(len(z["net"]), bool))):
        levels = {}
        for s in np.unique(z["net_snr"]):
            sel = (z["net_snr"] == s) & sel_pop
            if sel.sum():
                levels[float(s)] = dict(
                    n=int(sel.sum()),
                    recovered=int((z["net"][sel] >= NET_CUT).sum()),
                    frac=float((z["net"][sel] >= NET_CUT).mean()))
        out[pop] = levels
    return out


def main():
    rel = json.load(open(f"{HERE}/vt_relabel_release.json"))
    rep = {"net_cut": NET_CUT, "runs": {}}
    lines = ["STEP 3e PILOT REPORT - rescan go/no-go inputs", ""]
    for run in ("o4a", "o4b"):
        R = {"injections": {}, "background": {}}
        run_u = run.upper()
        cmed = [c for c in rel["runs"][run_u]["c_median"] if c] \
            if run_u in rel.get("runs", {}) else []
        # (a) injections
        for f in sorted(glob.glob(f"{SM}/inj_out_pilot3e_{run}/*_inj.npz")):
            ev = os.path.basename(f).replace("_inj.npz", "")
            R["injections"][ev] = {"corrected_physical_grid": inj_recovery(f)}
            asrun = f"{SM}/inj_out_{run}/{ev}_inj.npz"
            if os.path.exists(asrun):
                R["injections"][ev]["asrun_nominal_grid"] = inj_recovery(asrun)
            lines.append(f"[{run_u}] recovery (net>=4) vs PHYSICAL SNR - {ev} "
                         f"(corrected refs; as-run c_med span "
                         f"{min(cmed):.2f}-{max(cmed):.2f} -> grid lies in "
                         f"newly-accessible volume)" if cmed else
                         f"[{run_u}] recovery - {ev}")
            for pop in ("sig", "um"):
                lv = R["injections"][ev]["corrected_physical_grid"][pop]
                row = "  ".join(f"snr{int(s)}: {v['frac']:.2f}"
                                for s, v in sorted(lv.items()))
                lines.append(f"  {pop:3s}: {row}")
        # (b) background
        for variant in ("asrun", "corrected"):
            tot_n, tot_h, per_seg = 0, 0.0, {}
            for f in sorted(glob.glob(f"{PILOT_OUT}/{run}_*_{variant}_bg.npz")):
                z = np.load(f)
                h = len(z["net"]) / 3600.0
                nn = int((z["net"] >= NET_CUT).sum())
                seg = os.path.basename(f).split(f"_{variant}_bg")[0]
                per_seg[seg] = dict(hours=h, n_net4=nn)
                tot_n += nn
                tot_h += h
            lo, hi = poisson_ci90(tot_n)
            R["background"][variant] = dict(
                hours=tot_h, n_net4=tot_n,
                rate_per_h=tot_n / tot_h if tot_h else None,
                rate_ci90=[lo / tot_h, hi / tot_h] if tot_h else None,
                per_seg=per_seg)
        ba, bc = R["background"].get("asrun"), R["background"].get("corrected")
        if ba and bc and ba["hours"] and bc["hours"] and ba["n_net4"]:
            ratio = (bc["rate_per_h"] or 0) / ba["rate_per_h"]
            R["background"]["corrected_over_asrun"] = ratio
            lines.append(f"[{run_u}] bg net>=4 rate: as-run "
                         f"{ba['rate_per_h']:.2f}/h ({ba['n_net4']} in "
                         f"{ba['hours']:.1f} h) vs corrected "
                         f"{bc['rate_per_h']:.2f}/h ({bc['n_net4']} in "
                         f"{bc['hours']:.1f} h) -> x{ratio:.2f}")
        lines.append("")
        rep["runs"][run_u] = R
    json.dump(rep, open(f"{HERE}/pilot3e_report.json", "w"), indent=1)
    print("\n".join(lines), flush=True)
    print(f"[pilot3e] -> {HERE}/pilot3e_report.json", flush=True)


if __name__ == "__main__":
    main()

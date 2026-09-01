#!/usr/bin/env python
"""Downstream survival of the 3e corrected-refs background triggers
(2026-08-12): glitch-arm margins, coherence, and the FINAL statistic,
loud vs quiet segments, corrected vs as-run. Headline = elevation factor at
the final statistic (stage-1 x9.4 is not the decision number).

Chain layers reproduced from the saved per-window features
(pilot3e_bg_out/<seg>_<variant>_bg.npz):
  stage-1   net = (sigH+sigL)/sqrt(2) > 4           (as scanned)
  arms      gate(g, sig) = clip(g,+-6) * clip(sig/3,0,1)  (the LR features
            carrying the glitch-arm margins, betas +1.3/+2.6..2.9)
  coherence coh (beta +1.7..1.9)
  FINAL     loglr = frozen LR (cross-fit fold model, unchanged) >= 4.0
            (the blindscan floor).

Centroids (cenH/cenL) are not stored by the scan. Default mode fixes them
at the fold model's feature means (betas +0.40..0.45/-0.07..-0.13 per SD ->
loglr sensitivity ~ +-0.5 for a +-1 SD centroid excursion; reported as a
band). With --exact and cen_<seg>_<variant>.npz files present (step-3g GPU
job), exact centroids are used instead.

Out: pilot3e_survival.{json,txt}
Run: madgrav-venv python analyze_3e_survival.py [--exact]
"""
import glob
import json
import os
import sys

import numpy as np

MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
SM = f"{MG}/search_mode"
OUT = f"{SM}/pilot3e_bg_out"
sys.path.insert(0, HERE)
import pastro_final as PF
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


FLOOR = 4.0
NET_CUT = 4.0
CEN_SD_BAND = 1.0


def seg_fold(run):
    bg = np.load(f"{PF.RUNS[run.capitalize()]['out']}/bg_cache_{run}.npz")
    names = [str(n) for n in bg["seg_names"]]
    return {n: int(f) for n, f in zip(names, bg["seg_fold"])}


def loglr_fixed_cen(z, g, cen_shift_sd=0.0, cen=None):
    mu, sd, be = PF.MDL[1 - g]                     # cross-fit convention
    n = len(z["net"])
    cenH = np.full(n, mu[3] + cen_shift_sd * sd[3])
    cenL = np.full(n, mu[4] + cen_shift_sd * sd[4])
    if cen is not None:
        # exact centroids cover only the trigger windows (net > NET_CUT);
        # non-trigger windows keep the fold-mean default (never pass stage 1)
        t = z["net"] > NET_CUT
        if len(cen[0]) != int(t.sum()):
            raise ValueError(f"centroid/trigger count mismatch: "
                             f"{len(cen[0])} vs {int(t.sum())}")
        cenH[t] = cen[0]
        cenL[t] = cen[1]
    F = np.column_stack([z["sigH"], z["sigL"], z["coh"], cenH, cenL,
                         PF.gate(z["gH"], z["sigH"]),
                         PF.gate(z["gL"], z["sigL"])])
    return PF.loglr_of(F, 1 - g)


def main():
    exact = "--exact" in sys.argv
    runs = {}
    for f in sorted(glob.glob(f"{OUT}/*_bg.npz")):
        base = os.path.basename(f)[:-7]            # <seg>_<variant>
        seg, variant = base.rsplit("_", 1)
        run = seg.split("_")[0]
        runs.setdefault(run, {}).setdefault(variant, {})[seg] = f
    rep = {"mode": "exact-centroids" if exact else
           "fixed-centroid approximation (+-1 SD band reported)",
           "runs": {}}
    lines = [f"3e DOWNSTREAM SURVIVAL ({rep['mode']})", ""]
    for run, variants in runs.items():
        if "asrun" not in variants or "corrected" not in variants:
            lines.append(f"[{run}] incomplete (variants: "
                         f"{sorted(variants)}) - skipped")
            continue
        folds = seg_fold(run)
        # stage-1 elevation per segment -> loud/quiet classes
        elev = {}
        for seg in variants["corrected"]:
            if seg not in variants["asrun"]:
                continue
            za = np.load(variants["asrun"][seg])
            zc = np.load(variants["corrected"][seg])
            ra = (za["net"] > NET_CUT).sum() / max(len(za["net"]), 1)
            rc = (zc["net"] > NET_CUT).sum() / max(len(zc["net"]), 1)
            elev[seg] = rc / max(ra, 1e-9)
        classes = {"loud": [s for s, e in elev.items() if e >= 5.0],
                   "quiet": [s for s, e in elev.items() if e < 5.0]}
        R = {"segment_classes": {k: v for k, v in classes.items()},
             "stage1_elevation_per_seg": {s: round(e, 2)
                                          for s, e in elev.items()},
             "by_class": {}}
        for cls, segs in classes.items():
            if not segs:
                continue
            row = {}
            for variant in ("asrun", "corrected"):
                hours = 0.0
                n1 = 0
                nf = 0
                nf_lo = 0
                nf_hi = 0
                gateH, gateL, coh = [], [], []
                for seg in segs:
                    z = np.load(variants[variant][seg])
                    g = folds.get(seg, 0)
                    hours += len(z["net"]) / 3600.0
                    trig = z["net"] > NET_CUT
                    n1 += int(trig.sum())
                    cenf = f"{OUT}/cen_{seg}_{variant}.npz"
                    cen = None
                    if exact and os.path.exists(cenf):
                        cz = np.load(cenf)
                        cen = (cz["cenH"], cz["cenL"])
                    ll = loglr_fixed_cen(z, g, 0.0, cen)
                    nf += int((trig & (ll >= FLOOR)).sum())
                    if not exact:
                        nf_lo += int((trig & (loglr_fixed_cen(z, g, -CEN_SD_BAND)
                                              >= FLOOR)).sum())
                        nf_hi += int((trig & (loglr_fixed_cen(z, g, +CEN_SD_BAND)
                                              >= FLOOR)).sum())
                    gateH.append(PF.gate(z["gH"], z["sigH"])[trig])
                    gateL.append(PF.gate(z["gL"], z["sigL"])[trig])
                    coh.append(z["coh"][trig])
                gateH = np.concatenate(gateH) if gateH else np.array([])
                gateL = np.concatenate(gateL) if gateL else np.array([])
                coh = np.concatenate(coh) if coh else np.array([])
                row[variant] = dict(
                    hours=round(hours, 2), stage1=n1,
                    stage1_per_h=round(n1 / hours, 2),
                    final=nf, final_per_h=round(nf / hours, 4),
                    final_band_per_h=None if exact else
                    [round(min(nf_lo, nf_hi) / hours, 4),
                     round(max(nf_lo, nf_hi) / hours, 4)],
                    survival=round(nf / n1, 4) if n1 else None,
                    gateH_med=round(float(np.median(gateH)), 2) if len(gateH) else None,
                    gateL_med=round(float(np.median(gateL)), 2) if len(gateL) else None,
                    coh_med=round(float(np.median(coh)), 3) if len(coh) else None)
            a, c = row["asrun"], row["corrected"]
            row["elevation_stage1"] = round(
                c["stage1_per_h"] / max(a["stage1_per_h"], 1e-9), 2)
            row["elevation_final"] = (
                round(c["final_per_h"] / a["final_per_h"], 2)
                if a["final_per_h"] > 0 else
                (f">{c['final']}x (as-run final = 0)" if c["final"] else "0/0"))
            R["by_class"][cls] = row
            lines.append(f"[{run}/{cls}] ({len(segs)} segs)")
            for v in ("asrun", "corrected"):
                r = row[v]
                lines.append(
                    f"  {v:9s}: stage1 {r['stage1_per_h']:8.2f}/h -> "
                    f"final {r['final_per_h']:8.4f}/h "
                    + (f"(band {r['final_band_per_h']}) " if r["final_band_per_h"] else "")
                    + f"survival {r['survival']}  gateH~{r['gateH_med']} "
                    f"gateL~{r['gateL_med']} coh~{r['coh_med']}")
            lines.append(f"  ELEVATION: stage1 x{row['elevation_stage1']} -> "
                         f"FINAL x{row['elevation_final']}")
            lines.append("")
        rep["runs"][run] = R
    json.dump(rep, open(f"{HERE}/pilot3e_survival.json", "w"), indent=1)
    with open(f"{HERE}/pilot3e_survival.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

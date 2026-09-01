#!/usr/bin/env python
"""CAVEAT BRANCH (design decision 2026-08-12): O3a in-sample systematic BAND
+ band-mode gate.

The 3f measurement: the trigger-to-FAR conversion (det_frac/eps_trigger) is
segment-independent in every run, but O3a sits flat at ~0.90 while
out-of-sample O3b is mass-graded at 0.65-0.79 -> contrast 1.15-1.40,
matching the gate excess in shape and size. Decision: the contrast is
applied as an ASYMMETRIC DOWNWARD systematic band on the O3a numerator
ONLY - central value unchanged; the band's lower edge divides each
detector-frame cohort's contribution by the per-bin contrast (same rebuild
machinery as the pre-registered correction). UPPER-BOUND interpretation
(stated): the contrast conflates in-sample optimism with O3b's genuinely
harder background, so the true in-sample term is <= the band width.

BAND-MODE GATE: every O3a cell (and the named checks) is reported at BOTH
band edges. Expected per the decision: single-bin passes at the lower
edge; pooled marginal within the band.

Out: vt_o3a_band.json, gate_ratio_table_band.{json,txt}
Run: madgrav-venv python gate_band_mode.py
"""
import json
import sys

import numpy as np

MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
sys.path.insert(0, HERE)
from gate_preview_corrected import rebuild_numerator
from gate_ratio_vs_matrix import (MASS_EDGES, NEFF_MIN, PIPEMAP,
                                  count_ratio_ci, pipe_hit)
from vt_relabel_comoving import comoving_machinery
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


NB = len(MASS_EDGES) - 1


def main():
    far = json.load(open(f"{HERE}/pilot3d_far_report.json"))
    contrast = []
    for k in range(NB):
        a = far["runs"]["O3a"]["random"][k]
        b = far["runs"]["O3b"]["random"][k]
        contrast.append(a["ratio"] / b["ratio"]
                        if a and b and a["ratio"] and b["ratio"] else None)
    vmax, _ = comoving_machinery()
    vt_lower = rebuild_numerator(
        "O3a", vmax, [None if c is None else 1.0 / c for c in contrast])
    num = json.load(open(f"{HERE}/vt_relabel_comoving.json"))
    central = np.array([v if v is not None else 0.0 for v in
                        num["runs"]["O3a"]["vt_comoving_srcframe_gpc3yr"]])
    json.dump(dict(
        central_gpc3yr=central.tolist(), lower_gpc3yr=vt_lower.tolist(),
        contrast_per_detbin=contrast,
        interpretation="UPPER BOUND on the O3a in-sample FAR-calibration "
                       "systematic: the O3a/O3b conversion contrast "
                       "conflates in-sample optimism with O3b's genuinely "
                       "harder background (design decision 2026-08-12)"),
        open(f"{HERE}/vt_o3a_band.json", "w"), indent=1)

    den = json.load(open(f"{HERE}/vt_pipelines_target_zc.json"))
    mat = json.load(open(f"{HERE}/cross_recovery_matrix.json"))
    events = [e for e in mat["events"] if e["run"] == "o3a"]
    lines = ["BAND-MODE GATE: O3a cells at both band edges "
             "(central | lower); other runs unchanged (all PASS)", ""]
    cells = []
    for k in range(NB):
        lo, hi = MASS_EDGES[k], MASS_EDGES[k + 1]
        ebin = [e for e in events if lo <= e["mtot"] < hi]
        N = len(ebin)
        kM = sum(e["madgrav"] for e in ebin)
        for pipe, col in PIPEMAP["O3a"].items():
            pd = den["runs"]["O3a"]["pipelines"].get(pipe)
            if pd is None:
                continue
            ne, vt_p = pd["neff"][k], pd["vt_gpc3yr"][k]
            if ne is None or ne < NEFF_MIN or not vt_p or vt_p <= 0 \
                    or central[k] <= 0:
                continue
            Rc, Rl = central[k] / vt_p, vt_lower[k] / vt_p
            kP = sum(pipe_hit(e, "o3a", col) for e in ebin)
            r, rlo_, rhi = count_ratio_ci(kM, kP, N)
            powered = (N >= 3 and kP >= 2)
            okc = (not powered) or (rhi is not None and Rc <= rhi)
            okl = (not powered) or (rhi is not None and Rl <= rhi)
            cells.append(dict(bin=f"{lo:.0f}-{hi:.0f}", pipe=pipe,
                              R_central=round(Rc, 3), R_lower=round(Rl, 3),
                              N=N, k_madgrav=int(kM), k_pipe=int(kP),
                              ci90_hi=None if rhi is None else round(rhi, 3),
                              powered=powered,
                              verdict_central="PASS" if okc else "FAIL",
                              verdict_lower="PASS" if okl else "FAIL"))
            lines.append(f"  O3a {lo:3.0f}-{hi:3.0f} vs {pipe:12s}: "
                         f"R=[{Rl:5.2f},{Rc:5.2f}]  counts {kM}/{kP} N={N}"
                         + (f"  CIhi={rhi:.2f}" if rhi else "")
                         + f"  {'POWERED' if powered else 'unpowered'}"
                         f"  central:{'PASS' if okc else 'FAIL'}"
                         f" lower:{'PASS' if okl else 'FAIL'}")
    # named checks at both edges (amended expected-count weighting)
    def named(tag, bins, subset_all):
        pdc = den["runs"]["O3a"]["pipelines"]["cWB"]
        usable = [k for k in bins if central[k] > 0 and pdc["vt_gpc3yr"][k]
                  and pdc["neff"][k] and pdc["neff"][k] >= NEFF_MIN]
        nk = {k: sum(1 for e in events
                     if MASS_EDGES[k] <= e["mtot"] < MASS_EDGES[k + 1])
              for k in usable}
        w = sum(nk.values())
        Rc = sum(central[k] / pdc["vt_gpc3yr"][k] * nk[k] for k in usable) / w
        Rl = sum(vt_lower[k] / pdc["vt_gpc3yr"][k] * nk[k] for k in usable) / w
        ebin = events if subset_all else \
            [e for e in events
             if any(MASS_EDGES[k] <= e["mtot"] < MASS_EDGES[k + 1]
                    for k in bins)]
        N = len(ebin)
        kM = sum(e["madgrav"] for e in ebin)
        kP = sum(pipe_hit(e, "o3a", "cwb") for e in ebin)
        r, rlo_, rhi = count_ratio_ci(kM, kP, N)
        lines.append(f"NAMED {tag}: R=[{Rl:.2f},{Rc:.2f}] counts {kM}/{kP} "
                     f"N={N} CIhi={rhi:.2f} "
                     f"central:{'PASS' if Rc <= rhi else 'FAIL'} "
                     f"lower:{'PASS' if Rl <= rhi else 'FAIL'}")
        return dict(tag=tag, R_central=round(Rc, 3), R_lower=round(Rl, 3),
                    ci90_hi=round(rhi, 3),
                    verdict_central="PASS" if Rc <= rhi else "FAIL",
                    verdict_lower="PASS" if Rl <= rhi else "FAIL")
    n1 = named("O3a_100-130_vs_cWB", [4], False)
    n2 = named("O3a_pooled_vs_cWB", list(range(NB)), True)
    lines.append("")
    lines.append("Interpretation: band lower edge = upper-bound in-sample "
                 "correction; verdicts at both edges per the caveat-branch "
                 "decision (no central-value change).")
    json.dump(dict(cells=cells, named=[n1, n2]),
              open(f"{HERE}/gate_ratio_table_band.json", "w"), indent=1)
    with open(f"{HERE}/gate_ratio_table_band.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

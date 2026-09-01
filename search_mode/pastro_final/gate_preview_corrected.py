#!/usr/bin/env python
"""GATE PREVIEW under the 3d-corrected numerators (2026-08-12).

The SINGLE pre-registered correction: each run's source-frame numerator is
rebuilt with every injection's contribution scaled by the step-3d measured
efficiency ratio eps_random/eps_event of its DETECTOR-frame mass bin
(per-bin where both samples have >= 200 kept injections, else the
run-pooled ratio - the rule frozen in pilot3d_report.py). Applied
UNIFORMLY to all four runs. The gate of gate_ratio_vs_matrix.py is then
re-evaluated. This is a PREVIEW: the figure stays WITHHELD regardless, no
clearance markers are written, and no iteration happens beyond this one
correction (protocol).

Numerator rebuild (identical machinery, correction inserted):
  per injection i (relabel_inj_<run>.npz: drel, c, kept, det_bin, src_bin):
    contrib_i = (w0_i / W_b) * det_i * Vc(drel_i/c_i) * T * corr[b]
  scattered to src_bin - the uncorrected version reproduces
  vt_comoving_srcframe_gpc3yr exactly (asserted, 1e-6).

Out: gate_ratio_table_corrected.{json,txt}  (verdict = PREVIEW-PASS/FAIL)
Run: madgrav-venv python gate_preview_corrected.py
"""
import json
import sys

import numpy as np

MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
sys.path.insert(0, HERE)
from gate_ratio_vs_matrix import (MASS_EDGES, NEFF_MIN, PIPEMAP,
                                  count_ratio_ci, pipe_hit)
from vt_relabel_comoving import comoving_machinery
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


RHO_TH = 5.0
NB = len(MASS_EDGES) - 1


def rebuild_numerator(run, vmax, corr):
    rl = run.lower()
    inj = np.load(f"{HERE}/relabel_inj_{rl}.npz")
    sc = np.load(f"{HERE}/inj_scored_{rl}.npz")
    num = json.load(open(f"{HERE}/vt_relabel_comoving.json"))
    T = num["runs"][run]["T_obs_yr"]
    w0, det = sc["w0"], sc["det_frac"]
    vc = vmax(inj["drel"] / inj["c"])
    kept = inj["kept"].astype(bool)
    vt = np.zeros(NB)
    vt_ref = np.zeros(NB)
    for b in range(NB):
        sel_b = inj["det_bin"] == b
        if not sel_b.sum():
            continue
        Wb = w0[sel_b].sum()
        base = w0[sel_b] * kept[sel_b] * det[sel_b] * vc[sel_b] / Wb * T * 1e-9
        cb = corr[b] if corr[b] is not None else 1.0
        for k, v in zip(inj["src_bin"][sel_b], base):
            if k >= 0:
                vt_ref[k] += v
                vt[k] += v * cb
    # identity gate: uncorrected rebuild == stored srcframe numerator
    stored = np.array([v if v is not None else 0.0 for v in
                       num["runs"][run]["vt_comoving_srcframe_gpc3yr"]])
    dev = np.max(np.abs(vt_ref - stored) / np.maximum(stored, 1e-12))
    assert dev < 1e-6, f"{run}: rebuild identity fails ({dev:.2e})"
    return vt


def trend_test(rep3d):
    """Pre-registered (incident entry 2, 2026-08-12): the correction is
    applied ONLY if eps_random/eps_event FALLS WITH MASS in O3a AND O3b -
    Spearman rho < 0 with one-sided p < 0.10 in BOTH runs, over the
    per-bin MEASURED ratios (correction_source == 'per-bin')."""
    from scipy.stats import spearmanr
    mids = 0.5 * (MASS_EDGES[1:] + MASS_EDGES[:-1])
    out = {}
    for run in ("o3a", "o3b"):
        r3 = rep3d["runs"][run]
        xs, ys = [], []
        for k, (src, row) in enumerate(zip(r3["correction_source"],
                                           r3["bins"])):
            if src == "per-bin" and row.get("ratio") is not None:
                xs.append(mids[k])
                ys.append(row["ratio"])
        rho, p2 = spearmanr(xs, ys)
        p1 = p2 / 2 if rho < 0 else 1 - p2 / 2
        out[run] = dict(n_bins=len(xs), rho=round(float(rho), 3),
                        p_onesided=round(float(p1), 4),
                        falls=bool(rho < 0 and p1 < 0.10))
    out["confirmed"] = out["o3a"]["falls"] and out["o3b"]["falls"]
    return out


def main():
    vmax, _ = comoving_machinery()
    rep3d = json.load(open(f"{HERE}/pilot3d_report.json"))
    den = json.load(open(f"{HERE}/vt_pipelines_target_zc.json"))
    mat = json.load(open(f"{HERE}/cross_recovery_matrix.json"))
    events = mat["events"]

    lines = ["GATE PREVIEW under 3d-corrected numerators "
             "(single pre-registered correction; figure remains WITHHELD)",
             ""]
    trend = trend_test(rep3d)
    for run in ("o3a", "o3b"):
        t = trend[run]
        lines.append(f"[trend {run}] per-bin ratios n={t['n_bins']}: "
                     f"Spearman rho={t['rho']} p1={t['p_onesided']} -> "
                     f"{'FALLS with mass' if t['falls'] else 'no confirmed fall'}")
    lines.append(f"[trend] pre-registered condition (falls in BOTH runs): "
                 f"{'CONFIRMED' if trend['confirmed'] else 'NOT CONFIRMED'}")
    lines.append("")
    if not trend["confirmed"]:
        lines += ["CORRECTION NOT APPLIED (prediction not confirmed; "
                  "incident entry 2) - report only, no preview gate.", ""]
        json.dump(dict(verdict="CORRECTION-NOT-APPLIED", trend=trend),
                  open(f"{HERE}/gate_ratio_table_corrected.json", "w"),
                  indent=1)
        with open(f"{HERE}/gate_ratio_table_corrected.txt", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    failures = []
    corr_used = {}
    vt_corr = {}
    for run in ("O3a", "O3b", "O4a", "O4b"):
        rl = run.lower()
        r3 = rep3d["runs"].get(rl)
        if r3 is None or "correction_per_bin" not in r3:
            lines.append(f"[{run}] no 3d correction available - SKIPPED")
            continue
        corr = r3["correction_per_bin"]
        corr_used[run] = dict(per_bin=corr, source=r3["correction_source"],
                              pooled=r3.get("ratio_pooled"))
        vt_corr[run] = rebuild_numerator(run, vmax, corr)
        lines.append(f"[{run}] correction (per det-bin): "
                     + " ".join("-" if c is None else f"{c:.2f}"
                                for c in corr))
    lines.append("")
    for run, ours in vt_corr.items():
        rl = run.lower()
        evr = [e for e in events if e["run"] == rl]
        for k in range(NB):
            lo, hi = MASS_EDGES[k], MASS_EDGES[k + 1]
            ebin = [e for e in evr if lo <= e["mtot"] < hi]
            N = len(ebin)
            kM = sum(e["madgrav"] for e in ebin)
            for pipe, col in PIPEMAP[run].items():
                pd = den["runs"][run]["pipelines"].get(pipe)
                if pd is None:
                    continue
                ne, vt_p = pd["neff"][k], pd["vt_gpc3yr"][k]
                if ne is None or ne < NEFF_MIN or not vt_p or vt_p <= 0:
                    continue
                R = ours[k] / vt_p
                kP = sum(pipe_hit(e, rl, col) for e in ebin)
                r, rlo_, rhi = count_ratio_ci(kM, kP, N)
                powered = (N >= 3 and kP >= 2)
                ok = (not powered) or (rhi is not None and R <= rhi)
                if powered and not ok:
                    failures.append(dict(run=run, bin=f"{lo:.0f}-{hi:.0f}",
                                         pipe=pipe, R_vt=round(R, 3),
                                         ci_hi=round(rhi, 3)))
                lines.append(
                    f"  {run} {lo:3.0f}-{hi:3.0f} vs {pipe:12s}: "
                    f"R_vt={R:6.2f}  counts {int(kM)}/{int(kP)} N={N}"
                    + (f"  CI90hi={rhi:.2f}" if rhi is not None else "")
                    + f"  {'POWERED' if powered else 'unpowered'}"
                    + f"  {'PASS' if ok else '** FAIL **'}")
    verdict = "PREVIEW-PASS" if not failures else "PREVIEW-FAIL"
    lines += ["", f"PREVIEW VERDICT: {verdict} "
              f"({len(failures)} powered failures)"
              + " - figure remains WITHHELD either way; no further "
                "iteration (protocol)"]
    json.dump(dict(verdict=verdict, failures=failures,
                   corrections=corr_used,
                   vt_corrected={r: v.tolist() for r, v in vt_corr.items()}),
              open(f"{HERE}/gate_ratio_table_corrected.json", "w"), indent=1)
    with open(f"{HERE}/gate_ratio_table_corrected.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

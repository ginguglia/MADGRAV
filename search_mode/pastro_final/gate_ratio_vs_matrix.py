#!/usr/bin/env python
"""ACCEPTANCE GATE (2026-08-12; replaces all previous clearing
conditions on the ratio panels): the full VT-ratio table - source-frame
rebinned numerator over the Z-CONSISTENT pipeline denominator - verified
per bin against the real-event cross-recovery matrix.

Operational definitions (stated per the gate deliverable spec):
  VT ratio        R_vt = VT_MADGRAV_src[k] / VT_pipe_zc[k], pipeline bins
                  with N_eff >= 300 only.
  Count-implied   from matrix events of that run in that source-mass bin:
                  k_M vs k_P out of N shared events; r_c = k_M/k_P with
                  Katz 90% CI on ln r (0.5 continuity added when k_M = 0).
  Arbitrating power  bin has N >= 3 events AND k_P >= 2. Only powered cells
                  can fail the gate; unpowered cells are reported as such.
  Cell gate       PASS iff R_vt <= CI_hi(r_c)  (one-sided: the concern is
                  MADGRAV overstatement; the two-sided position is reported).
  Named checks    (i) O3a 100-130 vs cWB consistent with 6/11;
                  (ii) O3a pooled (all populated bins) vs cWB consistent
                  with the run totals 9/15. Both via the same CI rule.
  Overall verdict PASS iff every powered cell and both named checks pass.
                  On FAIL: stop and report - no fix iteration without a new
                  incident entry (protocol).

Out: gate_ratio_table.{json,txt}; markers campaign/step4e_gate.{pass,fail}.
Run: madgrav-venv python gate_ratio_vs_matrix.py
"""
import json
import os

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
CAMP = f"{HERE}/campaign"
Z90 = 1.6449
NEFF_MIN = 300
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
# pipeline name -> matrix column accessor
PIPEMAP = {
    "O3a": {"cWB": "cwb", "PyCBC-BBH": "PyCBC-BBH", "PyCBC-broad": "PyCBC-broad",
            "GstLAL": "GstLAL"},
    "O3b": {"cWB": "cwb", "PyCBC-BBH": "PyCBC-BBH", "PyCBC-broad": "PyCBC-broad",
            "GstLAL": "GstLAL", "MBTA": "MBTA"},
    "O4a": {"cWB": "cWB", "PyCBC": "PyCBC", "GstLAL": "GstLAL", "MBTA": "MBTA"},
    "O4b": {"cWB": "cWB", "PyCBC": "PyCBC", "GstLAL": "GstLAL", "MBTA": "MBTA"},
}


def count_ratio_ci(kM, kP, N):
    """Katz 90% CI for (kM/N)/(kP/N); 0.5 continuity when kM = 0."""
    if kP <= 0 or N <= 0:
        return None, None, None
    cm = kM if kM > 0 else 0.5
    p1, p2 = cm / N, kP / N
    r = p1 / p2
    se = np.sqrt(max(0.0, (1 - p1) / (N * p1)) + max(0.0, (1 - p2) / (N * p2)))
    return r, r * np.exp(-Z90 * se), r * np.exp(Z90 * se)


def pipe_hit(e, run, pipe_col):
    if pipe_col == "cwb":
        return bool(e["cwb_1yr"])
    if pipe_col == "cWB" and run in ("o4a", "o4b"):
        return bool(e["pipe_1yr"].get("cWB", False))
    return bool(e["pipe_1yr"].get(pipe_col, False))


def main():
    num = json.load(open(f"{HERE}/vt_relabel_comoving.json"))
    den = json.load(open(f"{HERE}/vt_pipelines_target_zc.json"))
    mat = json.load(open(f"{HERE}/cross_recovery_matrix.json"))
    events = mat["events"]

    cells, failures, lines = [], [], []
    lines.append("ACCEPTANCE GATE: VT ratio (src-frame num / z-consistent "
                 "den) vs count-implied ratio (Katz 90% CI)")
    lines.append("cell gate: R_vt <= CI_hi on powered cells "
                 "(N>=3 events, k_pipe>=2)")
    lines.append("")
    for run in ("O3a", "O3b", "O4a", "O4b"):
        rl = run.lower()
        ours = num["runs"][run]["vt_comoving_srcframe_gpc3yr"]
        evr = [e for e in events if e["run"] == rl]
        for k in range(len(MASS_EDGES) - 1):
            lo, hi = MASS_EDGES[k], MASS_EDGES[k + 1]
            ebin = [e for e in evr if lo <= e["mtot"] < hi]
            N = len(ebin)
            kM = sum(e["madgrav"] for e in ebin)
            for pipe, col in PIPEMAP[run].items():
                pd = den["runs"][run]["pipelines"].get(pipe)
                if pd is None:
                    continue
                ne = pd["neff"][k]
                vt_p = pd["vt_gpc3yr"][k]
                if ne is None or ne < NEFF_MIN or not vt_p or vt_p <= 0 \
                        or ours[k] is None:
                    continue
                R = ours[k] / vt_p
                kP = sum(pipe_hit(e, rl, col) for e in ebin)
                r, rlo_, rhi = count_ratio_ci(kM, kP, N)
                powered = (N >= 3 and kP >= 2)
                ok = (not powered) or (rhi is not None and R <= rhi)
                cell = dict(run=run, bin=f"{lo:.0f}-{hi:.0f}", pipe=pipe,
                            R_vt=round(R, 3), N=N, k_madgrav=int(kM),
                            k_pipe=int(kP),
                            r_counts=None if r is None else round(r, 3),
                            ci90=None if r is None else
                            [round(rlo_, 3), round(rhi, 3)],
                            powered=powered, gate="PASS" if ok else "FAIL")
                cells.append(cell)
                if powered and not ok:
                    failures.append(cell)
                lines.append(
                    f"  {run} {lo:3.0f}-{hi:3.0f} vs {pipe:12s}: "
                    f"R_vt={R:6.2f}  counts {int(kM)}/{int(kP)} of N={N}"
                    + (f"  r={r:.2f} CI90=[{rlo_:.2f},{rhi:.2f}]"
                       if r is not None else "  r=n/a")
                    + f"  {'POWERED' if powered else 'unpowered'}"
                    + f"  {'PASS' if ok else '** FAIL **'}")
    lines.append("")

    # named checks - AMENDED (incident entry 2026-08-12, pre-registered;
    # original-spec error acknowledged): the pooled comparison must weight
    # the per-bin VT ratios by the ASTROPHYSICAL event distribution so the
    # prediction is an expected-count ratio: r_pred = sum_k R_vt(k) *
    # w_astro(k), w_astro(k) = N_events(run, k)/sum over bins where R_vt is
    # defined. The old injected-population-weighted sum-ratio is reported
    # alongside (old vs new); the gate evaluates the AMENDED definition.
    named = {}
    def named_check(tag, run, bins, pipe, col, note):
        rl = run.lower()
        evr = [e for e in events if e["run"] == rl]
        ours = num["runs"][run]["vt_comoving_srcframe_gpc3yr"]
        pd = den["runs"][run]["pipelines"][pipe]
        usable = [k for k in bins
                  if ours[k] is not None and pd["vt_gpc3yr"][k]
                  and pd["neff"][k] and pd["neff"][k] >= NEFF_MIN]
        vt_n = sum(ours[k] for k in usable)
        vt_d = sum(pd["vt_gpc3yr"][k] for k in usable)
        R_old = vt_n / vt_d if vt_d > 0 else float("nan")
        nk = {k: sum(1 for e in evr
                     if MASS_EDGES[k] <= e["mtot"] < MASS_EDGES[k + 1])
              for k in usable}
        wsum = sum(nk.values())
        R_new = (sum((ours[k] / pd["vt_gpc3yr"][k]) * nk[k]
                     for k in usable) / wsum if wsum else float("nan"))
        if note == "all":
            ebin = evr
        else:
            ebin = [e for e in evr
                    if any(MASS_EDGES[k] <= e["mtot"] < MASS_EDGES[k + 1]
                           for k in bins)]
        N = len(ebin)
        kM = sum(e["madgrav"] for e in ebin)
        kP = sum(pipe_hit(e, rl, col) for e in ebin)
        r, rlo_, rhi = count_ratio_ci(kM, kP, N)
        ok = rhi is not None and np.isfinite(R_new) and R_new <= rhi
        named[tag] = dict(R_vt_amended=round(R_new, 3),
                          R_vt_old_injweighted=round(R_old, 3),
                          k_madgrav=int(kM), k_pipe=int(kP),
                          N=N, r_counts=round(r, 3), ci90=[round(rlo_, 3),
                                                           round(rhi, 3)],
                          gate="PASS" if ok else "FAIL")
        lines.append(f"NAMED {tag} [AMENDED]: r_pred={R_new:.2f} "
                     f"(old inj-weighted {R_old:.2f}) counts {kM}/{kP} of "
                     f"N={N} r={r:.2f} CI90=[{rlo_:.2f},{rhi:.2f}] "
                     f"{'PASS' if ok else '** FAIL **'}")
        if not ok:
            failures.append(named[tag] | {"run": run, "bin": tag, "pipe": pipe})
    named_check("O3a_100-130_vs_cWB", "O3a", [4], "cWB", "cwb", "bins")
    named_check("O3a_pooled_vs_cWB", "O3a", list(range(10)), "cWB", "cwb", "all")

    verdict = "PASS" if not failures else "FAIL"
    lines.append("")
    lines.append(f"OVERALL GATE: {verdict}"
                 + ("" if verdict == "PASS" else
                    f" ({len(failures)} powered cell(s) failed - STOPPED "
                    "per protocol; no fix iteration without a new incident "
                    "entry)"))
    out = dict(verdict=verdict, cells=cells, named=named,
               failures=failures,
               definitions="see module docstring; one-sided Katz 90% gate")
    json.dump(out, open(f"{HERE}/gate_ratio_table.json", "w"), indent=1)
    with open(f"{HERE}/gate_ratio_table.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    for m in ("step4e_gate.pass", "step4e_gate.fail"):
        try:
            os.remove(f"{CAMP}/{m}")
        except FileNotFoundError:
            pass
    open(f"{CAMP}/step4e_gate.{'pass' if verdict == 'PASS' else 'fail'}",
         "w").close()
    print(f"[gate] -> gate_ratio_table.json/.txt  verdict={verdict}")


if __name__ == "__main__":
    main()

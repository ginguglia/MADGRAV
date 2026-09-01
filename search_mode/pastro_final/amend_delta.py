#!/usr/bin/env python
"""HP-amendment delta report (decision 2026-08-13): old -> new for
c_median, per-bin VT, and gate verdicts after the 15 Hz HP rebuild of the O3
release PSDs flowed through relabel -> comoving -> gates.

Old = campaign/archive/amend_hp_pre/ snapshots; new = current files.
Out: amend_delta.{json,txt}
"""
import json

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


HERE = MADGRAV_ROOT + "/search_mode/pastro_final"
ARC = f"{HERE}/campaign/archive/amend_hp_pre"


def j(path):
    return json.load(open(path))


def fmt_list(v, nd=2):
    return "[" + " ".join(f"{x:.{nd}f}" if x is not None else "-" for x in v) + "]"


def main():
    old_r, new_r = j(f"{ARC}/vt_relabel_release.json"), j(f"{HERE}/vt_relabel_release.json")
    old_c, new_c = j(f"{ARC}/vt_relabel_comoving.json"), j(f"{HERE}/vt_relabel_comoving.json")
    rep = {"runs": {}}
    lines = ["HP-AMENDMENT DELTA REPORT (old -> new)",
             f"mass bins: {old_r['mass_edges']}", ""]
    for run in old_r["runs"]:
        o, n = old_r["runs"][run], new_r["runs"][run]
        oc, nc = old_c["runs"].get(run, {}), new_c["runs"].get(run, {})
        entry = {"c_median_old": o["c_median"], "c_median_new": n["c_median"],
                 "vt_old": o.get("vt_gpc3yr"), "vt_new": n.get("vt_gpc3yr")}
        rep["runs"][run] = entry
        co = np.array([x if x else np.nan for x in o["c_median"]], float)
        cn = np.array([x if x else np.nan for x in n["c_median"]], float)
        max_c_shift = np.nanmax(np.abs(cn / co - 1)) if np.isfinite(co).any() else 0.0
        lines += [f"[{run}] c_median old {fmt_list(o['c_median'])}",
                  f"       c_median new {fmt_list(n['c_median'])}   "
                  f"(max shift {max_c_shift:.1%})"]
        for tag, od, nd_ in (("release VT", o, n), ("comoving VT", oc, nc)):
            for key in ("vt_gpc3yr", "vtc_gpc3yr", "vt_comoving_gpc3yr"):
                if key in od and key in nd_:
                    ov = np.array([x if x else np.nan for x in od[key]], float)
                    nv = np.array([x if x else np.nan for x in nd_[key]], float)
                    with np.errstate(all="ignore"):
                        ratio = nv / ov
                    lines += [f"       {tag} old {fmt_list(od[key], 3)}",
                              f"       {tag} new {fmt_list(nd_[key], 3)}   "
                              f"(bin ratios {fmt_list(list(ratio), 2)})"]
                    entry[f"{tag}_{key}_ratio"] = [None if not np.isfinite(x)
                                                   else float(x) for x in ratio]
                    break
        lines.append("")

    # O4 invariance check: O4 inputs untouched -> identical labels expected
    for run in ("O4a", "O4b"):
        same = np.allclose(
            [x or 0 for x in old_r["runs"][run]["c_median"]],
            [x or 0 for x in new_r["runs"][run]["c_median"]], rtol=1e-9)
        rep[f"{run}_invariant"] = bool(same)
        lines.append(f"O4 invariance [{run}]: c_median identical = {same}")
    lines.append("")

    # gate verdict old -> new (band-mode table): the table is {"cells": [...],
    # "named": [...]} (all lists), so store the full content plus a derived
    # overall verdict — the archive must be self-contained.
    for tag, path in (("old", f"{ARC}/gate_ratio_table_band.json"),
                      ("new", f"{HERE}/gate_ratio_table_band.json")):
        try:
            g = j(path)
            cells = g.get("cells", [])
            named = g.get("named", [])
            fails = [c for c in cells if c.get("powered")
                     and "FAIL" in (c.get("verdict_central", ""),
                                    c.get("verdict_lower", ""))]
            nfail = [n for n in named if "FAIL" in (n.get("verdict_central", ""),
                                                    n.get("verdict_lower", ""))]
            verdict = {
                "overall": "FAIL" if (fails or nfail) else "PASS",
                "n_cells": len(cells),
                "n_powered_cell_fails": len(fails),
                "powered_cell_fails": [
                    f"{c['bin']}_vs_{c['pipe']}" for c in fails],
                "named_fails": [n["tag"] for n in nfail],
                "cells": cells, "named": named}
            rep[f"gate_{tag}"] = verdict
            lines.append(f"gate ({tag}): overall={verdict['overall']} "
                         f"powered_cell_fails={verdict['powered_cell_fails']} "
                         f"named_fails={verdict['named_fails']}")
        except FileNotFoundError:
            lines.append(f"gate ({tag}): table missing")
    json.dump(rep, open(f"{HERE}/amend_delta.json", "w"), indent=1)
    with open(f"{HERE}/amend_delta.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

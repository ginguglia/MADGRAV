#!/usr/bin/env python
"""O4b per-segment recovery diagnostic (2026-08-13): break the pilot-3e
corrected-refs recovery down per segment and set it against each segment's
background elevation, to separate 'thin because loud segments' from 'thin
because model'.

Recovery = stage-1 net >= 4.0 on the PHYSICAL SNR grid (identical definition
to pilot3e_report.py). Bg rates recomputed directly from the pilot3e_bg_out
npz files (not read from the report json). Wilson 90% CIs on fractions.

Inputs:  search_mode/inj_out_pilot3e_o4b/<seg>_inj.npz   (6 segments, step 3h)
         search_mode/pilot3e_bg_out/<seg>_{asrun,corrected}_bg.npz
Output:  pilot3e_perseg.{json,txt}
"""
import json
import os

import numpy as np
from scipy.stats import norm, spearmanr
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
SM = f"{MG}/search_mode"
INJ = f"{SM}/inj_out_pilot3e_o4b"
BG = f"{SM}/pilot3e_bg_out"
NET_CUT = 4.0
LOUD_CUT = 50.0  # corrected net>=4 rate /h separating the loud triple


def wilson90(k, n):
    if n == 0:
        return None
    z = norm.ppf(0.95)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def bg_rate(seg, variant):
    z = np.load(f"{BG}/{seg}_{variant}_bg.npz")
    h = len(z["net"]) / 3600.0
    n = int((z["net"] >= NET_CUT).sum())
    return n / h if h else None, n, h


def main():
    segs = list(json.load(open(f"{SM}/o4b_events_inj.json")))[:6]
    rep = {"net_cut": NET_CUT, "loud_cut_per_h": LOUD_CUT, "segments": {}}
    lines = ["O4B PER-SEGMENT RECOVERY vs BACKGROUND ELEVATION (step 3h)",
             "recovery = stage-1 net>=4, corrected (release run-median) prep,"
             " PHYSICAL SNR grid; Wilson 90% CI", ""]
    rec8, crate = [], []
    for seg in segs:
        f = f"{INJ}/{seg}_inj.npz"
        if not os.path.exists(f):
            lines.append(f"{seg}: INJECTION FILE MISSING - skipped")
            continue
        z = np.load(f)
        S = {"recovery": {}, "background": {}}
        for pop, sel_pop in (("sig", z["is_um"] == 0), ("um", z["is_um"] == 1),
                             ("all", np.ones(len(z["net"]), bool))):
            lv = {}
            for s in np.unique(z["net_snr"]):
                sel = (z["net_snr"] == s) & sel_pop
                if sel.sum():
                    k, n = int((z["net"][sel] >= NET_CUT).sum()), int(sel.sum())
                    lv[float(s)] = dict(n=n, recovered=k, frac=k / n,
                                        ci90=wilson90(k, n))
            S["recovery"][pop] = lv
        for variant in ("asrun", "corrected"):
            r, n, h = bg_rate(seg, variant)
            S["background"][variant] = dict(rate_per_h=r, n_net4=n, hours=h)
        ra = S["background"]["asrun"]["rate_per_h"]
        rc = S["background"]["corrected"]["rate_per_h"]
        S["background"]["elevation"] = (rc / ra) if (ra and rc is not None) else None
        S["loud"] = bool(rc is not None and rc > LOUD_CUT)
        rep["segments"][seg] = S
        a8 = S["recovery"]["all"].get(8.0)
        if a8 and rc is not None:
            rec8.append(a8["frac"])
            crate.append(rc)
        elev = S["background"]["elevation"]
        lines.append(
            f"{seg}  [{'LOUD ' if S['loud'] else 'quiet'}]  "
            f"bg corrected {rc:.1f}/h (as-run {ra:.2f}/h, "
            f"elev {'x%.1f' % elev if elev else 'n/a - as-run 0'})")
        for pop in ("sig", "um", "all"):
            lv = S["recovery"][pop]
            row = "  ".join(f"snr{int(s)}: {v['frac']:.2f}"
                            f"[{v['ci90'][0]:.2f},{v['ci90'][1]:.2f}]"
                            for s, v in sorted(lv.items()))
            lines.append(f"    {pop:3s}: {row}")
        lines.append("")

    # pooled loud-vs-quiet contrast + rank correlation at snr8
    for grp, sel in (("quiet", False), ("loud", True)):
        ks = ns = 0
        for seg, S in rep["segments"].items():
            if S["loud"] is sel and 8.0 in S["recovery"]["all"]:
                ks += S["recovery"]["all"][8.0]["recovered"]
                ns += S["recovery"]["all"][8.0]["n"]
        rep[f"pooled_{grp}_snr8"] = dict(recovered=ks, n=ns,
                                         frac=ks / ns if ns else None,
                                         ci90=wilson90(ks, ns))
        if ns:
            p = rep[f"pooled_{grp}_snr8"]
            lines.append(f"pooled {grp} snr8: {p['frac']:.3f} "
                         f"[{p['ci90'][0]:.3f},{p['ci90'][1]:.3f}] ({ks}/{ns})")
    if len(rec8) >= 4:
        rho, pval = spearmanr(rec8, crate)
        rep["spearman_rec8_vs_corrected_rate"] = dict(rho=float(rho),
                                                      p=float(pval))
        lines.append(f"Spearman recovery@8 vs corrected bg rate: "
                     f"rho={rho:.2f} (p={pval:.2f}, n={len(rec8)})")

    json.dump(rep, open(f"{HERE}/pilot3e_perseg.json", "w"), indent=1)
    with open(f"{HERE}/pilot3e_perseg.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

"""Source-frame detection efficiency, so the two panels of vt_vs_mass_paper share one mass axis.

`vt_relabel_comoving` reports `eff_covered` binned in DETECTOR-frame total mass, while the VT it
reports alongside is rebinned to SOURCE-frame mass. Plotting both against the same bin centres made
the panels non-comparable point-by-point (a ~115 Msun source-frame bin is ~150-170 Msun in the
detector frame at z~0.3-0.5). This recomputes efficiency in the SOURCE-frame bins the VT already
uses, from artifacts vt_relabel already wrote -- no re-run of the relabel layer:

    eff_src[k] = sum_i (w0_i * det_i) / sum_i w0_i     over injections with src_bin == k

i.e. the same weighted efficiency, over the cohort whose SOURCE-frame mass lands in bin k.

Usage: SM_VT_SUF=_x1cnnfixveto build_eff_srcframe.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUF = os.environ.get("SM_VT_SUF", "")
RUNS = ["O3a", "O3b", "O4a", "O4b"]


def main():
    rel = json.load(open(f"{HERE}/vt_relabel_comoving{SUF}.json"))
    edges = np.array(rel["mass_edges"], float)
    nb = len(edges) - 1
    out = {}
    for run in RUNS:
        ri = np.load(f"{HERE}/relabel_inj_{run.lower()}{SUF}.npz")
        zi = np.load(f"{HERE}/inj_scored_{run.lower()}{SUF}.npz")
        w0 = np.asarray(zi["w0"], float)
        det = np.asarray(zi["det_frac"], float)
        sb, db = ri["src_bin"], ri["det_bin"]
        eff_src, eff_det, n = [], [], []
        for k in range(nb):
            m = sb == k
            eff_src.append(float((w0[m] * det[m]).sum() / w0[m].sum()) if w0[m].sum() > 0 else None)
            m2 = db == k
            eff_det.append(float((w0[m2] * det[m2]).sum() / w0[m2].sum()) if w0[m2].sum() > 0 else None)
            n.append(int(m.sum()))
        out[run] = dict(eff_srcframe=eff_src, eff_detframe=eff_det, n_inj_srcframe=n)
        fmt = lambda a: " ".join("  -  " if x is None else f"{x:5.3f}" for x in a)
        print(f"[{run}] src {fmt(eff_src)}")
        print(f"      det {fmt(eff_det)}")
    json.dump(out, open(f"{HERE}/eff_srcframe{SUF}.json", "w"), indent=1)
    print(f"-> {HERE}/eff_srcframe{SUF}.json")


if __name__ == "__main__":
    main()

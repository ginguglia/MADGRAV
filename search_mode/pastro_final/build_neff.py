"""Build neff_srcframe<SUF>.json -- the source-frame SUPPORT N_eff that fig_vt_paper masks on.

Mirrors vt_compare_pipelines.py exactly: SUPPORT N_eff = (sum w)^2 / sum w^2 with
w = cohort-normalised w0 * kept, computed PRE-detection (the detection-weighted "contribution"
N_eff is stored alongside for transparency only, as there). Points below NEFF_MIN=300 are masked
in the figure because they are kinematically drained, not insensitive.

For the accepted _x1cnn build this was reused from a sibling suffix because `kept` did not move.
That is NOT true for the reference-PSD-corrected build: kept = net_snr * c >= RHO_TH and c changes
with the ASD, so the mask must be recomputed per suffix.

Usage: SM_VT_SUF=_x1cnnfixveto build_neff.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUF = os.environ.get("SM_VT_SUF", "")
RUNS = ["O3a", "O3b", "O4a", "O4b"]
NEFF_MIN = 300.0


def main():
    rel = json.load(open(f"{HERE}/vt_relabel_comoving{SUF}.json"))
    edges = np.array(rel["mass_edges"], float)
    out = {"neff_srcframe": {}}
    for run in RUNS:
        ri = np.load(f"{HERE}/relabel_inj_{run.lower()}{SUF}.npz")
        zi = np.load(f"{HERE}/inj_scored_{run.lower()}{SUF}.npz")
        w0, det = zi["w0"], zi["det_frac"]
        db, sb, kept = ri["det_bin"], ri["src_bin"], ri["kept"]
        W = np.zeros_like(w0, dtype=float)
        for b in range(len(edges) - 1):
            sel = db == b
            if sel.sum():
                W[sel] = w0[sel] / w0[sel].sum()

        def _neff(c):
            r = []
            for k in range(len(edges) - 1):
                ck = c[sb == k]
                r.append(float(ck.sum() ** 2 / (ck ** 2).sum()) if ck.sum() > 0 else 0.0)
            return r

        neff = _neff(W * kept)
        out["neff_srcframe"][run] = dict(support=neff, contribution=_neff(W * kept * det),
                                         n_inj=[int((sb == k).sum()) for k in range(len(edges) - 1)])

        # Detector-frame support, so a detector-frame panel can be masked in its OWN frame rather
        # than borrowing the source-frame mask (the two differ: the relabel moves weight down-mass).
        def _neff_det(c):
            r = []
            for k in range(len(edges) - 1):
                ck = c[db == k]
                r.append(float(ck.sum() ** 2 / (ck ** 2).sum()) if ck.sum() > 0 else 0.0)
            return r

        out.setdefault("neff_detframe", {})[run] = dict(
            support=_neff_det(W * kept), contribution=_neff_det(W * kept * det),
            n_inj=[int((db == k).sum()) for k in range(len(edges) - 1)])
        nd = out["neff_detframe"][run]["support"]
        print(f"[{run}] src N_eff {np.round(neff, 0).astype(int).tolist()} masked "
              f"{[k for k, v in enumerate(neff) if v < NEFF_MIN]}", flush=True)
        print(f"      det N_eff {np.round(nd, 0).astype(int).tolist()} masked "
              f"{[k for k, v in enumerate(nd) if v < NEFF_MIN]}", flush=True)
    json.dump(out, open(f"{HERE}/neff_srcframe{SUF}.json", "w"), indent=1)
    print(f"-> {HERE}/neff_srcframe{SUF}.json")


if __name__ == "__main__":
    main()

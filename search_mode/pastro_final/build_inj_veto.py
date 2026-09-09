"""Fold the local-ASD veto into the CNN-gated injection scoring -> inj_scored_<run>_x1cnnveto.npz.

The veto is a per-injection boolean that is INDEPENDENT of the CNN pair index, while det_frac is
the mean of the per-pair detection matrix over pairs. ANDing the veto into the trigger condition
is therefore EXACTLY det_frac * veto_keep -- no re-run of the FAR scoring is needed, and the far
column is untouched (the veto removes candidates, it does not move their FAR).

Guards:
  V1  draw columns (mtot, net_snr, is_um, off) identical between inj_asdveto and inj_cnn, file by file
  V2  concatenation order matches inj_scored_<run>_x1cnn.npz element-wise (the same G2 vt_relabel uses)
  V3  no injection has det_frac > 0 while the veto column was never evaluated (gate-fail fallback);
      inject.py writes veto_keep=0 + net_loc=NaN for gate-failures, which must already be det_frac=0
"""
import glob
import os

import numpy as np

import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = MADGRAV_SCRATCH
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = ["O3a", "O3b", "O4a", "O4b"]
# SM_VETO_SRC / SM_VETO_SUF: which campaign carries the veto column, and which inj_scored suffix
# it is folded into. Defaults reproduce the accepted _x1cnn -> _x1cnnveto build exactly.
SRC = os.environ.get("SM_VETO_SRC", "inj_asdveto")
SUF_IN = os.environ.get("SM_VETO_SUF", "_x1cnn")
REF = os.environ.get("SM_VETO_REF", "inj_cnn")
# SM_VETO_NOREF=1 -> skip guard V1 only. V1 proves the veto campaign used the SAME DRAWS as a
# shape-matched reference campaign; a MERGED campaign (e.g. inj_all = inj_fixed2 + inj_hi) has
# more rows than any reference by construction, so V1 is inapplicable rather than violated.
# V2 (element-wise alignment against the scored file) and V3 still run and are what actually
# protect correctness -- the veto column is read from the very files pastro_final consumed.
# DEFAULT UNSET = the accepted _x1cnn -> _x1cnnveto build is byte-identical.
NOREF = os.environ.get("SM_VETO_NOREF", "0") == "1"


def main():
    for run in RUNS:
        r = run.lower()
        dirs = [f"{SC}/{SRC}/{r}", f"{SC}/{SRC}/{r}_lowsnr"]
        keep, gate, mtot, snr = [], [], [], []
        nfile = 0
        for d in dirs:
            for f in sorted(glob.glob(f"{d}/*_inj.npz")):
                z = np.load(f)
                if not NOREF:
                    ref = np.load(f.replace(SRC, REF))
                    for c in ("mtot", "net_snr", "is_um", "off"):
                        assert np.array_equal(z[c], ref[c]), f"V1 FAIL {f}: column {c}"
                keep.append(z["veto_keep"].astype(bool))
                gate.append(np.maximum(z["cnn_hm"], z["cnn_lm"]) > 0.5)
                mtot.append(np.asarray(z["mtot"], float))
                snr.append(np.asarray(z["net_snr"], float))
                nfile += 1
        keep = np.concatenate(keep); gate = np.concatenate(gate)
        mtot = np.concatenate(mtot); snr = np.concatenate(snr)

        src = f"{HERE}/inj_scored_{r}{SUF_IN}.npz"
        z = np.load(src)
        assert np.array_equal(mtot, z["mtot"]) and np.array_equal(snr, z["net_snr"]), \
            f"V2 FAIL {run}: inj_asdveto concatenation does not align with {os.path.basename(src)}"
        df = np.asarray(z["det_frac"], float)
        bad = int(((df > 0) & ~gate).sum())
        assert bad == 0, f"V3 FAIL {run}: {bad} injections detected but never veto-evaluated"

        out = {k: z[k] for k in z.files}
        out["det_frac"] = df * keep
        if "det_frac_sweep" in out:          # same veto, every sweep threshold
            out["det_frac_sweep"] = np.asarray(out["det_frac_sweep"], float) * keep[None, :]
        out["veto_keep"] = keep
        np.savez(f"{HERE}/inj_scored_{r}{SUF_IN}veto.npz", **out)
        w = np.asarray(z["w0"], float)
        e0 = float((df * w).sum() / w.sum())
        e1 = float((out["det_frac"] * w).sum() / w.sum())
        print(f"[{run}] files={nfile} inj={len(df)} "
              f"{'V2/V3 pass (V1 skipped: merged campaign)' if NOREF else 'V1/V2/V3 pass'} | "
              f"gate-pass {gate.mean():.3f}  veto retention on gate-pass "
              f"{keep[gate].mean():.3f} | w-eff {e0:.4f} -> {e1:.4f} "
              f"({e1 / e0 if e0 else float('nan'):.3f}x)", flush=True)


if __name__ == "__main__":
    main()

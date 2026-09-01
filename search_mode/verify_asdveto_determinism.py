"""Construction proof for the ASD-veto injection pass.

The veto re-runs the injection campaign with SM_INJ_ASDVETO=1. Three things must hold, and they
are checked at three different strengths -- deliberately, because they have different natures:

  1. DRAW columns (net_snr, mtot, is_um, off) are pure RNG output. EXACT equality required.
     If these move, the injections are different signals and the run is worthless.
  2. PEAK-WINDOW columns (cen*, g*, morph, cnn_*) are read at the argmax-net grid index. EXACT
     equality required -- it is what proves the SAME peak window was selected for every
     injection, i.e. the veto acts on the window the original pass reported.
  3. SIGMA columns (sigH, sigL, net) come out of the CAE forward pass in float32. Exact equality
     is NOT required and NOT achievable: kernel selection differs between nodes/allocations, so
     reduction order differs and results move at the 1e-6 level (float32 eps = 1.2e-7). Bounded
     by TOL_REL instead.

  4. DECISIONS (CNN glitch gate, net floor) must be identical. This is the check that actually
     matters for the science: it is what says the float32 noise changed no outcome.

History: the first version of this file required exact equality everywhere and failed on (3) at
max rel 4.1e-6 while (1), (2) and (4) all held -- i.e. it flagged float32 kernel noise as if it
were injection drift. The tolerance below is set ~25x above that observed noise and ~1000x below
anything that could move a decision; it is NOT tuned to make a particular run pass.

Usage: verify_asdveto_determinism.py <new.npz> <reference.npz>
"""
import os
import sys

import numpy as np

DRAW_COLS = ("net_snr", "mtot", "is_um", "off")
PEAK_COLS = ("coh", "cenH", "cenL", "gH", "gL", "chirpH", "vertH", "eccH",
             "chirpL", "vertL", "eccL", "cnn_hm", "cnn_lm")
SIGMA_COLS = ("sigH", "sigL", "net")
NEW_COLS = ("net_loc", "hm_loc", "lm_loc", "veto_keep")

# Bound the sigma columns on ABSOLUTE difference (np.allclose semantics), not relative: sigma is a
# normalised deviation that passes through zero, so a relative bound explodes on near-zero values
# while the actual error stays at the float32 noise floor. Measured across runs: max |d| = 1.7e-6,
# regardless of SNR band; the relative "blow-up" was 6.1e-4 at a sigma of -6.7e-4, where |d| = 4e-7.
# TOL_ABS is ~60x that noise floor and ~100x below what could flip a decision (~1e-2), and the
# decision-invariance check below is asserted independently in any case.
TOL_ABS = 1e-4
TOL_REL = 1e-4
GATE = 0.5
NET_FLOOR = 4.0


DRAWS_ONLY = os.environ.get("SM_VERIFY_DRAWS_ONLY", "0") == "1"


def main(new_p, ref_p):
    """DRAWS_ONLY mode: for the reference-PSD-corrected campaign (SM_INJ_NORM_FIXED=1) the injected
    AMPLITUDES change by construction, so peak-window/sigma/decision equality is NOT the invariant --
    it would be a bug if they did not move. What must still hold exactly is that the SAME signals
    were drawn at the same times: DRAW_COLS. The other groups are reported, never asserted."""
    a = np.load(new_p); b = np.load(ref_p)
    bad = []

    groups = [("draw", DRAW_COLS, True)]
    if not DRAWS_ONLY:
        groups += [("peak-window", PEAK_COLS, True), ("sigma", SIGMA_COLS, False)]
    for group, cols, exact in groups:
        for c in cols:
            if c not in b:
                continue
            if c not in a:
                bad.append(f"[{group}] {c}: missing from new file"); continue
            x, y = a[c], b[c]
            if x.shape != y.shape:
                bad.append(f"[{group}] {c}: shape {x.shape} vs {y.shape}"); continue
            if exact:
                if not np.array_equal(x, y):
                    bad.append(f"[{group}] {c}: {int((x != y).sum())}/{x.size} differ -- must be exact")
            else:
                d = np.abs(x - y)
                if not np.all(d <= TOL_ABS + TOL_REL * np.abs(y)):
                    bad.append(f"[{group}] {c}: max |d| {np.nanmax(d):.3e} exceeds "
                               f"{TOL_ABS:.0e}+{TOL_REL:.0e}|y|")

    for c in NEW_COLS:
        if c not in a:
            bad.append(f"[veto] {c}: absent -- did SM_INJ_ASDVETO reach the writer?")

    # the check that matters: float noise must not have moved any outcome
    if not DRAWS_ONLY and all(c in a and c in b for c in ("cnn_hm", "cnn_lm", "net")):
        ga = np.maximum(a["cnn_hm"], a["cnn_lm"]) > GATE
        gb = np.maximum(b["cnn_hm"], b["cnn_lm"]) > GATE
        if not np.array_equal(ga, gb):
            bad.append(f"[decision] gate flipped for {int((ga != gb).sum())} injections")
        if not np.array_equal(a["net"] >= NET_FLOOR, b["net"] >= NET_FLOOR):
            bad.append(f"[decision] net>={NET_FLOOR} flipped for "
                       f"{int(((a['net'] >= NET_FLOOR) != (b['net'] >= NET_FLOOR)).sum())} injections")

    if bad:
        print("DETERMINISM FAIL " + new_p)
        for m in bad:
            print("  " + m)
        return 1

    gate = np.maximum(a["cnn_hm"], a["cnn_lm"]) > GATE
    keep = a["veto_keep"].astype(bool)
    if DRAWS_ONLY:
        gb = np.maximum(b["cnn_hm"], b["cnn_lm"]) > GATE
        dn = np.nanmedian(a["net"] / np.where(b["net"] == 0, np.nan, b["net"]))
        print(f"DRAWS OK {new_p}: {len(a['net'])} inj; draw columns bit-identical. "
              f"EXPECTED to move: median net ratio new/old {dn:.3f}, gate-pass "
              f"{int(gate.sum())} vs {int(gb.sum())} ({gate.sum()/max(1,gb.sum()):.3f}x), "
              f"veto-keep {int(keep.sum())} (retention {keep.sum()/max(1,gate.sum()):.4f})")
        return 0
    sig = max(np.nanmax(np.abs(a[c] - b[c])) for c in SIGMA_COLS)
    print(f"DETERMINISM OK {new_p}: {len(a['net'])} inj; draws + peak-window columns bit-identical, "
          f"sigma max |d| {sig:.2e}, all decisions unchanged | "
          f"gate-pass {int(gate.sum())}, veto-keep {int(keep.sum())} "
          f"(retention {keep.sum()/max(1, gate.sum()):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))

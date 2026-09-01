"""Aggregate the ASD-veto injection campaign into the retention table.

Reads every /scratch/.../inj_asdveto/<band>/*_inj.npz, re-verifies each against its inj_cnn
reference, and reports retention (kept / gate-pass) per run and per injected network SNR --
the numbers the VT efficiency correction needs.

Writes asdveto_retention.json + asdveto_retention.txt next to this file. Stdout is the report.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


INJ = MADGRAV_SCRATCH + "/inj_asdveto"
REF = MADGRAV_SCRATCH + "/inj_cnn"
HERE = os.path.dirname(os.path.abspath(__file__))
GATE = 0.5
EXPECTED = 110


def main():
    files = sorted(glob.glob(f"{INJ}/*/*_inj.npz"))
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out(f"MADGRAV -- local-ASD veto on the injection campaign")
    out(f"files: {len(files)}/{EXPECTED}")
    if len(files) < EXPECTED:
        out(f"WARNING: incomplete ({EXPECTED - len(files)} missing) -- numbers below are partial")

    run_snr = defaultdict(lambda: [0, 0])
    run_tot = defaultdict(lambda: [0, 0])
    all_tot = [0, 0]
    bad = []
    for p in files:
        band = os.path.basename(os.path.dirname(p))
        run = band.replace("_lowsnr", "")
        ref = p.replace("inj_asdveto", "inj_cnn")
        d = np.load(p)
        if not os.path.exists(ref):
            bad.append(f"{p}: no reference"); continue
        r = np.load(ref)
        for c in ("net_snr", "mtot", "is_um", "off"):     # draws must be exact
            if not np.array_equal(d[c], r[c]):
                bad.append(f"{p}: draw column {c} differs"); break
        g = np.maximum(d["cnn_hm"], d["cnn_lm"]) > GATE
        k = d["veto_keep"].astype(bool)
        run_tot[run][0] += int(g.sum()); run_tot[run][1] += int((g & k).sum())
        all_tot[0] += int(g.sum());      all_tot[1] += int((g & k).sum())
        for s in np.unique(d["net_snr"]):
            m = d["net_snr"] == s
            run_snr[(run, float(s))][0] += int(g[m].sum())
            run_snr[(run, float(s))][1] += int((g & k)[m].sum())

    runs = sorted({r for r, _ in run_snr})
    snrs = sorted({s for _, s in run_snr})
    out()
    out("retention = kept / gate-pass  (fraction of gate-passing injections surviving the local-ASD veto)")
    out()
    out(f"{'SNR':>5}  " + "".join(f"{r:>18}" for r in runs))
    for s in snrs:
        row = f"{s:>5.0f}  "
        for r in runs:
            n, kk = run_snr.get((r, s), [0, 0])
            row += f"{(f'{kk}/{n} {kk/n:.3f}' if n else '-'):>18}"
        out(row)
    out()
    out(f"{'pooled':>5}  " + "".join(
        f"{(f'{run_tot[r][1]}/{run_tot[r][0]} {run_tot[r][1]/run_tot[r][0]:.3f}' if run_tot[r][0] else '-'):>18}"
        for r in runs))
    out()
    out(f"ALL RUNS: {all_tot[1]}/{all_tot[0]} = {all_tot[1]/max(1,all_tot[0]):.4f} pooled retention")
    out("NOTE: pooled numbers mix SNR bands and are only comparable across runs when every band is in.")
    if bad:
        out()
        out(f"VERIFY FAILURES ({len(bad)}):")
        for b in bad[:20]:
            out("  " + b)

    res = dict(n_files=len(files), expected=EXPECTED, complete=len(files) >= EXPECTED,
               pooled=dict(gate_pass=all_tot[0], kept=all_tot[1],
                           retention=all_tot[1] / max(1, all_tot[0])),
               per_run={r: dict(gate_pass=run_tot[r][0], kept=run_tot[r][1],
                                retention=run_tot[r][1] / max(1, run_tot[r][0])) for r in runs},
               per_run_snr={f"{r}_{s:g}": dict(gate_pass=n, kept=k, retention=k / n)
                            for (r, s), (n, k) in sorted(run_snr.items()) if n},
               verify_failures=bad)
    json.dump(res, open(os.path.join(HERE, "asdveto_retention.json"), "w"), indent=1)
    open(os.path.join(HERE, "asdveto_retention.txt"), "w").write("\n".join(lines) + "\n")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

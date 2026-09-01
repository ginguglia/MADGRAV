"""Variance decomposition for the bottleneck replicate campaign (wave 4).

3 variants x 3 draws x 4 repeats. Waves 1-3 could not separate the architectures because BOTH
noise components are comparable to the effect: draw-to-draw (different 36 h training sets) and
run-to-run at FIXED data (GPU kernel nondeterminism; there is no seed flag). This splits them and
reports the A-vs-C contrast against the right error.
"""
import glob, re, sys
import numpy as np
from collections import defaultdict
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


PAT_HDR = re.compile(r'^\[bneckrep\] task \d+ variant=(\S+) latent_channels=\d+ draw=(\d+) rep=(\d+)', re.M)
PAT_CKPT = re.compile(r'weaksup best checkpoint: epoch=(\d+).*?val_inj>3sigma=(\d+) val_sep_k1=([\d.eE+-]+)')


def collect():
    out = defaultdict(list)   # variant -> [(draw, rep, epoch, n3sig, sep)]
    for f in sorted(glob.glob(MADGRAV_ROOT + '/launchers/bneckrep_*_*.log')):
        h = open(f, errors='replace').read()
        m = PAT_HDR.search(h)
        c = PAT_CKPT.search(h)
        if not m or not c:
            continue
        out[m.group(1)].append((int(m.group(2)), int(m.group(3)),
                                int(c.group(1)), int(c.group(2)), float(c.group(3))))
    return out


def components(vals_by_draw):
    """-> (between-draw sd, within-draw sd) of the draw means / residuals."""
    means = [np.mean(v) for v in vals_by_draw if len(v)]
    within = [np.std(v, ddof=1) for v in vals_by_draw if len(v) > 1]
    b = float(np.std(means, ddof=1)) if len(means) > 1 else float('nan')
    w = float(np.sqrt(np.mean(np.array(within) ** 2))) if within else float('nan')
    return b, w


def main():
    data = collect()
    if not data:
        print("no replicate logs parsed yet")
        return 1
    print(f"{'variant':<8}{'n':>4}{'val>3sigma mean+/-sd':>24}{'val_sep_k1 mean+/-sd':>24}{'best-epoch median':>20}")
    summ = {}
    for v in ('A', 'C_k32', 'C_k20'):
        r = data.get(v, [])
        if not r:
            continue
        g = np.array([x[3] for x in r], float); s = np.array([x[4] for x in r], float)
        e = np.array([x[2] for x in r], float)
        summ[v] = (g, s, r)
        print(f"{v:<8}{len(r):>4}{f'{g.mean():.1f} +/- {g.std(ddof=1):.1f}':>24}"
              f"{f'{s.mean():.2f} +/- {s.std(ddof=1):.2f}':>24}{np.median(e):>20.0f}")

    print()
    print("variance components (val_sep_k1):")
    for v, (g, s, r) in summ.items():
        by = [[x[4] for x in r if x[0] == d] for d in (1, 2, 3)]
        b, w = components(by)
        print(f"  {v:<8} between-draw sd {b:5.2f}   within-draw (run-to-run) sd {w:5.2f}")

    print()
    print("contrasts vs A (Welch, unpaired over all runs):")
    if 'A' in summ:
        ga, sa, _ = summ['A']
        for v in ('C_k32', 'C_k20'):
            if v not in summ:
                continue
            gv, sv, _ = summ[v]
            for nm, a, b_ in (("val>3sigma", ga, gv), ("val_sep_k1", sa, sv)):
                d = a.mean() - b_.mean()
                se = np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_))
                t = d / se if se > 0 else float('nan')
                print(f"  A - {v:<6} {nm:<12} diff {d:+8.2f}  SE {se:6.2f}  t {t:+5.2f}"
                      f"   {'RESOLVED' if abs(t) > 2 else 'not resolved'}")
    print()
    print("A positive t means A is BETTER. |t|>2 is the bar; anything less means this campaign")
    print("cannot tell the architectures apart, whatever the means happen to be.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

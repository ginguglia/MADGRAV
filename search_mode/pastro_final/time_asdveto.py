"""Timing probe: cost of the local-ASD veto (asd_consistency.recompute_local) per window.

Pure measurement -- writes no science output. Times N windows after a warm-up call so the
cached pipeline / arm / CNN loads are excluded, and separately times the local Welch ASD leg
so the ASD cost can be told apart from the rescore cost.
"""
import os, sys, time, json
import numpy as np

MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT")
for _p in ("search_mode", "improved", "spectrogram_cascade"):
    _ap = os.path.join(MADGRAV_ROOT, _p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)


if __name__ == "__main__":
    import driver_blindscan as B
    import asd_consistency as AC

    N = int(os.environ.get("PROBE_N", "12"))
    segjson = os.environ["SM_BGJSON"]
    segs = json.load(open(segjson))
    names = [s[3] if isinstance(s, (list, tuple)) else s for s in (segs["segments"] if isinstance(segs, dict) else segs)]
    seg = None
    for nm in names:
        try:
            r = B._strain(nm, "H1")
            if r is not None and len(r) > 4096 * 400:
                seg = nm
                break
        except Exception as e:
            continue
    if seg is None:
        print("[probe] no usable segment", flush=True)
        sys.exit(2)
    n = len(B._strain(seg, "H1"))
    print(f"[probe] seg={seg} nsamp={n} ({n/4096:.0f}s)", flush=True)

    rng = np.random.default_rng(0)
    lo, hi = 200, int(n / 4096) - 200
    idxs = rng.integers(lo, hi, size=N + 1)

    # warm-up (loads pipeline, glitch arm, CNN heads -- excluded from the timing)
    t0 = time.time()
    AC.recompute_local(seg, int(idxs[0]))
    print(f"[probe] warm-up (incl. model load) {time.time()-t0:.1f}s", flush=True)

    # full veto call
    ts = []
    for k, i in enumerate(idxs[1:]):
        t = time.time()
        AC.recompute_local(seg, int(i))
        ts.append(time.time() - t)
        print(f"[probe] window {k+1}/{N} {ts[-1]:.3f}s", flush=True)

    # ASD leg alone
    ta = []
    for i in idxs[1:1 + min(N, 6)]:
        t = time.time()
        AC._local_asds(seg, int(i))
        ta.append(time.time() - t)

    ts = np.array(ts); ta = np.array(ta)
    out = dict(seg=seg, n_windows=int(N),
               full_mean_s=float(ts.mean()), full_median_s=float(np.median(ts)),
               full_min_s=float(ts.min()), full_max_s=float(ts.max()),
               asd_only_mean_s=float(ta.mean()),
               rescore_mean_s=float(ts.mean() - ta.mean()))
    print("[probe] RESULT " + json.dumps(out), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "asdveto_timing.json"), "w") as f:
        json.dump(out, f, indent=1)

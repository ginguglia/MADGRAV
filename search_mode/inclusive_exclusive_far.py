"""Quote BOTH backgrounds for every detection: inclusive and exclusive of the zero-lag foreground.

INCLUSIVE (the number of record): the time-slide background as built. It contains each detection's
own Hanford and Livingston windows paired against other slides, so a real signal contributes to the
noise estimate that ranks it. Conservative -- it can only raise a FAR.

EXCLUSIVE: every background pair whose H1 or L1 window lies within +/-4 s of ANY detection's GPS in
that detection's segment is removed before counting, and families are re-evaluated on what remains.
This is the standard foreground-removed background; it is the less conservative of the two.

Counting is the production single-counting statistic, unchanged:
    FAR = min_c min(N_c^HM, N_c^LM) / T_f,  c in {lnLambda, sigma_net}
with the as-run whole-segment self-exclusion, so the INCLUSIVE column must reproduce
madgrav_far_final_x1.csv exactly. That reproduction is asserted as a gate before anything is
written -- if the counting has drifted, this fails loud rather than quoting two numbers from two
different pipelines.

Usage: inclusive_exclusive_far.py [trials]      (default 1.0; pass the adopted factor to rescale both)
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, MADGRAV_ROOT + "/search_mode")
import successor_stat as S
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
DET = f"{MG}/details/successor_statistic"
CSV = f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"
OUT = os.environ.get("SM_OUT", f"{MG}/figures/catalog_o3o4/madgrav_far_incl_excl.csv")

TRIALS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
# SM_NETMAX: veto background pairs at or above this sigma_net. Threshold set on the INJECTION
# population (61,528 gate-passing injections: p99.9 = 10.6, max 12.2), never on the candidates.
NETMAX = float(os.environ.get("SM_NETMAX", "0")) or None
# SM_NETMAX_PERRUN: JSON path with per-run thresholds, e.g. {"o3a":10.77,...}. Overrides SM_NETMAX.
_PR = os.environ.get("SM_NETMAX_PERRUN", "")
PERRUN = __import__("json").load(open(_PR)) if _PR else None
# SM_LR_ONLY=1 -> rank on the lnLambda channel alone (the channel that carries coherence).
# The sigma_net>4 TRIGGER is unchanged; only the FAR channel set changes.
LR_ONLY = os.environ.get("SM_LR_ONLY", "0") == "1"
UL90 = 2.302585


def counts(bg, cand, touch=None, netmax=None):
    """Production per-arm counting (whole-segment self-exclusion), optionally with extra pairs removed."""
    Sx = bg.seg_ix[cand["seg"]]; f = int(bg.seg_fold[Sx])
    keep = (bg.hseg != Sx) & (bg.lseg != Sx)
    nm = netmax if netmax is not None else NETMAX
    if nm is not None:
        keep &= bg.net < nm
    if touch is not None:
        keep &= ~touch
    m = (bg.fold == f) & keep
    ll, hm, lm, net = cand["loglr"], cand["cnn_hm"], cand["cnn_lm"], cand["net"]
    mm = m & (bg.ll > ll)
    n_lr = min(len(np.unique(bg.fam[mm & (bg.hm >= hm)])), len(np.unique(bg.fam[mm & (bg.lm >= lm)])))
    F = bg.F[f]; rep = F["rep_net"]
    rk = (bg.hseg[rep] != Sx) & (bg.lseg[rep] != Sx) & (bg.net[rep] >= net)
    if nm is not None:
        rk &= bg.net[rep] < nm
    if touch is not None:
        rk &= ~touch[rep]
    rr = rep[rk]
    n_net = min(int((bg.hm[rr] >= hm).sum()), int((bg.lm[rr] >= lm).sum()))
    if LR_ONLY:
        n_net = 10**9          # channel removed from the minimum
    return n_lr, n_net, float(F["T"])


def far_of(n_lr, n_net, T):
    N = min(n_lr, n_net)
    return (TRIALS * N / T, TRIALS * (N + UL90) / T, N)


def main():
    rows = [r for r in csv.DictReader(open(CSV))]
    print(f"[incl/excl] {len(rows)} detections, trials={TRIALS}")
    out, bad = [], []
    for run in ("O3a", "O3b", "O4a", "O4b"):
        dr = [r for r in rows if r["run"] == run]
        if not dr:
            continue
        bg = S.Background(run, f"{DET}/bg_veto_{run.lower()}.npz", verbose=False)
        touch = np.zeros(len(bg.ll), bool)
        for r in dr:
            Sx = bg.seg_ix[r["seg"]]; g = float(r["gps"])
            touch |= (bg.hseg == Sx) & (np.abs(bg.gpsH - g) <= 4.0)
            touch |= (bg.lseg == Sx) & (np.abs(bg.gpsL - g) <= 4.0)
        print(f"  {run}: {len(dr)} detections, {int(touch.sum())} background pairs touch a detection window")
        for r in dr:
            c = dict(seg=r["seg"], loglr=float(r["loglr"]), net=float(r["net"]),
                     cnn_hm=float(r["cnn_hm"]), cnn_lm=float(r["cnn_lm"]))
            nm = PERRUN[run.lower()] if PERRUN else None
            fi, ui, Ni = far_of(*counts(bg, c, None, nm))
            fe, ue, Ne = far_of(*counts(bg, c, touch, nm))
            ref = float(r["far"])
            if NETMAX is None and PERRUN is None and not LR_ONLY and TRIALS == 1.0 and abs(fi - ref) > max(1e-9, 1e-6 * abs(ref)):
                bad.append(f"{r['name']}: inclusive {fi:.6g} != table {ref:.6g}")
            out.append(dict(run=run, name=r["name"], N_incl=Ni, far_incl=fi, ul90_incl=ui,
                            N_excl=Ne, far_excl=fe, ul90_excl=ue,
                            ratio=(fi / fe if fe > 0 else float("inf"))))
    if bad:
        print("\nGATE FAILED -- inclusive column does not reproduce madgrav_far_final_x1.csv:")
        for b in bad[:10]:
            print("   ", b)
        raise SystemExit(1)
    print(f"\nGATE PASS: inclusive column reproduces all {len(out)} table FARs exactly\n")
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    fi = np.array([r["far_incl"] for r in out]); fe = np.array([r["far_excl"] for r in out])
    ui = np.array([r["ul90_incl"] for r in out]); ue = np.array([r["ul90_excl"] for r in out])
    di = int(((fi < 1) & (ui < 1)).sum()); de = int(((fe < 1) & (ue < 1)).sum())
    print(f"detections: inclusive {di}, exclusive {de}")
    r = fi / np.where(fe > 0, fe, np.nan)
    print(f"FAR ratio incl/excl: median {np.nanmedian(r):.2f}, max {np.nanmax(r):.2f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

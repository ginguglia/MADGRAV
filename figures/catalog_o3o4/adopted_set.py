"""The adopted-criterion detection set, in one place.

Frozen 2026-08-31. sigma_net>4 trigger -> CNN glitch gate -> sigma_net<10.6 veto ->
lnLambda-channel per-arm FAR against the FOREGROUND-EXCLUDED time-slide background,
multiplied by the per-run null-calibration factor K. A candidate is a detection when the
calibrated FAR and its calibrated 90% UL are both below 1/yr.

This is the same selection make_table_adopted.py implements for Table I; the figures import it
so a table row and a plotted point can never disagree.
"""
import csv
import json
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
CAT = f"{MG}/figures/catalog_o3o4"
NETMAX = 10.6
KE = json.load(open(f"{MG}/details/successor_statistic/ke_adopted.json"))


def flt(x):
    s = str(x).strip()
    return float(s) if s not in ("", "nan") else float("nan")


def load():
    """-> list of dicts, one per adopted detection, sorted run-then-calibrated-FAR."""
    base = list(csv.DictReader(open(f"{CAT}/madgrav_far_final_x1.csv")))
    far = {(r["run"], r["name"]): r for r in csv.DictReader(open(f"{CAT}/far_lronly_g106.csv"))}
    out = []
    for b in base:
        k = (b["run"], b["name"])
        if flt(b["net"]) >= NETMAX or k not in far:
            continue
        ke = KE[b["run"].lower()]
        f = far[k]
        fe, ue = flt(f["far_excl"]) * ke, flt(f["ul90_excl"]) * ke
        # Admission (design decision 2026-09-01): the calibrated point estimate alone.
        # The 90% upper limit is reported per event but is not part of the bar.
        if not (fe < 1.0):
            continue
        N = int(flt(f["N_excl"]))
        # background livetime implied by the count (623 yr/fold); recovered from whichever
        # of the two quantities is defined at this N, so N=0 events keep a valid T.
        T = N / (flt(f["far_excl"])) if N > 0 else 2.302585 / flt(f["ul90_excl"])
        out.append(dict(run=b["run"], name=b["name"], mtot=b["mtot"], snr=b["snr_cat"],
                        net=flt(b["net"]), cwb=b["cwb"], source=b.get("source", ""),
                        N=N, T=T, ke=ke, far=fe, ul90=ue,
                        far_incl=flt(f["far_incl"]) * ke))
    order = {"O3a": 0, "O3b": 1, "O4a": 2, "O4b": 3}
    out.sort(key=lambda r: (order[r["run"]], r["far"]))
    assert len(out) == 47, f"adopted set is {len(out)}, expected 47"
    return out


def poisson_ci(N, T, ke):
    """90% Poisson interval on FAR=N/T, scaled by the calibration factor.

    Same convention as build_far_x1.py: lo = chi2(0.05, 2N)/2T, hi = chi2(0.95, 2N+2)/2T,
    UL90 = chi2(0.90, 2N+2)/2T. K is a deterministic multiplier, so it scales the interval.
    """
    from scipy.stats import chi2
    lo = chi2.ppf(0.05, 2 * N) / 2 / T if N > 0 else 0.0
    hi = chi2.ppf(0.95, 2 * N + 2) / 2 / T
    ul = chi2.ppf(0.90, 2 * N + 2) / 2 / T
    return lo * ke, hi * ke, ul * ke

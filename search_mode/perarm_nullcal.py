"""Null-calibration gate for the PER-ARM (arm-conditioned) counting at a trials factor n.

The successor gate measured the UNCONDITIONED scalar rank and passed at the measured N_eff.
The 48-detection table uses a different rank function -- the as-run arm-conditioned per-arm counts,
FAR = min_c min(N_c^HM, N_c^LM)/T -- whose calibration has never been measured. This runs the same
pre-registered pseudo-foreground test on THAT counting so the two are compared on one footing.

Everything is held identical to the successor gate except the rank function: same background-only
pseudo-foreground population, same veto-symmetric background, same OWN-PAIR exclusion (not the
as-run whole-segment exclusion, which the successor audit showed removes a candidate's own sibling
population and biases the test), same K(x) vs E(x)=x*T, same [0.5,1.5] interval at 1/yr.

The foreground is never touched, so nothing here can tune a threshold.

Usage: perarm_nullcal.py [n_sample_per_fold]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, MADGRAV_ROOT + "/search_mode")
import successor_stat as S
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


DET = MADGRAV_ROOT + "/details/successor_statistic"
XS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
NS = [1.0, 1.5, 1.869, 2.0, 3.0, 4.0]
RUNS = ["O3a", "O3b", "O4a", "O4b"]
NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
# SM_EXCL_DET=1 -> remove every background pair within +/-4 s of a detection (same segment) from BOTH
# the ranking background and the pseudo-foreground population. Tests whether real signals leaking into
# the slid data are what inflate K, rather than the rank function.
EXCL_DET = os.environ.get("SM_EXCL_DET", "0") == "1"
# SM_NETMAX / SM_NETMAX_PERRUN: sigma_net upper veto, applied to the background AND to the
# pseudo-foreground population (a vetoed family cannot be a candidate either).
NETMAX = float(os.environ.get("SM_NETMAX", "0")) or None
LR_ONLY = os.environ.get("SM_LR_ONLY", "0") == "1"   # rank on lnLambda alone (drop the sigma_net channel)
_PR = os.environ.get("SM_NETMAX_PERRUN", "")
PERRUN = json.load(open(_PR)) if _PR else None
CSVDET = MADGRAV_ROOT + "/figures/catalog_o3o4/madgrav_far_final_x1.csv"
RNG = np.random.default_rng(20260831)


def perarm_counts(bg, f, p, extra=None, nm=None):
    """Arm-conditioned per-arm counts for pseudo-candidate p against fold f, OWN-PAIR exclusion."""
    F = bg.F[f]; gi = F["gi"]
    ll_q, net_q = float(bg.ll[p]), float(bg.net[p])
    hm_q, lm_q = float(bg.hm[p]), float(bg.lm[p])
    ex = (bg.hseg[gi] == bg.hseg[p]) & (np.abs(bg.gpsH[gi] - bg.gpsH[p]) <= S.EXCL_TOL) & \
         (bg.lseg[gi] == bg.lseg[p]) & (np.abs(bg.gpsL[gi] - bg.gpsL[p]) <= S.EXCL_TOL)
    msk = ~ex
    if nm is not None:
        msk &= bg.net[gi] < nm
    if extra is not None:
        msk &= ~extra[gi]
    k = gi[msk]
    out = {}
    if np.isfinite(ll_q) and ll_q >= S.LR_FLOOR:
        m = bg.gate[k] & bg.finll[k] & (bg.ll[k] > ll_q) & (~bg.cnn_vet[k])
        kk = k[m]
        out["lr_hm"] = len(np.unique(bg.fam[kk[bg.hm[kk] >= hm_q]]))
        out["lr_lm"] = len(np.unique(bg.fam[kk[bg.lm[kk] >= lm_q]]))
    if (not LR_ONLY) and np.isfinite(net_q) and net_q >= S.NET_FLOOR:
        rep = F["rep_net"]
        rex = (bg.hseg[rep] == bg.hseg[p]) & (np.abs(bg.gpsH[rep] - bg.gpsH[p]) <= S.EXCL_TOL) & \
              (bg.lseg[rep] == bg.lseg[p]) & (np.abs(bg.gpsL[rep] - bg.gpsL[p]) <= S.EXCL_TOL)
        sel = (~rex) & bg.gate[rep] & (~bg.net_vet[rep]) & (bg.net[rep] >= net_q)
        if nm is not None:
            sel &= bg.net[rep] < nm
        if extra is not None:
            sel &= ~extra[rep]
        rr = rep[sel]
        out["net_hm"] = int((bg.hm[rr] >= hm_q).sum())
        out["net_lm"] = int((bg.lm[rr] >= lm_q).sum())
    return out


def main():
    res = {"xs": XS, "ns": NS, "n_sample_per_fold": NSAMP, "runs": {}}
    pooled = {n: [np.zeros(len(XS)), np.zeros(len(XS))] for n in NS}   # [K, E]
    for run in RUNS:
        bg = S.Background(run, veto_path=f"{DET}/bg_veto_{run.lower()}.npz", verbose=False)
        touch = None
        if EXCL_DET:
            import csv as _csv
            touch = np.zeros(len(bg.ll), bool)
            for r in _csv.DictReader(open(CSVDET)):
                if r["run"] != run: continue
                Sx = bg.seg_ix[r["seg"]]; g = float(r["gps"])
                touch |= (bg.hseg == Sx) & (np.abs(bg.gpsH - g) <= 4.0)
                touch |= (bg.lseg == Sx) & (np.abs(bg.gpsL - g) <= 4.0)
            print(f"  [{run}] EXCL_DET: {int(touch.sum())} background pairs removed", flush=True)
        rk = {n: np.zeros(len(XS)) for n in NS}; rE = np.zeros(len(XS))
        for f in (0, 1):
            z = np.load(f"{DET}/pseudo_fg_{run.lower()}_f{f}.npz")
            pop = z["pop"][~z["vetoed"]]
            nmv = (PERRUN[run.lower()] if PERRUN else NETMAX)
            if nmv is not None:
                pop = pop[bg.net[pop] < nmv]     # a vetoed family cannot be a candidate either
            if touch is not None:
                pop = pop[~touch[pop]]          # and drop them from the pseudo-fg population itself
            T = float(z["T"])
            frac = 1.0
            if len(pop) > NSAMP:
                pop = RNG.choice(pop, NSAMP, replace=False); frac = NSAMP / int((~z["vetoed"]).sum())
            if LR_ONLY:            # a family with no lnLambda channel cannot be ranked at all
                pop = pop[np.isfinite(bg.ll[pop]) & (bg.ll[pop] >= S.LR_FLOOR)]
            nmin = np.full(len(pop), np.inf)
            for i, p in enumerate(pop):
                c = perarm_counts(bg, f, int(p), touch, nmv)
                vals = []
                if "lr_hm" in c: vals.append(min(c["lr_hm"], c["lr_lm"]))
                if "net_hm" in c: vals.append(min(c["net_hm"], c["net_lm"]))
                if vals: nmin[i] = min(vals)
            E = np.array([x * T for x in XS]); rE += E
            for n in NS:
                far = nmin / T * n
                rk[n] += np.array([int((far < x).sum()) for x in XS]) / frac
            print(f"  [{run} f{f}] scored {len(pop)} pseudo-fg (frac {frac:.3f}), T={T:.1f} yr", flush=True)
        res["runs"][run] = {}
        for n in NS:
            ke = rk[n] / rE
            res["runs"][run][str(n)] = dict(K=rk[n].tolist(), E=rE.tolist(), K_over_E=ke.tolist(),
                                            pass_at_1=bool(0.5 <= ke[0] <= 1.5))
            pooled[n][0] += rk[n]; pooled[n][1] += rE
        print(f"{run}: " + "  ".join(f"n={n}: K/E={rk[n][0]/rE[0]:.2f}" for n in NS), flush=True)
    print("\n" + "=" * 78)
    print("PER-ARM COUNTING -- pooled K/E at 1/yr (gate interval [0.5, 1.5])")
    print("=" * 78)
    res["pooled"] = {}
    for n in NS:
        K, E = pooled[n]; ke = K / E
        res["pooled"][str(n)] = dict(K_over_E=ke.tolist(), pass_at_1=bool(0.5 <= ke[0] <= 1.5))
        print(f"  trials n={n:<6} K/E(1/yr) = {ke[0]:6.2f}   {'PASS' if 0.5<=ke[0]<=1.5 else 'FAIL'}"
              f"   grid " + " ".join(f"{v:.2f}" for v in ke))
    tag = ('_excldet' if EXCL_DET else '') + ('_netmax' if (NETMAX or PERRUN) else '') + ('_lronly' if LR_ONLY else '')
    json.dump(res, open(f"{DET}/perarm_nullcal{tag}.json", "w"), indent=1)
    print(f"\n-> {DET}/perarm_nullcal.json")


if __name__ == "__main__":
    main()

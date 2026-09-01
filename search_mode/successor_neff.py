#!/usr/bin/env python
"""successor_neff.py -- Sec. 3 (N_eff measurement + freeze) and Sec. 4 (acceptance null calibration) of the
successor-statistic amendment (2026-08-18). Background only; uses successor_stat.py (the ONE shared module).

  python successor_neff.py neff   [runs]   -> details/successor_statistic/pseudo_fg_<run>.npz, neff_<run>_<date>.json,
                                             neff_table.txt, neff_freeze.json (+ homogeneity test)
  python successor_neff.py nullcal [runs]  -> null_calibration_successor.{json,txt,png}, ACCEPTANCE_{PASSED|FAILED}.txt

Pseudo-foreground (Sec. 3): every gate-passing, veto-surviving background family of fold f, represented by its
lnLambda rep pair (family with lnLambda channel) or its net rep pair (net-only family), scored with the SAME
score_pseudo_fg entry point (own leave-one-out through the narrowed exclusion). The foreground veto rule is
mirrored: a pseudo-fg whose winning channel is sigma_net and whose rep has net_loc < 4 is vetoed (dropped),
exactly as a real net-sigma-channel candidate would be.
N_eff(x) = R_min(x) / max_c R_c(x); Garwood 90% on the counts, endpoint rule for the ratio.
Homogeneity (Addendum C rerun): disjoint bins (x_{i+1}, x_i]; K_i = dR_min_i, E_i = d(max_c R_c)_i;
Poisson LR G-test (df = 5) + parametric bootstrap p; alpha = 0.05 -> constant N_eff := N_eff(1/yr) unless
rejected, then piecewise-in-x (value N_eff(x_i) on (x_{i+1}, x_i], N_eff(1) above 1/yr, N_eff(0.01) below).
"""
import os, sys, json, time, hashlib
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import successor_stat as S

DET = os.environ.get("SUCC_DET", S.DET); XS = S.XS; DATE = "2026-08-18"; ALPHA = 0.05; NMC = 20000
RNG = np.random.default_rng(20260818)


def veto_path(run):
    p = f"{DET}/bg_veto_{run.lower()}.npz"
    return p if os.path.exists(p) else None


def piecewise_fn(vals):
    """vals: dict x -> N_eff(x) on the grid; returns callable of x0."""
    xs = np.array(XS); v = np.array([vals[x] for x in XS])
    def fn(x0):
        if x0 > xs[0]: return float(v[0])
        i = int(np.searchsorted(-xs, -x0, "left"))       # first grid x_i <= ... careful: want x0 in (x_{i+1}, x_i]
        # xs desc: find smallest i with xs[i] >= x0 -> largest index with xs[i] >= x0
        j = np.where(xs >= x0)[0]
        return float(v[j[-1]]) if len(j) else float(v[-1])
    return fn


def neff_of(freeze, run, f):
    e = freeze[run][str(f)]
    if e["mode"] == "constant": return float(e["N_eff"])
    return piecewise_fn({float(k): v for k, v in e["piecewise"].items()})


def G_stat(K, E):
    K = np.asarray(K, float); E = np.asarray(E, float); kap = K.sum() / E.sum() if E.sum() > 0 else 0.0; mu = kap * E
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(K > 0, K * np.log(K / np.where(mu > 0, mu, 1)), 0.0)
    return float(2 * t.sum()), float(kap)


def lr_test(K, E):
    G, kap = G_stat(K, E); df = int(np.sum(np.asarray(E) > 0) - 1)
    p = float(chi2.sf(G, df)) if df > 0 else float("nan")
    mu = kap * np.asarray(E, float); sim = RNG.poisson(mu, size=(NMC, len(K)))
    Gs = np.array([G_stat(s, E)[0] for s in sim]); pmc = float((Gs >= G - 1e-12).mean())
    return dict(G=G, df=df, p_chi2=p, p_mc=pmc, kappa_hat=kap, reject_005=bool(p < ALPHA))


def score_population(bg, f):
    pop, has_lr = bg.pseudo_fg_population(f)
    # A3: complete the population to ALL families with >=1 surviving channel: families whose lnLambda rep is vetoed
    # but whose sigma_net rep survives were absent from pseudo_fg_population (neither valid_lr nor net-only); they are
    # counted in the net channel of every candidate, so they must be pseudo-candidates too (net rep, net channel only).
    Fp = bg.F[f]; extra = Fp["rep_net"][Fp["has_lr"] & ~Fp["valid_lr"] & Fp["valid_net"]]
    if len(extra):
        pop = np.concatenate([pop, extra]); has_lr = np.concatenate([has_lr, np.zeros(len(extra), bool)])
    print(f"  [{bg.run} f{f}] A3 population: +{len(extra)} families with vetoed lnL rep but surviving net rep (net channel only)", flush=True)
    n = len(pop); N_lr = np.full(n, -1, np.int64); N_net = np.full(n, -1, np.int64); nex = np.zeros(n, np.int64)
    t0 = time.time()
    for j, p in enumerate(pop):
        r = bg.score_pseudo_fg(f, int(p))
        N_lr[j] = -1 if r["N_lr"] is None else r["N_lr"]; N_net[j] = -1 if r["N_net"] is None else r["N_net"]; nex[j] = r["n_excluded_pairs"]
        if (j + 1) % 10000 == 0: print(f"  [{bg.run} f{f}] {j+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    T = bg.F[f]["T"]
    # ---- A3 (null-test amendment 2026-08-18): pseudo-fg CHANNEL AVAILABILITY = the FAMILY's surviving channels,
    #      mirroring the production path (a family competes in the sigma_net channel only if its net rep survives
    #      the symmetric veto; in the lnLambda channel only if its lr rep survives). Applied here, i.e. to the arrays
    #      that feed BOTH the N_eff legs (R_min, R_c; cmd_neff) and the K leg (cmd_nullcal); E = x*T is population-free.
    F = bg.F[f]; fid = np.searchsorted(F["fam_sorted"], bg.fam[pop]); assert np.all(F["fam_sorted"][fid] == bg.fam[pop])
    fam_valid_lr = F["valid_lr"][fid]; fam_valid_net = F["valid_net"][fid]
    n_lr_drop = int(((N_lr >= 0) & ~fam_valid_lr).sum()); n_net_drop = int(((N_net >= 0) & ~fam_valid_net).sum())
    N_lr = np.where(fam_valid_lr, N_lr, -1); N_net = np.where(fam_valid_net, N_net, -1)
    print(f"  [{bg.run} f{f}] A3 availability rule: lnL channel removed for {n_lr_drop} pseudo-fg, sigma_net channel removed for {n_net_drop} (family rep vetoed); "
          f"pseudo-fg with no channel left: {int(((N_lr < 0) & (N_net < 0)).sum())}", flush=True)
    keepany = (N_lr >= 0) | (N_net >= 0)
    pop, has_lr, N_lr, N_net, nex = pop[keepany], has_lr[keepany], N_lr[keepany], N_net[keepany], nex[keepany]
    # winning channel + mirrored net_loc veto
    ch = np.where(N_lr < 0, 1, np.where(N_net < 0, 0, np.where(N_net < N_lr, 1, 0)))     # 0 = loglr, 1 = net (ties -> loglr)
    vet = (ch == 1) & bg.net_vet[pop]
    Nmin = np.where(N_lr < 0, N_net, np.where(N_net < 0, N_lr, np.minimum(N_lr, N_net)))
    return dict(pop=pop, has_lr=has_lr, N_lr=N_lr, N_net=N_net, n_excl=nex, channel=ch, vetoed=vet, N_min=Nmin, T=T,
                ll=bg.ll[pop], net=bg.net[pop], hm=bg.hm[pop], lm=bg.lm[pop])


def tail_counts(P, x):
    keep = ~P["vetoed"]; T = P["T"]
    R_min = int((keep & (P["N_min"] / T < x)).sum())
    R_lr = int((keep & (P["N_lr"] >= 0) & (P["N_lr"] / T < x)).sum())
    R_net = int((keep & (P["N_net"] >= 0) & (P["N_net"] / T < x)).sum())
    return R_min, R_lr, R_net


def cmd_neff(runs):
    S.assert_spec(); md5 = S.module_md5()
    freeze = dict(date=DATE, module_md5=md5, spec_md5=S.SPEC_MD5, xs=XS, alpha=ALPHA, rule="N_eff := N_eff(x=1/yr) unless in-x constancy rejected (LR, alpha=0.05) -> piecewise", runs={})
    lines = ["SUCCESSOR N_eff measurement (Sec. 3) -- background only, veto-symmetric, narrowed exclusion", "=" * 100,
             f"generated {time.strftime('%F %T')}; successor_stat.py md5 {md5}; spec md5 {S.SPEC_MD5}",
             "N_eff(x) = R_min(x)/max_c R_c(x); Garwood 90% endpoint rule; grid x = " + ", ".join(f"{x:g}" for x in XS) + " /yr", ""]
    for run in runs:
        vp = veto_path(run); bg = S.Background(run, vp)
        out = dict(run=run, date=DATE, module_md5=md5, spec_md5=S.SPEC_MD5, veto_path=vp, folds={})
        for f in (0, 1):
            P = score_population(bg, f); T = P["T"]
            np.savez_compressed(f"{DET}/pseudo_fg_{run.lower()}_f{f}.npz", **{k: v for k, v in P.items() if isinstance(v, np.ndarray)}, T=np.array(T), veto_path=np.array(str(vp)))
            rows = {}
            for x in XS:
                Rm, Rl, Rn = tail_counts(P, x); den = max(Rl, Rn)
                lo_m, hi_m = S.garwood(Rm); lo_d, hi_d = S.garwood(den)
                rows[x] = dict(R_min=Rm, R_lr=Rl, R_net=Rn, den=den, N_eff=(Rm / den if den else None),
                               ci90=[(lo_m / hi_d if hi_d else None), (hi_m / lo_d if lo_d else None)])
            # homogeneity: disjoint bins
            K = [rows[XS[i]]["R_min"] - rows[XS[i + 1]]["R_min"] for i in range(len(XS) - 1)]
            E = [rows[XS[i]]["den"] - rows[XS[i + 1]]["den"] for i in range(len(XS) - 1)]
            hom = lr_test(K, E); hom.update(bins=[f"({XS[i+1]:g},{XS[i]:g}]" for i in range(len(XS) - 1)], dR_min=K, d_maxR_c=E,
                                            N_eff_bins=[(k / e if e else None) for k, e in zip(K, E)])
            mode = "piecewise" if hom["reject_005"] else "constant"
            n_vet = int(P["vetoed"].sum()); n_pop = int(len(P["pop"]))
            fo = dict(T_f_yr=T, n_pseudo_fg=n_pop, n_has_lr=int(P["has_lr"].sum()), n_vetoed_netloc=n_vet, n_excluded_pairs_mean=float(P["n_excl"].mean()),
                      grid={str(x): rows[x] for x in XS}, homogeneity=hom, mode=mode,
                      N_eff_frozen=rows[1.0]["N_eff"], ci90_frozen=rows[1.0]["ci90"],
                      piecewise={str(x): rows[x]["N_eff"] for x in XS})
            out["folds"][str(f)] = fo
            freeze["runs"].setdefault(run, {})[str(f)] = dict(mode=mode, N_eff=rows[1.0]["N_eff"], ci90=rows[1.0]["ci90"], T_f_yr=T,
                                                          piecewise={str(x): rows[x]["N_eff"] for x in XS}, p_homogeneity_chi2=hom["p_chi2"], p_homogeneity_mc=hom["p_mc"])
            lines.append(f"{run} fold {f}: T_f={T:.2f} yr, pseudo-fg {n_pop} (has-lnL {int(P['has_lr'].sum())}), net_loc-vetoed {n_vet}; mode={mode} "
                         f"(in-x LR G={hom['G']:.2f} df={hom['df']} p_chi2={hom['p_chi2']:.3g} p_mc={hom['p_mc']:.3g}); FROZEN N_eff={rows[1.0]['N_eff']:.3f} [{rows[1.0]['ci90'][0]:.3f}-{rows[1.0]['ci90'][1]:.3f}]")
            lines.append("      x[/yr]   " + "  ".join(f"{x:>14g}" for x in XS))
            lines.append("      R_min    " + "  ".join(f"{rows[x]['R_min']:>14d}" for x in XS))
            lines.append("      R_lr     " + "  ".join(f"{rows[x]['R_lr']:>14d}" for x in XS))
            lines.append("      R_net    " + "  ".join(f"{rows[x]['R_net']:>14d}" for x in XS))
            lines.append("      N_eff    " + "  ".join((f"{rows[x]['N_eff']:.2f}[{rows[x]['ci90'][0]:.2f}-{rows[x]['ci90'][1]:.2f}]" if rows[x]['N_eff'] is not None else 'n/a').rjust(14) for x in XS))
            lines.append("      bins     " + "  ".join(f"{b:>14s}" for b in hom["bins"]) + "   N_eff_i=" + " ".join(f"{v:.2f}" if v is not None else "n/a" for v in hom["N_eff_bins"]))
            print(lines[-6], flush=True)
        json.dump(out, open(f"{DET}/neff_{run.lower()}_{DATE}.json", "w"), indent=1)
    lines.append("")
    tp = f"{DET}/neff_table.txt"
    if os.path.exists(tp): lines = [f"# appended {time.strftime('%F %T')} for runs {runs}"] + lines[5:]   # keep the header once
    with open(tp, "a") as fh: fh.write("\n".join(lines) + "\n")
    # freeze file (merge with existing runs if re-run per run)
    fz = f"{DET}/neff_freeze.json"
    if os.path.exists(fz):
        old = json.load(open(fz)); old["runs"].update(freeze["runs"]); freeze = {**old, **{k: v for k, v in freeze.items() if k != "runs"}, "runs": old["runs"]}
    json.dump(freeze, open(fz, "w"), indent=1)
    print(f"-> {fz} md5 {hashlib.md5(open(fz,'rb').read()).hexdigest()}", flush=True)
    print("\n".join(lines))


def cmd_nullcal(runs):
    S.assert_spec(); md5 = S.module_md5()
    freeze = json.load(open(f"{DET}/neff_freeze.json")); fmd5 = hashlib.md5(open(f"{DET}/neff_freeze.json", "rb").read()).hexdigest()
    XG = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02]                       # gate bins
    out = dict(module_md5=md5, spec_md5=S.SPEC_MD5, neff_freeze_md5=fmd5, xs=XS, gate=dict(interval=[0.5, 1.5], at_x=1.0, bins=XG, mode="cross-fold"), runs={})
    L = ["SUCCESSOR null calibration (Sec. 4) -- pseudo-fg FAR = N_eff * min_c N_c / T_f; K(x) vs E(x) = x T_f", "=" * 100,
         f"generated {time.strftime('%F %T')}; successor_stat.py md5 {md5}; neff_freeze.json md5 {fmd5}",
         "IN-FOLD: N_eff of the same fold (implementation check); CROSS-FOLD: N_eff of fold f applied to pseudo-fg of fold f' (the GATE).", ""]
    all_pass = True; pooledK = {m: np.zeros(len(XS)) for m in ("in", "cross")}; pooledE = np.zeros(len(XS))
    for run in runs:
        fr = freeze["runs"][run]; rk = {m: np.zeros(len(XS)) for m in ("in", "cross")}; rE = np.zeros(len(XS)); per = {}
        for f in (0, 1):
            z = np.load(f"{DET}/pseudo_fg_{run.lower()}_f{f}.npz"); T = float(z["T"]); keep = ~z["vetoed"]; Nmin = z["N_min"][keep]
            x0 = Nmin / T; E = np.array([x * T for x in XS]); rE += E
            per[str(f)] = dict(T_f_yr=T, n_pseudo_fg=int(keep.sum()), E=E.tolist())
            for m, src in (("in", f), ("cross", 1 - f)):
                ne = neff_of(freeze["runs"], run, src)
                far = x0 * (np.array([ne(v) for v in x0]) if callable(ne) else ne)
                K = np.array([int((far < x).sum()) for x in XS]); rk[m] += K
                per[str(f)][m] = dict(neff_from_fold=src, N_eff=(fr[str(src)]["N_eff"] if fr[str(src)]["mode"] == "constant" else fr[str(src)]["piecewise"]),
                                      K=K.tolist(), K_over_E=(K / E).tolist(), ci90=[[a / e, b / e] for (a, b), e in ((S.garwood(int(k)), e) for k, e in zip(K, E))])
        pooledE += rE
        rd = dict(E=rE.tolist(), per_fold=per)
        for m in ("in", "cross"):
            K = rk[m]; ci = [S.garwood(int(k)) for k in K]; pooledK[m] += K
            rd[m] = dict(K=K.tolist(), K_over_E=(K / rE).tolist(), ci90=[[c[0] / e, c[1] / e] for c, e in zip(ci, rE)])
        ke = np.array(rd["cross"]["K_over_E"]); ci = np.array(rd["cross"]["ci90"])
        c1 = 0.5 <= ke[0] <= 1.5
        bad_bins = [x for x, (lo, hi) in zip(XS, ci) if x in XG and (hi < 0.5 or lo > 1.5)]
        rd["gate"] = dict(K_over_E_at_1=float(ke[0]), pass_at_1=bool(c1), bins_wholly_outside=bad_bins, PASS=bool(c1 and not bad_bins))
        all_pass &= rd["gate"]["PASS"]; out["runs"][run] = rd
        L.append(f"{run}: T_0+T_1={rE[0]:.1f} yr; pseudo-fg " + "+".join(str(per[f]['n_pseudo_fg']) for f in "01") + f"; N_eff frozen: fold0 {fr['0']['N_eff']:.3f} ({fr['0']['mode']}), fold1 {fr['1']['N_eff']:.3f} ({fr['1']['mode']})")
        L.append("      x[/yr]        " + "  ".join(f"{x:>14g}" for x in XS))
        L.append("      E             " + "  ".join(f"{e:>14.1f}" for e in rE))
        for m, lab in (("in", "IN-FOLD  K/E"), ("cross", "CROSS    K/E")):
            L.append(f"      {lab:13s} " + "  ".join(f"{k/e:.2f}[{c[0]:.2f}-{c[1]:.2f}]".rjust(14) for k, e, c in zip(rd[m]["K"], rE, rd[m]["ci90"])))
        for f in "01":
            for m in ("in", "cross"):
                d = per[f][m]; L.append(f"        fold {f} {m:5s} (N_eff from fold {d['neff_from_fold']}) K/E " + " ".join(f"{v:.2f}" for v in d["K_over_E"]))
        L.append(f"      GATE (cross-fold): K/E(1/yr)={ke[0]:.3f} in [0.5,1.5]: {c1}; bins wholly outside: {bad_bins or 'none'} -> {'PASS' if rd['gate']['PASS'] else 'FAIL'}")
        L.append("")
    for m in ("in", "cross"):
        K = pooledK[m]; ci = [S.garwood(int(k)) for k in K]
        out[f"pooled_{m}"] = dict(K=K.tolist(), E=pooledE.tolist(), K_over_E=(K / pooledE).tolist(), ci90=[[c[0] / e, c[1] / e] for c, e in zip(ci, pooledE)])
        L.append(f"POOLED {m:5s} K/E: " + "  ".join(f"{k/e:.3f}[{c[0]/e:.2f}-{c[1]/e:.2f}]" for k, e, c in zip(K, pooledE, ci)))
    out["ALL_PASS"] = bool(all_pass)
    L.append(f"\nACCEPTANCE (all runs, cross-fold): {'PASS' if all_pass else 'FAIL'}")
    txt = "\n".join(L); print(txt)
    open(f"{DET}/null_calibration_successor.txt", "w").write(txt + "\n")
    json.dump(out, open(f"{DET}/null_calibration_successor.json", "w"), indent=1)
    open(f"{DET}/ACCEPTANCE_{'PASSED' if all_pass else 'FAILED'}.txt", "w").write(txt + "\n")
    try: make_figure(out, runs)
    except Exception as e: print("figure failed:", e)


def make_figure(out, runs):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = {"O3a": "#2a78d6", "O3b": "#eb6834", "O4a": "#1baf7a", "O4b": "#eda100", "O4ars": "#4a3aa7"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, m, ttl in zip(axes, ("in", "cross"), ("IN-FOLD (implementation check)", "CROSS-FOLD (acceptance gate)")):
        for i, r in enumerate(runs):
            d = out["runs"][r][m]; x = np.array(XS) * (1 + 0.04 * (i - 2)); ke = np.array(d["K_over_E"]); ci = np.array(d["ci90"])
            ax.errorbar(x, ke, yerr=[ke - ci[:, 0], ci[:, 1] - ke], fmt="o-", ms=5, color=cols.get(r, "k"), label=r, capsize=2)
        d = out[f"pooled_{m}"]; ke = np.array(d["K_over_E"]); ci = np.array(d["ci90"])
        ax.errorbar(XS, ke, yerr=[ke - ci[:, 0], ci[:, 1] - ke], fmt="s--", ms=6, lw=2, color="k", label="pooled", capsize=3)
        ax.axhspan(0.5, 1.5, color="#e34948", alpha=0.08); ax.axhline(1, color="#e34948", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("x [1/yr]"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3, which="both"); ax.set_xlim(0.008, 1.3)
    axes[0].set_ylabel("K(x)/E(x)"); axes[0].legend(fontsize=8)
    fig.suptitle("Successor statistic null calibration (unconditioned rank, veto-symmetric bg, narrowed exclusion, measured N_eff)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{DET}/null_calibration_successor.png", dpi=150)


if __name__ == "__main__":
    what = sys.argv[1]; runs = sys.argv[2:] or S.MAIN_RUNS
    if what == "neff": cmd_neff(runs)
    elif what == "nullcal": cmd_nullcal(runs)

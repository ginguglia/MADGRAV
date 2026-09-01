#!/usr/bin/env python
"""successor_stat.py -- the MADGRAV SUCCESSOR detection statistic (pre-registered amendment 2026-08-18,
PREREG_AMENDMENT_successor_statistic.md, md5 f8ab2043a39df4ce630a5d944163f9a6).  ONE shared module for the
production foreground path and the pseudo-foreground null test (Sec. 2b consistency requirement).

Definitions implemented here (all frozen by the amendment):
  * background family  = L1 4-s family key fam = lseg_ix*1e7 + gps_L//4  (bg_cache convention, unchanged)
  * glitch gate        = max(cnn_hm, cnn_lm) > GATE (0.5), pair-level, unchanged
  * channel values per family (fold f):
        r_lr(b)  = max lnLambda over the family's GATE-PASSING finite-lnLambda pairs  (rep pair = argmax);
                   family has the lnLambda channel iff such a pair exists (cache complete above the floor 4.0)
        r_net(b) = sigma_net of the family's max-net cached pair (the famN rep, as run); family has the
                   sigma_net channel iff that rep passes the gate (as run)
  * veto-symmetric background (Change 2): per-pair local-ASD recomputation (asd_consistency_bg.py) gives
        cnn_vetoed(p) = evaluated & max(hm_loc,lm_loc) < GATE
        net_vetoed(p) = cnn_vetoed(p) | (evaluated & net_loc < NET_FLOOR)
    a family counts in the lnLambda channel iff its lr rep is not cnn_vetoed, in the sigma_net channel iff its
    net rep is not net_vetoed; un-evaluated pairs (beyond top-M) count as SURVIVING (conservative).
  * self-exclusion, AMENDMENT 2 (2026-08-18, PREREG_AMENDMENT_successor_statistic_A2.md; EXCL_MODE="own_pair"):
    EXCLUDE ONLY the candidate's own pair, i.e. cached pairs with
        hseg == hseg_q & |gps_H - gpsH_q| <= EXCL_TOL & lseg == lseg_q & |gps_L - gpsL_q| <= EXCL_TOL
    (for a zero-lag candidate no cross-segment background pair matches -> nothing excluded; for a pseudo-fg
    background pair exactly itself is excluded, leave-one-PAIR-out). NO own-window, NO own-family, NO
    whole-segment exclusion. Iteration history: whole-segment (as-run) -> own H1/L1 window +/-4 s + own family
    (Change 3, gate FAILED 2026-08-18 20:31, cross-fold K/E 1.5-2.3) -> own pair only (this). Families touched
    by the exclusion are re-evaluated on their remaining pairs (rep may change; a rep that was not
    veto-evaluated counts as surviving).
  * unconditioned scalar rank:  N_lr(q) = # families (with lnLambda channel, surviving) with r_lr > lnLambda_q
                                 N_net(q) = # families (with sigma_net channel, surviving) with r_net >= sigma_net_q
    (strict > for lnLambda, >= for sigma_net, as coded in the as-run pipeline)
  * FAR(q) = N_eff * min_c N_c(q) / T_f  over the candidate's available channels (lnLambda iff lnLambda_q >= 4.0,
    sigma_net iff sigma_net_q >= 4.0);  UL90 (all N_c = 0) = N_eff * 2.30 / T_f;  N_eff from neff_freeze.json.
Two entry points share _rank_core: score_candidate (production: candidate dict) and score_pseudo_fg (null test:
cached background pair as pseudo-candidate).  self_test() scores a background sample through both and asserts
identical N_c, and validates _rank_core against a brute-force recount.  count_asrun() is the OLD conditioned
per-arm counting (whole-segment exclusion) on the same arrays -- a plumbing regression check only.
"""
import os, sys, json, time, hashlib
import numpy as np
from scipy.stats import chi2
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")


MG = MADGRAV_ROOT
DET = f"{MG}/details/successor_statistic"
SC = MADGRAV_SCRATCH
RUNS = {"O3a": f"{SC}/search_out_o3a_far_f40", "O3b": f"{SC}/search_out_o3b_far_f40",
        "O4a": f"{SC}/search_out_o4a_far", "O4b": f"{SC}/search_out_o4b_far", "O4ars": f"{SC}/search_out_o4ars_far"}
MAIN_RUNS = ["O3a", "O3b", "O4a", "O4b"]
GATE = 0.5; LR_FLOOR = 4.0; NET_FLOOR = 4.0; EXCL_S = 4.0; MERGE_S = 4.0
EXCL_MODE = "own_pair"; EXCL_TOL = 0.51   # A2: exclude only the candidate's own pair (window match within one 1-s stride)
XS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
UL90_N0 = float(chi2.ppf(0.90, 2) / 2)          # 2.3026 (one-sided 90% Poisson UL on a zero count)
X_MAX = 2.0; NEFF_PRIOR = 2.0                    # top-M rule: M = ceil(X_MAX * T_f * NEFF_PRIOR) = ceil(4 T_f)
SPEC_MD5 = "f8ab2043a39df4ce630a5d944163f9a6"


def module_md5():
    return hashlib.md5(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def assert_spec():
    """Verify this module against the frozen specification, when that document is present.

    The specification is an internal development record and is not distributed with the code, so
    a missing file is not an error: the check is skipped and None returned. Where the document IS
    present its md5 must match SPEC_MD5 exactly.
    """
    spec = os.path.join(MG, "PREREG_AMENDMENT_successor_statistic.md")
    if not os.path.exists(spec):
        return None
    got = hashlib.md5(open(spec, "rb").read()).hexdigest()
    assert got == SPEC_MD5, f"spec md5 {got} != {SPEC_MD5}"
    return got


def garwood(k, cl=0.90):
    a = (1 - cl) / 2
    lo = chi2.ppf(a, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(1 - a, 2 * k + 2) / 2
    return float(lo), float(hi)


def top_M(T):
    return int(np.ceil(X_MAX * T * NEFF_PRIOR))


def _key(seg, gps):
    return seg.astype(np.int64) * (1 << 36) + np.rint(np.asarray(gps, np.float64) * 4.0).astype(np.int64)


class Background:
    """Per-run background (bg_cache + window cache + optional veto results), per-fold family structures."""

    def __init__(self, run, veto_path=None, verbose=True):
        d = RUNS[run]; self.run = run
        z = np.load(f"{d}/bg_cache_{run.lower()}.npz", allow_pickle=False)
        w = np.load(f"{DET}/bg_cache_{run.lower()}_win.npz", allow_pickle=False)
        self.n = n = len(z["hseg"]); assert int(w["n"]) == n
        self.hseg = z["hseg"].astype(np.int64); self.lseg = z["lseg"].astype(np.int64); self.fold = z["fold"].astype(np.int64)
        self.fam = z["fam"].astype(np.int64); self.ll = z["loglr"].astype(np.float64); self.net = z["net"].astype(np.float64)
        self.hm = z["cnn_hm"].astype(np.float64); self.lm = z["cnn_lm"].astype(np.float64)
        self.gpsH = w["gps_H"].astype(np.float64); self.gpsL = w["gps_L"].astype(np.float64)
        self.iH = w["iH"].astype(np.int64); self.iL = w["iL"].astype(np.int64)
        assert np.array_equal(self.fam, self.lseg * 10_000_000 + (self.gpsL // MERGE_S).astype(np.int64))
        self.seg_names = [str(s) for s in z["seg_names"]]; self.seg_ix = {s: i for i, s in enumerate(self.seg_names)}
        self.seg_fold = z["seg_fold"].astype(np.int64); self.far_live = z["far_live"].astype(np.float64)
        self.gate = np.maximum(self.hm, self.lm) > GATE
        self.finll = np.isfinite(self.ll)
        # veto results (per pair; nan = not evaluated -> surviving)
        self.hm_loc = np.full(n, np.nan); self.lm_loc = np.full(n, np.nan); self.net_loc = np.full(n, np.nan)
        self.veto_path = veto_path
        if veto_path is not None:
            v = np.load(veto_path, allow_pickle=False)
            ix = v["idx"].astype(np.int64)
            self.hm_loc[ix] = v["hm_loc"]; self.lm_loc[ix] = v["lm_loc"]; self.net_loc[ix] = v["net_loc"]
        ev = np.isfinite(self.hm_loc) & np.isfinite(self.lm_loc)
        self.evaluated = ev
        self.cnn_vet = ev & (np.maximum(self.hm_loc, self.lm_loc) < GATE)
        self.net_vet = self.cnn_vet | (ev & np.isfinite(self.net_loc) & (self.net_loc < NET_FLOOR))
        self.F = {f: self._build_fold(f) for f in (0, 1)}
        if verbose:
            for f in (0, 1):
                F = self.F[f]
                print(f"[successor:{run}] fold {f}: {len(F['gi'])} pairs, {F['nfam']} families, T_f={F['T']:.2f} yr, "
                      f"lr-channel families {int(F['has_lr'].sum())} (surviving {int(F['valid_lr'].sum())}), "
                      f"net-channel families {int(F['has_net'].sum())} (surviving {int(F['valid_net'].sum())}); "
                      f"veto evaluated pairs {int(ev[F['gi']].sum())}", flush=True)

    # ------------------------------------------------------------------ fold structures
    def _build_fold(self, f):
        g = np.where(self.fold == f)[0]
        o = np.argsort(self.fam[g], kind="stable"); gi = g[o]              # fold pairs sorted by family
        famk = self.fam[gi]
        first = np.ones(len(gi), bool); first[1:] = famk[1:] != famk[:-1]
        fid = np.cumsum(first) - 1; nfam = int(fid[-1]) + 1 if len(gi) else 0
        fstart = np.where(first)[0]; fend = np.append(fstart[1:], len(gi))
        fam_key = famk[first]
        # lr rep: argmax ll among gate & finite pairs, per family
        m = self.gate[gi] & self.finll[gi]
        llm = np.where(m, self.ll[gi], -np.inf)
        o2 = np.lexsort((-llm, fid)); f2 = np.ones(len(gi), bool); f2[1:] = fid[o2][1:] != fid[o2][:-1]
        cand = o2[f2]                                                       # position (in gi) of best per family
        rep_lr = np.where(np.isfinite(llm[cand]), gi[cand], -1)
        r_lr = np.where(rep_lr >= 0, llm[cand], -np.inf)
        # net rep: argmax net over all cached pairs of the family (famN rep)
        o3 = np.lexsort((-self.net[gi], fid)); f3 = np.ones(len(gi), bool); f3[1:] = fid[o3][1:] != fid[o3][:-1]
        cand3 = o3[f3]; rep_net = gi[cand3]; r_net = self.net[rep_net]
        has_lr = rep_lr >= 0
        has_net = self.gate[rep_net]
        valid_lr = has_lr & ~self.cnn_vet[np.maximum(rep_lr, 0)]
        valid_net = has_net & ~self.net_vet[rep_net]
        F = dict(f=f, gi=gi, fid=fid, nfam=nfam, fstart=fstart, fend=fend, fam_key=fam_key,
                 rep_lr=rep_lr, r_lr=r_lr, rep_net=rep_net, r_net=r_net, has_lr=has_lr, has_net=has_net,
                 valid_lr=valid_lr, valid_net=valid_net, T=float(self.far_live[f]))
        # sorted valid values (descending) for global counts
        F["lr_sorted"] = np.sort(r_lr[valid_lr])[::-1]
        F["net_sorted"] = np.sort(r_net[valid_net])[::-1]
        # exclusion indices: pairs sorted by (hseg, gpsH) and by (lseg, gpsL)
        kH = _key(self.hseg[gi], self.gpsH[gi]); oH = np.argsort(kH, kind="stable"); F["kH"] = kH[oH]; F["pH"] = oH   # positions in gi
        kL = _key(self.lseg[gi], self.gpsL[gi]); oL = np.argsort(kL, kind="stable"); F["kL"] = kL[oL]; F["pL"] = oL
        F["fam_sorted"] = fam_key                                          # sorted ascending (families sorted by key)
        # position lookup: global index -> position in gi (for touched-family bookkeeping)
        pos = np.full(self.n, -1, np.int64); pos[gi] = np.arange(len(gi)); F["pos"] = pos
        return F

    # ------------------------------------------------------------------ exclusion set
    def _excluded_positions(self, F, hseg_q, gpsH_q, lseg_q, gpsL_q, fam_q):
        """Positions (in F['gi']) of the pairs excluded for one candidate. A2 rule (EXCL_MODE='own_pair'):
        only pairs whose H1 window AND L1 window both coincide with the candidate's (within EXCL_TOL)."""
        assert EXCL_MODE == "own_pair"
        if hseg_q < 0 or lseg_q < 0: return np.zeros(0, np.int64)
        lo = int(hseg_q) * (1 << 36) + int(np.rint((gpsH_q - EXCL_TOL) * 4.0)); hi = int(hseg_q) * (1 << 36) + int(np.rint((gpsH_q + EXCL_TOL) * 4.0))
        a, b = np.searchsorted(F["kH"], lo, "left"), np.searchsorted(F["kH"], hi, "right")
        if b <= a: return np.zeros(0, np.int64)
        cand = F["pH"][a:b]; gi = F["gi"][cand]
        m = (self.lseg[gi] == lseg_q) & (np.abs(self.gpsL[gi] - gpsL_q) <= EXCL_TOL) & (np.abs(self.gpsH[gi] - gpsH_q) <= EXCL_TOL)
        return np.unique(cand[m])

    # ------------------------------------------------------------------ core rank
    def _rank_core(self, f, ll_q, net_q, hseg_q, gpsH_q, lseg_q, gpsL_q, fam_q):
        """Unconditioned ranks (N_lr, N_net) of one candidate against fold f with the narrowed exclusion.
        N_lr is None if the candidate has no lnLambda channel; N_net None if no sigma_net channel."""
        F = self.F[f]; gi = F["gi"]
        use_lr = bool(np.isfinite(ll_q) and ll_q >= LR_FLOOR); use_net = bool(np.isfinite(net_q) and net_q >= NET_FLOOR)
        N_lr = int(np.searchsorted(-F["lr_sorted"], -ll_q, "left")) if use_lr else None     # r_lr > ll_q   (-r < -ll_q)
        N_net = int(np.searchsorted(-F["net_sorted"], -net_q, "right")) if use_net else None  # r_net >= net_q (-r <= -net_q)
        E = self._excluded_positions(F, hseg_q, gpsH_q, lseg_q, gpsL_q, fam_q)
        n_excl = int(len(E)); n_touch = 0
        if n_excl:
            fids = np.unique(F["fid"][E]); n_touch = int(len(fids))
            # all positions of the touched families, minus the excluded ones (vectorized)
            st = F["fstart"][fids]; en = F["fend"][fids]; ln = en - st
            allpos = np.repeat(st - np.cumsum(ln) + ln, ln) + np.arange(int(ln.sum()))
            keepm = np.ones(len(allpos), bool); keepm[np.searchsorted(allpos, E)] = False   # E subset of allpos (sorted)
            kp = allpos[keepm]; kfid = F["fid"][kp]; kk = gi[kp]
            if use_lr:
                c0 = int((F["valid_lr"][fids] & (F["r_lr"][fids] > ll_q)).sum())
                m = self.gate[kk] & self.finll[kk]
                c1 = 0
                if m.any():
                    k2 = kk[m]; f2 = kfid[m]
                    o = np.lexsort((-self.ll[k2], f2)); first = np.ones(len(o), bool); first[1:] = f2[o][1:] != f2[o][:-1]
                    rep = k2[o[first]]
                    c1 = int(((~self.cnn_vet[rep]) & (self.ll[rep] > ll_q)).sum())
                N_lr += c1 - c0
            if use_net:
                c0 = int((F["valid_net"][fids] & (F["r_net"][fids] >= net_q)).sum())
                c1 = 0
                if len(kk):
                    o = np.lexsort((-self.net[kk], kfid)); first = np.ones(len(o), bool); first[1:] = kfid[o][1:] != kfid[o][:-1]
                    rep = kk[o[first]]
                    c1 = int((self.gate[rep] & (~self.net_vet[rep]) & (self.net[rep] >= net_q)).sum())
                N_net += c1 - c0
        return dict(N_lr=N_lr, N_net=N_net, n_excluded_pairs=n_excl, n_touched_families=n_touch, fold=f, T=F["T"])

    # ------------------------------------------------------------------ entry point 1: PRODUCTION
    def score_candidate(self, cand, neff=1.0):
        """cand: dict with loglr, net, cnn_hm, cnn_lm and either (seg, gps) [zero-lag: H1=L1 window] or
        (hseg, gpsH, lseg, gpsL) [explicit windows]; fold optional (else from seg). neff: frozen N_eff(run, fold)
        (float) or callable(x0)->N_eff (piecewise-in-x). Returns dict with N_c, FARs, UL, gate result."""
        if "seg" in cand and "hseg" not in cand:
            S = self.seg_ix[cand["seg"]]; gps = float(cand["gps"])
            hseg = lseg = S; gpsH = gpsL = gps
            f = int(cand.get("fold", self.seg_fold[S]))
        else:
            hseg = int(cand["hseg"]); lseg = int(cand["lseg"]); gpsH = float(cand["gpsH"]); gpsL = float(cand["gpsL"])
            f = int(cand["fold"])
        fam_q = lseg * 10_000_000 + int(gpsL // MERGE_S)
        ll = float(cand["loglr"]) if cand.get("loglr") is not None else -np.inf
        net = float(cand["net"]); hm = float(cand["cnn_hm"]); lm = float(cand["cnn_lm"])
        out = dict(fold=f, T=self.F[f]["T"], gate_pass=bool(max(hm, lm) > GATE), loglr=ll, net=net, cnn_hm=hm, cnn_lm=lm)
        r = self._rank_core(f, ll, net, hseg, gpsH, lseg, gpsL, fam_q)
        out.update(r)
        out.update(far_from_counts(r["N_lr"], r["N_net"], self.F[f]["T"], neff))
        return out

    # ------------------------------------------------------------------ entry point 2: NULL TEST
    def score_pseudo_fg(self, f, p, neff=1.0):
        """Background pair p (global cache index, fold f) as pseudo-candidate with its own lnLambda/net/HM/LM
        and its own H1/L1 windows (leave-one-out through the narrowed exclusion + own family)."""
        assert self.fold[p] == f
        r = self._rank_core(f, float(self.ll[p]), float(self.net[p]), int(self.hseg[p]), float(self.gpsH[p]),
                            int(self.lseg[p]), float(self.gpsL[p]), int(self.fam[p]))
        r.update(far_from_counts(r["N_lr"], r["N_net"], self.F[f]["T"], neff))
        r.update(pair=int(p), loglr=float(self.ll[p]), net=float(self.net[p]), cnn_hm=float(self.hm[p]), cnn_lm=float(self.lm[p]),
                 gate_pass=bool(self.gate[p]))
        return r

    # ------------------------------------------------------------------ pseudo-fg population
    def pseudo_fg_population(self, f):
        """Pseudo-foreground families of fold f (Sec. 3): gate-passing, veto-surviving families, each represented by
        (a) its lnLambda rep pair (family has the lnLambda channel) or (b) its net rep pair (net-only family).
        Returns global pair indices and a flag has_lr. The 'winning-channel' net_loc part of the veto is applied
        by the caller after FARs are known (mirrors the foreground rule)."""
        F = self.F[f]
        a = F["rep_lr"][F["valid_lr"]]
        b = F["rep_net"][(~F["has_lr"]) & F["valid_net"]]
        return np.concatenate([a, b]), np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])

    def top_M_pairs(self, f):
        """Rep pairs of the top-M families per channel (ranked by r_c) -> union (for the background veto)."""
        F = self.F[f]; M = top_M(F["T"])
        lr = np.where(F["has_lr"])[0]; lr = lr[np.argsort(-F["r_lr"][lr], kind="stable")][:M]
        nt = np.where(F["has_net"])[0]; nt = nt[np.argsort(-F["r_net"][nt], kind="stable")][:M]
        p_lr = F["rep_lr"][lr]; p_nt = F["rep_net"][nt]
        return dict(M=M, lr_pairs=p_lr, net_pairs=p_nt, union=np.unique(np.concatenate([p_lr, p_nt])),
                    lr_min_value=float(F["r_lr"][lr].min()) if len(lr) else None, net_min_value=float(F["r_net"][nt].min()) if len(nt) else None)

    # ------------------------------------------------------------------ OLD statistic (regression plumbing check)
    def count_asrun(self, cand):
        """As-run conditioned per-arm counts with WHOLE-SEGMENT exclusion (far_repro_check.perarm_counts logic)."""
        S = self.seg_ix[cand["seg"]]; f = int(self.seg_fold[S])
        notself = (self.hseg != S) & (self.lseg != S); m = (self.fold == f) & notself
        ll = cand["loglr"]; hm = cand["cnn_hm"]; lm = cand["cnn_lm"]; net = cand["net"]
        mm = m & (self.ll > ll)
        n_lr_hm = len(np.unique(self.fam[mm & (self.hm >= hm)])); n_lr_lm = len(np.unique(self.fam[mm & (self.lm >= lm)]))
        F = self.F[f]; rep = F["rep_net"]
        rm = (self.hseg[rep] != S) & (self.lseg[rep] != S) & (self.net[rep] >= net)
        rr = rep[rm]
        n_net_hm = int((self.hm[rr] >= hm).sum()); n_net_lm = int((self.lm[rr] >= lm).sum())
        return dict(lr_hm=n_lr_hm, lr_lm=n_lr_lm, net_hm=n_net_hm, net_lm=n_net_lm, T=F["T"])

    # ------------------------------------------------------------------ brute force (validation)
    def brute_rank(self, f, ll_q, net_q, hseg_q, gpsH_q, lseg_q, gpsL_q, fam_q):
        gi = self.F[f]["gi"]
        ex = (self.hseg[gi] == hseg_q) & (np.abs(self.gpsH[gi] - gpsH_q) <= EXCL_TOL) & \
             (self.lseg[gi] == lseg_q) & (np.abs(self.gpsL[gi] - gpsL_q) <= EXCL_TOL)   # A2: own pair only
        k = gi[~ex]; fam = self.fam[k]
        N_lr = N_net = None
        if np.isfinite(ll_q) and ll_q >= LR_FLOOR:
            m = self.gate[k] & self.finll[k]; kk = k[m]; fk = fam[m]
            o = np.lexsort((-self.ll[kk], fk)); first = np.ones(len(o), bool); first[1:] = fk[o][1:] != fk[o][:-1]
            rep = kk[o[first]]
            N_lr = int(((~self.cnn_vet[rep]) & (self.ll[rep] > ll_q)).sum())
        if np.isfinite(net_q) and net_q >= NET_FLOOR:
            o = np.lexsort((-self.net[k], fam)); first = np.ones(len(o), bool); first[1:] = fam[o][1:] != fam[o][:-1]
            rep = k[o[first]]
            N_net = int((self.gate[rep] & (~self.net_vet[rep]) & (self.net[rep] >= net_q)).sum())
        return N_lr, N_net


def far_from_counts(N_lr, N_net, T, neff=1.0):
    """FAR = N_eff * min_c N_c / T; UL90 = N_eff * chi2.ppf(.9, 2(N+1))/2 / T of the same channel;
    neff may be a float or a callable of the uncorrected x0 = min_c N_c / T (piecewise-in-x)."""
    ch = {}
    if N_lr is not None: ch["loglr"] = N_lr
    if N_net is not None: ch["net-sigma"] = N_net
    if not ch:
        return dict(channel=None, N_min=None, x0=None, N_eff=None, far=None, ul90=None, far_lr=None, far_net=None)
    c = min(ch, key=lambda k: (ch[k], k != "loglr"))            # ties -> loglr (as-run _chan convention)
    Nmin = ch[c]; x0 = Nmin / T
    ne = float(neff(x0)) if callable(neff) else float(neff)
    return dict(channel=c, N_min=int(Nmin), x0=x0, N_eff=ne, far=ne * Nmin / T,
                ul90=ne * float(chi2.ppf(0.90, 2 * (Nmin + 1)) / 2) / T,
                far_lr=(N_lr / T if N_lr is not None else None), far_net=(N_net / T if N_net is not None else None))


# ---------------------------------------------------------------------- self-test
def self_test(run, veto_path=None, n_sample=300, n_brute=60, seed=20260818, log=True):
    """(1) production vs null-test entry points give identical N_c on a background sample; (2) both agree with a
    brute-force recount; (3) as-run per-arm counts reproduce detections.json (plumbing). Returns a JSON-able dict."""
    rng = np.random.default_rng(seed); t0 = time.time()
    bg = Background(run, veto_path, verbose=log)
    res = dict(run=run, module_md5=module_md5(), spec_md5=assert_spec(), veto_path=veto_path, folds={})
    ok_all = True
    for f in (0, 1):
        pop, has_lr = bg.pseudo_fg_population(f)
        # sample: random + loudest by lnLambda and by net (the tail that matters)
        F = bg.F[f]
        loud_lr = pop[has_lr][np.argsort(-bg.ll[pop[has_lr]])[:40]] if has_lr.any() else np.zeros(0, np.int64)
        loud_nt = pop[np.argsort(-bg.net[pop])[:40]]
        samp = np.unique(np.concatenate([rng.choice(pop, min(n_sample, len(pop)), replace=False), loud_lr, loud_nt]))
        n_mis = 0; n_brute_mis = 0; nb = 0
        for j, p in enumerate(samp):
            a = bg.score_pseudo_fg(f, int(p))
            cand = dict(loglr=float(bg.ll[p]), net=float(bg.net[p]), cnn_hm=float(bg.hm[p]), cnn_lm=float(bg.lm[p]),
                        hseg=int(bg.hseg[p]), gpsH=float(bg.gpsH[p]), lseg=int(bg.lseg[p]), gpsL=float(bg.gpsL[p]), fold=f)
            b = bg.score_candidate(cand)
            if (a["N_lr"], a["N_net"]) != (b["N_lr"], b["N_net"]): n_mis += 1
            if j < n_brute:
                nb += 1
                br = bg.brute_rank(f, float(bg.ll[p]), float(bg.net[p]), int(bg.hseg[p]), float(bg.gpsH[p]), int(bg.lseg[p]), float(bg.gpsL[p]), int(bg.fam[p]))
                if br != (a["N_lr"], a["N_net"]): n_brute_mis += 1
        # zero-lag-form production candidates vs brute force (seg,gps form): use the fold's real triggers if any
        res["folds"][str(f)] = dict(n_pop=int(len(pop)), n_sample=int(len(samp)), n_entrypoint_mismatch=n_mis, n_brute=nb, n_brute_mismatch=n_brute_mis)
        ok_all &= (n_mis == 0 and n_brute_mis == 0)
        if log: print(f"[self-test:{run}] fold {f}: {len(samp)} pseudo-fg scored via BOTH entry points: {n_mis} mismatches; brute force on {nb}: {n_brute_mis} mismatches", flush=True)
    # (3) as-run plumbing: reproduce detections.json per-arm counts
    dets = json.load(open(f"{RUNS[run]}/detections.json")); nbad = 0; rows = []
    for d in dets:
        c = bg.count_asrun(d); T = c["T"]
        ref = {k: (None if d.get(f"far_{k}") is None else int(round(d[f"far_{k}"] * T))) for k in ("lr_hm", "lr_lm", "net_hm", "net_lm")}
        got = {k: c[k] for k in ("lr_hm", "lr_lm", "net_hm", "net_lm")}
        ok = all(ref[k] is None or ref[k] == got[k] for k in got); nbad += (not ok)
        rows.append(dict(seg=d["seg"], ok=ok, ref=ref, got=got))
        # brute-force check of the production zero-lag form vs brute_rank
        S = bg.seg_ix[d["seg"]]; f = int(bg.seg_fold[S]); gps = float(d["gps"])
        sc = bg.score_candidate(d); br = bg.brute_rank(f, float(d["loglr"]), float(d["net"]), S, gps, S, gps, S * 10_000_000 + int(gps // MERGE_S))
        if br != (sc["N_lr"], sc["N_net"]): nbad += 1; rows[-1]["zero_lag_brute_mismatch"] = True
    res["asrun_repro"] = dict(n=len(dets), n_bad=nbad, rows=rows)
    ok_all &= (nbad == 0)
    res["PASS"] = bool(ok_all); res["elapsed_s"] = time.time() - t0
    if log: print(f"[self-test:{run}] as-run per-arm counts reproduced {len(dets)-nbad}/{len(dets)} (+ zero-lag brute check); OVERALL {'PASS' if ok_all else 'FAIL'} ({res['elapsed_s']:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if what == "selftest":
        runs = sys.argv[2:] or list(RUNS)
        out = {r: self_test(r) for r in runs}
        out["module_md5"] = module_md5(); out["spec_md5"] = SPEC_MD5
        json.dump(out, open(f"{DET}/successor_stat_selftest_S0.json", "w"), indent=1)
        print(json.dumps({r: out[r]["PASS"] for r in runs}))
    elif what == "md5":
        print(module_md5())

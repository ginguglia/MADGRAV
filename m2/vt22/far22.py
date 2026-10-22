"""MADGRAV 2.2 injection efficiency, step 2: the merge's FAR arithmetic on flat arrays (far_tables_<run>.npz from
export_far_tables.py), applied to (a) the merge's own zero-lag triggers -> identity check against blindscan_s3rank.json,
(b) injection files -> inj_scored_<run>_m22.npz (per-injection detection at FAR < 1/yr under the 2.2 recipe).

The arithmetic mirrors m2/rank/driver_blindscan_merge_hook.py, S3 hook on, per-arm statistic:
  loglr channel  : open iff stat >= floor and S3 >= S3_THR; n = distinct L1 families among same-fold bank rows with
                   stat > trigger stat, bank S3 >= trigger S3, neither leg in the trigger's segment; FAR = n/T, UL90 chi2.
  net-sigma chan.: open iff net >= NETSIG_FLOOR, max(June hm, lm) >= GLITCH_THRESH and lm > NETSIG_LM_MIN; over same-fold
                   families with net >= trigger net, neither leg in the trigger's segment, lm > NETSIG_LM_MIN:
                   nhm = #(hm >= trig hm), nlm = #(lm >= trig lm); FAR = 2 min(nhm,nlm)/T, UL90 = 2 min(ul90).
  best           : n_open_channels x min over channels (point and UL90); detected iff both < 1/yr.
Not mirrored: the local-ASD consistency veto (post-processing of the detection list; MADGRAV1 measured >= 97% signal
retention at SNR >= 8). veto_keep is written as 1 and veto_applied=0 so the figure can say so.
Usage: far22.py validate --run o4b --tables F --blindscan B
       far22.py score    --run o4b --tables F --inj-dirs D1 D2 --out O [--assume-gates]   (campaign env: imports driver_search_multi)
Written 2026-10-04."""
import os, sys, json, glob, time, argparse, numpy as np
from scipy.stats import chi2

def ul90(n, flt): return float(chi2.ppf(0.90, 2 * (n + 1)) / 2 / flt)

class Tables:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True); self.z = z; self.path = path
        self.seg_name = [str(s) for s in z["seg_name"]]; self.name2i = {n: i for i, n in enumerate(self.seg_name)}; self.fold = z["seg_fold"].astype(int)
        self.live = {0: float(z["far_live0"]), 1: float(z["far_live1"])}; self.floor = float(z["floor"]); self.s3_thr = float(z["s3_thr"])
        self.netsig_floor = float(z["netsig_floor"]); self.lm_min = float(z["netsig_lm_min"]); self.glitch = float(z["glitch_thresh"]); self.net_cut = float(z["net_cut"])
        self.zl_gates = bool(int(z["zl_gates"])); self.arm_rule = str(z["arm_rule"]); self.lr_model = str(z["lr_model"])
        self.B = {}; self.F = {}
        for g in (0, 1):
            m = z["bank_fold"] == g; o = np.argsort(-z["bank_stat"][m], kind="stable")
            key = z["bank_bi"][m][o].astype(np.int64) * 10**10 + z["bank_l1bin"][m][o].astype(np.int64)
            _, fam = np.unique(key, return_inverse=True)
            self.B[g] = dict(stat=z["bank_stat"][m][o], s3=z["bank_s3"][m][o], ai=z["bank_ai"][m][o], bi=z["bank_bi"][m][o], fam=fam.astype(np.int64), nfam=int(fam.max()) + 1 if len(fam) else 0)
            m = z["fam_fold"] == g; o = np.argsort(-z["fam_net"][m], kind="stable")
            hm = z["fam_hm"][m][o].copy(); lm = z["fam_lm"][m][o].copy(); hm[~np.isfinite(hm)] = -np.inf; lm[~np.isfinite(lm)] = -np.inf
            self.F[g] = dict(net=z["fam_net"][m][o], ai=z["fam_ai"][m][o], bi=z["fam_bi"][m][o], hm=hm, lm=lm)
    def _ndistinct(self, fam, nfam):
        if len(fam) == 0: return 0
        seen = np.zeros(nfam, bool); seen[fam] = True; return int(seen.sum())
    def lr_counts(self, g, si, stat, s3):
        """(n_stage1, n_s3): distinct L1 families with bank stat > stat; stage1 = no other condition (merge Stage-1,
        'n_louder_L1fam'), n_s3 = + bank S3 >= s3 and neither leg in segment si (merge per-arm S3 count)."""
        B = self.B[g]; k = int(np.searchsorted(-B["stat"], -stat, side="left"))
        if k == 0: return 0, 0
        fam = B["fam"][:k]; n1 = self._ndistinct(fam, B["nfam"])
        m = (B["s3"][:k] >= s3) & (B["ai"][:k] != si) & (B["bi"][:k] != si)
        return n1, self._ndistinct(fam[m], B["nfam"])
    def net_counts(self, g, si, net, hm, lm):
        """(nk, nhm, nlm): same-fold families with net >= trigger net. nk = CNN-passing (max(hm,lm) > GLITCH and lm > LM_MIN,
        no segment exclusion; merge 'n_louder_netfam_cnn'); nhm/nlm = per-arm counts with both legs outside si and lm > LM_MIN."""
        F = self.F[g]; k = int(np.searchsorted(-F["net"], -net, side="right"))
        if k == 0: return 0, 0, 0
        H = F["hm"][:k]; L = F["lm"][:k]
        p = np.maximum(H, L) > self.glitch
        if self.lm_min > 0: p &= L > self.lm_min
        m = (F["ai"][:k] != si) & (F["bi"][:k] != si)
        if self.lm_min > 0: m &= L > self.lm_min
        return int(p.sum()), int((m & (H >= hm)).sum()), int((m & (L >= lm)).sum())
    def trigger(self, g, si, stat, s3, net, hm, lm):
        """Full per-trigger result dict in the merge's field names (None where the merge writes None)."""
        flt = self.live[g]; r = dict(fold=g, si=si, below_floor=bool(stat < self.floor))
        if stat >= self.floor:
            n1, n = self.lr_counts(g, si, stat, s3 if s3 is not None else np.inf)
            r["n_louder_L1fam"] = n1; r["far_lr_per_yr"] = n1 / flt; r["far_lr_ul90"] = ul90(n1, flt)
            r["is_glitch_lr"] = bool(s3 is None or s3 < self.s3_thr)
            if not r["is_glitch_lr"]:
                r["n_louder_L1fam_cnn"] = n; r["far_lr_perarm"] = n / flt; r["far_lr_perarm_ul90"] = ul90(n, flt)
            else: r["n_louder_L1fam_cnn"] = None; r["far_lr_perarm"] = None; r["far_lr_perarm_ul90"] = None
        else:
            r["n_louder_L1fam"] = None; r["far_lr_per_yr"] = None; r["far_lr_ul90"] = None; r["is_glitch_lr"] = True; r["n_louder_L1fam_cnn"] = None; r["far_lr_perarm"] = None; r["far_lr_perarm_ul90"] = None
        hm = float(hm) if hm is not None else 0.0; lm = float(lm) if lm is not None else 0.0
        r["is_glitch"] = bool(max(hm, lm) < self.glitch); r["netsig_lm_vetoed"] = bool(lm <= self.lm_min) if self.lm_min > 0 else False
        # merge order: a net-candidate (net >= floor, LM rule passed) is counted; then, S3 hook on, a June-gate failure
        # (is_glitch) nulls the far_net_* fields but leaves the diagnostic count n_louder_netfam_cnn in place (merge lines 384-389)
        if net >= self.netsig_floor and not r["netsig_lm_vetoed"]:
            nk, nhm, nlm = self.net_counts(g, si, net, hm, lm)
            r["n_louder_netfam_cnn"] = nk
            if not r["is_glitch"]:
                r["far_net_per_yr"] = nk / flt; r["far_net_ul90"] = ul90(nk, flt)
                r["far_net_hm"] = nhm / flt; r["far_net_lm"] = nlm / flt; r["far_net_perarm"] = 2 * min(nhm, nlm) / flt; r["far_net_perarm_ul90"] = 2 * min(ul90(nhm, flt), ul90(nlm, flt))
            else:
                for k_ in ("far_net_per_yr", "far_net_ul90", "far_net_hm", "far_net_lm", "far_net_perarm", "far_net_perarm_ul90"): r[k_] = None
        else:
            for k_ in ("n_louder_netfam_cnn", "far_net_per_yr", "far_net_ul90", "far_net_hm", "far_net_lm", "far_net_perarm", "far_net_perarm_ul90"): r[k_] = None
        ch = [(r["far_lr_perarm"], r["far_lr_perarm_ul90"], "loglr"), (r["far_net_perarm"], r["far_net_perarm_ul90"], "net-sigma")]
        ch = [c for c in ch if c[0] is not None]
        if ch:
            n = len(ch); pt, ul, name = min(ch, key=lambda c: c[0]); r["best_far"] = n * pt; r["best_ul90"] = n * ul; r["channel"] = name
            r["detected"] = bool(r["best_far"] < 1.0 and r["best_ul90"] < 1.0)
        else: r["best_far"] = None; r["best_ul90"] = None; r["channel"] = None; r["detected"] = False
        return r

def _cmp(name, a, b, tol, bad):
    if (a is None) != (b is None): bad.append((name, a, b)); return
    if a is None: return
    if abs(float(a) - float(b)) > tol * max(1.0, abs(float(b))): bad.append((name, a, b))

def validate(args):
    T = Tables(args.tables); bs = json.load(open(args.blindscan)); trig = bs["triggers"]
    assert abs(bs["far_live_yr"]["0"] - T.live[0]) < 1e-9 and abs(bs["far_live_yr"]["1"] - T.live[1]) < 1e-9, ("livetime mismatch", bs["far_live_yr"], T.live)
    FIELDS = ("n_louder_L1fam", "far_lr_per_yr", "far_lr_perarm", "far_lr_perarm_ul90", "n_louder_netfam_cnn", "far_net_per_yr", "far_net_hm", "far_net_lm", "far_net_perarm", "far_net_perarm_ul90")
    nbad = 0; nchk = {k: 0 for k in FIELDS}; ndet = 0; ndet_ref = 0; worst = []
    for t in trig:
        g = int(t["fold"]); si = int(t["si"]); assert T.seg_name[si] == t["seg"], (si, t["seg"])
        r = T.trigger(g, si, float(t["loglr"]), t.get("s3"), float(t["net"]), t.get("cnn_hm"), t.get("cnn_lm"))
        bad = []
        for k in FIELDS:
            if t.get(k) is not None or r.get(k) is not None: nchk[k] += 1
            _cmp(k, r.get(k), t.get(k), 1e-9, bad)
        # the merge's best_far/UL90 are not stored per trigger in blindscan.json; recompute them from its own fields
        ch = [(t.get("far_lr_perarm"), t.get("far_lr_perarm_ul90")), (t.get("far_net_perarm"), t.get("far_net_perarm_ul90"))]; ch = [c for c in ch if c[0] is not None]
        if ch:
            n = len(ch); pt, ul = min(ch, key=lambda c: c[0]); ref_det = bool(n * pt < 1 and n * ul < 1); ndet_ref += ref_det
            if ref_det != r["detected"]: bad.append(("detected", r["detected"], ref_det))
        ndet += r["detected"]
        if bad: nbad += 1; worst.append((t["seg"], t["gps"], t.get("matches_known", ""), bad[:4]))
    print(f"[vt22-validate] {args.tables}: {len(trig)} merge triggers; fields checked {nchk}")
    print(f"[vt22-validate] triggers with any mismatch: {nbad}; detected (pre-veto) reproduced {ndet} vs merge {ndet_ref}")
    for w in worst[:15]: print("   MISMATCH", w)
    print("[vt22-validate] " + ("PASS" if nbad == 0 and ndet == ndet_ref else "FAIL"), flush=True)
    return 0 if (nbad == 0 and ndet == ndet_ref) else 1

RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1")   # == m2/arm/s3off5/score_rank_off5.py
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

def score(args):
    import joblib
    MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]
    for _p in ("search_mode", "improved", "spectrogram_cascade"):
        _ap = os.path.join(MADGRAV_ROOT, _p)
        if _ap not in sys.path: sys.path.insert(0, _ap)
    os.chdir(MADGRAV_ROOT); import driver_search_multi as M
    T = Tables(args.tables); t0 = time.time()
    lrm = np.load(args.lr_model or T.lr_model); mdl = {g: (lrm[f"mu{g}"], lrm[f"sd{g}"], lrm[f"be{g}"]) for g in (0, 1)}
    models = joblib.load(args.ranker or f"{MADGRAV_SCRATCH}/madgrav2_spec/rank/models_{args.run}.joblib"); assert set(models) == {0, 1}
    def rule_pass(gH, gL, sH, sL):
        if T.arm_rule == "none": return np.ones(len(gH), bool)
        name, thr = T.arm_rule.split(":"); thr = float(thr)
        if name == "max": v = np.where(sH >= sL, gH, gL)
        elif name == "mean": v = 0.5 * (gH + gL)
        elif name == "both": v = np.minimum(gH, gL)
        elif name == "ramp": v = np.clip(sH / 3.0, 0, 1) * gH + np.clip(sL / 3.0, 0, 1) * gL
        else: raise SystemExit(T.arm_rule)
        return v >= thr
    files = [f for d in args.inj_dirs for f in sorted(glob.glob(f"{d}/*_inj.npz"))]
    if not files: raise SystemExit(f"no *_inj.npz in {args.inj_dirs}")
    cols = {k: [] for k in ("mtot", "net_snr", "is_um", "ev", "fold", "net", "lnl", "stat", "s3", "cnn_hm", "cnn_lm", "trig", "lr_open", "net_open", "far_lr", "far_net", "best_far", "best_ul90", "channel", "detected", "n_lr", "nhm", "nlm", "gate_fail")}
    n_nonfinite = 0; have_cnn = None
    for f in files:
        z = np.load(f); name = os.path.basename(f).replace("_inj.npz", ""); si = T.name2i[name]; g = int(T.fold[si]); n = len(z["net"])
        sH, sL, coh, gx = (np.asarray(z[k], float) for k in ("sigH", "sigL", "coh", "gx")); cH2, cL2, gH2, gL2 = (np.asarray(z[k], float) for k in ("cenH2", "cenL2", "gH2", "gL2"))
        cH1, cL1, gH1, gL1 = (np.asarray(z[k], float) for k in ("cenH", "cenL", "gH", "gL")); net = np.asarray(z["net"], float)
        if have_cnn is None: have_cnn = ("cnn_hm" in z.files and "s3" in z.files)
        if have_cnn: hm = np.asarray(z["cnn_hm"], float); lm = np.asarray(z["cnn_lm"], float); s3 = np.asarray(z["s3"], float)
        elif args.assume_gates: hm = np.ones(n); lm = np.ones(n); s3 = np.ones(n)
        else: raise SystemExit(f"{f}: no cnn_hm/cnn_lm/s3 columns (inject with SM_INJ_CNN=1 SM_INJ_S3=1) and --assume-gates not given")
        # trigger formation: build_setup zero-lag rule (net > NET_CUT, optional centroid-v2 + arm gates) + merge arm rule on the pair itself
        trig = net > T.net_cut
        if T.zl_gates: trig &= M.cen_gate(cH2, cL2) & M.arm_gate(gH2, gL2, sH, sL)
        trig &= rule_pass(gH2, gL2, sH, sL)
        fin = np.isfinite(coh) & np.isfinite(gx) & np.isfinite(sH) & np.isfinite(sL); n_nonfinite += int((trig & ~fin).sum()); trig &= fin
        lnl = np.full(n, np.nan); stat = np.full(n, np.nan)
        if trig.any():
            F = M.feats(sH[trig], sL[trig], coh[trig], cH2[trig], cL2[trig], gH2[trig], gL2[trig], gx=gx[trig]); mu, sd, be = mdl[1 - g]
            lnl[trig] = M.loglr(mu, sd, be, F)
            X = np.column_stack([sH, sL, coh, gx, cH2, cL2, gH2, gL2, cH1, cL1, gH1, gL1])[trig]
            p = np.mean([m_.predict_proba(X)[:, 1] for m_ in models[1 - g]], 0)
            stat[trig] = np.where(lnl[trig] < 3.0, lnl[trig], 3.0 + 20.0 * p)      # == score_rank_off5.py (FLOOR 3, SCALE 20)
        for i in range(n):
            if not trig[i]:
                r = dict(far_lr_perarm=None, far_net_perarm=None, best_far=None, best_ul90=None, channel=None, detected=False, is_glitch_lr=True, is_glitch=True, netsig_lm_vetoed=False, n_louder_L1fam_cnn=None, far_net_hm=None, far_net_lm=None)
                nhm = nlm = None
            else:
                r = T.trigger(g, si, float(stat[i]), float(s3[i]), float(net[i]), float(hm[i]), float(lm[i]))
                nhm = r["far_net_hm"] * T.live[g] if r["far_net_hm"] is not None else None; nlm = r["far_net_lm"] * T.live[g] if r["far_net_lm"] is not None else None
            cols["mtot"].append(float(z["mtot"][i])); cols["net_snr"].append(float(z["net_snr"][i])); cols["is_um"].append(int(z["is_um"][i])); cols["ev"].append(name); cols["fold"].append(g)
            cols["net"].append(float(net[i])); cols["lnl"].append(float(lnl[i])); cols["stat"].append(float(stat[i])); cols["s3"].append(float(s3[i])); cols["cnn_hm"].append(float(hm[i])); cols["cnn_lm"].append(float(lm[i]))
            cols["trig"].append(bool(trig[i])); cols["lr_open"].append(bool(trig[i] and not r["is_glitch_lr"])); cols["net_open"].append(bool(trig[i] and not r["is_glitch"] and not r["netsig_lm_vetoed"] and r["far_net_perarm"] is not None))
            cols["far_lr"].append(r["far_lr_perarm"] if r["far_lr_perarm"] is not None else np.inf); cols["far_net"].append(r["far_net_perarm"] if r["far_net_perarm"] is not None else np.inf)
            cols["best_far"].append(r["best_far"] if r["best_far"] is not None else np.inf); cols["best_ul90"].append(r["best_ul90"] if r["best_ul90"] is not None else np.inf)
            cols["channel"].append(r["channel"] or ""); cols["detected"].append(bool(r["detected"])); cols["n_lr"].append(r["n_louder_L1fam_cnn"] if r["n_louder_L1fam_cnn"] is not None else -1)
            cols["nhm"].append(int(round(nhm)) if nhm is not None else -1); cols["nlm"].append(int(round(nlm)) if nlm is not None else -1)
            cols["gate_fail"].append("" if trig[i] else ("nonfinite" if not fin[i] else "trig"))
        d_ = np.array(cols["detected"][-n:]); print(f"[vt22-score] {name} fold {g}: {n} inj, trig {int(trig.sum())}, detected {int(d_.sum())} ({(time.time()-t0)/60:.1f} min)", flush=True)
    A = {k: np.array(v) for k, v in cols.items()}
    # rho^-4 population weights over the levels actually present (== pastro_final.py w_snr_of on its SNR_GRID)
    lev = np.unique(A["net_snr"]); mids = (lev[1:] + lev[:-1]) / 2; dr = np.diff(np.concatenate([[lev[0]], mids, [lev[-1]]])) if len(lev) > 1 else np.ones(1)
    w = lev ** -4.0 * dr; w /= w.sum(); w_of = {float(s): float(x) for s, x in zip(lev, w)}; A["w0"] = np.array([w_of[float(s)] for s in A["net_snr"]])
    A["det_frac"] = A["detected"].astype(float); A["far_mean"] = A["best_far"]; A["veto_keep"] = np.ones(len(A["det_frac"]), bool)
    meta = dict(npair=1, veto_applied=0, assume_gates=int(bool(args.assume_gates and not have_cnn)), tables=args.tables, lr_model=str(args.lr_model or T.lr_model), n_nonfinite_triggers=n_nonfinite,
                inj_dirs=np.array(args.inj_dirs), files=np.array(files), built=time.strftime("%F %T"), recipe="off5 arm max:-2.05 + S3 gate + HGB ranker, per-arm, S3 hook, LM rule", snr_levels=lev)
    np.savez(args.out, **A, **meta)
    for s in lev:
        m = A["net_snr"] == s; print(f"[vt22-score]  SNR {s:5.1f}: {int(m.sum())} inj, trig {A['trig'][m].mean():.3f}, det_frac {A['det_frac'][m].mean():.3f} (loglr {np.mean(A['channel'][m]=='loglr'):.3f}, net-sigma {np.mean(A['channel'][m]=='net-sigma'):.3f})")
    print(f"[vt22-score] wrote {args.out}: {len(A['det_frac'])} injections, {int(A['det_frac'].sum())} detected; assume_gates={meta['assume_gates']}; non-finite triggers dropped {n_nonfinite} ({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate"); v.add_argument("--run", required=True); v.add_argument("--tables", required=True); v.add_argument("--blindscan", required=True)
    s = sp.add_parser("score"); s.add_argument("--run", required=True); s.add_argument("--tables", required=True); s.add_argument("--inj-dirs", nargs="+", required=True); s.add_argument("--out", required=True)
    s.add_argument("--assume-gates", action="store_true"); s.add_argument("--lr-model", default=""); s.add_argument("--ranker", default="")
    a = ap.parse_args(); sys.exit(validate(a) if a.cmd == "validate" else score(a))

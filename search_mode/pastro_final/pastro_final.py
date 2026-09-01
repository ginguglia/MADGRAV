#!/usr/bin/env python
"""FINAL blind-search p_astro (panel-spec): FGMC on the BEST_FAR axis.

Panel verdict (2026-07): the blind detection statistic is best_far (per-arm
CNN-rank, x4 trials), NOT loglr. This script therefore:

  p_s : injections (inj_out_<run>) pushed through the IDENTICAL per-arm
        counting that produced madgrav_far_final.csv (gate: far_repro_check.py
        reproduces 44/44 frozen rows) -> injection best_far distribution.
  p_n : analytic on the FAR axis -- a noise candidate's FAR is uniform on
        [0, 1] /yr by construction (trials corrections are Bonferroni ->
        sub-uniform at low FAR -> assuming uniform is CONSERVATIVE for p_astro).
  L_n : FIXED to the null expectation DET_FAR * T_fg (self-calibrating FGMC:
        the time-slides give the noise RATE, it is not a free parameter).
        T_fg = analyzed zero-lag coincident time (segment sum x ANALYZED_FRAC).
        A free-(Ls,Ln) fit is recorded per run as a sensitivity (Ln_free).
  fit : per-run Poisson mixture over the run's detections, binned on the
        best_far axis (first bin isolates the N_louder=0 atom); only L_s free.

Injection CNN arm scores are not stored in the inj npz (the campaign predates
the per-arm statistic). The per-arm rank needs (cnn_hm, cnn_lm) for the event;
we AVERAGE the counting over the empirical joint (hm, lm) of ALL blind
detections, and quote a 5-95% bootstrap interval (resampling the det cnn
pairs AND the injection weights, refitting the mixture per replicate).

Population weights on injections: SNR grid -> Euclidean-volume rho^-4 * d_rho;
detected-injection Mtot -> reweighted to the published-catalog Mtot
distribution (merged_plot_v2.csv). O3b has no injection campaign -> the O3a
injection set is scored against the O3b background (same O3 detectors), both
folds averaged (flagged in the output).

Outputs (this directory):
  inj_scored_<run>.npz  : per-injection mtot/net_snr/loglr/net + emp best_far
                          (feeds the VT analysis)
  pastro_final.json     : per-detection p_astro + bracket + fit metadata
  pastro_final.csv      : run,name,far,p_astro,p_astro_lo,p_astro_hi
"""
import os, json, glob, csv
import numpy as np
from scipy.stats import chi2
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

from scipy.optimize import minimize

SC = MADGRAV_SCRATCH
MG = MADGRAV_ROOT
HERE = f"{MG}/search_mode/pastro_final"
SM = f"{MG}/search_mode"
RUNS = {
    "O3a": dict(out=f"{SC}/search_out_o3a_far_f40",
                inj=[f"{SC}/inj_out_o3a_56", f"{SM}/inj_out_o3a_lowsnr"]),
    "O3b": dict(out=f"{SC}/search_out_o3b_far_f40",
                inj=[f"{SM}/inj_out_o3b", f"{SM}/inj_out_o3b_lowsnr"]),
    "O4a": dict(out=f"{SC}/search_out_o4a_far",
                inj=[f"{SM}/inj_out_o4a", f"{SM}/inj_out_o4a_lowsnr"]),
    "O4b": dict(out=f"{SC}/search_out_o4b_far",
                inj=[f"{SM}/inj_out_o4b", f"{SM}/inj_out_o4b_lowsnr"]),
}
# RULING 2026-08-15 (flagged deviation, rescan_report VT-after leg): the pooled
# empirical CNN-pair multiset is ALWAYS the as-run detections of these four
# runs (47 pairs). Extending RUNS (e.g. "O4ars") scores that run's injections
# against the SAME 47 pairs - rescan detections are never pooled in
# (out-of-sample by construction; O3a in-sample lesson).
PAIR_RUNS = tuple(RUNS)
# SM_TRIALS_OFF=1 -> the single-counting statistic of the 48-detection accounting (trials = 1).
# Unset (default) reproduces the as-run x4 run byte-identically.
X1 = os.environ.get("SM_TRIALS_OFF", "0") == "1"
# SM_INJ_CNN_GATE=1 -> use the CNN-scored injection campaign (/scratch/.../inj_cnn) and apply the CNN
# glitch gate max(cnn_hm,cnn_lm) > GLITCH_THRESH to injections, exactly as real candidates must pass it.
# Without this the efficiency/VT omit the gate entirely and are upper bounds.
INJCNN = os.environ.get("SM_INJ_CNN_GATE", "0") == "1"
GLITCH_THRESH = float(os.environ.get("SM_GLITCH_THRESH", "0.5"))
# SM_INJ_DIR -> campaign root under $SCRATCH (default inj_cnn); SM_SUF_EXTRA -> appended to the
# output suffix so a re-scored campaign never lands on an accepted file. The reference-PSD-corrected
# campaign uses SM_INJ_DIR=inj_fixed SM_SUF_EXTRA=fix -> reads inj_fixed/, writes *_x1cnnfix.*
INJDIR = os.environ.get("SM_INJ_DIR", "inj_cnn")
# ---- adopted detection criterion (2026-08-31) --------------------------------------------------
# SM_LR_ONLY=1   rank on the lnLambda channel alone (the channel carrying coherence). The
#                sigma_net>4 TRIGGER is unchanged; only the FAR channel set changes.
# SM_NETMAX      upper sigma_net veto, threshold set on the injection population (p99.9 = 10.6),
#                applied to candidates and injections alike.
# SM_KE          JSON of per-run null-calibration factors; an injection counts as detected only if
#                FAR x K/E and UL x K/E are both below the 1/yr threshold -- i.e. the injection
#                efficiency is measured against the SAME calibrated criterion as the detections.
LR_ONLY = os.environ.get("SM_LR_ONLY", "0") == "1"
NETMAX = float(os.environ.get("SM_NETMAX", "0")) or None
_KE = os.environ.get("SM_KE", "")
KEMAP = json.load(open(_KE)) if _KE else None
SUF = ("_x1cnn" if INJCNN else "_x1") if X1 else ("_cnn" if INJCNN else "")
SUF += os.environ.get("SM_SUF_EXTRA", "")
if INJCNN:
    _IC = f"{SC}/{INJDIR}"
    for _r in RUNS:
        RUNS[_r]["inj"] = [f"{_IC}/{_r.lower()}", f"{_IC}/{_r.lower()}_lowsnr"]
CSVF = f"{MG}/figures/catalog_o3o4/madgrav_far_final{'_x1' if X1 else ''}.csv"
# SM_ADOPTED_AXIS=1 -> put the CANDIDATES on the same ruler as the injections. Without it,
# dets_of() reads best_far from the frozen x1 CSV (both channels, no veto) while the injections
# are scored lnLambda-only with the sigma_net veto -- the two halves of the FGMC then sit on
# different statistics. The injections are scored against the background AS BUILT (foreground
# inclusive) and are not K-scaled, so the matching candidate quantity is raw far_incl.
ADOPTED_AXIS = os.environ.get("SM_ADOPTED_AXIS", "0") == "1"
# SM_AXIS_COL: which background the candidate axis uses. "far_excl" (default) matches the
# foreground-excluded background of the detection criterion and of Table I; "far_incl" matches the
# background the injections are actually scored against. Neither is fully consistent -- the
# injection side applies no foreground exclusion -- and the two differ only for the events whose
# own windows enter their background.
AXIS_COL = os.environ.get("SM_AXIS_COL", "far_excl")
# SM_ADMIT: which candidates enter the fit / carry a p_astro.
#   "and" (frozen 2026-08-31) calibrated FAR < 1 AND calibrated UL90 < 1  -> 46
#   "far" (design decision 2026-09-01) calibrated FAR < 1, central value only -> 47
#   "all" every gate-passing candidate of the frozen table                 -> 48
ADMIT = os.environ.get("SM_ADMIT", "and")
# SM_DET_RULE: the admission rule applied to INJECTIONS, i.e. what the efficiency/VT measure.
# It must match the rule used to admit candidates or the signal model and the detection list
# describe different searches. "and" = FAR<1 AND UL90<1 (frozen); "far" = central FAR<1 only.
DET_RULE = os.environ.get("SM_DET_RULE", "and")
# SM_FAR_LIVE_SCALE: multiplies the background livetime used as the INJECTION FAR denominator.
# The caches are built with ANALYZED_FRAC=0.5, so every quoted FAR is 2x the wall-clock rate.
# Setting this to 2.0 undoes that halving and measures how much of the injection-side VT deficit
# the convention accounts for. Default 1.0 = the production convention, unchanged.
FAR_LIVE_SCALE = float(os.environ.get("SM_FAR_LIVE_SCALE", "1.0"))
ADOPTED = {}
if ADOPTED_AXIS:
    _kem = json.load(open(os.environ["SM_KE"]))
    _lr = {(r["run"], r["name"]): r for r in
           csv.DictReader(open(f"{MG}/figures/catalog_o3o4/far_lronly_g106.csv"))}
    _base = {(r["run"], r["name"]): r for r in
             csv.DictReader(open(f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"))}
    _f = lambda x: float(x) if str(x).strip() not in ("", "nan") else float("nan")
    for _k, _b in _base.items():
        if _f(_b["net"]) >= NETMAX or _k not in _lr:
            continue
        _ke = _kem[_k[0].lower()]
        _fc, _uc = _f(_lr[_k]["far_excl"]) * _ke, _f(_lr[_k]["ul90_excl"]) * _ke
        _ok = {"and": _fc < 1.0 and _uc < 1.0, "far": _fc < 1.0, "all": True}[ADMIT]
        if _ok:
            ADOPTED[_k] = _f(_lr[_k][AXIS_COL])           # raw, un-K-scaled
    _exp = {"and": 46, "far": 47, "all": 48}[ADMIT]
    assert len(ADOPTED) == _exp, f"admitted {len(ADOPTED)}, expected {_exp} for SM_ADMIT={ADMIT}"
    print(f"[pastro] SM_ADMIT={ADMIT}: {len(ADOPTED)} candidates on the raw {AXIS_COL} axis")
CATF = f"{MG}/figures/catalog_o3o4/merged_plot_v2.csv"
LRM = np.load(f"{SC}/o3a_frozen_lr_off200.npz")
MDL = {0: (LRM["mu0"], LRM["sd0"], LRM["be0"]), 1: (LRM["mu1"], LRM["sd1"], LRM["be1"])}
FLOOR, NETSIG_FLOOR, NET_CUT, GCLIP = 4.0, 4.0, 4.0, 6.0   # o*_far_merge launchers (f40)
DET_FAR = 1.0
SNR_GRID = np.array([5., 6., 7., 8., 10., 12., 15., 20., 25.])   # original grid + low-SNR extension
MASS_EDGES = np.array([0., 50., 80., 120., 180., 1e9])
FAR_EDGES = np.array([0., 0.005, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
EV_GPS = {"GW190521": 1242442967}   # o3a inj event not among detections

def gate(g, s): return np.clip(g, -GCLIP, GCLIP) * np.clip(np.asarray(s, float) / 3.0, 0, 1)
def feats(z):
    return np.column_stack([z["sigH"], z["sigL"], z["coh"], z["cenH"], z["cenL"],
                            gate(z["gH"], z["sigH"]), gate(z["gL"], z["sigL"])])
def loglr_of(F, g):
    mu, sd, be = MDL[g]
    return be[0] + ((np.asarray(F, float) - mu) / sd) @ be[1:]
def ul90(n, flt): return chi2.ppf(0.90, 2 * (np.asarray(n) + 1)) / 2 / flt


class RunBG:
    """Per-run background with per-event-segment, per-arm-threshold cumulative
    distinct-L1-family counts (the counting validated by far_repro_check.py)."""
    def __init__(self, run):
        z = np.load(f"{RUNS[run]['out']}/bg_cache_{run.lower()}.npz", allow_pickle=False)
        self.z = {k: z[k] for k in ("hseg", "lseg", "fold", "fam", "loglr", "net", "cnn_hm", "cnn_lm")}
        self.seg_names = [str(n) for n in z["seg_names"]]
        self.seg_ix = {n: i for i, n in enumerate(self.seg_names)}
        self.seg_fold = z["seg_fold"]; self.far_live = z["far_live"] * FAR_LIVE_SCALE
        key = self.z["fold"].astype(np.int64) * (1 << 60) + self.z["fam"]
        order = np.lexsort((-self.z["net"], key))
        ks = key[order]; first = np.ones(len(ks), bool); first[1:] = ks[1:] != ks[:-1]
        self.rep = order[first]                     # famN representatives

    def curves(self, six, g, qs_hm, qs_lm):
        """Counting curves for one event segment (six, -1 = none) in fold g:
        arm-thresholded cumulative distinct-family counts vs loglr (lr channel)
        and cumulative famN-rep counts vs net (net channel)."""
        z = self.z
        m = (z["fold"] == g) & (z["hseg"] != six) & (z["lseg"] != six)
        ll = z["loglr"][m]; fam = z["fam"][m]; hm = z["cnn_hm"][m]; lm = z["cnn_lm"][m]
        o = np.argsort(-ll)
        ll, fam, hm, lm = ll[o], fam[o], hm[o], lm[o]
        rep = self.rep
        rm = (z["fold"][rep] == g) & (z["hseg"][rep] != six) & (z["lseg"][rep] != six)
        rnet = z["net"][rep][rm]; rhm = z["cnn_hm"][rep][rm]; rlm = z["cnn_lm"][rep][rm]
        ro = np.argsort(-rnet)
        rnet, rhm, rlm = rnet[ro], rhm[ro], rlm[ro]
        out = {"lr": {}, "net": {}}
        for arm, scores, rscores, qs in (("hm", hm, rhm, qs_hm), ("lm", lm, rlm, qs_lm)):
            for q in np.unique(qs):
                am = scores >= q
                f2 = fam[am]
                _, fi = np.unique(f2, return_index=True)
                ind = np.zeros(len(f2)); ind[fi] = 1
                out["lr"][(arm, q)] = (ll[am], np.cumsum(ind))
                out["net"][(arm, q)] = (rnet, np.cumsum((rscores >= q).astype(float)))
        return out

    @staticmethod
    def n_at(curve, x, strict):
        """#entries with stat > x (strict) / >= x from a (desc-stat, cum) curve."""
        thr, cum = curve
        if len(cum) == 0:
            return np.zeros(np.shape(x), int)
        side = "left" if strict else "right"
        k = np.searchsorted(-thr, -np.asarray(x, float), side=side)
        return np.where(k > 0, cum[np.minimum(k - 1, len(cum) - 1)], 0).astype(int)


def best_far_vec(nlrh, nlrl, nnth, nntl, flt, lr_ok, nt_ok):
    """best_far + best_ul90 (per-arm 2*min, then n_channels trials) — verbatim
    cumulative_far_snapshot.py:259-268, vectorized."""
    A = 1.0 if X1 else 2.0                      # per-arm factor
    lr = np.where(lr_ok, A * np.minimum(nlrh, nlrl) / flt, np.inf)
    lru = np.where(lr_ok, A * np.minimum(ul90(nlrh, flt), ul90(nlrl, flt)), np.inf)
    nt = np.where(nt_ok, A * np.minimum(nnth, nntl) / flt, np.inf)
    ntu = np.where(nt_ok, A * np.minimum(ul90(nnth, flt), ul90(nntl, flt)), np.inf)
    if LR_ONLY:
        nt = np.full_like(nt, np.inf); ntu = np.full_like(ntu, np.inf)
        nt_ok = np.zeros_like(nt_ok)
    nch = lr_ok.astype(int) + nt_ok.astype(int)
    chf = np.ones_like(nch) if X1 else nch      # channel factor
    pick_nt = nt < lr
    far = np.where(pick_nt, nt, lr) * chf
    ul = np.where(pick_nt, ntu, lru) * chf
    far = np.where(nch == 0, np.nan, far); ul = np.where(nch == 0, np.nan, ul)
    return far, ul


def fit_fgmc(xs, ps, pn, widths, Ln_fixed=None):
    """Poisson-mixture MLE on binned best_far. Ln_fixed pins the noise count to
    its time-slide null expectation (primary); Ln_fixed=None refits both
    (sensitivity only)."""
    bi = np.clip(np.digitize(xs, FAR_EDGES) - 1, 0, len(ps) - 1)
    psv, pnv = ps[bi] / widths[bi], pn[bi] / widths[bi]
    if Ln_fixed is None:
        def nll(th):
            Ls, Ln = np.exp(th)
            return (Ls + Ln) - np.sum(np.log(Ls * psv + Ln * pnv))
        r = minimize(nll, np.log([max(0.5, 0.8 * len(xs)), max(0.5, 0.2 * len(xs))]),
                     method="Nelder-Mead")
        Ls, Ln = np.exp(r.x)
    else:
        Ln = float(Ln_fixed)
        def nll(th):
            Ls = np.exp(th[0])
            return (Ls + Ln) - np.sum(np.log(Ls * psv + Ln * pnv))
        r = minimize(nll, [np.log(max(0.5, len(xs)))], method="Nelder-Mead")
        Ls = float(np.exp(r.x[0]))
    return Ls, Ln, Ls * psv / (Ls * psv + Ln * pnv)


def main():
    frozen = list(csv.DictReader(open(CSVF)))
    cat_mtot = np.array([float(r["total_mass_source"]) for r in csv.DictReader(open(CATF))
                         if r["total_mass_source"].strip()])
    f_cat = np.histogram(cat_mtot, MASS_EDGES)[0] / len(cat_mtot)

    # pooled empirical detection (cnn_hm, cnn_lm) pairs: PAIR_RUNS only
    # (the accepted 47-pair as-run multiset; see RULING above)
    def dets_of(run):
        """Detection list for `run`. X1: the 48-row table of record (includes the four
        trials-promoted candidates, excludes the local-ASD veto rejects that detections.json
        still carries for O3b). Default: the run's detections.json, unchanged."""
        if not X1:
            return json.load(open(f"{RUNS[run]['out']}/detections.json"))
        return [dict(seg=r["seg"], gps=float(r["gps"]), net=float(r["net"]),
                     loglr=float(r["loglr"]), cnn_hm=float(r["cnn_hm"]), cnn_lm=float(r["cnn_lm"]),
                     best_far=(ADOPTED[(r["run"], r["name"])] if ADOPTED_AXIS
                               else float(r["far"])),
                     matches_known=r["name"])
                for r in frozen if r["run"] == run
                and (not ADOPTED_AXIS or (r["run"], r["name"]) in ADOPTED)]

    pairs = []
    for run in PAIR_RUNS:
        for d in dets_of(run):
            pairs.append((d["cnn_hm"], d["cnn_lm"]))
    pairs = np.array(pairs)
    qs_hm = np.unique(pairs[:, 0]); qs_lm = np.unique(pairs[:, 1])
    npair = len(pairs)
    print(f"[pastro] {npair} pooled det cnn pairs; hm range "
          f"[{pairs[:,0].min():.4f},{pairs[:,0].max():.4f}]")

    mids = (SNR_GRID[1:] + SNR_GRID[:-1]) / 2
    dr = np.diff(np.concatenate([[SNR_GRID[0]], mids, [SNR_GRID[-1]]]))
    w_snr_grid = SNR_GRID ** -4.0 * dr; w_snr_grid /= w_snr_grid.sum()
    w_snr_of = {float(s): w for s, w in zip(SNR_GRID, w_snr_grid)}

    pn = np.diff(FAR_EDGES) / (FAR_EDGES[-1] - FAR_EDGES[0])
    widths = np.diff(FAR_EDGES)
    results = []
    for run in RUNS:
        bg = RunBG(run)
        dets = dets_of(run)
        borrowed = False                 # every run now has its own campaign
        ev2seg = {d["matches_known"]: d["seg"]
                  for d in json.load(open(f"{RUNS['O3a']['out']}/detections.json"))
                  if d.get("matches_known")}
        files = []
        for d in RUNS[run]["inj"]:
            fl = sorted(glob.glob(f"{d}/*_inj.npz"))
            assert fl, f"{run}: no injection files in {d}"
            files += fl
        far_mat = []; det_mat = []       # (npair, ninj) blocks per event/fold
        inj_mtot = []; inj_snr = []; inj_w0 = []; inj_ev = []
        for f in files:
            ev = os.path.basename(f)[:-8]
            if ev in bg.seg_ix:
                seg = ev
            elif ev in ev2seg and ev2seg[ev] in bg.seg_ix:
                seg = ev2seg[ev]
            elif ev in EV_GPS:
                by_start = sorted(bg.seg_names, key=lambda n: int(n.rsplit("_", 1)[1]))
                starts = np.array([int(n.rsplit("_", 1)[1]) for n in by_start])
                seg = by_start[int(np.searchsorted(starts, EV_GPS[ev], side="right") - 1)]
            else:
                raise SystemExit(f"{run}: cannot resolve inj event {ev} to a segment")
            z = np.load(f)
            F = feats(z)
            net = z["net"].astype(float)
            mtot = z["mtot"].astype(float); snr = z["net_snr"].astype(float)
            w0 = np.array([w_snr_of[s] for s in snr])
            folds = (int(bg.seg_fold[bg.seg_ix[seg]]),)
            for g in folds:
                six = bg.seg_ix[seg]
                x = loglr_of(F, 1 - g)     # cross-fit: fold-g candidates scored by the other fold's model
                cur = bg.curves(six, g, qs_hm, qs_lm)
                flt = float(bg.far_live[g])
                lr_ok = x >= FLOOR; nt_ok = net >= NETSIG_FLOOR; trig = net > NET_CUT
            if INJCNN:                      # CNN glitch gate, per injection, same rule as candidates
                trig = trig & (np.maximum(z["cnn_hm"], z["cnn_lm"]) > GLITCH_THRESH)
            if NETMAX is not None:          # sigma_net upper veto, same rule as candidates
                trig = trig & (net < NETMAX)
                fblk = np.empty((npair, len(x))); dblk = np.empty((npair, len(x)), bool)
                for pi, (phm, plm) in enumerate(pairs):
                    nlrh = RunBG.n_at(cur["lr"][("hm", phm)], x, strict=True)
                    nlrl = RunBG.n_at(cur["lr"][("lm", plm)], x, strict=True)
                    nnth = RunBG.n_at(cur["net"][("hm", phm)], net, strict=False)
                    nntl = RunBG.n_at(cur["net"][("lm", plm)], net, strict=False)
                    far, ul = best_far_vec(nlrh, nlrl, nnth, nntl, flt, lr_ok, nt_ok)
                    _ke = KEMAP[run.lower()] if KEMAP else 1.0
                    det = trig & np.isfinite(far) & (far * _ke < DET_FAR)
                    if DET_RULE == "and":
                        det = det & (ul * _ke < DET_FAR)
                    fblk[pi] = far; dblk[pi] = det
                far_mat.append(fblk); det_mat.append(dblk)
                inj_mtot.append(mtot); inj_snr.append(snr); inj_ev.append(np.full(len(x), f"{ev}:f{g}"))
                inj_w0.append(w0 / len(folds))
        far_mat = np.concatenate(far_mat, axis=1); det_mat = np.concatenate(det_mat, axis=1)
        inj_mtot = np.concatenate(inj_mtot); inj_snr = np.concatenate(inj_snr)
        inj_w0 = np.concatenate(inj_w0); inj_ev = np.concatenate(inj_ev)
        mbins = np.clip(np.digitize(inj_mtot, MASS_EDGES) - 1, 0, len(MASS_EDGES) - 2)

        def ps_of(pair_ix, w_inj):
            """Signal density over FAR bins for a multiset of cnn pairs and
            per-injection weights (mass-reweighted to the catalog)."""
            F = far_mat[pair_ix]; D = det_mat[pair_ix]
            W = np.broadcast_to(w_inj, F.shape)[D]
            MTb = np.broadcast_to(mbins, F.shape)[D]
            f_inj = np.bincount(MTb, weights=W, minlength=len(MASS_EDGES) - 1)
            f_inj = f_inj / max(f_inj.sum(), 1e-300)
            wm = np.where(f_inj > 0, f_cat / np.maximum(f_inj, 1e-12), 0.0)
            ps, _ = np.histogram(F[D], FAR_EDGES, weights=W * wm[MTb])
            ps = ps / max(ps.sum(), 1e-300)
            ps = np.maximum(ps, 1e-6)
            return ps / ps.sum()

        # null noise count: analyzed zero-lag time of the segments actually scanned
        segj = {s[3]: s[2] for s in json.load(
            open(f"{SC}/{run.lower()}_full_coincident.json"))["segments"]}
        T_fg = sum(segj.get(n, 0.0) for n in bg.seg_names) * 0.5 / 3.1557e7   # ANALYZED_FRAC
        Ln_null = DET_FAR * T_fg
        xs = np.array([d["best_far"] for d in dets])
        all_ix = np.arange(npair)
        ps = ps_of(all_ix, inj_w0)
        Ls, Ln, pa = fit_fgmc(xs, ps, pn, widths, Ln_fixed=Ln_null)
        Ls_f, Ln_f, pa_f = fit_fgmc(xs, ps, pn, widths)      # sensitivity: free Ln
        eff = float((det_mat.mean(axis=0) * inj_w0).sum() / inj_w0.sum())
        # ---- bootstrap: resample cnn pairs + injection weights, refit ----
        B = 200
        rng = np.random.default_rng(20260811)
        pa_boot = np.empty((B, len(xs)))
        for b in range(B):
            pix = rng.integers(0, npair, npair)
            wb = inj_w0 * rng.dirichlet(np.ones(len(inj_w0))) * len(inj_w0)
            _, _, pab = fit_fgmc(xs, ps_of(pix, wb), pn, widths, Ln_fixed=Ln_null)
            pa_boot[b] = pab
        lo, hi = np.percentile(pa_boot, [5, 95], axis=0)
        print(f"[{run}] eff={eff:.3f} T_fg={T_fg:.4f}yr Ln={Ln:.3f} Ls={Ls:.2f} "
              f"(free fit: Ls={Ls_f:.2f} Ln={Ln_f:.2f}) "
              f"(n={len(xs)}{' BORROWED-INJ' if borrowed else ''}) "
              f"ps={np.array2string(ps, precision=3)}")
        for i, d in enumerate(dets):
            results.append(dict(run=run, seg=d["seg"], loglr=d["loglr"], net=d["net"],
                                matches_known=d.get("matches_known", ""),
                                best_far=float(xs[i]), p_astro=float(pa[i]),
                                p_astro_lo=float(lo[i]), p_astro_hi=float(hi[i]),
                                p_astro_freeLn=float(pa_f[i]),
                                borrowed_inj=borrowed, Ls=float(Ls), Ln=float(Ln),
                                Ls_free=float(Ls_f), Ln_free=float(Ln_f),
                                T_fg_yr=float(T_fg)))
        np.savez(f"{HERE}/inj_scored_{run.lower()}{SUF}.npz",
                 far_mean=np.nanmean(np.where(np.isfinite(far_mat), far_mat, np.nan), axis=0),
                 det_frac=det_mat.mean(axis=0), mtot=inj_mtot, net_snr=inj_snr,
                 w0=inj_w0, ev=inj_ev, npair=npair)

    json.dump(results, open(f"{HERE}/pastro_final{SUF}.json", "w"), indent=1)
    with open(f"{HERE}/pastro_final{SUF}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "name", "far", "p_astro", "p_astro_lo", "p_astro_hi"])
        for r in frozen:
            m = [x for x in results if x["run"] == r["run"]
                 and abs(x["loglr"] - float(r["loglr"])) < 1e-9
                 and abs(x["net"] - float(r["net"])) < 1e-9]
            if ADOPTED_AXIS and not m:
                continue                      # candidate not in the adopted set
            assert len(m) == 1, f"{r['run']} {r['name']}: {len(m)} matches"
            w.writerow([r["run"], r["name"], r["far"], f"{m[0]['p_astro']:.4f}",
                        f"{m[0]['p_astro_lo']:.4f}", f"{m[0]['p_astro_hi']:.4f}"])
            print(f"  {r['run']} {r['name']:24s} far={float(r['far']):.4g} "
                  f"p_astro={m[0]['p_astro']:.4f} [{m[0]['p_astro_lo']:.4f},{m[0]['p_astro_hi']:.4f}]")
    print(f"[pastro] -> {HERE}/pastro_final{SUF}.json / pastro_final{SUF}.csv")

if __name__ == "__main__":
    main()

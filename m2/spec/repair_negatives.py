"""A3 re-paired hard negatives + assembly of the FINAL negative set (PREREG_SPECIALISTS_20260923 A3). CPU.
Legs = distinct H1 / L1 legs of the S0-passing campaign pairs that reach the specialists (same admission as select_negatives.py),
outside the quarantine (quarantine.py + neg_sideband.csv legs). Re-pairs are formed within run and day-fold, admitted by the
production rule (sigma_net >= 3, louder-leg g2 >= -2.05, centroid2 < 220 Hz both legs), |gps_H - gps_L| >= 10 s, caps 40 per H1
leg / 6 per L1 leg, no duplicate of a campaign pair. Final set = 2,029 S0-pass campaign + re-pairs + S0-fail campaign (sub-sampled
from neg_select.npz per run in proportion) = 300k. Output neg_final.npz: neg_select fields + source (campaign_pass |
campaign_fail | repair); re-pairs carry cnn_hm = cnn_lm = loglr = NaN, s0_pass = -1 (unknown until rendered)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M = MADGRAV_SCRATCH + "/madgrav2"; OUT = MADGRAV_SCRATCH + "/madgrav2_spec"; SEED = 20260923; N_TOTAL = 300000
STREAMS = {"o3a": f"{M}/streams_o3a_full", "o3b": f"{M}/streams_o3b_full", "o4a": f"{M}/streams_o4a_full", "o4b": f"{M}/streams_o4b_full_cat2"}
CAP_H, CAP_L, MIN_LAG, TRIES = 40, 6, 10.0, 400
rng = np.random.default_rng(SEED); Q = QU.build(extra_csv=[f"{OUT}/neg_sideband.csv"])
mc = {}
def meta(run, seg, det):
    k = (run, seg, det)
    if k not in mc: z = np.load(f"{STREAMS[run]}/{seg}_{det}_meta.npz"); mc[k] = (np.asarray(z["g2"], float), np.asarray(z["centroid2"], float))
    return mc[k]
def cen_ok(a, b): return not ((np.isfinite(a) and a >= 220) or (np.isfinite(b) and b >= 220))
R = {k: [] for k in ("run", "H1_seg", "L1_seg", "iH", "iL", "gps_H", "gps_L", "sigma_H", "sigma_L", "g2_H", "g2_L", "c2_H", "c2_L", "fold")}
for run in STREAMS:
    d = json.load(open(f"{M}/search_out_{run}_m2_gx/survivors_bg.json"))["survivors"]; legs = {"H1": {}, "L1": {}}; existing = set()
    for r in d:
        existing.add((r["H1_seg"], r["iH"], r["L1_seg"], r["iL"]))
        gH, cH = meta(run, r["H1_seg"], "H1"); gL, cL = meta(run, r["L1_seg"], "L1"); sH, sL = r["sigma_H"], r["sigma_L"]
        lead = gH[r["iH"]] if sH >= sL else gL[r["iL"]]
        if not (lead >= -2.05 and cen_ok(cH[r["iH"]], cL[r["iL"]]) and (sH + sL) / np.sqrt(2) >= 3): continue
        if max(r["cnn_hm"], r["cnn_lm"]) <= 0.5: continue
        legs["H1"][(r["H1_seg"], r["iH"])] = (r["gps_H"], sH, gH[r["iH"]], cH[r["iH"]])
        legs["L1"][(r["L1_seg"], r["iL"])] = (r["gps_L"], sL, gL[r["iL"]], cL[r["iL"]])
    del d; mc.clear(); made = 0
    pools = {}
    for det in ("H1", "L1"):
        ks = list(legs[det]); g = np.array([legs[det][k][0] for k in ks]); q = QU.quarantined(Q, det, g)
        f = np.array([QU.fold_of(int(x // 86400)) for x in g])
        pools[det] = {fo: [ks[i] for i in np.where((~q) & (f == fo))[0]] for fo in (0, 1)}
    for fo in (0, 1):
        H = pools["H1"][fo]; L = pools["L1"][fo]
        if not H or not L: continue
        useL = {k: 0 for k in L}
        for h in rng.permutation(len(H)):
            hk = H[h]; gh, sh, g2h, ch = legs["H1"][hk]; n = 0
            for _ in range(TRIES):
                if n >= CAP_H: break
                lk = L[int(rng.integers(len(L)))]
                if useL[lk] >= CAP_L or (hk[0], hk[1], lk[0], lk[1]) in existing: continue
                gl, sl, g2l, cl = legs["L1"][lk]
                if abs(gh - gl) < MIN_LAG or (sh + sl) / np.sqrt(2) < 3 or (g2h if sh >= sl else g2l) < -2.05 or not cen_ok(ch, cl): continue
                existing.add((hk[0], hk[1], lk[0], lk[1])); useL[lk] += 1; n += 1; made += 1
                for k, v in (("run", run), ("H1_seg", hk[0]), ("L1_seg", lk[0]), ("iH", hk[1]), ("iL", lk[1]), ("gps_H", gh), ("gps_L", gl), ("sigma_H", sh),
                             ("sigma_L", sl), ("g2_H", g2h), ("g2_L", g2l), ("c2_H", ch), ("c2_L", cl), ("fold", fo)): R[k].append(v)
    print(f"[{run}] re-pairs {made} (H1 legs {len(pools['H1'][0])+len(pools['H1'][1])}, L1 legs {len(pools['L1'][0])+len(pools['L1'][1])})", flush=True)
# ---- assemble the final set ----
S = np.load(f"{OUT}/neg_select.npz", allow_pickle=True); n_rep = len(R["run"]); cp = S["s0_pass"] == 1
n_fail = N_TOTAL - int(cp.sum()) - n_rep; fail_idx = np.where(~cp)[0]; keep_fail = []
for run in STREAMS:
    fi = fail_idx[S["run"][fail_idx] == run]; k = int(round(n_fail * len(fi) / len(fail_idx)))
    keep_fail += list(rng.choice(fi, size=min(k, len(fi)), replace=False))
keep = np.concatenate([np.where(cp)[0], np.sort(np.array(keep_fail, int))])
F = {k: S[k][keep] for k in S.files if k != "seed"}; F["source"] = np.where(F["s0_pass"] == 1, "campaign_pass", "campaign_fail").astype("<U13")
nan = np.full(n_rep, np.nan)
for k in F:
    if k in R: F[k] = np.concatenate([F[k], np.array(R[k])])
    elif k in ("loglr", "cnn_hm", "cnn_lm"): F[k] = np.concatenate([F[k], nan])
    elif k == "s0_pass": F[k] = np.concatenate([F[k], np.full(n_rep, -1)])
    elif k == "source": F[k] = np.concatenate([F[k], np.full(n_rep, "repair", "<U13")])
F["iH"] = F["iH"].astype(int); F["iL"] = F["iL"].astype(int)
o = np.lexsort((F["source"] != "campaign_pass", F["run"]))       # per run: campaign_pass rows first (renderer tripwire sees them first)
F = {k: v[o] for k, v in F.items()}
np.savez(f"{OUT}/neg_final.npz", **F, seed=SEED)
src, cnt = np.unique(F["source"], return_counts=True)
print(f"[neg-final] {len(F['run'])} negatives: " + ", ".join(f"{s} {c}" for s, c in zip(src, cnt)) + f"; folds {np.bincount(F['fold'])}; per run "
      + ", ".join(f"{r} {int((F['run']==r).sum())}" for r in STREAMS), flush=True)

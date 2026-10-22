"""Ranker pilot, step 2b: FAR of every S0-gated zero-lag candidate under the campaign lnL and under the HGB ranker, counted the way
driver_blindscan_merge.py counts the loglr-channel FAR (mirrored, not imported):
  bank      = campaign background pairs that reach the specialists with a finite lnL (>= prune floor 3.0), per fold (H1 segment parity)
  louder    = bank pairs of the candidate's fold with statistic > candidate's statistic
  family    = (L1 segment, floor(gps_L / 4 s));  FAR = #families / far_live_yr[fold]
  per-arm   = 2 * min(#fam with cnn_hm >= cand hm, #fam with cnn_lm >= cand lm) over louder pairs not sharing the candidate's segment
  FAR_CNN   = #families among louder pairs passing the S0 gate (max(hm, lm) > 0.5)
Statistics: A = campaign lnL (stored). B = HGB RAW12 (cross-fit: fold-h rows scored by the model trained on fold 1-h noise + injections)
RESTRICTED to lnL >= 3.0 on both sides (the bank holds nothing below the prune floor, so a pure B FAR would be under-counted).
Candidates: S0-gated triggers with lnL >= 3.0 (same set for A and B). Sanity: A's per-arm FAR of the campaign detections vs the
stored far_lr_perarm. Output: rank/far_readout.txt, rank/far_candidates.csv"""
import os, json, csv, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec/rank"; SC = MADGRAV_SCRATCH + "/madgrav2"; RAW = ("sH", "sL", "coh", "gx", "cH2", "cL2", "gH2", "gL2", "cH1", "cL1", "gH1", "gL1"); FLOOR = 3.0
def raw(Z, p, idx=None): X = np.column_stack([Z[f"{p}_{k}"] for k in RAW]); return X if idx is None else X[idx]
def hgb(): return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, class_weight="balanced", random_state=0)
out = []; rows = []; TOT = {}
for run in ("o3a", "o3b", "o4a", "o4b"):
    if not os.path.exists(f"{D}/bgfeats_{run}.npz"): out.append(f"== {run}: no bank features yet"); continue
    B = np.load(f"{D}/bgfeats_{run}.npz", allow_pickle=True); Z = np.load(f"{D}/feats_{run}.npz", allow_pickle=True); live = {0: float(B["live0"]), 1: float(B["live1"])}
    ok = np.isfinite(B["bg_loglr"]) & (B["bg_loglr"] >= FLOOR) & np.isfinite(B["bg_coh"]) & np.isfinite(B["bg_gx"])
    bank = {k: B[f"bg_{k}"][ok] for k in ("fold", "loglr", "cnn_hm", "cnn_lm", "gps_L", "H1_seg", "L1_seg")}; Xb = raw({f"bg_{k}": B[f"bg_{k}"][ok] for k in RAW}, "bg")
    segs = np.unique(np.concatenate([bank["H1_seg"], bank["L1_seg"], Z["trig_seg"]])); sid = {s: i for i, s in enumerate(segs)}
    fam = np.array([sid[s] for s in bank["L1_seg"]], np.int64) * 10_000_000 + (bank["gps_L"] // 4.0).astype(np.int64); hseg = np.array([sid[s] for s in bank["H1_seg"]]); lseg = np.array([sid[s] for s in bank["L1_seg"]])
    # cross-fitted HGB scores for bank rows and triggers
    hb = np.full(len(fam), np.nan); ht = np.full(len(Z["trig_gps"]), np.nan)
    for g in (0, 1):
        h = 1 - g; tr = Z["inj_fold"] == g
        mdl = hgb().fit(np.vstack([raw(Z, f"noise{g}"), raw(Z, "inj", tr)]), np.concatenate([np.zeros(len(Z[f"noise{g}_sH"])), np.ones(int(tr.sum()))]))
        mb = bank["fold"] == h; hb[mb] = mdl.predict_proba(Xb[mb])[:, 1]; mt = Z["trig_fold"] == h; ht[mt] = mdl.predict_proba(raw(Z, "trig", mt))[:, 1]
    gate_t = np.nan_to_num(np.maximum(Z["trig_cnn_hm"], Z["trig_cnn_lm"]), nan=0) > 0.5; cand = np.where(gate_t & np.isfinite(Z["trig_loglr"]) & (Z["trig_loglr"] >= FLOOR))[0]
    gate_b = np.maximum(bank["cnn_hm"], bank["cnn_lm"]) > 0.5
    det = {d.get("matches_known") or f"{d['gps']:.0f}": d for d in json.load(open(f"{SC}/search_out_{run}_m2_gx/detections_lm.json"))}
    def far(stat_b, stat_c, f, cs, chm, clm):
        m = (bank["fold"] == f) & (stat_b > stat_c); ns = m & (hseg != cs) & (lseg != cs)
        n_cnn = len(np.unique(fam[m & gate_b])); n_hm = len(np.unique(fam[ns & (bank["cnn_hm"] >= chm)])); n_lm = len(np.unique(fam[ns & (bank["cnn_lm"] >= clm)]))
        return n_cnn / live[f], 2 * min(n_hm, n_lm) / live[f]
    res = []
    for i in cand:
        f = int(Z["trig_fold"][i]); cs = sid[Z["trig_seg"][i]]; chm, clm = float(Z["trig_cnn_hm"][i]), float(Z["trig_cnn_lm"][i]); name = Z["trig_known"][i]
        fa_cnn, fa_pa = far(bank["loglr"], float(Z["trig_loglr"][i]), f, cs, chm, clm); fb_cnn, fb_pa = far(hb, float(ht[i]), f, cs, chm, clm)
        stored = det[name]["far_lr_perarm"] if name in det else np.nan
        res.append(dict(run=run, name=name or f"{run}_{Z['trig_gps'][i]:.0f}", known=bool(name), gps=float(Z["trig_gps"][i]), loglr=float(Z["trig_loglr"][i]), hgb=float(ht[i]), farA_cnn=fa_cnn, farA_perarm=fa_pa, farB_cnn=fb_cnn, farB_perarm=fb_pa, stored_perarm=stored))
    rows += res
    kn = [r for r in res if r["known"]]; nk = [r for r in res if not r["known"]]
    chk = [(r["farA_perarm"], r["stored_perarm"]) for r in kn if np.isfinite(r["stored_perarm"]) and r["stored_perarm"] is not None]
    out.append(f"== {run}: bank {len(fam)} pairs (fold0 {int((bank['fold']==0).sum())}, fold1 {int((bank['fold']==1).sum())}), livetime {live[0]:.0f}/{live[1]:.0f} yr; candidates (S0-gated, lnL>=3) {len(cand)}: catalogue {len(kn)}, other {len(nk)}")
    if chk: out.append(f"  sanity: recomputed A per-arm FAR vs stored (campaign detections, n={len(chk)}): median |log10 ratio| {np.median([abs(np.log10((a+1e-4)/(b+1e-4))) for a, b in chk]):.2f}, max {max(abs(np.log10((a+1e-4)/(b+1e-4))) for a, b in chk):.2f}")
    out.append(f"  {'':<22} {'FAR<1/yr':>9} {'<0.1/yr':>8} {'<0.01/yr':>9} | non-catalogue: {'<1/yr':>6} {'<0.1/yr':>8}")
    for lab, key in (("A lnL per-arm", "farA_perarm"), ("B HGB per-arm", "farB_perarm"), ("A lnL FAR_CNN", "farA_cnn"), ("B HGB FAR_CNN", "farB_cnn")):
        c = [sum(r[key] < t for r in kn) for t in (1, 0.1, 0.01)]; n = [sum(r[key] < t for r in nk) for t in (1, 0.1)]
        out.append(f"  {lab:<22} {c[0]:9d} {c[1]:8d} {c[2]:9d} | {n[0]:22d} {n[1]:8d}"); TOT.setdefault(lab, np.zeros(5)); TOT[lab] += np.array(c + n)
    gains = [r for r in kn if r["farB_perarm"] < 1 <= r["farA_perarm"]]; losses = [r for r in kn if r["farA_perarm"] < 1 <= r["farB_perarm"]]
    if gains: out.append("  catalogue GAINED at FAR<1/yr under B: " + "; ".join(f"{r['name']} (A {r['farA_perarm']:.3g} -> B {r['farB_perarm']:.3g})" for r in gains))
    if losses: out.append("  catalogue LOST at FAR<1/yr under B: " + "; ".join(f"{r['name']} (A {r['farA_perarm']:.3g} -> B {r['farB_perarm']:.3g})" for r in losses))
    imp = sorted([r for r in kn if r["farA_perarm"] < 1], key=lambda r: r["farA_perarm"] / (r["farB_perarm"] + 1e-6))
    if imp: out.append(f"  catalogue detections: median FAR ratio A/B {np.median([r['farA_perarm'] / (r['farB_perarm'] + 1e-6) for r in imp if r['farA_perarm'] > 0]):.2f} over those with A>0; zero-FAR under A {sum(r['farA_perarm']==0 for r in imp)}, under B {sum(r['farB_perarm']==0 for r in imp)}")
    newc = sorted([r for r in nk if r["farB_perarm"] < 1], key=lambda r: r["farB_perarm"])[:5]
    if newc: out.append("  non-catalogue with B per-arm FAR<1/yr: " + "; ".join(f"{r['name']} (A {r['farA_perarm']:.3g} -> B {r['farB_perarm']:.3g}, lnL {r['loglr']:.1f})" for r in newc))
out.append("== TOTAL over runs (catalogue <1 / <0.1 / <0.01 | non-cat <1 / <0.1)"); [out.append(f"  {k:<22} " + " ".join(f"{int(x):5d}" for x in v)) for k, v in TOT.items()]
txt = "\n".join(out); print(txt); open(f"{D}/far_readout.txt", "w").write(txt + "\n")
with open(f"{D}/far_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

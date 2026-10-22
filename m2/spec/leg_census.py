"""Distinct-leg census of the hard negative class (gate-passing pairs that reach the specialists), per run / detector / fold,
before and after the quarantine (incl. sideband legs), and the within-fold re-pairing capacity. CPU only; reuses the
select_negatives.py admission and sideband definitions verbatim (import-free copy of the loop)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import quarantine as QU
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M = MADGRAV_SCRATCH + "/madgrav2"; OUT = MADGRAV_SCRATCH + "/madgrav2_spec"
STREAMS = {"o3a": f"{M}/streams_o3a_full", "o3b": f"{M}/streams_o3b_full", "o4a": f"{M}/streams_o4a_full", "o4b": f"{M}/streams_o4b_full_cat2"}
Q = QU.build(extra_csv=[f"{OUT}/neg_sideband.csv"])
mc = {}
def meta(run, seg, det):
    k = (run, seg, det)
    if k not in mc: z = np.load(f"{STREAMS[run]}/{seg}_{det}_meta.npz"); mc[k] = (np.asarray(z["g2"], float), np.asarray(z["centroid2"], float), np.asarray(z["sigma"], float))
    return mc[k]
tot = {}
for run in STREAMS:
    d = json.load(open(f"{M}/search_out_{run}_m2_gx/survivors_bg.json"))["survivors"]; legs = {"H1": {}, "L1": {}}
    for r in d:
        gH, cH, _ = meta(run, r["H1_seg"], "H1"); gL, cL, _ = meta(run, r["L1_seg"], "L1")
        sH, sL = r["sigma_H"], r["sigma_L"]; net = (sH + sL) / np.sqrt(2); lead = gH[r["iH"]] if sH >= sL else gL[r["iL"]]
        c2h, c2l = cH[r["iH"]], cL[r["iL"]]; cen_ok = not ((np.isfinite(c2h) and c2h >= 220) or (np.isfinite(c2l) and c2l >= 220))
        if not (lead >= -2.05 and cen_ok and net >= 3): continue
        if max(r["cnn_hm"], r["cnn_lm"]) <= 0.5: continue
        for det, seg, i, g, s, gg, cc in (("H1", r["H1_seg"], r["iH"], r["gps_H"], sH, gH[r["iH"]], c2h), ("L1", r["L1_seg"], r["iL"], r["gps_L"], sL, gL[r["iL"]], c2l)):
            k = (seg, i); e = legs[det].setdefault(k, [g, s, gg, cc, 0]); e[4] += 1
    del d; mc.clear(); line = [f"[{run}]"]
    for det in ("H1", "L1"):
        L = legs[det]; g = np.array([v[0] for v in L.values()]); n = np.array([v[4] for v in L.values()])
        q = QU.quarantined(Q, det, g); f = np.array([QU.fold_of(int(x // 86400)) for x in g])
        tot[(run, det)] = {fo: int(((~q) & (f == fo)).sum()) for fo in (0, 1)}
        line.append(f"{det}: {len(L)} distinct legs in gate-passing pairs (pairs/leg median {np.median(n):.0f}, max {n.max()}), "
                    f"outside quarantine {int((~q).sum())} (f0 {tot[(run,det)][0]}, f1 {tot[(run,det)][1]})")
    rep = sum(tot[(run, 'H1')][fo] * tot[(run, 'L1')][fo] for fo in (0, 1))
    print(" | ".join(line) + f" | within-fold re-pairing capacity {rep:,}", flush=True)

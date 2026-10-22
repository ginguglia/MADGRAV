"""Score every centroid definition. Convention: signal LOW, glitch HIGH; a tile 'passes' if centroid < cut.
Operating point: cut = p99 of the injection population (1% single-detector injection loss, the A5 discipline),
set separately per detector. Reported per definition:
  AUC(pos vs glitch), AUC(pos vs murky), AUC(pos vs top50)          (higher = better)
  detections passing at the cut (must be 96/96 legs -> 47/47 events for a viable gate)
  removal at the cut: glitch tiles, murky tiles (per run), top-50 families (family removed if EITHER leg fails,
  as in the frozen morphology gate), per glitch class
  injection retention per log-Mtot bin (8 bins 15-800)
Also: frozen reference = old_proxy at the same 1% loss, and old_proxy at 190.79 Hz.
"""
import numpy as np, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centroid_defs import DEFS
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

O = MADGRAV_SCRATCH + "/madgrav2/centroid"
SM = "_smoke" if "--smoke" in sys.argv else ""
L = lambda p: np.load(f"{O}/feat_{p}{SM}.npz", allow_pickle=True)
pos, det, gl, top, mk = L("pos"), L("det47"), L("glitch"), L("top50"), L("murky")

def auc(a, b):
    """P(score_glitch > score_signal): a = signal, b = glitch (nan -> treated as +inf for glitch, -inf for signal? no: drop)."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return np.nan
    allv = np.concatenate([a, b]); r = allv.argsort().argsort().astype(float) + 1
    ra = r[:len(a)].sum()
    return 1.0 - (ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))

MBINS = np.array([15, 24, 38, 60, 95, 150, 240, 380, 800], float)
lines = []; P = lambda s="": lines.append(s)
P("# MADGRAV2 item 1 — centroid pilot on cached tiles"); P()
P(f"pos {len(pos['old_proxy'])} injection tiles (mtot {pos['mtot'].min():.0f}-{pos['mtot'].max():.0f}, sigma "
  f"{pos['sigma'].min():.1f}-{pos['sigma'].max():.1f}); det47 {len(det['old_proxy'])} legs; glitch {len(gl['old_proxy'])}; "
  f"top50 {len(top['old_proxy'])} legs; murky {len(mk['old_proxy'])} tiles"); P()
hdr = "| def | AUC pos/glitch | AUC pos/murky | AUC pos/top50 | cut H1/L1 [Hz] | det legs pass | events kept | glitch removed | murky removed | top50 fam removed |"
P(hdr); P("|" + "---|" * 10)
summary = {}
for k in DEFS:
    a = {}; a["auc_gl"] = auc(pos[k], gl[k]); a["auc_mk"] = auc(pos[k], mk[k]); a["auc_top"] = auc(pos[k], top[k])
    cut = {}
    for d in ("H1", "L1"):
        v = pos[k][pos["det"] == d]; v = v[np.isfinite(v)]
        if len(v) < 50: v = pos[k][np.isfinite(pos[k])]          # fall back to pooled cut (smoke mode / sparse det)
        cut[d] = float(np.percentile(v, 99.0))
    def passes(z):
        c = np.array([cut[d] for d in z["det"]]); v = z[k]
        return np.where(np.isfinite(v), v < c, True)   # undefined centroid (no excess) passes: nothing to veto on
    dp = passes(det); ev = det["event"]; ev_ok = {e: True for e in ev}
    for e, ok in zip(ev, dp): ev_ok[e] &= bool(ok)
    a["det_legs"] = int(dp.sum()); a["det_events"] = sum(ev_ok.values()); a["n_events"] = len(ev_ok)
    a["gl_removed"] = 1 - passes(gl).mean(); a["mk_removed"] = 1 - passes(mk).mean()
    tp = passes(top); fam_ok = {}
    for f, ok in zip(top["fam"], tp): fam_ok[f] = fam_ok.get(f, True) & bool(ok)
    a["top_removed"] = 1 - np.mean(list(fam_ok.values())); a["cut"] = cut
    a["mk_by_run"] = {r: float(1 - passes({"det": mk["det"][mk["run"] == r], k: mk[k][mk["run"] == r]}).mean()) for r in np.unique(mk["run"])}
    a["gl_by_class"] = {}
    for c in np.unique(gl["label"]):
        m = gl["label"] == c
        a["gl_by_class"][str(c)] = (int(m.sum()), float(1 - passes({"det": gl["det"][m], k: gl[k][m]}).mean()))
    pp = passes(pos); ret = []
    for lo, hi in zip(MBINS[:-1], MBINS[1:]):
        m = (pos["mtot"] >= lo) & (pos["mtot"] < hi); ret.append(float(pp[m].mean()) if m.any() else np.nan)
    a["ret_mbin"] = ret
    # watchlist operating point: cut = max detection-leg centroid per detector (+1 Hz) -> all 48 events kept by construction
    wc = {}
    for d in ("H1", "L1"):
        v = det[k][det["det"] == d]; v = v[np.isfinite(v)]; wc[d] = float(v.max()) + 1.0 if len(v) else np.inf
    def passes_w(z):
        c = np.array([wc[d] for d in z["det"]]); v = z[k]; return np.where(np.isfinite(v), v < c, True)
    tpw = passes_w(top); famw = {}
    for f, ok in zip(top["fam"], tpw): famw[f] = famw.get(f, True) & bool(ok)
    a["w_cut"] = wc; a["w_inj_loss"] = float(1 - passes_w(pos).mean()); a["w_gl_removed"] = float(1 - passes_w(gl).mean())
    a["w_mk_removed"] = float(1 - passes_w(mk).mean()); a["w_top_removed"] = float(1 - np.mean(list(famw.values())))
    summary[k] = a
    P(f"| {k} | {a['auc_gl']:.3f} | {a['auc_mk']:.3f} | {a['auc_top']:.3f} | {cut['H1']:.0f}/{cut['L1']:.0f} | {a['det_legs']}/{len(dp)} | "
      f"{a['det_events']}/{a['n_events']} | {100*a['gl_removed']:.1f}% | {100*a['mk_removed']:.1f}% | {100*a['top_removed']:.0f}% |")
P()
P("## Watchlist operating point: loosest cut keeping every detection leg (per detector)"); P()
P("| def | cut H1/L1 [Hz] | injection loss | glitch removed | murky removed | top50 fam removed |"); P("|---|---|---|---|---|---|")
for k in DEFS:
    a = summary[k]
    P(f"| {k} | {a['w_cut']['H1']:.0f}/{a['w_cut']['L1']:.0f} | {100*a['w_inj_loss']:.1f}% | {100*a['w_gl_removed']:.1f}% | {100*a['w_mk_removed']:.1f}% | {100*a['w_top_removed']:.0f}% |")
P()
# frozen reference: old_proxy at 190.79 Hz both legs
k = "old_proxy"; c = 190.79
def passes_fixed(z): v = z[k]; return np.where(np.isfinite(v), v < c, True)
dp = passes_fixed(det); ev_ok = {}
for e, ok in zip(det["event"], dp): ev_ok[e] = ev_ok.get(e, True) & bool(ok)
tp = passes_fixed(top); fam_ok = {}
for f, ok in zip(top["fam"], tp): fam_ok[f] = fam_ok.get(f, True) & bool(ok)
P(f"Frozen reference, CAVEAT: tile proxy is offset ~+10 Hz from the strain-PSD centroid, so this fixed-cut line is indicative only (old_proxy < 190.79 Hz on the tile): injections pass {100*passes_fixed(pos).mean():.1f}%, det legs {dp.sum()}/{len(dp)}, "
  f"events {sum(ev_ok.values())}/{len(ev_ok)}, glitch removed {100*(1-passes_fixed(gl).mean()):.1f}%, murky removed {100*(1-passes_fixed(mk).mean()):.1f}%, "
  f"top50 families removed {100*(1-np.mean(list(fam_ok.values()))):.0f}%"); P()
P("## Best definitions: top-50 family removal at the 1% injection cut, among those keeping all events; then by murky AUC"); P()
keep_all = [k for k in summary if summary[k]["det_events"] == summary[k]["n_events"]]
best = sorted(keep_all or summary, key=lambda k: (-summary[k]["top_removed"], -np.nan_to_num(summary[k]["auc_mk"])))[:4]
for k in best:
    a = summary[k]; P(f"### {k}"); P()
    P("murky removed per run at the 1% cut: " + ", ".join(f"{r} {100*v:.1f}%" for r, v in a["mk_by_run"].items()))
    P("injection retention per Mtot bin [" + ", ".join(f"{int(lo)}-{int(hi)}" for lo, hi in zip(MBINS[:-1], MBINS[1:])) + "]: "
      + ", ".join("nan" if not np.isfinite(r) else f"{100*r:.1f}%" for r in a["ret_mbin"]))
    P("glitch classes removed (n, %): " + ", ".join(f"{c} ({n}, {100*v:.0f}%)" for c, (n, v) in sorted(a["gl_by_class"].items(), key=lambda x: -x[1][0])[:9]))
    P()
P("## Detection legs failing the 1% cut (top-3 by top-50 removal, any def)"); P()
for k in sorted(summary, key=lambda k: -summary[k]["top_removed"])[:3]:
    c = np.array([summary[k]["cut"][d] for d in det["det"]]); bad = [(n, v, cc) for n, v, cc in zip(det["name"], det[k], c) if np.isfinite(v) and v >= cc]
    P(f"- {k} (cut {summary[k]['cut']['H1']:.0f}/{summary[k]['cut']['L1']:.0f} Hz): {len(bad)} legs fail: " + ", ".join(f"{n} {v:.0f}" for n, v, cc in bad[:12]) + (" ..." if len(bad) > 12 else ""))
P()
# spread of detections vs cut, to see the margin
P("## Detection centroid quantiles (best def)"); v = det[k]; v = v[np.isfinite(v)]
P(f"min {v.min():.0f}, p50 {np.median(v):.0f}, p90 {np.percentile(v,90):.0f}, max {v.max():.0f} Hz; cuts {summary[k]['cut']}")
open(f"{O}/pilot_summary{SM}.md", "w").write("\n".join(lines))
json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) for kk, vv in a.items()} for k, a in summary.items()},
          open(f"{O}/pilot_summary{SM}.json", "w"), indent=1, default=float)
print("\n".join(lines))

# ======================= extension: lower-sigma positives (g8 shakedown) as cut population + full catalog ===================
lines2 = []; Q = lambda s="": lines2.append(s)
try: pg = np.load(f"{O}/feat_posg8.npz", allow_pickle=True)
except FileNotFoundError: pg = None
try: cat = np.load(f"{O}/feat_cat{SM}.npz", allow_pickle=True)
except FileNotFoundError:
    try: cat = np.load(f"{O}/feat_cat.npz", allow_pickle=True)
    except FileNotFoundError: cat = None
if pg is not None:
    Q("## Cut set on the lower-sigma positives (g8 shakedown: sigma %.1f-%.1f, median %.1f, mtot %.0f-%.0f, n=%d)" %
      (pg["sigma"].min(), pg["sigma"].max(), np.median(pg["sigma"]), pg["mtot"].min(), pg["mtot"].max(), len(pg["sigma"]))); Q()
    Q("| def | cut H1/L1 (p99 g8) | det legs pass | events kept | glitch removed | murky removed | top50 fam removed | AUC g8/murky |"); Q("|---|---|---|---|---|---|---|---|")
    for k in DEFS:
        cut = {}
        for d in ("H1", "L1"):
            v = pg[k][pg["det"] == d]; v = v[np.isfinite(v)]; cut[d] = float(np.percentile(v, 99.0))
        def passes(z):
            c = np.array([cut[d] for d in z["det"]]); v = z[k]; return np.where(np.isfinite(v), v < c, True)
        dp = passes(det); ev_ok = {}
        for e, ok in zip(det["event"], dp): ev_ok[e] = ev_ok.get(e, True) & bool(ok)
        tp = passes(top); fam_ok = {}
        for f, ok in zip(top["fam"], tp): fam_ok[f] = fam_ok.get(f, True) & bool(ok)
        Q(f"| {k} | {cut['H1']:.0f}/{cut['L1']:.0f} | {dp.sum()}/{len(dp)} | {sum(ev_ok.values())}/{len(ev_ok)} | {100*(1-passes(gl).mean()):.1f}% | "
          f"{100*(1-passes(mk).mean()):.1f}% | {100*(1-np.mean(list(fam_ok.values()))):.0f}% | {auc(pg[k], mk[k]):.3f} |")
    Q()
if cat is not None:
    ad = cat["adopted"].astype(bool); Q(f"## Full catalog footprint: {len(set(cat['name']))} events / {len(cat['name'])} legs rendered "
      f"({len(set(cat['name'][ad]))} adopted, {len(set(cat['name'][~ad]))} missed)"); Q()
    Q("Per definition: max leg over adopted events (= watchlist cut), and how many missed events have BOTH legs below it "
      "(would pass a morphology gate at that cut) vs at least one leg above."); Q()
    Q("| def | adopted max H1/L1 | adopted p90 | missed events both legs below | missed with a leg above | missed legs above [Hz, worst 3] |"); Q("|---|---|---|---|---|---|")
    names = cat["name"]
    for k in DEFS:
        v = cat[k]; wc = {}
        for d in ("H1", "L1"):
            a = v[ad & (cat["det"] == d)]; a = a[np.isfinite(a)]; wc[d] = float(a.max()) if len(a) else np.nan
        c = np.array([wc[d] for d in cat["det"]]); below = np.where(np.isfinite(v), v <= c, True)
        ok = {}
        for n, a_, b in zip(names, ad, below):
            if not a_: ok[n] = ok.get(n, True) & bool(b)
        nb = sum(ok.values()); na = len(ok) - nb
        worst = sorted([(float(v[i]), names[i], cat["det"][i]) for i in range(len(v)) if not ad[i] and np.isfinite(v[i]) and v[i] > c[i]], reverse=True)[:3]
        pa = v[ad]; pa = pa[np.isfinite(pa)]
        Q(f"| {k} | {wc['H1']:.0f}/{wc['L1']:.0f} | {np.percentile(pa, 90) if len(pa) else np.nan:.0f} | {nb} | {na} | " + ", ".join(f"{n} {d} {x:.0f}" for x, n, d in worst) + " |")
    Q()
    # where-to-cut view for the leading cluster definition: legs sorted by centroid with mtot, snr, adopted flag
    for k in ("clu98", "clu99"):
        if k not in cat.files: continue
        Q(f"### {k}: all catalog legs above 150 Hz (name det mtot snr_det adopted centroid)"); Q()
        idx = [i for i in np.argsort(-cat[k]) if np.isfinite(cat[k][i]) and cat[k][i] > 150]
        for i in idx[:40]:
            Q(f"- {names[i]} {cat['det'][i]} Mtot {float(cat['mtot'][i]):.0f} snr {float(cat['snr_det'][i]):.1f} {'ADOPTED' if ad[i] else 'missed:'+str(cat['miss_cause'][i])} -> {float(cat[k][i]):.0f} Hz")
        Q()
        # by mass bin: median/p90 centroid of adopted vs missed legs
        Q(f"### {k}: centroid by Mtot bin (median / p90 of legs) adopted | missed"); Q()
        for lo, hi in zip(MBINS[:-1], MBINS[1:]):
            m = (cat["mtot"] >= lo) & (cat["mtot"] < hi)
            a = cat[k][m & ad]; a = a[np.isfinite(a)]; b = cat[k][m & ~ad]; b = b[np.isfinite(b)]
            fa = f"{np.median(a):.0f}/{np.percentile(a,90):.0f} (n={len(a)})" if len(a) else "-"
            fb = f"{np.median(b):.0f}/{np.percentile(b,90):.0f} (n={len(b)})" if len(b) else "-"
            Q(f"- Mtot {int(lo)}-{int(hi)}: adopted {fa} | missed {fb}")
        Q()
with open(f"{O}/pilot_summary{SM}.md", "a") as fh: fh.write("\n" + "\n".join(lines2))
print("\n".join(lines2))

"""vt96p: paired per-injection comparison of the frozen 96 recipe at the search's own placement (vt96p, merger uniform in a 1 s tile, two
windows) against the same field at the 0.25 s three-window placement (vt96). Same seed 20261004 -> identical sources, SNRs, data; only the
scored windows differ. Per run / source-frame mass bin / SNR: detected by both, grid-only (placement loss), prod-only; losses and gains by
cause from the per-injection columns (trigger = net floor 2.75 on the production grid first). Also the net-SNR shift grid -> prod.
Inputs: vt96/inj_scored_<run>_m96.npz, vt96/inj_scored_<run>_m96p.npz, vt96/relabel_inj_<run>_m96p.npz. Output: COMPARE_M96P_vs_M96_<date>.md."""
import os, datetime, numpy as np
from collections import Counter
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

V = MADGRAV_SCRATCH + "/madgrav2/vt96"
OUT = os.environ.get("VT96P_COMPARE", f"{MADGRAV_ROOT}/m2/vt96p/COMPARE_M96P_vs_M96_{datetime.date.today():%Y%m%d}.md")
EDGES = np.array([10, 20, 40, 60, 80, 100, 130, 160, 200, 260, 330, 400], float); LAB = [f"{int(a)}-{int(b)}" for a, b in zip(EDGES[:-1], EDGES[1:])]
FLOOR = 2.75
L = [f"# Paired comparison: frozen 96 recipe at production placement (vt96p) vs grid placement (vt96) ({datetime.date.today()}, seed 20261004, identical injections)", ""]
def cause_loss(g, p, i):   # grid detected, prod not: why (prod-side columns)
    if not p["trig"][i]: return f"no trigger at production placement (net < {FLOOR} on the 1 s grid)"
    if not p["det_merge"][i]:
        if (g["lr_open"][i] and not p["lr_open"][i]) or (g["net_open"][i] and not p["net_open"][i]): return "gate at the production window (CNN/S3 or LM rule)"
        return "FAR tables / ranking (same gates open, lower statistic)"
    if not p["veto_keep"][i]: return "local-ASD net clause" if p["veto_reason"][i] == "net<floor" else "local-ASD feature clause (O3)"
    if not p["guard_keep"][i]: return "net-sigma guard"
    return "other"
def cause_gain(g, p, i):   # prod detected, grid not
    if not g["trig"][i]: return "trigger only at production placement"
    if p["cap_adm"][i] and not p["det_guard"][i]: return "cap bypass"
    if (p["lr_open"][i] and not g["lr_open"][i]) or (p["net_open"][i] and not g["net_open"][i]): return "gate opens at the production window"
    return "FAR tables / ranking (same gates open)"
tot = dict(n=0, both=0, onlyg=0, onlyp=0, dg=0, dp=0)
for run in ("o3a", "o3b", "o4a", "o4b"):
    g = np.load(f"{V}/inj_scored_{run}_m96.npz", allow_pickle=True); p = np.load(f"{V}/inj_scored_{run}_m96p.npz", allow_pickle=True)
    assert len(g["net"]) == len(p["net"]) and np.array_equal(g["ev"], p["ev"]), "pairing: different files/order"
    assert np.allclose(g["mtot"], p["mtot"]) and np.allclose(g["net_snr"], p["net_snr"]), "pairing: injections differ"
    rl = np.load(f"{V}/relabel_inj_{run}_m96p.npz"); kept = rl["kept"]; msrc = rl["msrc"]
    dg = g["detected"].astype(bool); dp = p["detected"].astype(bool); both = dg & dp; og = dg & ~dp; op = dp & ~dg
    tot["n"] += len(dg); tot["both"] += int(both.sum()); tot["onlyg"] += int(og.sum()); tot["onlyp"] += int(op.sum()); tot["dg"] += int(dg.sum()); tot["dp"] += int(dp.sum())
    L.append(f"## {run}: {len(dg)} paired injections (relabel-kept {int(kept.sum())}); detected grid {int(dg.sum())}, prod {int(dp.sum())}; both {int(both.sum())}, grid-only {int(og.sum())}, prod-only {int(op.sum())} (net {int(dp.sum())-int(dg.sum()):+d}, {100*(int(dp.sum())-int(dg.sum()))/max(1,int(dg.sum())):+.1f} %)"); L.append("")
    dn = p["net"] - g["net"]; tg = g["trig"].astype(bool); tp = p["trig"].astype(bool)
    L.append(f"net SNR shift prod - grid: median {np.median(dn):+.3f}, q10 {np.quantile(dn, .1):+.3f}, q90 {np.quantile(dn, .9):+.3f}; triggers (net >= {FLOOR}) grid {int(tg.sum())} -> prod {int(tp.sum())} "
             f"({int((tg & ~tp).sum())} lost, {int((~tg & tp).sum())} new); grid triggers with prod net < {FLOOR}: {100*np.mean(~tp[tg]):.1f} %"); L.append("")
    L.append("| source-frame Mtot bin | n (kept) | det grid | det prod | both | grid only | prod only | det_frac grid | det_frac prod | prod/grid |"); L.append("|---|---|---|---|---|---|---|---|---|---|")
    bi = np.digitize(msrc, EDGES) - 1
    for j, lb in enumerate(LAB):
        m = kept & (bi == j)
        if m.sum() == 0: continue
        L.append(f"| {lb} | {int(m.sum())} | {int(dg[m].sum())} | {int(dp[m].sum())} | {int(both[m].sum())} | {int(og[m].sum())} | {int(op[m].sum())} | {dg[m].mean():.3f} | {dp[m].mean():.3f} | {(dp[m].mean()/dg[m].mean() if dg[m].mean()>0 else float('nan')):.3f} |")
    L.append(""); L.append("| SNR | n | det grid | det prod | grid only | prod only | prod/grid |"); L.append("|---|---|---|---|---|---|---|")
    for s in np.unique(g["net_snr"]):
        m = g["net_snr"] == s; L.append(f"| {s:g} | {int(m.sum())} | {int(dg[m].sum())} | {int(dp[m].sum())} | {int(og[m].sum())} | {int(op[m].sum())} | {(dp[m].sum()/dg[m].sum() if dg[m].sum()>0 else float('nan')):.3f} |")
    L.append("")
    cl = Counter(cause_loss(g, p, i) for i in np.flatnonzero(og)); cg = Counter(cause_gain(g, p, i) for i in np.flatnonzero(op))
    L.append(f"grid-only losses by cause ({int(og.sum())}): " + "; ".join(f"{k} {v}" for k, v in cl.most_common()))
    L.append(f"prod-only gains by cause ({int(op.sum())}): " + "; ".join(f"{k} {v}" for k, v in cg.most_common()))
    L.append("")
L.append(f"## All runs: {tot['n']} paired injections; detected grid {tot['dg']}, prod {tot['dp']} (ratio {tot['dp']/max(1,tot['dg']):.3f}); both {tot['both']}, grid-only {tot['onlyg']}, prod-only {tot['onlyp']} (net {tot['onlyp']-tot['onlyg']:+d}).")
L.append(""); L.append("Reading: 'grid only' = recovered only when the injection is scored on the 0.25 s three-window grid (the vt96/vt22 convention), i.e. the optimism of those "
         "fields relative to the frozen 1 s search; 'prod only' = recovered only at the production placement (noise, or a neighbouring tile that happens to score higher). "
         "vt96p is the field consistent with the search; quote its VT.")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"[compare] -> {OUT}")

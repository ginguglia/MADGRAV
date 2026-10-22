"""vt96: paired per-injection comparison 2.2 vs 96 recipe (same seed 20261004 -> same injections, row by row). Per run and per source-frame
mass bin: detected by both / 2.2 only / 96 only; the 96-only gains split by cause (floor 2.75 trigger, cap bypass, gate at the coincident crop,
FAR tables/ranking) and the 2.2-only losses by cause (gate at the coincident crop, FAR tables/ranking, local-ASD net clause, local-ASD feature
clause (O3), net-sigma guard). Inputs: vt22/inj_scored_<run>_m22.npz, vt96/inj_scored_<run>_m96.npz, vt96/relabel_inj_<run>_m96.npz (msrc, kept).
Output: COMPARE_M96_vs_M22_<date>.md (+ stdout)."""
import os, datetime, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

V22 = MADGRAV_SCRATCH + "/madgrav2/vt22"; V96 = MADGRAV_SCRATCH + "/madgrav2/vt96"
OUT = os.environ.get("VT96_COMPARE", f"{MADGRAV_ROOT}/m2/vt96/COMPARE_M96_vs_M22_{datetime.date.today():%Y%m%d}.md")
EDGES = np.array([10, 20, 40, 60, 80, 100, 130, 160, 200, 260, 330, 400], float); LAB = [f"{int(a)}-{int(b)}" for a, b in zip(EDGES[:-1], EDGES[1:])]
L = [f"# Paired comparison of the 96-recipe and 2.2 injection fields ({datetime.date.today()}, seed 20261004, identical injections)", ""]
G = {"O3a": {}, "O3b": {}, "O4a": {}, "O4b": {}}
def causes_loss(a, b, i):   # 2.2 detected, 96 not: why (96-side columns)
    if not b["trig"][i]: return "trigger lost (should not happen)"
    if not b["det_merge"][i]:
        if (a["lr_open"][i] and not b["lr_open"][i]) or (a["net_open"][i] and not b["net_open"][i]): return "gate at coincident crop (CNN/S3 or LM rule)"
        return "FAR tables / ranking (same gates open)"
    if not b["veto_keep"][i]: return "local-ASD net clause" if b["veto_reason"][i] == "net<floor" else "local-ASD feature clause (O3)"
    if not b["guard_keep"][i]: return "net-sigma guard"
    return "other"
def causes_gain(a, b, i):   # 96 detected, 2.2 not
    if not a["trig"][i]: return "floor 2.75 (new trigger)"
    if b["cap_adm"][i] and not b["det_guard"][i]: return "cap bypass"
    if (b["lr_open"][i] and not a["lr_open"][i]) or (b["net_open"][i] and not a["net_open"][i]): return "gate at coincident crop (CNN/S3 or LM rule)"
    return "FAR tables / ranking (same gates open)"
tot = dict(n=0, both=0, only22=0, only96=0)
for run in ("o3a", "o3b", "o4a", "o4b"):
    a = np.load(f"{V22}/inj_scored_{run}_m22.npz", allow_pickle=True); b = np.load(f"{V96}/inj_scored_{run}_m96.npz", allow_pickle=True)
    assert len(a["net"]) == len(b["net"]) and np.array_equal(a["ev"], b["ev"]), "pairing: different files/order"
    assert np.allclose(a["mtot"], b["mtot"]) and np.allclose(a["net_snr"], b["net_snr"]) and np.allclose(a["net"], b["net"]), "pairing: injections differ"
    rl = np.load(f"{V96}/relabel_inj_{run}_m96.npz"); kept = rl["kept"]; msrc = rl["msrc"]
    d22 = a["detected"].astype(bool); d96 = b["detected"].astype(bool); both = d22 & d96; o22 = d22 & ~d96; o96 = d96 & ~d22
    tot["n"] += len(d22); tot["both"] += int(both.sum()); tot["only22"] += int(o22.sum()); tot["only96"] += int(o96.sum())
    L.append(f"## {run}: {len(d22)} paired injections (relabel-kept {int(kept.sum())}); detected 2.2 {int(d22.sum())}, 96 {int(d96.sum())}; both {int(both.sum())}, 2.2-only {int(o22.sum())}, 96-only {int(o96.sum())} (net {int(d96.sum())-int(d22.sum()):+d})"); L.append("")
    L.append("| source-frame Mtot bin | n (kept) | det 2.2 | det 96 | both | 2.2 only | 96 only | det_frac 2.2 | det_frac 96 |"); L.append("|---|---|---|---|---|---|---|---|---|")
    bi = np.digitize(msrc, EDGES) - 1
    for j, lb in enumerate(LAB):
        m = kept & (bi == j)
        if m.sum() == 0: continue
        L.append(f"| {lb} | {int(m.sum())} | {int(d22[m].sum())} | {int(d96[m].sum())} | {int(both[m].sum())} | {int(o22[m].sum())} | {int(o96[m].sum())} | {d22[m].mean():.3f} | {d96[m].mean():.3f} |")
    L.append(""); L.append("| SNR | n | det 2.2 | det 96 | 2.2 only | 96 only |"); L.append("|---|---|---|---|---|---|")
    for s in np.unique(a["net_snr"]):
        m = a["net_snr"] == s; L.append(f"| {s:g} | {int(m.sum())} | {int(d22[m].sum())} | {int(d96[m].sum())} | {int(o22[m].sum())} | {int(o96[m].sum())} |")
    L.append("")
    from collections import Counter
    cl = Counter(causes_loss(a, b, i) for i in np.flatnonzero(o22)); cg = Counter(causes_gain(a, b, i) for i in np.flatnonzero(o96))
    L.append(f"2.2-only losses by cause ({int(o22.sum())}): " + "; ".join(f"{k} {v}" for k, v in cl.most_common()))
    L.append(f"96-only gains by cause ({int(o96.sum())}): " + "; ".join(f"{k} {v}" for k, v in cg.most_common()))
    vet = b["det_merge"].astype(bool) & ~b["veto_keep"].astype(bool); gd = b["det_veto"].astype(bool) & ~b["guard_keep"].astype(bool); cap = b["cap_adm"].astype(bool) & ~b["det_guard"].astype(bool)
    L.append(f"96-side stage counts over all injections: merge detections {int(b['det_merge'].sum())}; local-ASD vetoed {int(vet.sum())} "
             f"(net clause {int((vet & (b['veto_reason']=='net<floor')).sum())}, feature clause {int((vet & (b['veto_reason']=='cnn')).sum())}); guard removed {int(gd.sum())}; cap-bypass admitted {int(cap.sum())}; final {int(d96.sum())}.")
    L.append("")
L.append(f"## All runs: {tot['n']} paired injections; both {tot['both']}, 2.2-only {tot['only22']}, 96-only {tot['only96']} (net {tot['only96']-tot['only22']:+d}).")
L.append(""); L.append("Reading: a '2.2 only' row is an injection the frozen 96 recipe no longer recovers at FAR < 1/yr; a '96 only' row is a new recovery. Causes are read from the "
         "96-side per-injection columns in the production order (trigger -> merge channels -> local-ASD veto -> guard; cap bypass as a separate admission path).")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"[compare] -> {OUT}")

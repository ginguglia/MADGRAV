"""Score the PROTECTED + SIDEBAND tiles (build_protected.py -> render_negatives.py R_PREFIX=prot) with S0 / S1 / S2.
Models: the single-split nets (train_single.py, seed 0). Detection and sideband legs were quarantined (+-64 s) from all
training, so these scores are clean; a top-50 pair outside the sideband may have been a training negative -> flagged.
Thresholds (fixed from the TEST set, never from these pairs): S0 = 0.5 (production); S1 / S2 at (a) S0 @ 0.5's test signal
efficiency ("matched") and (b) 90% test signal efficiency ("eff90").
Output: madgrav2_spec/models_single/protected_readout_s0.txt (+ per-pair CSV)."""
import os, glob, csv, hashlib, numpy as np, torch, torch.nn as nn
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec"; MD = f"{D}/models_single"; SEED = int(os.environ.get("SPEC_SEED", "0"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
class Net2(nn.Module):
    def __init__(s):
        super().__init__(); ch = [2, 16, 32, 64, 128]
        s.b = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
        s.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        s.register_buffer("temperature", torch.ones(1))
    def forward(s, x):
        for bl in s.b: x = bl(x)
        return s.h(x).squeeze(-1)
BANDS = {"hm": (20.0, 300.0), "lm": (50.0, 500.0), "s2": (20.0, 500.0)}; fax = 20.0 + 0.5 * np.arange(961)
files = sorted(glob.glob(f"{D}/prot/prot_*.npz")); Z = [np.load(f, allow_pickle=True) for f in files]
keys = [k for k in Z[0].files if k not in ("tiles", "flo", "fhi")]; P = {k: np.concatenate([z[k] for z in Z]) for k in keys}; T = np.concatenate([z["tiles"] for z in Z])
print(f"[prot] {len(T)} pairs from {len(files)} shards", flush=True)
def score(cfg):
    m = Net2().to(dev); m.load_state_dict(torch.load(f"{MD}/spec_{cfg}_single_s{SEED}.pt", map_location=dev)); m.eval()
    r = np.where((fax >= BANDS[cfg][0]) & (fax <= BANDS[cfg][1]))[0]; out = []
    with torch.no_grad():
        for i in range(0, len(T), 512):
            x = torch.from_numpy(T[i:i + 512][:, :, r[0]:r[-1] + 1].astype(np.float32)).to(dev)
            mn = x.amin((2, 3), keepdim=True); mx = x.amax((2, 3), keepdim=True); x = (x - mn) / (mx - mn + 1e-9)
            out.append(torch.sigmoid(m(x) / m.temperature).float().cpu().numpy())
    return np.concatenate(out)
p = {c: score(c) for c in BANDS}; s0 = np.maximum(P["s0_hm"], P["s0_lm"]); s1 = np.maximum(p["hm"], p["lm"]); s2 = p["s2"]
# thresholds from the TEST set
tz = {c: np.load(f"{MD}/spec_{c}_single_s{SEED}_test.npz") for c in BANDS}
Dm = f"{D}/train"; mp0, mp1 = (dict(np.load(f"{Dm}/pos_f{f}_meta.npz", allow_pickle=True)) for f in (0, 1))
s0_test = np.maximum(np.concatenate([mp0["s0_hm"], mp1["s0_hm"]]), np.concatenate([mp0["s0_lm"], mp1["s0_lm"]]))[tz["s2"]["g_pos"]]
eff0 = float((s0_test >= 0.5).mean()); tp1 = np.maximum(tz["hm"]["p_pos"], tz["lm"]["p_pos"]); tp2 = tz["s2"]["p_pos"]
TH = {"S1": {"matched": float(np.percentile(tp1, 100 * (1 - eff0))), "eff90": float(np.percentile(tp1, 10.0))},
      "S2": {"matched": float(np.percentile(tp2, 100 * (1 - eff0))), "eff90": float(np.percentile(tp2, 10.0))}}
G = {"S0": s0, "S1": s1, "S2": s2}
def passes(g, op): return G[g] >= (0.5 if g == "S0" else TH[g][op])
def h(tag, d, mod): return int(hashlib.sha1((tag + str(int(d))).encode()).hexdigest(), 16) % mod
dH = (P["gps_H"] // 86400).astype(int); dL = (P["gps_L"] // 86400).astype(int)
train_day = np.array([not (h("L", a, 20) == 0 or h("L", b, 20) == 0 or h("T", a, 20) < 2 or h("T", b, 20) < 2) for a, b in zip(dH, dL)])
out = [f"PROTECTED + SIDEBAND read-out (single-split nets seed {SEED}; thresholds from the TEST set: S0 @ 0.5 keeps {100*eff0:.2f}% of test injections)",
       f"  S1 thresholds: matched {TH['S1']['matched']:.4f} | eff90 {TH['S1']['eff90']:.4f};  S2 thresholds: matched {TH['S2']['matched']:.4f} | eff90 {TH['S2']['eff90']:.4f}"]
det = P["source"] == "det"; nd = int(det.sum())
out.append(f"\n== (b) {nd} catalogue detections: pass counts")
for op in ("matched", "eff90"):
    out.append(f"  {op:<8} S0 {int(passes('S0', op)[det].sum())}/{nd} | S1 {int(passes('S1', op)[det].sum())}/{nd} | S2 {int(passes('S2', op)[det].sum())}/{nd}")
out.append("  lost (any gate/op), name run | S0 hm/lm | S1 hm/lm | S2 | lost under")
for i in np.where(det)[0]:
    lost = [f"{g}@{op}" for g in ("S1", "S2") for op in ("matched", "eff90") if not passes(g, op)[i]]
    if lost or s0[i] < 0.9 or min(s1[i], s2[i]) < 0.9:
        out.append(f"  {P['name'][i]:<18} {P['run'][i]} | {P['s0_hm'][i]:.3f}/{P['s0_lm'][i]:.3f} | {p['hm'][i]:.3f}/{p['lm'][i]:.3f} | {s2[i]:.3f} | {' '.join(lost) or '-'}")
WATCH = ("GW190602_175927", "GW190706_222641", "GW250119_025138", "GW250118_170523", "GW230922_040658", "GW240406_062847", "GW230628_231200", "GW230707_124047")
out.append("\n== (a) watchlist (8 of record)")
for w in WATCH:
    ii = np.where(det & (P["name"] == w))[0]
    if not len(ii): out.append(f"  {w}: NOT IN RENDER"); continue
    i = ii[0]; out.append(f"  {w:<18} S0 {s0[i]:.3f} | S1 {s1[i]:.3f} ({'P' if passes('S1','matched')[i] else 'F'}/{'P' if passes('S1','eff90')[i] else 'F'}) | S2 {s2[i]:.3f} ({'P' if passes('S2','matched')[i] else 'F'}/{'P' if passes('S2','eff90')[i] else 'F'})  [matched/eff90]")
sb = P["source"] == "sideband"
out.append(f"\n== counted FAR families (sideband, {int(sb.sum())} pairs; S0 passes all by construction)")
for op in ("matched", "eff90"):
    out.append(f"  {op:<8} S0 {100*passes('S0',op)[sb].mean():.1f}% | S1 {100*passes('S1',op)[sb].mean():.1f}% | S2 {100*passes('S2',op)[sb].mean():.1f}%   by run: "
               + " ".join(f"{r}: S1 {int(passes('S1',op)[sb&(P['run']==r)].sum())} S2 {int(passes('S2',op)[sb&(P['run']==r)].sum())} /{int((sb&(P['run']==r)).sum())}" for r in np.unique(P["run"])))
t5 = P["top50"] == 1; clean = t5 & ~(train_day & (P["source"] == "top50"))
out.append(f"\n== top-50 FAR pairs (by lnL, S0-passing, reach specialists): {int(t5.sum())} ({int((t5&sb).sum())} in sideband; {int((t5&~clean).sum())} possibly seen in training -> excluded below)")
for op in ("matched", "eff90"):
    out.append(f"  {op:<8} passing: S0 {int(passes('S0',op)[clean].sum())} | S1 {int(passes('S1',op)[clean].sum())} | S2 {int(passes('S2',op)[clean].sum())}  of {int(clean.sum())}")
txt = "\n".join(out); print(txt); open(f"{MD}/protected_readout_s{SEED}.txt", "w").write(txt + "\n")
with open(f"{MD}/protected_scores_s{SEED}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["source", "name", "run", "gps_H", "gps_L", "loglr", "top50", "s0_hm", "s0_lm", "hm", "lm", "s2", "train_day"])
    for i in range(len(T)): w.writerow([P["source"][i], P["name"][i], P["run"][i], P["gps_H"][i], P["gps_L"][i], P["loglr"][i], P["top50"][i], P["s0_hm"][i], P["s0_lm"][i], p["hm"][i], p["lm"][i], s2[i], int(train_day[i])])

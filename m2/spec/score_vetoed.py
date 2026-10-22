"""Score the S0-VETOED catalogue triggers (build_vetoed.py -> render R_PREFIX=vet) with the single-split S1/S2 nets.
Operating points fixed from the TEST set / detections (never from these events): equal background vs S0 @ 0.5
(S1 0.8307, S2 0.5331; production-weighted test bg) and keep-all-67 (S1 0.786, S2 0.430).
'loglr >= weakest det' = its lnL reaches the weakest loglr-channel detection of its run (a rough detection hint, not a FAR)."""
import glob, json, numpy as np, torch, torch.nn as nn
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec"; MD = f"{D}/models_single"; M2 = MADGRAV_SCRATCH + "/madgrav2"; dev = "cuda:0" if torch.cuda.is_available() else "cpu"
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
Z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{D}/vet/vet_*.npz"))]; T = np.concatenate([z["tiles"] for z in Z])
P = {k: np.concatenate([z[k] for z in Z]) for k in Z[0].files if k not in ("tiles", "flo", "fhi")}
def score(cfg):
    m = Net2().to(dev); m.load_state_dict(torch.load(f"{MD}/spec_{cfg}_single_s0.pt", map_location=dev)); m.eval(); r = np.where((fax >= BANDS[cfg][0]) & (fax <= BANDS[cfg][1]))[0]
    with torch.no_grad():
        x = torch.from_numpy(T[:, :, r[0]:r[-1] + 1].astype(np.float32)).to(dev); mn = x.amin((2, 3), keepdim=True); mx = x.amax((2, 3), keepdim=True)
        return torch.sigmoid(m((x - mn) / (mx - mn + 1e-9)) / m.temperature).float().cpu().numpy()
p = {c: score(c) for c in BANDS}; s1 = np.maximum(p["hm"], p["lm"]); s2 = p["s2"]
weak = {r: min(d["loglr"] for d in json.load(open(f"{M2}/search_out_{r}_m2_gx/detections_lm.json")) if d.get("channel") == "loglr" and d.get("loglr") is not None) for r in ("o3a", "o3b", "o4a", "o4b")}
TH = {"S1": (0.8307, 0.786), "S2": (0.5331, 0.430)}
out = [f"S0-VETOED catalogue triggers ({len(T)}), single-split nets seed 0. Pass = equal-bg / keep-all-67 thresholds. Weakest loglr-channel det lnL: " + " ".join(f"{r}:{v:.2f}" for r, v in weak.items()),
       f"  {'event':<18} run | S0 hm/lm      | S1 hm/lm      S2    | S1 eqbg/keep67 | S2 eqbg/keep67 | lnL   (>= weakest det?)"]
for i in np.argsort(-np.maximum(s1, s2)):
    f = lambda g, k: "P" if (s1 if g == "S1" else s2)[i] >= TH[g][k] else "-"
    out.append(f"  {P['name'][i]:<18} {P['run'][i]} | {P['s0_hm'][i]:.3f}/{P['s0_lm'][i]:.3f} | {p['hm'][i]:.3f}/{p['lm'][i]:.3f} {s2[i]:.3f} |   {f('S1',0)}/{f('S1',1)}          |   {f('S2',0)}/{f('S2',1)}          | {P['loglr'][i]:5.2f} ({'yes' if P['loglr'][i] >= weak[P['run'][i]] else 'no'})")
for g in ("S1", "S2"):
    s = s1 if g == "S1" else s2
    for k, nm in ((0, "equal-bg"), (1, "keep-67")):
        ok = s >= TH[g][k]; out.append(f"  {g} {nm:<9}: recovers {int(ok.sum())}/{len(T)}; of those with lnL >= weakest det of their run: {int((ok & (P['loglr'] >= np.array([weak[r] for r in P['run']]))).sum())}")
txt = "\n".join(out); print(txt); open(f"{MD}/vetoed_readout_s0.txt", "w").write(txt + "\n")

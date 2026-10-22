"""Score the catalogue sets with specialist ENSEMBLES (2026-09-30): armvet (catalogue triggers vetoed by the production CNN gate,
build_vetoed.py) and prot (detected events 'det', 'top50', 'sideband') with one or more 2x5 hm ensembles, e.g. production S3
(madgrav2_spec/models) vs a retrain (models_gneg). Per ensemble: threshold = 8th percentile of its own cross-fitted held-out
injection scores (the A6 rule = 92% efficiency), plus the production threshold for reference; background pass as drawn from the
held-out negatives. Catalogue events are quarantined from all training, so every model scores them cleanly (mean over 10 models).
Env: SPEC_ENS = 'label=dir,label=dir' (default 'prod=.../models'), SPEC_CFG (hm), SPEC_SFX (nfull). Args: [--mail] [--out FILE]."""
import os, sys, glob, json, numpy as np, torch, torch.nn as nn
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = MADGRAV_SCRATCH + "/madgrav2_spec"; dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CFG = os.environ.get("SPEC_CFG", "hm"); SFX = os.environ.get("SPEC_SFX", "nfull")
ENS = dict(kv.split("=", 1) for kv in os.environ.get("SPEC_ENS", f"prod={D}/models").split(","))
BANDS = {"hm": (20.0, 300.0), "lm": (50.0, 500.0), "s2": (20.0, 500.0)}; fax = 20.0 + 0.5 * np.arange(961); r = np.where((fax >= BANDS[CFG][0]) & (fax <= BANDS[CFG][1]))[0]
PROD_THR = json.load(open(MADGRAV_ROOT + "/m2/rank/s3_threshold.json"))["threshold"]
class Net2(nn.Module):
    def __init__(s):
        super().__init__(); ch = [2, 16, 32, 64, 128]
        s.b = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
        s.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)); s.register_buffer("temperature", torch.ones(1))
    def forward(s, x):
        for bl in s.b: x = bl(x)
        return s.h(x).squeeze(-1)
def load_set(prefix):
    Z = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{D}/{prefix}/{prefix}_*.npz"))]
    T = np.concatenate([z["tiles"] for z in Z]); P = {k: np.concatenate([np.atleast_1d(z[k]) if np.asarray(z[k]).ndim else np.full(len(z["tiles"]), z[k].item()) for z in Z]) for k in Z[0].files if k not in ("tiles", "flo", "fhi")}
    return T, P
def ens_scores(md, T):
    ps = []
    for f in (0, 1):
        for s_ in range(5):
            m = Net2().to(dev); m.load_state_dict(torch.load(f"{md}/spec_{CFG}_f{f}_s{s_}_{SFX}.pt", map_location=dev)); m.eval()
            with torch.no_grad():
                out = []
                for i in range(0, len(T), 512):
                    x = torch.from_numpy(T[i:i + 512, :, r[0]:r[-1] + 1].astype(np.float32)).to(dev); mn = x.amin((2, 3), keepdim=True); mx = x.amax((2, 3), keepdim=True)
                    out.append(torch.sigmoid(m((x - mn) / (mx - mn + 1e-9)) / m.temperature).float().cpu().numpy())
            ps.append(np.concatenate(out))
    return np.mean(ps, 0)
def ens_threshold(md):
    P, N = [], []
    for f in (0, 1):
        Z = [np.load(f"{md}/spec_{CFG}_f{f}_s{s_}_{SFX}_heldout.npz") for s_ in range(5)]; P.append(np.mean([z["p_pos"] for z in Z], 0)); N.append(np.mean([z["p_neg"] for z in Z], 0))
    P = np.concatenate(P); N = np.concatenate(N); thr = float(np.percentile(P, 8.0))
    return thr, float((N >= thr).mean()), float((P >= PROD_THR).mean()), float((N >= PROD_THR).mean()), len(P), len(N)
L = []; Pr = L.append
TA, PA = load_set("armvet"); TP, PP = load_set("prot")
Pr(f"Specialist ensembles on the catalogue sets ({os.popen('date +%F_%H:%M').read().strip()}); cfg {CFG}; production S3 threshold {PROD_THR:.4f}")
Pr(f"armvet = {len(TA)} catalogue triggers vetoed by the production CNN gate (S0); prot = {len(TP)} pairs: " + ", ".join(f"{s}:{int((PP['source']==s).sum())}" for s in np.unique(PP["source"])))
res = {}
for lab, md in ENS.items():
    thr, bgp, eff_prod, bgp_prod, npos, nneg = ens_threshold(md); sa = ens_scores(md, TA); sp = ens_scores(md, TP); res[lab] = (thr, sa, sp)
    Pr(f"\n== {lab} ({md}): own 92%-eff threshold {thr:.4f} (held-out bg pass as drawn {bgp:.4f}; at prod thr: eff {eff_prod:.4f}, bg pass {bgp_prod:.4f}; n_pos {npos}, n_neg {nneg})")
    for T_, nm, s in ((thr, "own thr", sa), (PROD_THR, "prod thr", sa)):
        ok = s >= T_; Pr(f"  armvet pass at {nm} {T_:.3f}: {int(ok.sum())}/{len(s)}  by run: " + " ".join(f"{rn}:{int((ok & (PA['run']==rn)).sum())}/{int((PA['run']==rn).sum())}" for rn in ("o3a", "o3b", "o4a", "o4b")))
    det = PP["source"] == "det"
    for T_, nm in ((thr, "own thr"), (PROD_THR, "prod thr")):
        Pr(f"  prot pass at {nm}: det {int((sp[det] >= T_).sum())}/{int(det.sum())}, top50 {int((sp[PP['source']=='top50'] >= T_).sum())}/{int((PP['source']=='top50').sum())}, sideband {int((sp[PP['source']=='sideband'] >= T_).sum())}/{int((PP['source']=='sideband').sum())}")
Pr("\n== armvet per event (score per ensemble; P = passes its own threshold)")
labs = list(ENS); order = np.argsort(-res[labs[-1]][1])
Pr("  " + f"{'event':<18} {'run':<4} {'net':>5} {'loglr':>6} {'S0hm':>5} " + " ".join(f"{l:>9}" for l in labs))
for i in order:
    Pr("  " + f"{str(PA['name'][i]):<18} {str(PA['run'][i]):<4} {float(PA['net'][i]):5.2f} {float(PA['loglr'][i]) if PA['loglr'][i] is not None else float('nan'):6.2f} {float(PA['s0_hm'][i]):5.2f} " + " ".join(f"{res[l][1][i]:7.3f}{'P' if res[l][1][i] >= res[l][0] else '-':>2}" for l in labs))
if len(labs) > 1:
    a, b = labs[0], labs[-1]; pa = res[a][1] >= res[a][0]; pb = res[b][1] >= res[b][0]
    Pr(f"\n{b} vs {a}: armvet NEW {int((pb & ~pa).sum())} " + ", ".join(str(PA['name'][i]) for i in np.where(pb & ~pa)[0]) + f" | LOST {int((pa & ~pb).sum())} " + ", ".join(str(PA['name'][i]) for i in np.where(pa & ~pb)[0]))
    det = PP["source"] == "det"; da = res[a][2][det] >= res[a][0]; db = res[b][2][det] >= res[b][0]
    Pr(f"{b} vs {a}: detected events pass {int(db.sum())}/{int(det.sum())} vs {int(da.sum())}/{int(det.sum())}; lost: " + ", ".join(str(PP['name'][det][i]) for i in np.where(da & ~db)[0]))
txt = "\n".join(L); print(txt); out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else f"{list(ENS.values())[-1]}/READOUT_SETS.txt"; open(out, "w").write(txt + "\n")
if "--mail" in sys.argv:
    import subprocess; r_ = subprocess.run([sys.executable, MADGRAV_SCRATCH + "/MADGRAV1/search_mode/pastro_final/campaign/mailer.py", f"[MADGRAV] specialist ensembles on catalogue sets: {', '.join(labs)}", out], capture_output=True, text=True); print("mail rc", r_.returncode, r_.stdout[-200:], r_.stderr[-200:])

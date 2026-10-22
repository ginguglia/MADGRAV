"""MADGRAV2 specialist retrain (PREREG_SPECIALISTS_20260923 §1, §5): Net2 (the production specialist architecture,
driver_blindscan.Net2), binary head, day-fold cross-fitting, recipe of m2/arm/train_arm.py.
Configs (SPEC_CFG): hm = S1 HM branch 20-300 Hz | lm = S1 LM branch 50-500 Hz | s2 = one pair net 20-500 Hz.
Input = the stored RAW 20-500 crop (production cnn_from_wh placement), band rows sliced, per-channel min-max on the GPU
(= driver_blindscan._crop), so train and inference share one path.
Train on fold F (calib days held aside), watch on the OTHER fold every epoch:
  t99 = 1st percentile of P(sig) over held-out positives (1% signal loss); bg_pass = held-out negatives with P >= t99;
  hard_pass = the same on the hard negatives (campaign S0-pass + A3 re-pairs).  Selection = bg_pass, tiebreak hard_pass
  (the counted FAR families are quarantined sideband tiles, not rendered here); patience 10; never AUC.
Balanced classes per batch, negatives uniform over the fold subset. AdamW 3e-4, cosine, wd 1e-4, batch 256, <=60 epochs,
grad clip 1.0, bf16 autocast. Temperature scaling (BCE) on the training fold's calib days.
Env: SPEC_CFG, SPEC_FOLD, SPEC_SEED, SPEC_N (per-class cap on the training subset, 0 = full), SPEC_EPOCHS, SPEC_SMOKE."""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = os.environ.get("SPEC_DATA", MADGRAV_SCRATCH + "/madgrav2_spec/train"); OUT = os.environ.get("SPEC_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/models"); os.makedirs(OUT, exist_ok=True)
CFG = os.environ.get("SPEC_CFG", "s2"); FOLD = int(os.environ.get("SPEC_FOLD", "0")); SEED = int(os.environ.get("SPEC_SEED", "0"))
NCAP = int(os.environ.get("SPEC_N", "0")); EPOCHS = int(os.environ.get("SPEC_EPOCHS", "60")); SMOKE = os.environ.get("SPEC_SMOKE") == "1"
BANDS = {"hm": (20.0, 300.0), "lm": (50.0, 500.0), "s2": (20.0, 500.0)}; FLO, FHI = BANDS[CFG]
NAME = f"spec_{CFG}_f{FOLD}_s{SEED}_n{NCAP // 1000 if NCAP else 'full'}{'k' if NCAP else ''}{'_smoke' if SMOKE else ''}"
BATCH = 256; LR = 3e-4; WD = 1e-4; PATIENCE = 10; EPOCH_SAMPLES = 3000 if SMOKE else 60000; WATCH_N = 2000 if SMOKE else 30000
dev = "cuda:0" if torch.cuda.is_available() else "cpu"; torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class Net2(nn.Module):                              # identical to search_mode/driver_blindscan.Net2 (+ temperature buffer)
    def __init__(s):
        super().__init__(); ch = [2, 16, 32, 64, 128]
        s.b = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
        s.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        s.register_buffer("temperature", torch.ones(1))
    def forward(s, x):
        for bl in s.b: x = bl(x)
        return s.h(x).squeeze(-1)
    def p(s, x): return torch.sigmoid(s(x) / s.temperature)

fax = 20.0 + 0.5 * np.arange(961); rows = np.where((fax >= FLO) & (fax <= FHI))[0]; R0, R1 = int(rows[0]), int(rows[-1]) + 1
X = {c: {f: np.load(f"{D}/{c}_f{f}.npy", mmap_mode="r") for f in (0, 1)} for c in ("pos", "neg")}
M = {c: {f: dict(np.load(f"{D}/{c}_f{f}_meta.npz", allow_pickle=True)) for f in (0, 1)} for c in ("pos", "neg")}
EXTRA = os.environ.get("SPEC_EXTRA", "")                # 2026-09-30: extra rendered sets (assemble_spec.py with SPEC_SRC_*) appended behind the base memmaps
class _Cat:                                             # two memmaps read as one: fancy row index + slices, rows of the base first (indices are sorted by load())
    def __init__(s, a, b): s.a, s.b, s.n0 = a, b, len(a); s.shape = (len(a) + len(b),) + tuple(a.shape[1:]); s.dtype = a.dtype
    def __len__(s): return s.shape[0]
    def __getitem__(s, key):
        idx = np.asarray(key[0]); rest = tuple(key[1:]); ia = idx[idx < s.n0]; ib = idx[idx >= s.n0] - s.n0; parts = []
        if len(ia): parts.append(np.asarray(s.a[(ia,) + rest]))
        if len(ib): parts.append(np.asarray(s.b[(ib,) + rest]))
        return np.concatenate(parts) if parts else np.empty((0,) + np.asarray(s.a[(np.array([0]),) + rest]).shape[1:], s.a.dtype)
if EXTRA:
    for c in ("pos", "neg"):
        for f in (0, 1):
            fx = f"{EXTRA}/{c}_f{f}.npy"
            if not os.path.exists(fx): continue
            Xe = np.load(fx, mmap_mode="r"); Me = dict(np.load(f"{EXTRA}/{c}_f{f}_meta.npz", allow_pickle=True)); nb = len(Xe)
            def _fill(k):
                base = np.asarray(M[c][f][k])
                if k in Me: return np.asarray(Me[k]).astype(base.dtype, copy=False) if base.dtype.kind != "U" else np.asarray(Me[k]).astype(str)
                return np.full(nb, np.nan if base.dtype.kind == "f" else (0 if base.dtype.kind in "iub" else ""), dtype=base.dtype)
            M[c][f] = {k: np.concatenate([np.asarray(M[c][f][k]), _fill(k)]) for k in M[c][f]}; X[c][f] = _Cat(X[c][f], Xe)
            print(f"[spec] extra {c} f{f}: +{nb} rows from {EXTRA}", flush=True)
assert X["pos"][0].shape[2] == 961, X["pos"][0].shape
def hard(m): return np.isin(m["source"], ["campaign_pass", "repair"])

def load(c, f, idx):                                # band rows of the raw crop -> CPU fp16 tensor (sorted reads from the memmap)
    idx = np.sort(np.asarray(idx)); out = np.empty((len(idx), 2, R1 - R0, X[c][f].shape[3]), np.float16)
    for i in range(0, len(idx), 2048): out[i:i + 2048] = X[c][f][idx[i:i + 2048], :, R0:R1]
    return idx, torch.from_numpy(out)
def prep(xb):                                       # per-channel min-max = driver_blindscan._crop
    x = xb.to(dev, non_blocking=True).float(); mn = x.amin((2, 3), keepdim=True); mx = x.amax((2, 3), keepdim=True)
    return (x - mn) / (mx - mn + 1e-9)

# ---- training subset (fold F, calib days out, per-class cap) and calibration / watch sets ----
tr, cal = {}, {}
for c in ("pos", "neg"):
    m = M[c][FOLD]; ok = np.where(~m["calib"].astype(bool))[0]; cal[c] = np.where(m["calib"].astype(bool))[0]
    if SMOKE: ok = rng.choice(ok, size=min(3000, len(ok)), replace=False)
    elif NCAP and NCAP < len(ok): ok = rng.choice(ok, size=NCAP, replace=False)
    tr[c] = ok
t0 = time.time()
Itr, Xtr = {}, {}
for c in ("pos", "neg"): Itr[c], Xtr[c] = load(c, FOLD, tr[c])
HO = 1 - FOLD; wsub = {c: rng.choice(len(X[c][HO]), size=min(WATCH_N, len(X[c][HO])), replace=False) for c in ("pos", "neg")}
Iho, Xho = {}, {}
for c in ("pos", "neg"): Iho[c], Xho[c] = load(c, HO, wsub[c])
ho_hard = hard({k: v[Iho["neg"]] for k, v in M["neg"][HO].items()})
tsub = {c: rng.choice(len(Itr[c]), size=min(WATCH_N // 3, len(Itr[c])), replace=False) for c in ("pos", "neg")}
print(f"[spec {NAME}] band {FLO:.0f}-{FHI:.0f} Hz rows {R0}:{R1} | train pos {len(Itr['pos'])} neg {len(Itr['neg'])} (hard {int(hard({k: v[Itr['neg']] for k, v in M['neg'][FOLD].items()}).sum())}) "
      f"| calib pos {len(cal['pos'])} neg {len(cal['neg'])} | watch pos {len(Iho['pos'])} neg {len(Iho['neg'])} (hard {int(ho_hard.sum())}) | load {(time.time()-t0)/60:.1f} min | {dev}", flush=True)

def scores(model, Xt, idx=None, bs=1024):
    model.eval(); out = []; n = len(Xt) if idx is None else len(idx)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        for i in range(0, n, bs):
            xb = Xt[i:i + bs] if idx is None else Xt[torch.from_numpy(np.asarray(idx[i:i + bs]))]
            out.append(model.p(prep(xb)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])
def watch(model, Xp, Xn, ip=None, inn=None, hard_mask=None):
    pp = scores(model, Xp, ip); pn = scores(model, Xn, inn); t99 = float(np.percentile(pp, 1.0))
    r = dict(t99=t99, bg_pass=float((pn >= t99).mean()))
    if hard_mask is not None: r["hard_pass"] = float((pn[hard_mask] >= t99).mean()) if hard_mask.any() else np.nan
    return r

model = Net2().to(dev); nparam = sum(p.numel() for p in model.parameters()); print(f"[spec] params {nparam:,d}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); spe = EPOCH_SAMPLES // BATCH
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * spe)
hist = []; best = dict(bg_pass=np.inf, hard_pass=np.inf, epoch=-1, state=None); es = 0; t0 = time.time()
for epoch in range(EPOCHS):
    model.train(); losses = []
    for _ in range(spe):
        ip = torch.from_numpy(rng.choice(len(Itr["pos"]), BATCH // 2, replace=False)); inn = torch.from_numpy(rng.choice(len(Itr["neg"]), BATCH // 2, replace=False))
        x = prep(torch.cat([Xtr["pos"][ip], Xtr["neg"][inn]])); y = torch.cat([torch.ones(BATCH // 2), torch.zeros(BATCH // 2)]).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")): loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); losses.append(float(loss.item()))
    wh = watch(model, Xho["pos"], Xho["neg"], hard_mask=ho_hard); wt = watch(model, Xtr["pos"], Xtr["neg"], tsub["pos"], tsub["neg"])
    rec = dict(epoch=epoch + 1, loss=float(np.mean(losses)), ho=wh, tr=wt, lr=float(sched.get_last_lr()[0]), min=(time.time() - t0) / 60); hist.append(rec)
    if (wh["bg_pass"], wh["hard_pass"]) < (best["bg_pass"], best["hard_pass"]):
        best.update(bg_pass=wh["bg_pass"], hard_pass=wh["hard_pass"], epoch=epoch + 1, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); es = 0; tag = " <-- best"
    else: es += 1; tag = f" ({es}/{PATIENCE})"
    print(f"[spec {NAME} {epoch+1:02d}/{EPOCHS}] loss {rec['loss']:.4f} | held-out @1%loss: bg pass {wh['bg_pass']:.4f} hard pass {wh['hard_pass']:.4f} t99 {wh['t99']:.3f} | "
          f"train-fold bg {wt['bg_pass']:.4f} | {rec['min']:.1f} min{tag}", flush=True)
    if es >= PATIENCE: print(f"[spec] early stop (best epoch {best['epoch']})", flush=True); break
model.load_state_dict(best["state"])
# ---- temperature scaling on the training fold's calib days ----
ci = {c: cal[c][:(200 if SMOKE else 4000)] for c in ("pos", "neg")}
if len(ci["pos"]) and len(ci["neg"]):
    model.eval(); lg = []
    with torch.no_grad():
        for c in ("pos", "neg"):
            _, xc = load(c, FOLD, ci[c])
            for i in range(0, len(xc), 1024): lg.append(model(prep(xc[i:i + 1024])).float())
    lg = torch.cat(lg); y = torch.cat([torch.ones(len(ci["pos"])), torch.zeros(len(ci["neg"]))]).to(dev)
    logT = torch.zeros(1, device=dev, requires_grad=True); o2 = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
    def closure():
        o2.zero_grad(); l = F.binary_cross_entropy_with_logits(lg / logT.exp(), y); l.backward(); return l
    o2.step(closure); logT.data.clamp_(np.log(0.2), np.log(20.0)); model.temperature.fill_(float(logT.exp().detach()))
    print(f"[spec] temperature {float(model.temperature):.3f} (calib pos {len(ci['pos'])} neg {len(ci['neg'])})", flush=True)
# ---- full held-out fold scores (streamed from the memmap) for the cross-fit evaluation ----
full = {}
for c in ("pos", "neg"):
    n = len(X[c][HO]); ps = []
    for i in range(0, n, 16384): ps.append(scores(model, load(c, HO, np.arange(i, min(n, i + 16384)))[1]))
    full[c] = np.concatenate(ps)
t99 = float(np.percentile(full["pos"], 1.0)); hm_ = hard(M["neg"][HO])
summary = dict(name=NAME, cfg=CFG, band=[FLO, FHI], fold=FOLD, seed=SEED, ncap=NCAP, n_train={c: int(len(Itr[c])) for c in Itr}, params=nparam, best_epoch=best["epoch"],
               temperature=float(model.temperature), t99=t99, bg_pass=float((full["neg"] >= t99).mean()), hard_pass=float((full["neg"][hm_] >= t99).mean()),
               n_heldout={c: int(len(full[c])) for c in full}, history=hist)
torch.save(model.state_dict(), f"{OUT}/{NAME}.pt"); json.dump(summary, open(f"{OUT}/{NAME}.json", "w"), indent=1)
np.savez(f"{OUT}/{NAME}_heldout.npz", p_pos=full["pos"], p_neg=full["neg"], heldout_fold=HO)
print(f"[spec {NAME}] DONE best epoch {best['epoch']}: held-out FULL @1% signal loss -> bg pass {summary['bg_pass']:.4f}, hard pass {summary['hard_pass']:.4f}; {(time.time()-t0)/60:.1f} min", flush=True)

"""MADGRAV2 specialist retrain, SINGLE-SPLIT variant (2026-09-23: pool both folds, keep 10% of days as a sealed
post-mortem lockbox, do training / testing / calibration in the rest). Same Net2, input path, loss and selection as
train_spec.py; only the partition differs.
Partition (deterministic, identical for every config). Unit = UTC day; a pair takes the role of EITHER leg's day, so a
leg (which lives on one day) can never appear in two roles, even though time slides reuse one glitch leg in up to ~3,500 pairs:
  lockbox = H1 day or L1 day has sha1("L"+day) % 20 == 0 (5% of days -> ~10% of pairs). SEALED: never trained, never
            scored here; its role list is saved for the post-mortem.
  test    = otherwise, H1 day or L1 day has sha1("T"+day) % 20 < 2 (10% of days -> ~17% of pairs); early stopping + read-out.
  calib   = otherwise, calib_of(H1 day) (existing "c" hash; temperature scaling only).
  train   = the rest (both legs on train days).
Per epoch: train loss (mean BCE over the epoch's balanced batches) and TEST loss (class-balanced BCE on the watch subset).
Env: SPEC_CFG (hm | lm | s2), SPEC_SEED, SPEC_EPOCHS, SPEC_WATCH (default 20000 per class), SPEC_SMOKE, SPEC_SPLIT_ONLY."""
import os, sys, json, time, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = os.environ.get("SPEC_DATA", MADGRAV_SCRATCH + "/madgrav2_spec/train"); OUT = os.environ.get("SPEC_OUT", MADGRAV_SCRATCH + "/madgrav2_spec/models_single"); os.makedirs(OUT, exist_ok=True)
CFG = os.environ.get("SPEC_CFG", "s2"); SEED = int(os.environ.get("SPEC_SEED", "0")); EPOCHS = int(os.environ.get("SPEC_EPOCHS", "60")); SMOKE = os.environ.get("SPEC_SMOKE") == "1"
BANDS = {"hm": (20.0, 300.0), "lm": (50.0, 500.0), "s2": (20.0, 500.0)}; FLO, FHI = BANDS[CFG]; NAME = f"spec_{CFG}_single_s{SEED}{'_smoke' if SMOKE else ''}"
BATCH = 256; LR = 3e-4; WD = 1e-4; PATIENCE = 10; EPOCH_SAMPLES = 3000 if SMOKE else 60000; WATCH_N = 2000 if SMOKE else int(os.environ.get("SPEC_WATCH", "20000"))
dev = "cuda:0" if torch.cuda.is_available() else "cpu"; torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
torch.backends.cudnn.benchmark = True               # speed (not bit-reproducible; the fold pilot keeps deterministic cuDNN)

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
X = {c: [np.load(f"{D}/{c}_f{f}.npy", mmap_mode="r") for f in (0, 1)] for c in ("pos", "neg")}
M = {}
for c in ("pos", "neg"):                             # global index = fold-0 rows then fold-1 rows
    m0, m1 = (dict(np.load(f"{D}/{c}_f{f}_meta.npz", allow_pickle=True)) for f in (0, 1)); M[c] = {k: np.concatenate([m0[k], m1[k]]) for k in m0}
N0 = {c: len(X[c][0]) for c in X}
def h(tag, d, mod): return int(hashlib.sha1((tag + str(int(d))).encode()).hexdigest(), 16) % mod

# ---- partition ----
ROLE = {}
for c in ("pos", "neg"):
    dH = (M[c]["gps_H"] // 86400).astype(int); dL = (M[c]["gps_L"] // 86400).astype(int); ud = np.unique(np.concatenate([dH, dL]))
    LK = dict(zip(ud, [h("L", d, 20) == 0 for d in ud])); TS = dict(zip(ud, [h("T", d, 20) < 2 for d in ud]))
    lk = np.array([LK[a_] or LK[b_] for a_, b_ in zip(dH, dL)]); ts = np.array([TS[a_] or TS[b_] for a_, b_ in zip(dH, dL)]); cb = M[c]["calib"].astype(bool)
    role = np.where(lk, "lock", np.where(ts, "test", np.where(cb, "calib", "train"))).astype(object); n_drop = 0
    ROLE[c] = role
    print(f"[split] {c}: " + " ".join(f"{r} {int((role==r).sum())}" for r in ("train", "calib", "test", "lock", "drop"))
          + f" | test by run " + " ".join(f"{r}:{int(((role=='test')&(M[c]['run']==r)).sum())}" for r in np.unique(M[c]["run"]))
          + " | lock by run " + " ".join(f"{r}:{int(((role=='lock')&(M[c]['run']==r)).sum())}" for r in np.unique(M[c]["run"]))
          + ("" if c == "pos" else " | test by source " + " ".join(f"{s}:{int(((role=='test')&(M[c]['source']==s)).sum())}" for s in np.unique(M[c]["source"]))
             + " | lock by source " + " ".join(f"{s}:{int(((role=='lock')&(M[c]['source']==s)).sum())}" for s in np.unique(M[c]["source"]))), flush=True)
if os.environ.get("SPEC_SPLIT_ONLY") == "1": sys.exit(0)
def hard(c, g): return np.isin(M[c]["source"][g], ["campaign_pass", "repair"]) if c == "neg" else np.zeros(len(g), bool)

def load(c, g):                                     # global indices -> (sorted g, CPU fp16 tensor of the band rows)
    g = np.sort(np.asarray(g)); out = np.empty((len(g), 2, R1 - R0, X[c][0].shape[3]), np.float16); o = 0
    for f, sel in ((0, g[g < N0[c]]), (1, g[g >= N0[c]] - N0[c])):
        for i in range(0, len(sel), 2048):
            blk = sel[i:i + 2048]; out[o:o + len(blk)] = X[c][f][blk, :, R0:R1]; o += len(blk)
    return g, torch.from_numpy(out)
def prep(xb):                                       # per-channel min-max = driver_blindscan._crop
    x = xb.to(dev, non_blocking=True).float(); mn = x.amin((2, 3), keepdim=True); mx = x.amax((2, 3), keepdim=True)
    return (x - mn) / (mx - mn + 1e-9)

t0 = time.time(); G = {r: {c: np.where(ROLE[c] == r)[0] for c in ROLE} for r in ("train", "calib", "test")}
if SMOKE: G["train"] = {c: rng.choice(v, size=min(3000, len(v)), replace=False) for c, v in G["train"].items()}
Itr, Xtr, Iw, Xw = {}, {}, {}, {}
for c in ("pos", "neg"):
    Itr[c], Xtr[c] = load(c, G["train"][c]); Iw[c], Xw[c] = load(c, rng.choice(G["test"][c], size=min(WATCH_N, len(G["test"][c])), replace=False))
w_hard = hard("neg", Iw["neg"]); tsub = {c: rng.choice(len(Itr[c]), size=min(WATCH_N // 3, len(Itr[c])), replace=False) for c in ("pos", "neg")}
print(f"[spec {NAME}] band {FLO:.0f}-{FHI:.0f} Hz rows {R0}:{R1} | train pos {len(Itr['pos'])} neg {len(Itr['neg'])} (hard {int(hard('neg', Itr['neg']).sum())}) | "
      f"watch pos {len(Iw['pos'])} neg {len(Iw['neg'])} (hard {int(w_hard.sum())}) | load {(time.time()-t0)/60:.1f} min | {dev}", flush=True)

def logits(model, Xt, idx=None, bs=1024):
    model.eval(); out = []; n = len(Xt) if idx is None else len(idx)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        for i in range(0, n, bs):
            xb = Xt[i:i + bs] if idx is None else Xt[torch.from_numpy(np.asarray(idx[i:i + bs]))]
            out.append((model(prep(xb)) / model.temperature).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])
def sig(z): return 1.0 / (1.0 + np.exp(-z))
def watch(model, Xp, Xn, ip=None, inn=None, hard_mask=None):
    zp = logits(model, Xp, ip); zn = logits(model, Xn, inn); pp, pn = sig(zp), sig(zn); t99 = float(np.percentile(pp, 1.0))
    loss = 0.5 * (float(np.mean(np.logaddexp(0, -zp))) + float(np.mean(np.logaddexp(0, zn))))   # class-balanced BCE
    r = dict(t99=t99, bg_pass=float((pn >= t99).mean()), loss=loss)
    if hard_mask is not None: r["hard_pass"] = float((pn[hard_mask] >= t99).mean()) if hard_mask.any() else np.nan
    return r

model = Net2().to(dev); nparam = sum(p.numel() for p in model.parameters())
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
    wh = watch(model, Xw["pos"], Xw["neg"], hard_mask=w_hard); wt = watch(model, Xtr["pos"], Xtr["neg"], tsub["pos"], tsub["neg"])
    rec = dict(epoch=epoch + 1, loss=float(np.mean(losses)), test_loss=wh["loss"], ho=wh, tr=wt, min=(time.time() - t0) / 60); hist.append(rec)
    if (wh["bg_pass"], wh["hard_pass"]) < (best["bg_pass"], best["hard_pass"]):
        best.update(bg_pass=wh["bg_pass"], hard_pass=wh["hard_pass"], epoch=epoch + 1, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); es = 0; tag = " <-- best"
    else: es += 1; tag = f" ({es}/{PATIENCE})"
    print(f"[spec {NAME} {epoch+1:02d}/{EPOCHS}] train loss {rec['loss']:.4f} test loss {wh['loss']:.4f} | test @1%loss: bg pass {wh['bg_pass']:.4f} hard pass {wh['hard_pass']:.4f} "
          f"t99 {wh['t99']:.3f} | train bg {wt['bg_pass']:.4f} | {rec['min']:.1f} min{tag}", flush=True)
    if es >= PATIENCE: print(f"[spec] early stop (best epoch {best['epoch']})", flush=True); break
model.load_state_dict(best["state"])
ci = {c: np.sort(rng.choice(G["calib"][c], size=min(200 if SMOKE else 4000, len(G["calib"][c])), replace=False)) for c in ("pos", "neg")}
model.eval(); lg = []
with torch.no_grad():
    for c in ("pos", "neg"):
        _, xc = load(c, ci[c])
        for i in range(0, len(xc), 1024): lg.append(model(prep(xc[i:i + 1024])).float())
lg = torch.cat(lg); y = torch.cat([torch.ones(len(ci["pos"])), torch.zeros(len(ci["neg"]))]).to(dev)
logT = torch.zeros(1, device=dev, requires_grad=True); o2 = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
def closure():
    o2.zero_grad(); l = F.binary_cross_entropy_with_logits(lg / logT.exp(), y); l.backward(); return l
o2.step(closure); logT.data.clamp_(np.log(0.2), np.log(20.0)); model.temperature.fill_(float(logT.exp().detach()))
print(f"[spec] temperature {float(model.temperature):.3f} (calib pos {len(ci['pos'])} neg {len(ci['neg'])})", flush=True)
full, gi = {}, {}                                    # full TEST set (the lockbox is NOT scored)
for c in ("pos", "neg"):
    g = np.sort(G["test"][c]); ps = []
    for i in range(0, len(g), 16384): ps.append(sig(logits(model, load(c, g[i:i + 16384])[1])))
    full[c] = np.concatenate(ps); gi[c] = g
t99 = float(np.percentile(full["pos"], 1.0)); hm_ = hard("neg", gi["neg"])
summary = dict(name=NAME, cfg=CFG, band=[FLO, FHI], seed=SEED, split="single (lockbox L%20==0 either leg day, test T%20<2 either leg day)", n={r: {c: int(len(G[r][c])) for c in G[r]} for r in G},
               n_lock={c: int((ROLE[c] == "lock").sum()) for c in ROLE}, n_drop={c: int((ROLE[c] == "drop").sum()) for c in ROLE}, params=nparam, best_epoch=best["epoch"],
               temperature=float(model.temperature), t99=t99, bg_pass=float((full["neg"] >= t99).mean()), hard_pass=float((full["neg"][hm_] >= t99).mean()), history=hist)
torch.save(model.state_dict(), f"{OUT}/{NAME}.pt"); json.dump(summary, open(f"{OUT}/{NAME}.json", "w"), indent=1)
np.savez(f"{OUT}/{NAME}_test.npz", p_pos=full["pos"], p_neg=full["neg"], g_pos=gi["pos"], g_neg=gi["neg"])
np.savez(f"{OUT}/split_manifest{'_smoke' if SMOKE else ''}.npz", **{f"role_{c}": ROLE[c].astype(str) for c in ROLE}, N0_pos=N0["pos"], N0_neg=N0["neg"])
print(f"[spec {NAME}] DONE best epoch {best['epoch']}: TEST @1% signal loss -> bg pass {summary['bg_pass']:.4f}, hard pass {summary['hard_pass']:.4f}; {(time.time()-t0)/60:.1f} min", flush=True)

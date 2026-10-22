"""MADGRAV2 glitch arm (a), charter architecture, 3-class head, time-partitioned 2-fold cross-fitting.
Train on fold F (sig/glitch/murky, calib days held aside), watch on the OTHER fold every epoch:
  t99 = 1st percentile of P(sig) over held-out signals (1% signal loss);  pass_murky / pass_glitch = fraction of held-out
  murky / glitch tiles with P(sig) >= t99 (background surviving a 1%-loss cut).  Selection metric = pass_murky (the
  FAR-setting population), tiebreak pass_glitch; patience 10 on it; NEVER on global val AUC.  The same metric on the
  training fold is logged (tail divergence = overfit).  Temperature scaling fit on the training fold's calib days.
Network: 4 stages [32,64,128,256] x 2 (3x3 conv, BN, GELU), 2x2 maxpool, GAP, 256-128-3 with dropout 0.3 (1,206,883 params).
AdamW 3e-4, cosine, wd 1e-4, batch 256, <=60 epochs, grad clip 1.0, AMP.  Env: ARM_FOLD, ARM_SEED, ARM_EPOCHS, ARM_SMOKE."""
import os, sys, json, time, hashlib, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

D = os.environ.get("ARM_DATA", MADGRAV_SCRATCH + "/madgrav2/arm/data"); OUT = MADGRAV_SCRATCH + "/madgrav2/arm/models"; os.makedirs(OUT, exist_ok=True)
FOLD = int(os.environ.get("ARM_FOLD", "0")); SEED = int(os.environ.get("ARM_SEED", "0")); EPOCHS = int(os.environ.get("ARM_EPOCHS", "60"))
SMOKE = os.environ.get("ARM_SMOKE") == "1"; sfx = "_smoke" if SMOKE else ""; TAG = os.environ.get("ARM_TAG", ""); CLASSES = ("sig", "glitch", "murky")
BATCH = 256; LR = 3e-4; WD = 1e-4; PATIENCE = 10; EPOCH_SAMPLES = 3000 if SMOKE else 60000; WATCH_N = 2000 if SMOKE else 30000
dev = "cuda:0" if torch.cuda.is_available() else "cpu"; torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class Arm(nn.Module):
    def __init__(self, chs=(32, 64, 128, 256), ncls=3, drop=0.3):
        super().__init__(); layers = []; c = 1
        for i, co in enumerate(chs):
            layers += [nn.Conv2d(c, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU()]
            if i < len(chs) - 1: layers.append(nn.MaxPool2d(2))
            c = co
        self.trunk = nn.Sequential(*layers); self.head = nn.Sequential(nn.Linear(c, 128), nn.GELU(), nn.Dropout(drop), nn.Linear(128, ncls))
        self.register_buffer("temperature", torch.ones(1))
    def forward(self, x): return self.head(self.trunk(x).mean((2, 3)))
    def psig(self, x): return torch.softmax(self(x) / self.temperature, 1)[:, 0]

def load_fold(f):
    X = {c: np.load(f"{D}/{c}_f{f}{sfx}.npy", mmap_mode="r") for c in CLASSES}
    M = {c: np.load(f"{D}/{c}_f{f}{sfx}_meta.npz", allow_pickle=True) for c in CLASSES}
    return X, M
Xtr, Mtr = load_fold(FOLD); Xho, Mho = load_fold(1 - FOLD)
tr_idx = {c: np.where(~Mtr[c]["calib"].astype(bool))[0] for c in CLASSES}; cal_idx = {c: np.where(Mtr[c]["calib"].astype(bool))[0] for c in CLASSES}
GB = float(os.environ.get("ARM_GLITCH_BALANCE", "0"))   # 0 = uniform over glitch tiles; 0.5 = 1/sqrt(n_label); 1 = uniform over morphology labels
glab = np.asarray(Mtr["glitch"]["label"])[tr_idx["glitch"]]; ulab, inv, cnt = np.unique(glab, return_inverse=True, return_counts=True)
gw = (1.0 / cnt[inv] ** GB); gw /= gw.sum()
if GB > 0: print("[arm] glitch sampling weights per label: " + ", ".join(f"{l}:{100*gw[inv==i].sum():.1f}%" for i, l in enumerate(ulab)), flush=True)
print(f"[arm f{FOLD} s{SEED}] train " + ", ".join(f"{c} {len(tr_idx[c])}" for c in CLASSES) + " | calib " + ", ".join(f"{c} {len(cal_idx[c])}" for c in CLASSES) + " | held-out " + ", ".join(f"{c} {len(Xho[c])}" for c in CLASSES) + f" | {dev}", flush=True)
ho_sub = {c: np.sort(rng.choice(len(Xho[c]), size=min(WATCH_N, len(Xho[c])), replace=False)) for c in CLASSES}
tr_sub = {c: np.sort(rng.choice(tr_idx[c], size=min(WATCH_N // 3, len(tr_idx[c])), replace=False)) for c in CLASSES}

def batch_from(X, idx):
    return torch.from_numpy(np.stack([np.asarray(X[int(i)], np.float32) for i in idx])[:, None])
def scores(model, X, idx, bs=512):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
        for i in range(0, len(idx), bs): out.append(model.psig(batch_from(X, idx[i:i + bs]).to(dev)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])
def watcher(model, X, sub):
    p = {c: scores(model, X[c], sub[c]) for c in CLASSES}; t99 = float(np.percentile(p["sig"], 1.0))
    return dict(t99=t99, pass_glitch=float((p["glitch"] >= t99).mean()), pass_murky=float((p["murky"] >= t99).mean()),
                auc_murky=float(np.mean(p["sig"][:, None] > p["murky"][None, :2000])) if len(p["murky"]) else np.nan), p

model = Arm().to(dev); nparam = sum(p.numel() for p in model.parameters()); print(f"[arm] params {nparam:,d}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); steps_per_epoch = EPOCH_SAMPLES // BATCH
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
hist = []; best = dict(pass_murky=np.inf, pass_glitch=np.inf, epoch=-1, state=None); es = 0; t0 = time.time()
for epoch in range(EPOCHS):
    model.train(); losses = []
    for _ in range(steps_per_epoch):
        xs, ys = [], []
        for ci, c in enumerate(CLASSES):                                   # balanced classes (glitch: morphology-weighted if ARM_GLITCH_BALANCE>0)
            idx = rng.choice(tr_idx[c], size=BATCH // 3, replace=False, p=(gw if c == "glitch" and GB > 0 else None)); xs.append(batch_from(Xtr[c], idx)); ys.append(torch.full((len(idx),), ci, dtype=torch.long))
        x = torch.cat(xs).to(dev, non_blocking=True); y = torch.cat(ys).to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")): loss = F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); losses.append(float(loss.item()))
    wh, _ = watcher(model, Xho, ho_sub); wt, _ = watcher(model, Xtr, tr_sub)
    rec = dict(epoch=epoch + 1, loss=float(np.mean(losses)), ho=wh, tr=wt, lr=float(sched.get_last_lr()[0]), min=(time.time() - t0) / 60); hist.append(rec)
    better = (wh["pass_murky"], wh["pass_glitch"]) < (best["pass_murky"], best["pass_glitch"])
    if better: best.update(pass_murky=wh["pass_murky"], pass_glitch=wh["pass_glitch"], epoch=epoch + 1, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); es = 0; tag = " <-- best"
    else: es += 1; tag = f" ({es}/{PATIENCE})"
    print(f"[arm f{FOLD} s{SEED} {epoch+1:02d}/{EPOCHS}] loss {rec['loss']:.4f} | held-out @1%loss: murky pass {wh['pass_murky']:.4f} glitch pass {wh['pass_glitch']:.4f} auc {wh['auc_murky']:.3f} | "
          f"train-fold: murky {wt['pass_murky']:.4f} glitch {wt['pass_glitch']:.4f} | {rec['min']:.1f} min{tag}", flush=True)
    if es >= PATIENCE: print(f"[arm] early stop (best epoch {best['epoch']})", flush=True); break
model.load_state_dict(best["state"])
# temperature scaling on the training fold's calib days
xs, ys = [], []
for ci, c in enumerate(CLASSES):
    idx = cal_idx[c][: (2000 if not SMOKE else 200)]
    if len(idx): xs.append(batch_from(Xtr[c], idx)); ys.append(torch.full((len(idx),), ci, dtype=torch.long))
if xs:
    model.eval(); x = torch.cat(xs); y = torch.cat(ys).to(dev); logits = []
    with torch.no_grad():
        for i in range(0, len(x), 512): logits.append(model(x[i:i + 512].to(dev)).float())
    logits = torch.cat(logits); logT = torch.zeros(1, device=dev, requires_grad=True); o2 = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
    def closure():
        o2.zero_grad(); l = F.cross_entropy(logits / logT.exp(), y); l.backward(); return l
    o2.step(closure); logT.data.clamp_(np.log(0.2), np.log(20.0)); model.temperature.fill_(float(logT.exp().detach())); print(f"[arm] temperature {float(logT.exp()):.3f} (calib {len(y)} tiles)", flush=True)
# final full held-out scores (for cross-fit evaluation) + save
full = {c: scores(model, Xho[c], np.arange(len(Xho[c]))) for c in CLASSES}; t99 = float(np.percentile(full["sig"], 1.0))
summary = dict(fold=FOLD, seed=SEED, params=nparam, best_epoch=best["epoch"], t99=t99, pass_murky=float((full["murky"] >= t99).mean()), pass_glitch=float((full["glitch"] >= t99).mean()),
               n_heldout={c: int(len(full[c])) for c in CLASSES}, temperature=float(model.temperature), history=hist)
torch.save(model.state_dict(), f"{OUT}/arm{TAG}_f{FOLD}_s{SEED}{sfx}.pt"); json.dump(summary, open(f"{OUT}/arm{TAG}_f{FOLD}_s{SEED}{sfx}.json", "w"), indent=1)
np.savez(f"{OUT}/arm{TAG}_f{FOLD}_s{SEED}{sfx}_heldout.npz", **{f"p_{c}": full[c] for c in CLASSES})
print(f"[arm f{FOLD} s{SEED}] DONE best epoch {best['epoch']}: held-out FULL @1% signal loss -> murky pass {summary['pass_murky']:.4f}, glitch pass {summary['pass_glitch']:.4f}; {(time.time()-t0)/60:.1f} min", flush=True)

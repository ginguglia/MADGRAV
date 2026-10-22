"""MADGRAV2 glitch arm v2: the day-fold cross-fit ensemble trained in m2/arm (arm_def: 2 folds x 10 seeds, 3-class head,
temperature-scaled). For a window at GPS t, fold_of(day(t)) selects the fold whose models NEVER saw that day; g2 = mean
over those 10 models of logit P(sig). Same sign convention as the v1 arm logit g (positive = signal-like), so it enters the
ranking through driver_search_multi.gate(g2, sigma) unchanged. Env: SM_ARM_V2_DIR (models dir), SM_ARM_V2_TAG (_def)."""
import os, glob, json, hashlib, numpy as np, torch, torch.nn as nn
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

DIR = os.environ.get("SM_ARM_V2_DIR", MADGRAV_SCRATCH + "/madgrav2/arm/models"); TAG = os.environ.get("SM_ARM_V2_TAG", "_def")
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
def fold_of_day(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(), 16) % 2
class ArmV2:
    def __init__(self, device="cuda:0"):
        self.dev = device; self.models = {0: [], 1: []}; self.meta = []
        for js in sorted(glob.glob(f"{DIR}/arm{TAG}_f*_s*.json")):
            if "smoke" in js: continue
            S = json.load(open(js)); m = Arm().to(device); m.load_state_dict(torch.load(js.replace(".json", ".pt"), map_location=device)); m.eval()
            self.models[S["fold"]].append(m); self.meta.append(dict(fold=S["fold"], seed=S["seed"], t99=S["t99"], best_epoch=S["best_epoch"]))
        assert self.models[0] and self.models[1], f"arm v2 ensemble incomplete in {DIR} ({TAG})"
        self.n = {f: len(v) for f, v in self.models.items()}
    def g2_from_qt(self, qt, gps, bs=512):
        """qt: (N,256,128) numpy; gps: (N,) window GPS. Returns mean logit P(sig) from the models of the OTHER fold."""
        qt = np.asarray(qt, np.float32); gps = np.asarray(gps, float); out = np.empty(len(qt))
        folds = np.array([fold_of_day(d) for d in (gps // 86400).astype(int)])
        with torch.no_grad():
            for f in (0, 1):
                idx = np.where(folds == f)[0]
                if not len(idx): continue
                acc = np.zeros(len(idx))
                for m in self.models[1 - f]:
                    for i in range(0, len(idx), bs):
                        x = torch.from_numpy(qt[idx[i:i + bs]][:, None]).to(self.dev); p = m.psig(x).float().cpu().numpy().astype(np.float64)
                        acc[i:i + bs] += np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
                out[idx] = acc / len(self.models[1 - f])
        return out

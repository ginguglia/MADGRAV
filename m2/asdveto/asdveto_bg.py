"""Background side of the local-ASD veto check (2026-10-07).

Takes the loudest CNN-passing time-slid pairs of a run (the net-sigma FAR setters: is_glitch False, cnn_lm > SM_NETSIG_LM_MIN,
ranked by net = (sigma_H + sigma_L)/sqrt(2)) from survivors_bg_s3rank.json and re-scores each under the local +-64 s median-Welch
ASD of ITS OWN leg (two-leg version of asd_consistency.recompute_local): net_loc, June hm_loc / lm_loc, and S3 under the local ASD
(same crop path as inject._cnn_s3_from_wh). Frozen scores come from the survivors file and the s3 bank. Output: one npz per run.
Env: the chain EXP of the run (SM_STRAIN, SM_PREP, ...), SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1, AV_K (default 400), AV_OUT.
"""
import os, sys, json, time, hashlib, numpy as np
ROOT = os.environ["MADGRAV_ROOT"]
for p in ("search_mode", "improved", "spectrogram_cascade"):
    if os.path.join(ROOT, p) not in sys.path: sys.path.insert(0, os.path.join(ROOT, p))
import torch
import driver_blindscan as B
import asd_consistency as AV
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

RUN = os.environ["SM_RUN"]; SC = os.environ.get("M2SC", MADGRAV_SCRATCH + "/madgrav2")
K = int(os.environ.get("AV_K", "400")); LM_MIN = float(os.environ.get("SM_NETSIG_LM_MIN", "0.0382"))
OUT = os.environ.get("AV_OUT", f"{SC}/asdveto"); os.makedirs(OUT, exist_ok=True)
SRC = f"{SC}/search_out_{RUN}_m2_gx_off5/survivors_bg_s3rank.json"

def log(*a): print(f"[asdveto-bg {RUN}]", *a, flush=True)

# ---- S3 ensemble (== inject._s3_ens / score_bank_s3) ----
S3_FLO, S3_FHI = 20.0, 300.0
class _Net2(torch.nn.Module):
    def __init__(s_):
        nn = torch.nn; super().__init__(); ch = [2, 16, 32, 64, 128]
        s_.b = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
        s_.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)); s_.register_buffer("temperature", torch.ones(1))
    def forward(s_, x):
        for bl in s_.b: x = bl(x)
        return s_.h(x).squeeze(-1)
MD = os.environ.get("SM_S3_MODELS", MADGRAV_SCRATCH + "/madgrav2_spec/models"); ENS = {}
for f in (0, 1):
    ms = []
    for sd in range(5):
        m = _Net2().to(B.DEV); m.load_state_dict(torch.load(f"{MD}/spec_hm_f{f}_s{sd}_nfull.pt", map_location=B.DEV)); m.eval(); ms.append(m)
    ENS[f] = ms
def fold_of(g): return int(hashlib.sha1(str(int(g // 86400)).encode()).hexdigest(), 16) % 2
def s3_from_wh(whH, whL, gH, gL):
    magH, t0, T = B._fullmag(whH); magL, _, _ = B._fullmag(whL); a = int(round((t0 - 14) / 128 * T))
    x = torch.from_numpy(np.stack([B._crop(magH, a, T, S3_FLO, S3_FHI), B._crop(magL, a, T, S3_FLO, S3_FHI)])[None]).float().to(B.DEV)
    fH, fL = fold_of(gH), fold_of(gL)
    with torch.no_grad():
        sc = lambda f: float(np.mean([torch.sigmoid(m(x) / m.temperature).item() for m in ENS[f]]))
        return sc(1 - fH) if fH == fL else 0.5 * (sc(0) + sc(1))      # fold-pure pair -> the other fold's models; mixed -> mean of both

def sigma_det(pipe, seg, det, idx):
    mu, sd = (pipe.norm["muH"], pipe.norm["sdH"]) if det == "H1" else (pipe.norm["muL"], pipe.norm["sdL"])
    wh = pipe._whiten(B._win(seg, det, idx)[None, :], det)
    return float((pipe._recon(AV._build_qt(pipe, wh)).reshape(-1)[0] - mu) / sd), wh

def recompute_two_leg(segH, iH, segL, iL, gH, gL):
    pipe = B.cpipe(); sav = {d: pipe.asd[d] for d in ("H1", "L1")}
    try:
        pipe.asd["H1"] = AV._local_asds(segH, iH)["H1"]; pipe.asd["L1"] = AV._local_asds(segL, iL)["L1"]
        sH, whH = sigma_det(pipe, segH, "H1", iH); sL, whL = sigma_det(pipe, segL, "L1", iL)
        hm, lm = B.cnn_hm_lm(segH, iH, segL, iL)
        s3 = s3_from_wh(whH, whL, gH, gL)
    finally:
        pipe.asd["H1"], pipe.asd["L1"] = sav["H1"], sav["L1"]
    return dict(net_loc=(sH + sL) / np.sqrt(2.0), sH_loc=sH, sL_loc=sL, hm_loc=float(hm), lm_loc=float(lm), s3_loc=float(s3))

def main():
    t0 = time.time(); log("loading", SRC); S = json.load(open(SRC))["survivors"]; log(f"{len(S)} survivors in {time.time()-t0:.0f}s")
    rows = [r for r in S if not r.get("is_glitch") and float(r.get("cnn_lm") or 0) > LM_MIN]
    for r in rows: r["net"] = (r["sigma_H"] + r["sigma_L"]) / np.sqrt(2.0)
    rows.sort(key=lambda r: -r["net"]); rows = rows[:K]; del S
    log(f"{len(rows)} loudest CNN-passing (lm>{LM_MIN}) slid pairs; net range {rows[-1]['net']:.2f}-{rows[0]['net']:.2f}")
    z = np.load(f"{SC}/s3_off5/s3bank/s3_{RUN}.npz", allow_pickle=True)
    s3map = {(str(a), int(b), str(c), int(d)): float(e) for a, b, c, d, e in zip(z["H1_seg"], z["iH"], z["L1_seg"], z["iL"], z["s3"])}
    R = {k: [] for k in ("H1_seg", "iH", "L1_seg", "iL", "gps_H", "gps_L", "net", "sigma_H", "sigma_L", "cnn_hm", "cnn_lm", "s3", "loglr",
                         "net_loc", "sH_loc", "sL_loc", "hm_loc", "lm_loc", "s3_loc")}
    t1 = time.time()
    for n, r in enumerate(rows):
        try:
            loc = recompute_two_leg(r["H1_seg"], int(r["iH"]), r["L1_seg"], int(r["iL"]), float(r["gps_H"]), float(r["gps_L"]))
        except Exception as e:
            log(f"row {n} failed: {e!r}"); loc = dict(net_loc=np.nan, sH_loc=np.nan, sL_loc=np.nan, hm_loc=np.nan, lm_loc=np.nan, s3_loc=np.nan)
        for k in ("H1_seg", "iH", "L1_seg", "iL", "gps_H", "gps_L", "net", "sigma_H", "sigma_L", "cnn_hm", "cnn_lm", "loglr"): R[k].append(r.get(k))
        R["s3"].append(s3map.get((r["H1_seg"], int(r["iH"]), r["L1_seg"], int(r["iL"])), np.nan))
        for k in ("net_loc", "sH_loc", "sL_loc", "hm_loc", "lm_loc", "s3_loc"): R[k].append(loc[k])
        if n % 25 == 0:
            log(f"{n+1}/{len(rows)} net {r['net']:.2f}->{loc['net_loc']:.2f} lm {r['cnn_lm']:.2f}->{loc['lm_loc']:.2f} s3 {R['s3'][-1]:.2f}->{loc['s3_loc']:.2f} ({(time.time()-t1)/60:.1f} min)")
    np.savez(f"{OUT}/bg_{RUN}.npz", **{k: np.array(v) for k, v in R.items()})
    A = {k: np.array(R[k], float) for k in ("net", "net_loc", "cnn_hm", "cnn_lm", "hm_loc", "lm_loc", "s3", "s3_loc")}
    cnn_veto = np.maximum(A["hm_loc"], A["lm_loc"]) < 0.5; net_veto = A["net_loc"] < 3.0
    log(f"SUMMARY K={len(rows)}: CNN clause vetoes {cnn_veto.mean()*100:.1f}%, net<3 clause {net_veto.mean()*100:.1f}%, either {(cnn_veto|net_veto).mean()*100:.1f}%; "
        f"S3 frozen>=0.5357 {np.nanmean(A['s3']>=0.5357)*100:.1f}% -> local {np.nanmean(A['s3_loc']>=0.5357)*100:.1f}%  -> {OUT}/bg_{RUN}.npz")
if __name__ == "__main__": main()

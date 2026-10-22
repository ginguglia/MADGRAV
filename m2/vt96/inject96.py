"""vt96 injection stage (2026-10-19): search_mode/inject.py of MADGRAV2-gx run UNCHANGED except for two overrides.

1. inject._cnn_s3_from_wh (the SM_INJ_S3=1 path that scores the June HM/LM gate and the S3 specialist on one QT magnitude) hard-codes
   the Grad-CAM crop start a=(t0H-14)/128*T. The frozen version-2 recipe places the CNN/S3 crop with driver_blindscan.crop_start()
   (SM_CROP=coinc: argmax of the H1 x L1 band-summed 20-140 Hz native-QT power; DECISIONS_PLACEMENT.md 2026-10-10), which is what
   B.cnn_from_wh, _pb_cnn_precompute and the production S3 bank (madgrav2_spec/s3_crop) use. Under SM_CROP=coinc the production
   inject.py would either trip its own tripwire or score the gate at the wrong placement. The override scores hm/lm/s3 at crop_start()
   (same calls, same models, same S3 fold rule) and keeps the tripwire against B.cnn_from_wh (first SM_INJ96_NTRIP, default 64).
2. decision 2026-10-19 (production-form net-sigma guard): the SAME call also scores the June HM/LM at the Grad-CAM placement
   (a_gc = round((t0H-14)/128*T), the SM_CROP=gradcam rule of crop_start(); same whitened data, same QT magnitudes, same band crops, same
   models) -> extra columns cnn_hm_gc, cnn_lm_gc (one CNN forward per band per injection; no S3 at the Grad-CAM crop). The columns are
   appended to the saved npz after inject.main() returns (inject.py's R dict has fixed keys). The local-ASD re-score (_veto_local) calls
   the same function; its Grad-CAM values are not stored (the guard uses the frozen-ASD scores, as in production).
Production code is not edited. Usage (cwd = MADGRAV_ROOT): python m2/vt96/inject96.py --event <name>   (env as inject.py)."""
import os, sys
MADGRAV_ROOT = os.environ["MADGRAV_ROOT"]
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

for _p in ("search_mode", "improved", "spectrogram_cascade"):
    _ap = os.path.join(MADGRAV_ROOT, _p)
    if _ap not in sys.path: sys.path.insert(0, _ap)
if os.environ.get("SM_CROP") != "coinc": raise SystemExit("[inj96] SM_CROP must be 'coinc' (frozen version-2 recipe)")
if os.environ.get("SM_INJ_S3") != "1" or os.environ.get("SM_INJ_CNN") != "1": raise SystemExit("[inj96] needs SM_INJ_CNN=1 SM_INJ_S3=1")
import numpy as np, torch
import inject as inj
NTRIP = int(os.environ.get("SM_INJ96_NTRIP", "64")); _ntrip = [0]; _in_veto = [False]; GC = []   # GC: (hm_gc, lm_gc) per main-path call, in R-row order

def _cnn_s3_from_wh(B, whH, whL, gps, tripwire):
    """(hm,lm,s3,fold) at crop_start() (SM_CROP=coinc) exactly as B.cnn_from_wh; also June HM/LM at the Grad-CAM placement -> GC."""
    magH, t0, T = B._fullmag(whH); magL, _, _ = B._fullmag(whL); a = B.crop_start(magH, magL, t0, T)
    a_gc = int(round((t0 - 14) / 128 * T))                                   # == crop_start() under SM_CROP=gradcam (frozen placement)
    def stk(a_, flo, fhi): return np.stack([B._crop(magH, a_, T, flo, fhi), B._crop(magL, a_, T, flo, fhi)])
    with torch.no_grad():
        hm = float(torch.sigmoid(B.cnet()(torch.from_numpy(stk(a, B.FLO, B.FHI)[None]).float().to(B.DEV))).item())
        lm = float(torch.sigmoid(B.lmnet()(torch.from_numpy(stk(a, B.LFLO, B.LFHI)[None]).float().to(B.DEV))).item())
        f = inj._s3_fold_of_gps(gps); x = torch.from_numpy(stk(a, inj.S3_FLO, inj.S3_FHI)[None]).float().to(B.DEV)
        s3 = float(np.mean([torch.sigmoid(m(x) / m.temperature).item() for m in inj._s3_ens(B)[1 - f]]))
        if not _in_veto[0]:
            hm_gc = float(torch.sigmoid(B.cnet()(torch.from_numpy(stk(a_gc, B.FLO, B.FHI)[None]).float().to(B.DEV))).item())
            lm_gc = float(torch.sigmoid(B.lmnet()(torch.from_numpy(stk(a_gc, B.LFLO, B.LFHI)[None]).float().to(B.DEV))).item())
            GC.append((hm_gc, lm_gc))
    if tripwire or _ntrip[0] < NTRIP:
        _ntrip[0] += 1
        hm0, lm0 = B.cnn_from_wh(whH, whL)
        if max(abs(hm - hm0), abs(lm - lm0)) > 1e-6:
            raise SystemExit(f"[inj96] crop TRIPWIRE: June CNN at crop_start differs from B.cnn_from_wh ({hm:.6f},{lm:.6f}) vs ({hm0:.6f},{lm0:.6f})")
    return hm, lm, s3, f

_veto_local0 = inj._veto_local
def _veto_local(*a, **k):
    _in_veto[0] = True
    try: return _veto_local0(*a, **k)
    finally: _in_veto[0] = False

inj._cnn_s3_from_wh = _cnn_s3_from_wh; inj._veto_local = _veto_local
print(f"[inj96] inject.py {os.path.join(MADGRAV_ROOT,'search_mode','inject.py')} with crop_start() override + Grad-CAM-crop columns, SM_CROP={os.environ['SM_CROP']}, "
      f"ASDVETO={os.environ.get('SM_INJ_ASDVETO','0')} LOCAL_ALL={os.environ.get('SM_INJ_LOCAL_ALL','0')} tripwire on first {NTRIP}", flush=True)
if __name__ == "__main__":
    inj.main()
    name = sys.argv[sys.argv.index("--event") + 1]; path = f"{inj.OUT}/{name}_inj.npz"
    if os.path.exists(path):                                                 # main() returns without saving for events not in the configs
        z = dict(np.load(path)); n = len(z["net"])
        if len(GC) != n: raise SystemExit(f"[inj96] Grad-CAM column count {len(GC)} != {n} injections -- NOT appended, file left as written")
        z["cnn_hm_gc"] = np.array([g[0] for g in GC]); z["cnn_lm_gc"] = np.array([g[1] for g in GC]); np.savez(path, **z)
        print(f"[inj96] appended cnn_hm_gc/cnn_lm_gc ({n}) to {path}; tripwire comparisons made: {_ntrip[0]} (all agreed)", flush=True)

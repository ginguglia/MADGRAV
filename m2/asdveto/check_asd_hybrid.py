"""Band-matched variant of the local-ASD veto (2026-10-07): the local +/-64 s ASD enters only where it is HIGHER than the frozen reference
ASD (hybrid = max(frozen, local) on the frozen grid), so the frozen suppression below ~50 Hz is never undone and the clause tests ASD
mismatch only in the band the models see. Re-scores the 26 census rows (8 catalogue, 18 non-cat) under local (reproduces the census) and
hybrid ASDs: net, June hm/lm. Env: the chain EXP of the run + SM_DEV=cpu BLIND_DEV=cpu SM_ALLOW_CPU=1. Usage: check_asd_hybrid.py <run> <out.json>"""
import os, sys, json, numpy as np
ROOT = os.environ["MADGRAV_ROOT"]
for p in ("search_mode", "improved", "spectrogram_cascade"):
    if os.path.join(ROOT, p) not in sys.path: sys.path.insert(0, os.path.join(ROOT, p))
from gwpy.frequencyseries import FrequencySeries
import driver_blindscan as B, asd_consistency as AV
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

def main():
    RUN, OUT = sys.argv[1], sys.argv[2]
    rows = [r for r in json.load(open(MADGRAV_ROOT + "/m2/asdveto/census_rows_26.json")) if r["run"] == RUN]
    pipe = B.cpipe(); FRZ = {d: pipe.asd[d] for d in ("H1", "L1")}
    def hybrid(frz, loc):
        f = np.asarray(frz.frequencies.value); li = np.interp(f, np.asarray(loc.frequencies.value), np.asarray(loc.value), left=np.inf, right=np.inf)
        return FrequencySeries(np.maximum(np.asarray(frz.value), np.where(np.isfinite(li), li, 0.0)), f0=float(f[0]), df=float(f[1] - f[0]))
    def score(seg, idx, asds):
        sav = {d: pipe.asd[d] for d in ("H1", "L1")}
        try:
            for d in ("H1", "L1"): pipe.asd[d] = asds[d]
            net = AV._netsigma(pipe, seg, idx); hm, lm = B.cnn_hm_lm(seg, idx, seg, idx)
        finally:
            for d in ("H1", "L1"): pipe.asd[d] = sav[d]
        return dict(net=float(net), hm=float(hm), lm=float(lm))
    out = []
    for r in rows:
        idx = AV.idx_of(r); loc = AV._local_asds(r["seg"], idx)
        diag = {}
        for d in ("H1", "L1"):
            f = np.asarray(FRZ[d].frequencies.value); li = np.interp(f, np.asarray(loc[d].frequencies.value), np.asarray(loc[d].value))
            ratio = np.asarray(FRZ[d].value) / np.maximum(li, 1e-30)
            diag[d] = dict(ratio_20_50=float(np.median(ratio[(f >= 20) & (f < 50)])), ratio_50_500=float(np.median(ratio[(f >= 50) & (f < 500)])))
        L = score(r["seg"], idx, loc); Hy = score(r["seg"], idx, {d: hybrid(FRZ[d], loc[d]) for d in ("H1", "L1")})
        rec = dict(r, idx=idx, local=L, hybrid=Hy, asd_ratio_frozen_over_local=diag)
        out.append(rec)
        print(f"[hybrid {RUN}] {r['name']:18s} {r['kind']:9s} frozen net {r['net']:.2f} hm {r['cnn_hm']:.2f} lm {r['cnn_lm']:.2f} | local net {L['net']:.2f} hm {L['hm']:.2f} lm {L['lm']:.2f} "
              f"| hybrid net {Hy['net']:.2f} hm {Hy['hm']:.2f} lm {Hy['lm']:.2f} | frz/loc ASD 20-50 Hz H {diag['H1']['ratio_20_50']:.1f} L {diag['L1']['ratio_20_50']:.1f}, 50-500 Hz H {diag['H1']['ratio_50_500']:.2f} L {diag['L1']['ratio_50_500']:.2f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)

if __name__ == "__main__":
    main()

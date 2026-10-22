"""MADGRAV2.1 gx validation, step 1 (CPU): compute the guided second-leg excess (guided_excess.py, production code path) + the
re-whitened coherence + the LR scalar features for EVERY banked background survivor of a finished prune (shard_*.npz, columns
loglr, ai, iH, bi, iL) and for the zero-lag triggers of a blindscan.json (si, idx). Rows come from driver_search_multi.series_for
(SM_COH_REWHITEN=1 -> flattened), exactly as the prune sees them. Output npz -> step 2 (LR refit with gx, re-ranking, matched-
efficiency comparison) runs in seconds. Env: the run's chain EXP block (+ SM_GX=1, SM_DEV=cpu); SHARDS, TRIGJSON, OUTNPZ, NPROC."""
import os, sys, glob, json, time, numpy as np
from multiprocessing import Pool
ROOT = os.environ["MADGRAV_ROOT"]; sys.path[:0] = [f"{ROOT}/search_mode", f"{ROOT}/improved", f"{ROOT}/spectrogram_cascade"]; os.chdir(ROOT)
import driver_search_multi as M, guided_excess as GX
assert GX.ON, "SM_GX=1 required"
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC = os.environ.get("SCRATCH", MADGRAV_SCRATCH); t0 = time.time()
SHARDS = os.environ["SHARDS"]; TRIGJSON = os.environ.get("TRIGJSON", ""); OUTNPZ = os.environ["OUTNPZ"]; NPROC = int(os.environ.get("NPROC", "32"))
TMP = OUTNPZ + "_parts"; os.makedirs(TMP, exist_ok=True)
def log(s): print(f"[gx-rows {(time.time()-t0)/60:5.1f} m] {s}", flush=True)
avail = [(a, nm) for a, b, d, nm in M.SEG["segments"] if M.have(a, nm)]; assert len(avail) == len(M.SEG["segments"]), "streams incomplete"
segs = [M.load_seg(a, nm) for a, nm in avail]; fold = np.array([i % 2 for i in range(len(segs))]); log(f"{len(segs)} segments")
bg = []
for f in sorted(glob.glob(f"{SHARDS}/shard_*.npz")):
    z = np.load(f); bg += [z["bg0"], z["bg1"]]
bg = np.concatenate([b for b in bg if len(b)]); log(f"{len(bg)} background rows from {len(glob.glob(f'{SHARDS}/shard_*.npz'))} shards")
kind = np.zeros(len(bg), np.int8)                                            # 0 = time-slid background
ai = bg[:, 1].astype(np.int64); iH = bg[:, 2].astype(np.int64); bi = bg[:, 3].astype(np.int64); iL = bg[:, 4].astype(np.int64); ll_old = bg[:, 0].copy()
if TRIGJSON:                                                                 # zero-lag triggers: (si, idx) pairs, same segment order
    T = json.load(open(TRIGJSON))["triggers"]; si = np.array([t["si"] for t in T], np.int64); ix = np.array([t["idx"] for t in T], np.int64)
    for t, s_, i_ in zip(T, si, ix): assert segs[s_]["name"] == t["seg"] and abs(segs[s_]["gps"][i_] - t["gps"]) < 0.6, (t["seg"], segs[s_]["name"])
    ai = np.concatenate([ai, si]); iH = np.concatenate([iH, ix]); bi = np.concatenate([bi, si]); iL = np.concatenate([iL, ix])
    ll_old = np.concatenate([ll_old, [t["loglr"] for t in T]]); kind = np.concatenate([kind, np.ones(len(T), np.int8)]); log(f"{len(T)} zero-lag triggers appended")
# ---- per-(segment, detector) distinct rows: prepare() once per key in worker processes ----
keysH = {int(s): np.unique(iH[ai == s]) for s in np.unique(ai)}; keysL = {int(s): np.unique(iL[bi == s]) for s in np.unique(bi)}
jobs = [(s, "H1", w) for s, w in keysH.items()] + [(s, "L1", w) for s, w in keysL.items()]
log(f"{len(jobs)} (seg,det) keys, {sum(len(w) for w in keysH.values())} distinct H rows, {sum(len(w) for w in keysL.values())} distinct L rows")
def work(job):
    s, det, w = job; out = f"{TMP}/{s}_{det}.npz"
    if os.path.exists(out): return out
    rows = M.series_for(segs[s]["name"], det, w.tolist()); X = np.stack([rows[int(k)] for k in w])
    P = GX.prepare(X); np.savez(out + ".tmp.npz", w=w, rows=X.astype(np.float32), Q=P["Q"], qmax=P["qmax"], idx=P["idx"]); os.replace(out + ".tmp.npz", out); return out
with Pool(NPROC) as pool:
    for k, _ in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
        if k % 200 == 0: log(f"prepared {k}/{len(jobs)} keys")
log("all keys prepared; assembling")
def assemble(keys, det):
    Q, qmax, idx, rows, off = [], [], [], [], {}; n = 0
    for s in sorted(keys):
        z = np.load(f"{TMP}/{s}_{det}.npz"); off[s] = (n, z["w"]); n += len(z["w"]); Q.append(z["Q"]); qmax.append(z["qmax"]); idx.append(z["idx"]); rows.append(z["rows"])
    return dict(Q=np.concatenate(Q), qmax=np.concatenate(qmax), idx=np.concatenate(idx)), np.concatenate(rows), off
PH, RH, offH = assemble(keysH, "H1"); PL, RL, offL = assemble(keysL, "L1"); log(f"assembled: H {len(RH)} rows, L {len(RL)} rows ({RH.nbytes/1e9+RL.nbytes/1e9:.1f} GB rows)")
def gidx(off, seg, w):
    g = np.empty(len(seg), np.int64)
    for s in np.unique(seg):
        m = seg == s; base, ws = off[int(s)]; g[m] = base + np.searchsorted(ws, w[m])
    return g
gH = gidx(offH, ai, iH); gL = gidx(offL, bi, iL)
gx = GX.gx_pairs(PH, gH, PL, gL); log(f"gx done: median {np.median(gx):.2f} p99 {np.percentile(gx,99):.2f} max {gx.max():.1f}")
coh = np.empty(len(gx))
for c0 in range(0, len(gx), 20000): coh[c0:c0+20000] = M.coh_vec(RH[gH[c0:c0+20000]].astype(np.float64), RL[gL[c0:c0+20000]].astype(np.float64))
log(f"coh done: median {np.median(coh):.3f}")
def gv(key, s, w):
    out = np.empty(len(s))
    for u in np.unique(s): m = s == u; out[m] = segs[int(u)][key][w[m]]
    return out
np.savez(OUTNPZ, kind=kind, loglr_old=ll_old, ai=ai, iH=iH, bi=bi, iL=iL, fold=fold[ai], sH=gv("sH", ai, iH), sL=gv("sL", bi, iL), cH=gv("cH", ai, iH), cL=gv("cL", bi, iL),
         gH=gv("gH", ai, iH), gL=gv("gL", bi, iL), gpsH=gv("gps", ai, iH), gpsL=gv("gps", bi, iL), coh=coh, gx=gx, gx_params=json.dumps(GX.PARAMS))
log(f"saved {OUTNPZ} ({len(gx)} rows: {int((kind==0).sum())} bg + {int((kind==1).sum())} zero-lag)")

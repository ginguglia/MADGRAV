"""STAGE-2 CNN PRECOMPUTE (drop-in for driver_blindscan.py).

Bit-identical replacement for the lazy per-pair `cnn_hm_lm` louder-bg scoring.
Builds, for a given set of (ai,iH,bi,iL) pairs, a dict cnn_cache[(ai,iH,bi,iL)]=(hm,lm)
by (1) deduping per-detector QT magnitudes -- magH/t0/T by (ai,iH), magL by (bi,iL) --
computing each _fullmag ONCE, then (2) BATCHing the HM/LM CNN forwards.

Reproduces driver_blindscan's _fullmag / _crop / stacking VERBATIM so the CNN input
tensors are byte-for-byte identical to the per-pair path. The shared time placement
`a=round((t0-14)/128*T)` uses H1's t0 and magH's T -- exactly as the lazy path.

cam_t0_batch, GlitchArm, Net2 (cnet/lmnet) are all in eval() -> BatchNorm uses running
stats -> per-sample / batch-composition independent. Verified by _pb_cnn_harness.py
(max|hm diff|=max|lm diff|=0.0 over real bg pairs).

Usage in STAGE-2 (driver_blindscan.py), replacing the lazy surv_cnn/cnn_cache:

    import _pb_cnn_precompute as PB
    need_pairs=set()
    for i in cand_idx:
        if fgt[i].get("far_lr_per_yr") is not None: need_pairs.update(fgt[i]["_louder"])
    # net-sigma channel families >= each netcand's net, same fold (mirror STAGE-1b selection):
    if NETSIG_FLOOR>0:
        _fams=sorted(famN.values(),key=lambda r:-r[0])
        for i in (j for j in cand_idx if fgt[j]["net"]>=NETSIG_FLOOR):
            g=fgt[i]["fold"]
            for r in _fams:
                if r[0]>=fgt[i]["net"] and fold[r[1]]==g: need_pairs.add((r[1],r[2],r[3],r[4]))
    cnn_cache=PB.precompute_cnn(need_pairs, segs,
                                cpipe, carm, cnet, lmnet,
                                _win, _fullmag_batch=None)  # see precompute_cnn docstring
    def surv_cnn(ai,iH,bi,iL):
        hm,lm=cnn_cache[(ai,iH,bi,iL)]; return max(hm,lm)

The module re-derives the per-detector magnitudes itself (it does not call the lazy
_fullmag) so it can batch the whiten + gradcam + qt + forwards. It imports the SAME
module-level objects (FS,WN,WT,FLO,FHI,LFLO,LFHI, mr,ip,DS,torch,np) from
driver_blindscan at call time to stay locked to that file's constants.
"""
import os
import numpy as np, torch


def precompute_cnn(need_pairs, segs, cpipe, carm, cnet, lmnet, _win, bs=2048):
    """Return cnn_cache: {(ai,iH,bi,iL):(hm,lm)} for every pair in need_pairs.

    need_pairs : iterable of (ai,iH,bi,iL) int tuples (ai/bi seg indices into `segs`).
    segs       : the driver's segs list (segs[ai]["name"] -> npz basename).
    cpipe,carm,cnet,lmnet : the driver's lazy-singleton accessors (already eval()).
    _win       : the driver's _win(name,det,idx) raw-window loader (-> [WN] float32).
    bs         : CNN forward batch size.
    """
    import driver_blindscan as B   # lock to the LIVE constants/helpers in the driver
    FS, WN, WT = B.FS, B.WN, B.WT
    FLO, FHI, LFLO, LFHI = B.FLO, B.FHI, B.LFLO, B.LFHI
    mr, ip, DS = B.mr, B.ip, B.DS
    DEV = B.DEV

    pairs = list(dict.fromkeys((int(a), int(h), int(b), int(l)) for a, h, b, l in need_pairs))
    out = {}
    if not pairs:
        return out

    # ---- (1) dedup per-detector windows ----
    h_keys = list(dict.fromkeys((a, h) for a, h, b, l in pairs))   # (ai,iH)
    l_keys = list(dict.fromkeys((b, l) for a, h, b, l in pairs))   # (bi,iL)

    # MEMORY: the old code materialized ALL h_keys' full float64 QT magnitudes at once into magH_d
    # (~17.3k keys x 10.25 MB ~= 177 GB) -> OOM/hang on a 7-GB-swap box. The fix below NEVER holds the
    # full set: it (a) stores magnitudes as float32 (bit-identical through _crop -- the min-max norm
    # there runs in float32 regardless, verified 0.0-diff), (b) retains only the SMALL detector side,
    # and (c) STREAMS the large side in chunks, scoring each chunk's pairs immediately then dropping it.
    # Chunking is bit-identical: build_qt/_whiten are per-sample-independent (BatchNorm eval) and
    # cam_t0_batch is already internally chunked at 256, so the per-key mag/t0/T do NOT depend on which
    # other keys share the call. Verified empirically (chunked 4+4 == all-8, exact array equality).

    def _fullmag_batch(keys, det):
        """Batched _fullmag for a list of (si,idx). Returns:
           mags : list of [F,T] float32 arrays (float32 is bit-identical through _crop)
           t0s  : np.int array of gradcam t0 (only meaningful/used for H1)
        Reproduces _fullmag verbatim: whiten -> build_qt -> cam_t0_batch (gradcam t0),
        and SEPARATELY center_crop_waveforms -> _compute_qt_image_worker for the mag.
        BatchNorm in eval -> per-sample independent -> batching is bit-identical."""
        pipe = cpipe()
        raw = np.stack([_win(segs[s]["name"], det, i) for s, i in keys])   # [n,WN] float32
        wh = pipe._whiten(raw, det)                                        # [n,WN] (batched; per-row identical)
        # gradcam t0 over the min-max build_qt tiles (256x128), batched exactly like _fullmag
        qt = DS.build_qt(pipe, wh)                                         # [n,256,128]
        t0s = mr.cam_t0_batch(carm(), qt, DEV)                            # [n] ints
        # full QT magnitudes for the CNN crop (NOT the zoomed tile): center-crop then worker.
        # PARALLEL: this gwpy q_transform per key is THE CPU cost; the old serial list comprehension ran
        # it single-threaded in the parent (~2/64 cores). Route it through the SAME forkserver pool
        # (DS.pool(), SM_QT_WORKERS) via an ORDER-PRESERVING pool.map so mags[k] <-> keys[k] (required for
        # bit-identity). The worker is byte-identical (same fn/args); only WHERE it runs changes.
        # float32 (was float): _crop copies m[:,sa:sb] into a float32 buffer and min-max-norms in
        # float32, so a float32 mag yields a byte-identical crop (verified). Halves the retained bytes.
        qi = ip.center_crop_waveforms(wh, sample_rate=FS, context_seconds=pipe.ctx)
        args = [(qi[k], FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0) for k in range(len(keys))]
        if len(args) <= 1:                                                  # tiny/probe call: skip pool overhead
            raw_mags = [ip._compute_qt_image_worker(a) for a in args]
        else:
            cs = max(1, len(args) // (DS._QT_WORKERS * 4))                 # even load, bounded buffering
            raw_mags = DS.pool().map(ip._compute_qt_image_worker, args, chunksize=cs)
        mags = [np.asarray(m, np.float32) for m in raw_mags]               # order preserved by map
        return mags, np.asarray(t0s, int)

    # ---- _crop verbatim (copy of driver_blindscan._crop) ----
    def _crop(mag, a, T, flo, fhi):
        fax = ip.QTRANSFORM_FRANGE[0] + 0.5 * np.arange(mag.shape[0]); m = mag[(fax >= flo) & (fax <= fhi)]
        o = np.zeros((m.shape[0], WT), dtype=np.float32); sa = max(0, a); sb = min(T, a + WT)
        if sb > sa: o[:, sa - a:sb - a] = m[:, sa:sb]
        return ((o - o.min()) / (o.max() - o.min() + 1e-9)).astype(np.float32)

    net_hm = cnet(); net_lm = lmnet()

    def _score_one(magH, t0H, TH, magL):
        """The VERBATIM per-pair lazy forward (single-sample [None] + .item()). Kept exactly as the
        original loop: cuDNN conv-algo is batch-composition dependent, so DO NOT batch the forward."""
        aa = int(round((t0H - 14) / 128 * TH))
        hm_in = np.stack([_crop(magH, aa, TH, FLO, FHI), _crop(magL, aa, TH, FLO, FHI)])
        lm_in = np.stack([_crop(magH, aa, TH, LFLO, LFHI), _crop(magL, aa, TH, LFLO, LFHI)])
        with torch.no_grad():
            xh = torch.from_numpy(hm_in[None]).float().to(DEV)
            xl = torch.from_numpy(lm_in[None]).float().to(DEV)
            hm = float(torch.sigmoid(net_hm(xh)).item())
            lm = float(torch.sigmoid(net_lm(xl)).item())
        return hm, lm

    # ---- (2) MEMORY-BOUNDED streaming. Retain the SMALL side fully; chunk the LARGE side. ----
    # Measure per-key magnitude bytes from a single key so the chunk size is sized to the REAL F*T*4
    # (not assumed). Budget is an env knob (default 50 GB); chunk size C is solved from it.
    MAXGB = float(os.environ.get("SM_PRECOMPUTE_MAXGB", "50"))
    GB = 1024.0 ** 3
    # SIZING FIX: per-key magnitude bytes differ by detector (T differs -> e.g. H1 ~5.1 MB, L1 ~7.8 MB).
    # Probe BOTH sides and size the chunk from the STREAMED side's REAL per-key (and account the retained
    # side with its OWN per-key) so the chunk never overshoots the budget like it did (120->150 GB).
    pkb = {}
    if h_keys:
        _m, _ = _fullmag_batch([h_keys[0]], "H1"); pkb["H1"] = float(_m[0].nbytes); del _m
    if l_keys:
        _m, _ = _fullmag_batch([l_keys[0]], "L1"); pkb["L1"] = float(_m[0].nbytes); del _m

    # choose which side to RETAIN (the smaller key count) and which to STREAM
    if len(l_keys) <= len(h_keys):
        small_keys, small_det, large_keys, large_det = l_keys, "L1", h_keys, "H1"
        large_is_H = True
    else:
        small_keys, small_det, large_keys, large_det = h_keys, "H1", l_keys, "L1"
        large_is_H = False
    small_pk = pkb[small_det]; large_pk = pkb[large_det]   # bytes/key for retained and streamed sides

    SAFETY_GB = 4.0                                          # raw windows + zoomed qt + workers + python overhead
    retained_gb = len(small_keys) * small_pk / GB           # retained side uses ITS OWN per-key
    # If even the retained (small) side blows the budget, chunk it too and fall back to a single-pass
    # streaming-of-both scheme (rare on these runs; defensive/general). Otherwise retain small fully.
    avail_for_large = MAXGB - retained_gb - SAFETY_GB
    if avail_for_large <= large_pk / GB:                    # small side alone ~exhausts budget
        # general fallback: per-pair lazy scoring (constant memory, slower) -- still bit-identical.
        print(f"[precompute] retained side {retained_gb:.1f} GB exceeds budget {MAXGB} GB -> "
              f"per-pair lazy fallback (constant mem)", flush=True)
        H = {}; Ht = {}; HT = {}; Lm = {}
        for k, p in enumerate(pairs):
            ah = (p[0], p[1]); bl = (p[2], p[3])
            if ah not in H:
                mm, tt = _fullmag_batch([ah], "H1"); H[ah] = mm[0]; Ht[ah] = int(tt[0]); HT[ah] = mm[0].shape[1]
                if len(H) > 2: H.pop(next(iter(H)))          # keep memory ~constant
            if bl not in Lm:
                mm, _ = _fullmag_batch([bl], "L1"); Lm[bl] = mm[0]
                if len(Lm) > 2: Lm.pop(next(iter(Lm)))
            out[p] = _score_one(H[ah], Ht[ah], HT[ah], Lm[bl])
        return out

    C = max(1, int(avail_for_large * GB / large_pk))        # large-side chunk size, from STREAMED per-key
    est_peak_gb = retained_gb + min(len(large_keys), C) * large_pk / GB + SAFETY_GB
    print(f"[precompute] {len(pairs)} pairs; per-key H1={pkb.get('H1',0)/1e6:.2f} L1={pkb.get('L1',0)/1e6:.2f} MB (f32); "
          f"retain {small_det}={len(small_keys)} ({retained_gb:.1f} GB), stream {large_det}={len(large_keys)} "
          f"in chunks of C={C}; budget {MAXGB} GB; est peak ~{est_peak_gb:.1f} GB", flush=True)

    # build the retained (small) side once. For L1 only the mag is needed; for H1 we also need t0/T.
    smallmag = {}; small_t0 = {}; small_T = {}
    for c0 in range(0, len(small_keys), C):                 # chunk the build too (bounds the transient build_qt stack)
        ck = small_keys[c0:c0 + C]
        mm, tt = _fullmag_batch(ck, small_det)
        for key, m, t in zip(ck, mm, tt):
            smallmag[key] = m; small_t0[key] = int(t); small_T[key] = m.shape[1]

    # index pairs by their LARGE-side key so a streamed chunk scores exactly its pairs, then is dropped.
    from collections import defaultdict
    by_large = defaultdict(list)
    for p in pairs:
        lk = (p[0], p[1]) if large_is_H else (p[2], p[3])
        by_large[lk].append(p)

    for c0 in range(0, len(large_keys), C):
        ck = large_keys[c0:c0 + C]
        mm, tt = _fullmag_batch(ck, large_det)
        chunkmag = {}; chunk_t0 = {}; chunk_T = {}
        for key, m, t in zip(ck, mm, tt):
            chunkmag[key] = m; chunk_t0[key] = int(t); chunk_T[key] = m.shape[1]
        del mm, tt
        for lk in ck:
            for p in by_large[lk]:
                hk = (p[0], p[1]); bl = (p[2], p[3])
                if large_is_H:                               # H1 streamed -> H1 t0/T from chunk, L1 mag retained
                    magH = chunkmag[hk]; t0H = chunk_t0[hk]; TH = chunk_T[hk]; magL = smallmag[bl]
                else:                                        # L1 streamed -> H1 t0/T retained, L1 mag from chunk
                    magH = smallmag[hk]; t0H = small_t0[hk]; TH = small_T[hk]; magL = chunkmag[bl]
                out[p] = _score_one(magH, t0H, TH, magL)
        del chunkmag, chunk_t0, chunk_T                      # drop the chunk before loading the next
    return out

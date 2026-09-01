"""Local-ASD consistency veto (veto #1).

A net-sigma / CNN candidate must remain significant when its QT tile is re-whitened with a
LOCAL +/-64s median-Welch ASD instead of the fixed reference ASD. If it collapses, it was an
ASD-mismatch artifact (the reference ASD mis-estimated that segment's noise floor -> inflated
CAE reconstruction sigma and/or a corrupted tile that fools the CNN).

Validated 2026-07-06 on the O3b FAR run (reference ASD = O3a): net27 net 27->4.0, net23 22.9->0.9
(net-sigma collapse), net16 cnn 1.0->0.12 (CNN flips once whitened correctly); all 4 real events
unchanged (cnn ~1.0, net > floor). See gw-o3b-far-retune-config memory.

Pure post-processing on the candidate list -- only the handful of detections are recomputed, never
the background -- so it is a VETO (drop artifacts), not a re-ranking. Surviving candidates keep their
original reference-ASD FAR. Opt-in (SM_ASD_VETO=1) so default behaviour is byte-identical.
"""
import os
import numpy as np
from scipy.ndimage import zoom
from gwpy.timeseries import TimeSeries
import driver_blindscan as B
import improved_pipeline as ip

FS = 4096
WIN = 4.0
ASD_HALF = float(os.environ.get("SM_ASD_HALF", "64.0"))   # +/- seconds for the local Welch ASD


def _build_qt(pipe, wh):
    qi = ip.center_crop_waveforms(wh, sample_rate=FS, context_seconds=pipe.ctx)
    mags = [ip._compute_qt_image_worker((w, FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0)) for w in qi]
    return ip.min_max_norm(np.stack([zoom(m, (256 / m.shape[0], 128 / m.shape[1]), order=1)
                                     for m in mags]).astype(np.float32)).astype(np.float32)


def _netsigma(pipe, seg, idx):
    tot = 0.0
    for det in ("H1", "L1"):
        mu, sd = (pipe.norm["muH"], pipe.norm["sdH"]) if det == "H1" else (pipe.norm["muL"], pipe.norm["sdL"])
        wh = pipe._whiten(B._win(seg, det, idx)[None, :], det)
        tot += float((pipe._recon(_build_qt(pipe, wh)).reshape(-1)[0] - mu) / sd)
    return tot / np.sqrt(2.0)


def _local_asds(seg, idx):
    gc = idx + WIN / 2.0
    out = {}
    for det in ("H1", "L1"):
        r = B._strain(seg, det)
        j0 = max(0, int((gc - ASD_HALF) * FS)); j1 = min(len(r), int((gc + ASD_HALF) * FS))
        out[det] = TimeSeries(r[j0:j1].astype(np.float64), sample_rate=FS).asd(fftlength=4, overlap=2, method="median")
    return out


def recompute_local(seg, idx):
    """net-sigma + cnn (hm, lm) recomputed for one window under a local median-Welch ASD."""
    pipe = B.cpipe()
    sav = {d: pipe.asd[d] for d in ("H1", "L1")}
    la = _local_asds(seg, idx)
    try:
        pipe.asd["H1"], pipe.asd["L1"] = la["H1"], la["L1"]
        net = _netsigma(pipe, seg, idx)
        hm, lm = B.cnn_hm_lm(seg, idx, seg, idx)
    finally:
        pipe.asd["H1"], pipe.asd["L1"] = sav["H1"], sav["L1"]
    return dict(net_loc=float(net), hm_loc=float(hm), lm_loc=float(lm))


def idx_of(det):
    """Window index of a detection dict from its seg name (o3b_<t0>) and gps."""
    if "idx" in det and det["idx"] is not None:
        return int(det["idx"])
    t0 = int(str(det["seg"]).split("_")[1])
    return int(round(det["gps"] - WIN / 2.0 - t0))


def survives(det, floor=4.0, glitch_thresh=None):
    """True (keep) unless the candidate is an ASD-mismatch artifact under local whitening.

    Rule: (1) CNN gate must still pass locally (max(hm,lm) > glitch_thresh) -- catches broadband
    glitches whose tiles were corrupted by mis-whitening (net16). (2) a net-sigma-channel candidate
    must keep net > floor locally -- catches inflated-sigma artifacts (net27, net23).
    """
    if glitch_thresh is None:
        glitch_thresh = B.GLITCH_THRESH
    r = recompute_local(det["seg"], idx_of(det))
    keep = True
    reason = "keep"
    if max(r["hm_loc"], r["lm_loc"]) < glitch_thresh:
        keep = False; reason = "cnn"
    elif det.get("channel") == "net-sigma" and r["net_loc"] < floor:
        keep = False; reason = "net<floor"
    r["keep"] = keep; r["reason"] = reason
    return keep, r

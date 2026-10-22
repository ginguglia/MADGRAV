"""Injection campaign v2 (BLIND + mass-stratified) — fixes validation M12/UM findings.
Signals injected at a uniformly-random SUB-STRIDE GPS into the stretch, then recovered through
the SAME 0.25 s sliding grid (score the grid windows covering the injection, take the peak-net
pairing = what the blind scan + clustering would report). Carries the sub-tile placement loss.
Mass-stratified draw (50% ultra-massive bank) so GW231028 (Mtot153) & GW231123 (Mtot238) have
support. -> inj_out/<event>_inj.npz (recovered features + injected net SNR, mtot, offset).
Run: python inject.py --event NAME
"""
import os,sys,json,time,argparse
import numpy as np, torch
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import improved_pipeline as ip
from massive_pipeline import MassiveEventPipeline
import driver_streams as DS
import coh_rewhiten as CR
import guided_excess as GX     # MADGRAV2.1: guided second-leg excess, computed on the same re-whitened rows as the coherence
import centroid_v2 as C2
import arm_v2
ARM2=None
import morph_roi as mr
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
def _morph_one(qt, wl):   # ROI morphology [chirpslope, verticality, eccentricity]
    import numpy as _np
    P=qt**2; Et=gaussian_filter1d(P.sum(0),2); t0=int(Et.argmax()); wt=14
    roiP=P[:,max(0,t0-wt):min(128,t0+wt+1)]; thr=_np.percentile(roiP,90); ys,xs=_np.where(roiP>=thr); ww=roiP[ys,xs]
    if len(ww)<8: vert,ecc=1.0,0.0
    else:
        xm=(xs*ww).sum()/ww.sum(); ym=(ys*ww).sum()/ww.sum()
        vx=(ww*(xs-xm)**2).sum()/ww.sum(); vy=(ww*(ys-ym)**2).sum()/ww.sum(); vxy=(ww*(xs-xm)*(ys-ym)).sum()/ww.sum()
        l1=0.5*((vx+vy)+_np.sqrt((vx-vy)**2+4*vxy**2)); l2=0.5*((vx+vy)-_np.sqrt((vx-vy)**2+4*vxy**2))
        vert=float(vy/(vx+1e-9)); ecc=float((l1-l2)/(l1+l2+1e-9))
    FS_=4096; X=_np.fft.rfft(wl); f=_np.fft.rfftfreq(len(wl),1/FS_); X[(f<20)|(f>150)]=0
    xb=_np.fft.irfft(X,n=len(wl)); env=_np.abs(hilbert(xb)); c=len(env)//2; win=int(0.5*FS_)
    pk=c-win+int(env[c-win:c+win].argmax()); s=xb[max(0,pk-int(0.12*FS_)):pk+int(0.12*FS_)]
    a=hilbert(s); en=_np.abs(a); fi=_np.diff(_np.unwrap(_np.angle(a)))/(2*_np.pi)*FS_; w2=en[:-1]**2; m=w2>0.3*w2.max()
    if m.sum()<6: slope=0.0
    else:
        t=_np.arange(len(fi))[m]; fw=fi[m]; w2m=w2[m]; tm=(t*w2m).sum()/w2m.sum(); fm=(fw*w2m).sum()/w2m.sum()
        slope=abs(((w2m*(t-tm)*(fw-fm)).sum()/((w2m*(t-tm)**2).sum()+1e-9))*FS_)
    return float(slope),vert,ecc

O4A=DS.O4A; LRD=DS.LRD; SC=DS.SC; FS=4096; WN=4*FS; STRIDE=0.25; STEP=int(STRIDE*FS); DEV=os.environ.get("SM_DEV","cuda:1")
STR=os.environ.get("SM_STRAIN","search_mode/strain"); OUT=os.environ.get("SM_INJ","search_mode/inj_out"); os.makedirs(OUT,exist_ok=True)
SEG=json.load(open(os.environ.get("SM_SEGJSON_EV","search_mode/o3a_segments_event.json"))); EVENTS=DS.EVENTS
BANK_SIG=os.environ.get("SM_BANK_SIG",os.path.join(MADGRAV_ROOT,"data","o1_o3_signal_bank_projected_2s_x10"))
BANK_UM=os.environ.get("SM_BANK_UM",os.path.join(MADGRAV_ROOT,"bank","ultramassive_bank"))
# SM_BANK_LM -> a THIRD low-mass stratum (Mtot 10-22), drawn with probability SM_LM_FRAC.
# Its bank CSVs carry optimal_snr_full_ref / snr_full_over_crop, so with SM_INJ_SNR_FULL=1 the
# amplitude is scaled to the target FULL-SIGNAL network SNR (the quantity the catalog quotes)
# instead of the SNR retained by the stored 2 s crop. At Mtot 10-22 the crop holds only ~90%
# of the signal (measured ratio median 1.110), so the two differ; above Mtot 20 they do not
# (0.997). The ratio was computed against reference_psd_fixed2 from o4a_search_prep, which is
# EXACTLY the PSD this script normalises against under SM_INJ_NORM_FIXED=1, so the conversion
# is exact rather than approximate. DEFAULTS UNSET = existing campaigns byte-identical.
BANK_LM=os.environ.get("SM_BANK_LM","")
LM_FRAC=float(os.environ.get("SM_LM_FRAC","0.0"))
SNR_FULL=os.environ.get("SM_INJ_SNR_FULL","0")=="1"
NET_SNR_GRID=[float(x) for x in os.environ.get("SM_SNR_GRID","8,10,12,15,20,25").split(",")]  # default = original campaign grid, bit-identical
N_PER=int(os.environ.get("SM_INJ_NPER","300")); UM_FRAC=float(os.environ.get("SM_UM_FRAC","0.5")); NGRID=3            # score +-1 grid window around the injection -> peak
# SM_INJ_PLACE (2026-10-22, vt96p): "grid" (default, byte-identical) = three 0.25 s-spaced 4 s windows around the merger, peak = argmax net
# (the accepted campaigns; the merger is always within 0.125 s of one window centre). "prod" = PRODUCTION placement: the merger sits at a
# uniform offset u in [-0.5, 0.5) s from the centre of a tile of the fixed 1 s search grid, and the scored windows are that tile and its
# neighbour on the merger's side (1 s apart; peak = argmax net over them = the search's cluster representative on the 1 s grid). Every
# rng draw (source, SNR, sub-stride offset, GPS) is unchanged -- the coarse quarter of u comes from a second generator -- so a "prod" field
# with the same seed stays paired row by row with the "grid" field (same injections in the same data, placement is the only difference).
INJ_PLACE=os.environ.get("SM_INJ_PLACE","grid")
if INJ_PLACE not in ("grid","prod"): raise SystemExit(f"[inj] SM_INJ_PLACE={INJ_PLACE} not in grid|prod")
if INJ_PLACE=="prod": NGRID=2
# SM_INJ_CNN=1 -> also score each injection's peak window with the HM/LM specialists, so the injection
# set can be pushed through the CNN glitch gate (and the local-ASD veto) exactly as real candidates are.
# Default OFF: the campaign is byte-identical to the accepted one.
INJ_CNN = os.environ.get("SM_INJ_CNN","0")=="1"
# SM_INJ_ASDVETO=1 -> ALSO re-score each injection's peak window under a LOCAL +/-64s median-Welch ASD
# (the same veto the real candidates pass through, asd_consistency.py), so the injection efficiency that
# feeds VT is measured through the SAME chain as the detections instead of stopping at the CNN gate.
# Only injections that pass the CNN glitch gate are re-scored -- the others are already lost at the gate.
# Requires SM_INJ_CNN=1 (the gate value comes from it). Default OFF: byte-identical to the accepted campaign.
ASDVETO = os.environ.get("SM_INJ_ASDVETO","0")=="1"
# SM_INJ_LOCAL_ALL=1 -> run the local-ASD recompute for EVERY injection, not just gate-passers.
# The accepted campaign skipped gate-failures ("already lost"), which makes the complement --
# how many gate-FAILURES the local ASD would rescue -- unmeasured. Default 0 keeps it byte-identical.
LOCAL_ALL = os.environ.get("SM_INJ_LOCAL_ALL","0")=="1"
# SM_INJ_NORM_FIXED=1 -> normalise the injection AMPLITUDE against the leakage-free
# reference_psd_fixed_*.npz instead of the frozen reference_psd_*.npz, while the pipeline keeps
# whitening and gating with the frozen one. The two roles are different and must not be conflated:
#   whitening/gating = what the as-run search DID  -> must stay on the frozen ASD, or the measured
#                      efficiency stops describing the pipeline that produced the detections;
#   amplitude norm   = what "network SNR = 8" MEANS physically -> must use the true noise, or the
#                      same nominal SNR is a different physical loudness in each run (measured: the
#                      frozen ASD is wrong by up to 136x at 20-35 Hz in O4b, ~1x in O3a).
NORM_FIXED = os.environ.get("SM_INJ_NORM_FIXED","0")=="1"
ASD_TAG = os.environ.get("SM_ASD_TAG","fixed")   # reference_psd_<tag>_<det>.npz
# SM_INJ_SAVE_SERIES=1 -> also store each injection's peak-window re-whitened 1 s series (float32 [n,4096] per detector, ~180 MB per
# event file at 5400 injections) so future pair-statistic pilots (MADGRAV2.1) run on CPU from the files instead of re-injecting on GPU.
SAVE_SERIES = os.environ.get("SM_INJ_SAVE_SERIES","0")=="1"

def _load_asd_fixed(prep, det):
    """load_detector_asd_o1, but reading reference_psd_fixed_<det>.npz."""
    from gwpy.frequencyseries import FrequencySeries
    z = np.load(os.path.join(prep, f"reference_psd_{ASD_TAG}_{det}.npz"))
    psd = z["psd"].astype(np.float64)
    pos = psd[np.isfinite(psd) & (psd > 0.0)]
    if len(pos) == 0:
        raise ValueError(f"fixed PSD for {det} has no positive finite bins")
    psd = np.maximum(psd, float(np.median(pos) * 1e-10))
    return FrequencySeries(np.sqrt(psd), f0=float(z["freq"][0]),
                           df=float(z["freq"][1] - z["freq"][0]))
ASD_HALF = float(os.environ.get("SM_ASD_HALF","64.0"))   # +/- seconds, matches asd_consistency.ASD_HALF
if ASDVETO and not INJ_CNN:   # fail FAST: the veto branch lives inside the INJ_CNN block, so without it
    raise SystemExit("SM_INJ_ASDVETO=1 requires SM_INJ_CNN=1 -- refusing to write empty veto columns")
INJ_SEED=int(os.environ.get("SM_INJ_SEED","20260614"))   # 2026-10-04: SM_INJ_SEED -> independent draw (default = the accepted campaign seed, byte-identical)
rng=np.random.default_rng(INJ_SEED)
# SM_INJ_S3=1 (2026-10-04, MADGRAV 2.2 injection efficiency): ALSO score each injection's peak window with the S3 specialist
# ensemble (hm 20-300 Hz, madgrav2_spec/models, 2 folds x 5 seeds, cross-fit by the GPS day of the injection centre), on the
# SAME QT magnitude / Grad-CAM time placement / band-crop path as the June CNN (driver_blindscan._fullmag/_crop == the crop path
# score_bank_s3*.py tripwire-verified against the production cache). Requires SM_INJ_CNN=1. Default OFF: byte-identical.
INJ_S3 = os.environ.get("SM_INJ_S3","0")=="1"
if INJ_S3 and not INJ_CNN: raise SystemExit("SM_INJ_S3=1 requires SM_INJ_CNN=1 (same QT/crop path)")
S3_FLO,S3_FHI=20.0,300.0; _S3ENS=None; _ntrip=[0]
def _s3_fold_of_gps(g):
    import hashlib; return int(hashlib.sha1(str(int(g//86400)).encode()).hexdigest(),16)%2
def _s3_ens(B):
    global _S3ENS
    if _S3ENS is None:
        import torch.nn as nn
        class _Net2(nn.Module):                      # == m2/rank/driver_blindscan_merge_hook.py _Net2 (temperature buffer)
            def __init__(s_):
                super().__init__(); ch=[2,16,32,64,128]
                s_.b=nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i],ch[i+1],3,padding=1),nn.BatchNorm2d(ch[i+1]),nn.ReLU(),nn.MaxPool2d(2)) for i in range(4)])
                s_.h=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1)); s_.register_buffer("temperature",torch.ones(1))
            def forward(s_,x):
                for bl in s_.b: x=bl(x)
                return s_.h(x).squeeze(-1)
        MD=os.environ.get("SM_S3_MODELS",MADGRAV_SCRATCH + "/madgrav2_spec/models"); _S3ENS={}
        for f in (0,1):
            ms=[]
            for sd in range(5):
                m=_Net2().to(B.DEV); m.load_state_dict(torch.load(f"{MD}/spec_hm_f{f}_s{sd}_nfull.pt",map_location=B.DEV)); m.eval(); ms.append(m)
            _S3ENS[f]=ms
        print(f"[inj] S3 ensemble loaded from {MD} (2 folds x 5 seeds), band {S3_FLO}-{S3_FHI} Hz",flush=True)
    return _S3ENS
def _cnn_s3_from_wh(B,whH,whL,gps,tripwire):
    """(hm,lm,s3,fold): June HM/LM exactly as B.cnn_from_wh (same calls, same order) + S3 on the 20-300 Hz crop of the same mags."""
    magH,t0,T=B._fullmag(whH); magL,_,_=B._fullmag(whL); a=int(round((t0-14)/128*T))
    def stk(flo,fhi): return np.stack([B._crop(magH,a,T,flo,fhi),B._crop(magL,a,T,flo,fhi)])
    with torch.no_grad():
        hm=float(torch.sigmoid(B.cnet()(torch.from_numpy(stk(B.FLO,B.FHI)[None]).float().to(B.DEV))).item())
        lm=float(torch.sigmoid(B.lmnet()(torch.from_numpy(stk(B.LFLO,B.LFHI)[None]).float().to(B.DEV))).item())
        f=_s3_fold_of_gps(gps); x=torch.from_numpy(stk(S3_FLO,S3_FHI)[None]).float().to(B.DEV)
        s3=float(np.mean([torch.sigmoid(m(x)/m.temperature).item() for m in _s3_ens(B)[1-f]]))   # fold-pure pair -> models of the OTHER fold (== score_bank_s3)
    if tripwire:
        hm0,lm0=B.cnn_from_wh(whH,whL)
        if max(abs(hm-hm0),abs(lm-lm0))>1e-6: raise SystemExit(f"[inj] S3 path TRIPWIRE: June CNN differs from B.cnn_from_wh ({hm:.6f},{lm:.6f}) vs ({hm0:.6f},{lm0:.6f})")
    return hm,lm,s3,f

def _veto_local(pipe,rawH,rawL,base,off,wHs,wLs,xH,xL,gps=None):
    """Re-score ONE injection's peak window under a local +/-ASD_HALF median-Welch ASD.

    The ASD window is cut from the real strain and carries the injected signal, exactly as the
    local-ASD veto (asd_consistency.py) sees a real candidate's data. Welch parameters are kept
    identical to asd_consistency._local_asds (fftlength=4, overlap=2, median).
    """
    import driver_blindscan as B
    from gwpy.timeseries import TimeSeries
    L=len(wHs); cen=base+STEP+WN//2+off; half=int(ASD_HALF*FS)
    sav={d:pipe.asd[d] for d in ("H1","L1")}
    try:
        loc={}
        for det,raw,w in (("H1",rawH,wHs),("L1",rawL,wLs)):
            j0=max(0,cen-half); j1=min(len(raw),cen+half)
            sl=raw[j0:j1].astype(np.float64).copy()
            a=cen-L//2-j0                                  # signal start inside the slice
            if a>=0 and a+L<=len(sl): sl[a:a+L]+=w         # ASD sees the signal, as for a real event
            loc[det]=TimeSeries(sl,sample_rate=FS).asd(fftlength=4,overlap=2,method="median")
        pipe.asd["H1"],pipe.asd["L1"]=loc["H1"],loc["L1"]
        whH=pipe._whiten(xH[None,:].astype(np.float32),"H1"); whL=pipe._whiten(xL[None,:].astype(np.float32),"L1")
        sH=DS.sigma_from_qt(pipe,DS.build_qt(pipe,whH),"H1"); sL=DS.sigma_from_qt(pipe,DS.build_qt(pipe,whL),"L1")
        if INJ_S3 and gps is not None:                  # 2026-10-07: S3 under the local ASD too (same crop path; hm/lm identical to cnn_from_wh by the tripwire)
            hm_l,lm_l,s3_l,_=_cnn_s3_from_wh(B,whH,whL,gps,False)
        else:
            hm_l,lm_l=B.cnn_from_wh(whH,whL); s3_l=np.nan
    finally:
        pipe.asd["H1"],pipe.asd["L1"]=sav["H1"],sav["L1"]
    return dict(net_loc=float((sH[0]+sL[0])/np.sqrt(2.0)),hm_loc=float(hm_l),lm_loc=float(lm_l),
                veto_keep=int(max(hm_l,lm_l)>B.GLITCH_THRESH),s3_loc=float(s3_l))

def score_block(pipe,arms,X,det):
    wh=pipe._whiten(X.astype(np.float32),det); qt=DS.build_qt(pipe,wh)
    return DS.sigma_from_qt(pipe,qt,det), DS.g_from_qt(arms,qt), pipe._centroid(wh), wh, qt, C2.cluster_centroid(qt)   # MADGRAV2: + centroid2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--event",required=True); a=ap.parse_args(); name=a.event
    # SM_CALIB: alternative calibration JSON (model_path + sigma_norm [+ latent_channels]); default = the frozen BA calibration, byte-identical.
    pipe=MassiveEventPipeline(calib_path=os.environ.get("SM_CALIB",f"{SC}/massive_calibration_BA.json"),prep=O4A,device=DEV)
    if os.environ.get("SM_CALIB"): print(f"[inj] calibration override: {os.environ['SM_CALIB']} -> model {pipe.calib['model_path']}",flush=True)
    arms=[DS.GlitchArm().to(DEV) for _ in range(5)]
    for i,arm in enumerate(arms): arm.load_state_dict(torch.load(f"{LRD}/p1v42/arm_deploy_seed{i}.pt",map_location=DEV)); arm.eval()
    asd=pipe.asd                      # whitening/gating ASD -- frozen, never swapped
    norm_asd=pipe.asd
    if NORM_FIXED:
        norm_asd={d:_load_asd_fixed(O4A,d) for d in ("H1","L1")}
        print(f"[inj] amplitude normalisation uses reference_psd_fixed from {O4A} "
              f"(whitening still uses the frozen reference_psd)",flush=True)
    pb=ip.load_o1_signal_bank(BANK_SIG); ub=ip.load_o1_signal_bank(BANK_UM)
    banks={"sig":(pb["H1"],pb["L1"],pb.get("total_mass",[np.nan]*len(pb["H1"]))),
           "um":(ub["H1"],ub["L1"],ub.get("total_mass",[np.nan]*len(ub["H1"])))}
    ratio={}                                        # per-bank full/crop SNR, 1.0 where unknown
    if BANK_LM:
        lb=ip.load_o1_signal_bank(BANK_LM)
        banks["lm"]=(lb["H1"],lb["L1"],lb.get("total_mass",[np.nan]*len(lb["H1"])))
        if SNR_FULL:
            import csv as _csv, glob as _glob      # noqa: PLC0415
            r=[]
            for c in sorted(_glob.glob(os.path.join(BANK_LM,"**","signals_*.csv"),recursive=True)):
                with open(c,newline="") as fh: r+= [float(x["snr_full_over_crop"]) for x in _csv.DictReader(fh)]
            assert len(r)==len(lb["H1"]), f"lm bank: {len(r)} csv rows vs {len(lb['H1'])} waveforms"
            ratio["lm"]=np.asarray(r,np.float64)     # same order load_o1_signal_bank concatenates
        print(f"[inj] low-mass stratum: {len(lb['H1'])} sources, frac {LM_FRAC:.0%}"
              f"{' , scaling to FULL-signal SNR' if SNR_FULL else ''}",flush=True)
    print(f"[inj] {name}: sig {len(pb['H1'])} / UM {len(ub['H1'])} sources; blind, {UM_FRAC:.0%} UM; seed {INJ_SEED}; CNN {int(INJ_CNN)} S3 {int(INJ_S3)}",flush=True)
    if name not in EVENTS or name not in SEG:
        print(f"[inj] {name} not in pruned event configs (no strain / DQ gap) -- skipping",flush=True); return
    gps0=EVENTS[name]; t0=SEG[name]["coincident_lock"][0]
    rawH=np.load(f"{STR}/{name}_H1.npz")["strain"]; rawL=np.load(f"{STR}/{name}_L1.npz")["strain"]
    REG=WN+2*STEP                                   # region holding NGRID windows
    global ARM2
    if os.environ.get("SM_ARM_V2")=="1": ARM2=arm_v2.ArmV2(DEV); print(f"[inj] arm v2 ensemble loaded ({ARM2.n})",flush=True)
    R={k:[] for k in ("net_snr","mtot","is_um","off","sigH","sigL","net","coh","gx","cenH","cenL","cenH2","cenL2","gH","gL","gH2","gL2",
                      "chirpH","vertH","eccH","chirpL","vertL","eccL")+(("cnn_hm","cnn_lm") if INJ_CNN else ())+(("s3","s3_fold") if INJ_S3 else ())+(("serH","serL") if SAVE_SERIES else ())
                      +(("net_loc","hm_loc","lm_loc","veto_keep") if ASDVETO else ())+(("s3_loc",) if (ASDVETO and INJ_S3) else ())
                      +(("place_u",) if INJ_PLACE=="prod" else ())}
    rng2=np.random.default_rng([INJ_SEED,20261022]) if INJ_PLACE=="prod" else None; nedge=0   # "prod": coarse-offset generator (main rng untouched)
    ta=time.time()
    for snr in NET_SNR_GRID:
        XH=[];XL=[];meta=[];plc=[]
        for _ in range(N_PER):
            _u=rng.random()
            if BANK_LM and _u<LM_FRAC: _tag="lm"; um=0
            else: um=int((_u-LM_FRAC)/(1.0-LM_FRAC) < UM_FRAC) if BANK_LM else int(_u<UM_FRAC); _tag="um" if um else "sig"
            WH,WL,MT=banks[_tag]; k=int(rng.integers(0,len(WH)))
            wH=np.asarray(WH[k],np.float32); wL=np.asarray(WL[k],np.float32); L=len(wH)
            s0=np.sqrt(ip.compute_optimal_snr(wH,norm_asd["H1"])**2+ip.compute_optimal_snr(wL,norm_asd["L1"])**2)
            if s0<=0: continue
            # target is the FULL-signal SNR: crop SNR must be snr / (full/crop)
            _r=float(ratio[_tag][k]) if _tag in ratio else 1.0
            sc=np.float32((snr/_r)/s0); off=int(rng.integers(0,STEP))     # sub-stride offset
            while True:
                base=int(rng.integers(STEP,len(rawH)-REG-STEP)); cg=t0+(base+STEP+WN//2)/FS
                if abs(cg-gps0)>10: break
            if INJ_PLACE=="prod":
                q=int(rng2.integers(0,4)); u=off+STEP*q-2*STEP          # merger offset from the 1 s tile centre: uniform integer in [-2 STEP, 2 STEP) = [-0.5, 0.5) s
                r0=max(0,base-4*STEP); r1=min(len(rawH),base+REG+5*STEP)
                regH=rawH[r0:r1].copy(); regL=rawL[r0:r1].copy(); c=base+STEP+WN//2+off-r0   # SAME absolute signal centre as "grid" (base+STEP+WN//2+off, as _veto_local)
                regH[c-L//2:c-L//2+L]+=wH*sc; regL[c-L//2:c-L//2+L]+=wL*sc
                wsA=c-WN//2-u; wsB=wsA+(4*STEP if u>=0 else -4*STEP)   # tile holding the merger (centre at c-u), and the neighbouring 1 s tile on the merger's side
                if not (0<=wsB and wsB+WN<=len(regH)): wsB=wsA; nedge+=1   # neighbour outside the segment -> the tile alone (row duplicated: NGRID stays 2)
                for ws in (wsA,wsB): XH.append(regH[ws:ws+WN]); XL.append(regL[ws:ws+WN])
                plc.append(u)
            else:
                regH=rawH[base:base+REG].copy(); regL=rawL[base:base+REG].copy()
                c=STEP+WN//2+off                                          # signal center in region
                regH[c-L//2:c-L//2+L]+=wH*sc; regL[c-L//2:c-L//2+L]+=wL*sc
                for gi in range(NGRID):
                    XH.append(regH[gi*STEP:gi*STEP+WN]); XL.append(regL[gi*STEP:gi*STEP+WN])
            meta.append((snr,float(MT[k]),um,off)+((base,wH*sc,wL*sc) if ASDVETO else ())+(cg,))   # MADGRAV2: keep the injection centre GPS (last element)
        XH=np.stack(XH); XL=np.stack(XL)
        sH,gH,cH,whH,qtH,c2H=score_block(pipe,arms,XH,"H1"); sL,gL,cL,whL,qtL,c2L=score_block(pipe,arms,XL,"L1")
        camH=mr.cam_t0_batch(arms[0],qtH,DEV); camL=mr.cam_t0_batch(arms[0],qtL,DEV)   # Grad-CAM ROI centers
        if ARM2 is not None:
            _g=np.repeat(np.array([m_[-1] for m_ in meta],float),NGRID); g2H=ARM2.g2_from_qt(qtH,_g).reshape(len(meta),NGRID); g2L=ARM2.g2_from_qt(qtL,_g).reshape(len(meta),NGRID)
        else: g2H=np.full((len(meta),NGRID),np.nan); g2L=np.full((len(meta),NGRID),np.nan)
        if CR.REWHITEN:                                                 # MADGRAV2 re-whitened coherence == series_for path (coh_rewhiten.py)
            _half=int(pipe.coh_win*FS/2); _cc=whH.shape[1]//2; _cache=os.environ.get("SM_SERIESCACHE")
            def _ser(wb,det):
                a=wb[:,_cc-_half:_cc+_half].astype(np.float64); a=pipe._bandlimit(a); a-=a.mean(1,keepdims=True)   # == driver_search_multi._compute_series
                return CR.flatten(a,CR.get_ref(name,det,f"{_cache}/{name}_{det}_series.npy" if _cache else None))
            _sH=_ser(whH,"H1"); _sL=_ser(whL,"L1"); coh=(CR.coh_v2_np(_sH,_sL) if CR.V2 else CR.coh_vec_np(_sH,_sL,pipe.lag))   # 2026-09-27: SM_COH_V2
            gx=GX.gx_vec(_sH,_sL)                                       # MADGRAV2.1 guided excess on the same rows (always stored; the LR uses it under SM_GX=1)
        else:
            coh=pipe._coherence(whH,whL); gx=np.full(len(coh),np.nan); _sH=_sL=None
            if SAVE_SERIES: raise SystemExit("[inj] SM_INJ_SAVE_SERIES=1 needs SM_COH_REWHITEN=1 (the stored rows are the re-whitened series)")
        net=(sH+sL)/np.sqrt(2.0)
        ninj=len(meta)
        net=net.reshape(ninj,NGRID); sH=sH.reshape(ninj,NGRID); sL=sL.reshape(ninj,NGRID)
        gH=gH.reshape(ninj,NGRID); gL=gL.reshape(ninj,NGRID); cH=cH.reshape(ninj,NGRID)
        cL=cL.reshape(ninj,NGRID); coh=coh.reshape(ninj,NGRID); gx=gx.reshape(ninj,NGRID); pk=np.argmax(net,1); c2H=c2H.reshape(ninj,NGRID); c2L=c2L.reshape(ninj,NGRID)
        for i in range(ninj):
            j=pk[i]; sn,mt,um,off=meta[i][:4]; fi_=i*NGRID+j   # flat index of peak grid window
            R["net_snr"].append(sn); R["mtot"].append(mt); R["is_um"].append(um); R["off"].append(off)
            if INJ_PLACE=="prod": R["place_u"].append(plc[i])
            R["sigH"].append(float(sH[i,j])); R["sigL"].append(float(sL[i,j])); R["net"].append(float(net[i,j]))
            R["coh"].append(float(coh[i,j])); R["gx"].append(float(gx[i,j])); R["cenH"].append(float(cH[i,j])); R["cenL"].append(float(cL[i,j])); R["cenH2"].append(float(c2H[i,j])); R["cenL2"].append(float(c2L[i,j]))
            R["gH"].append(float(gH[i,j])); R["gL"].append(float(gL[i,j])); R["gH2"].append(float(g2H[i,j])); R["gL2"].append(float(g2L[i,j]))
            csH,vH,eH=mr.morph_one_cam(qtH[fi_],whH[fi_].astype(np.float64),int(camH[fi_])); csL,vL,eL=mr.morph_one_cam(qtL[fi_],whL[fi_].astype(np.float64),int(camL[fi_]))
            R["chirpH"].append(csH); R["vertH"].append(vH); R["eccH"].append(eH)
            R["chirpL"].append(csL); R["vertL"].append(vL); R["eccL"].append(eL)
            if SAVE_SERIES: R["serH"].append(_sH[fi_].astype(np.float32)); R["serL"].append(_sL[fi_].astype(np.float32))
            if INJ_CNN:
                import driver_blindscan as B
                if INJ_S3:
                    hm_,lm_,s3_,f3_=_cnn_s3_from_wh(B,whH[fi_][None,:],whL[fi_][None,:],float(meta[i][-1]),_ntrip[0]<16); _ntrip[0]+=1
                    R["s3"].append(s3_); R["s3_fold"].append(f3_)
                else:
                    hm_,lm_=B.cnn_from_wh(whH[fi_][None,:],whL[fi_][None,:])
                R["cnn_hm"].append(hm_); R["cnn_lm"].append(lm_)
                if ASDVETO:
                    if LOCAL_ALL or max(hm_,lm_)>B.GLITCH_THRESH:   # LOCAL_ALL also pays for gate-failures (the complement test)
                        r=_veto_local(pipe,rawH,rawL,meta[i][4],off,meta[i][5],meta[i][6],XH[fi_],XL[fi_],gps=(float(meta[i][-1]) if INJ_S3 else None))
                    else:
                        r=dict(net_loc=np.nan,hm_loc=np.nan,lm_loc=np.nan,veto_keep=0,s3_loc=np.nan)
                    for kk in ("net_loc","hm_loc","lm_loc","veto_keep")+(("s3_loc",) if INJ_S3 else ()): R[kk].append(r[kk])
        print(f"  snr_net={snr}: {ninj} inj, peak-net med {np.median(net.max(1)):.1f} | net>=4 frac {np.mean(net.max(1)>=4.0):.2f} ({(time.time()-ta)/60:.1f}m)",flush=True)
    off=np.array(R["off"]); print(f"[inj] offset uniformity: min {off.min()} max {off.max()} (step {STEP}) mean {off.mean():.0f}~{STEP/2:.0f}",flush=True)
    if INJ_PLACE=="prod":
        u=np.array(R["place_u"]); print(f"[inj] PRODUCTION placement: merger offset from the 1 s tile centre min {u.min()} max {u.max()} (samples; +-{2*STEP}) mean {u.mean():.0f}~0, quarters {np.bincount((u+2*STEP)//STEP,minlength=4).tolist()}; neighbour tile outside the segment {nedge}x",flush=True)
    np.savez(f"{OUT}/{name}_inj.npz",**{k:np.array(v) for k,v in R.items()})
    print(f"[inj] saved {OUT}/{name}_inj.npz ({len(R['net'])} inj, {int(np.sum(R['is_um']))} UM)",flush=True)

if __name__=="__main__": main()

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
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import improved_pipeline as ip
from massive_pipeline import MassiveEventPipeline
import driver_streams as DS
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
NET_SNR_GRID=[float(x) for x in os.environ.get("SM_SNR_GRID","8,10,12,15,20,25").split(",")]  # default = original campaign grid, bit-identical
N_PER=int(os.environ.get("SM_INJ_NPER","300")); UM_FRAC=0.5; NGRID=3            # score +-1 grid window around the injection -> peak
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
rng=np.random.default_rng(20260614)

def _veto_local(pipe,rawH,rawL,base,off,wHs,wLs,xH,xL):
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
        hm_l,lm_l=B.cnn_from_wh(whH,whL)
    finally:
        pipe.asd["H1"],pipe.asd["L1"]=sav["H1"],sav["L1"]
    return dict(net_loc=float((sH[0]+sL[0])/np.sqrt(2.0)),hm_loc=float(hm_l),lm_loc=float(lm_l),
                veto_keep=int(max(hm_l,lm_l)>B.GLITCH_THRESH))

def score_block(pipe,arms,X,det):
    wh=pipe._whiten(X.astype(np.float32),det); qt=DS.build_qt(pipe,wh)
    return DS.sigma_from_qt(pipe,qt,det), DS.g_from_qt(arms,qt), pipe._centroid(wh), wh, qt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--event",required=True); a=ap.parse_args(); name=a.event
    pipe=MassiveEventPipeline(calib_path=f"{SC}/massive_calibration_BA.json",prep=O4A,device=DEV)
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
    print(f"[inj] {name}: sig {len(pb['H1'])} / UM {len(ub['H1'])} sources; blind, {UM_FRAC:.0%} UM",flush=True)
    if name not in EVENTS or name not in SEG:
        print(f"[inj] {name} not in pruned event configs (no strain / DQ gap) -- skipping",flush=True); return
    gps0=EVENTS[name]; t0=SEG[name]["coincident_lock"][0]
    rawH=np.load(f"{STR}/{name}_H1.npz")["strain"]; rawL=np.load(f"{STR}/{name}_L1.npz")["strain"]
    REG=WN+2*STEP                                   # region holding NGRID windows
    R={k:[] for k in ("net_snr","mtot","is_um","off","sigH","sigL","net","coh","cenH","cenL","gH","gL",
                      "chirpH","vertH","eccH","chirpL","vertL","eccL")+(("cnn_hm","cnn_lm") if INJ_CNN else ())
                      +(("net_loc","hm_loc","lm_loc","veto_keep") if ASDVETO else ())}
    ta=time.time()
    for snr in NET_SNR_GRID:
        XH=[];XL=[];meta=[]
        for _ in range(N_PER):
            um=int(rng.random()<UM_FRAC); WH,WL,MT=banks["um" if um else "sig"]; k=int(rng.integers(0,len(WH)))
            wH=np.asarray(WH[k],np.float32); wL=np.asarray(WL[k],np.float32); L=len(wH)
            s0=np.sqrt(ip.compute_optimal_snr(wH,norm_asd["H1"])**2+ip.compute_optimal_snr(wL,norm_asd["L1"])**2)
            if s0<=0: continue
            sc=np.float32(snr/s0); off=int(rng.integers(0,STEP))     # sub-stride offset
            while True:
                base=int(rng.integers(STEP,len(rawH)-REG-STEP)); cg=t0+(base+STEP+WN//2)/FS
                if abs(cg-gps0)>10: break
            regH=rawH[base:base+REG].copy(); regL=rawL[base:base+REG].copy()
            c=STEP+WN//2+off                                          # signal center in region
            regH[c-L//2:c-L//2+L]+=wH*sc; regL[c-L//2:c-L//2+L]+=wL*sc
            for gi in range(NGRID):
                XH.append(regH[gi*STEP:gi*STEP+WN]); XL.append(regL[gi*STEP:gi*STEP+WN])
            meta.append((snr,float(MT[k]),um,off)+((base,wH*sc,wL*sc) if ASDVETO else ()))
        XH=np.stack(XH); XL=np.stack(XL)
        sH,gH,cH,whH,qtH=score_block(pipe,arms,XH,"H1"); sL,gL,cL,whL,qtL=score_block(pipe,arms,XL,"L1")
        camH=mr.cam_t0_batch(arms[0],qtH,DEV); camL=mr.cam_t0_batch(arms[0],qtL,DEV)   # Grad-CAM ROI centers
        coh=pipe._coherence(whH,whL); net=(sH+sL)/np.sqrt(2.0)
        ninj=len(meta)
        net=net.reshape(ninj,NGRID); sH=sH.reshape(ninj,NGRID); sL=sL.reshape(ninj,NGRID)
        gH=gH.reshape(ninj,NGRID); gL=gL.reshape(ninj,NGRID); cH=cH.reshape(ninj,NGRID)
        cL=cL.reshape(ninj,NGRID); coh=coh.reshape(ninj,NGRID); pk=np.argmax(net,1)
        for i in range(ninj):
            j=pk[i]; sn,mt,um,off=meta[i][:4]; fi_=i*NGRID+j   # flat index of peak grid window
            R["net_snr"].append(sn); R["mtot"].append(mt); R["is_um"].append(um); R["off"].append(off)
            R["sigH"].append(float(sH[i,j])); R["sigL"].append(float(sL[i,j])); R["net"].append(float(net[i,j]))
            R["coh"].append(float(coh[i,j])); R["cenH"].append(float(cH[i,j])); R["cenL"].append(float(cL[i,j]))
            R["gH"].append(float(gH[i,j])); R["gL"].append(float(gL[i,j]))
            csH,vH,eH=mr.morph_one_cam(qtH[fi_],whH[fi_].astype(np.float64),int(camH[fi_])); csL,vL,eL=mr.morph_one_cam(qtL[fi_],whL[fi_].astype(np.float64),int(camL[fi_]))
            R["chirpH"].append(csH); R["vertH"].append(vH); R["eccH"].append(eH)
            R["chirpL"].append(csL); R["vertL"].append(vL); R["eccL"].append(eL)
            if INJ_CNN:
                import driver_blindscan as B
                hm_,lm_=B.cnn_from_wh(whH[fi_][None,:],whL[fi_][None,:])
                R["cnn_hm"].append(hm_); R["cnn_lm"].append(lm_)
                if ASDVETO:
                    if LOCAL_ALL or max(hm_,lm_)>B.GLITCH_THRESH:   # LOCAL_ALL also pays for gate-failures (the complement test)
                        r=_veto_local(pipe,rawH,rawL,meta[i][4],off,meta[i][5],meta[i][6],XH[fi_],XL[fi_])
                    else:
                        r=dict(net_loc=np.nan,hm_loc=np.nan,lm_loc=np.nan,veto_keep=0)
                    for kk in ("net_loc","hm_loc","lm_loc","veto_keep"): R[kk].append(r[kk])
        print(f"  snr_net={snr}: {ninj} inj, peak-net med {np.median(net.max(1)):.1f} | net>=4 frac {np.mean(net.max(1)>=4.0):.2f} ({(time.time()-ta)/60:.1f}m)",flush=True)
    off=np.array(R["off"]); print(f"[inj] offset uniformity: min {off.min()} max {off.max()} (step {STEP}) mean {off.mean():.0f}~{STEP/2:.0f}",flush=True)
    np.savez(f"{OUT}/{name}_inj.npz",**{k:np.array(v) for k,v in R.items()})
    print(f"[inj] saved {OUT}/{name}_inj.npz ({len(R['net'])} inj, {int(np.sum(R['is_um']))} UM)",flush=True)

if __name__=="__main__": main()

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
SEG=json.load(open(os.environ.get("SM_SEGJSON_EV","search_mode/segments.json"))); EVENTS=DS.EVENTS
BANK_SIG=os.environ.get("SM_BANK_SIG",os.path.join(MADGRAV_ROOT,"data","o1_o3_signal_bank_projected_2s_x10"))
BANK_UM=os.environ.get("SM_BANK_UM",os.path.join(MADGRAV_ROOT,"bank","ultramassive_bank"))
NET_SNR_GRID=[8.,10.,12.,15.,20.,25.]
N_PER=300; UM_FRAC=0.5; NGRID=3            # score +-1 grid window around the injection -> peak
rng=np.random.default_rng(20260614)

def score_block(pipe,arms,X,det):
    wh=pipe._whiten(X.astype(np.float32),det); qt=DS.build_qt(pipe,wh)
    return DS.sigma_from_qt(pipe,qt,det), DS.g_from_qt(arms,qt), pipe._centroid(wh), wh, qt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--event",required=True); a=ap.parse_args(); name=a.event
    pipe=MassiveEventPipeline(calib_path=f"{SC}/massive_calibration_BA.json",prep=O4A,device=DEV)
    arms=[DS.GlitchArm().to(DEV) for _ in range(5)]
    for i,arm in enumerate(arms): arm.load_state_dict(torch.load(f"{LRD}/p1v42/arm_deploy_seed{i}.pt",map_location=DEV)); arm.eval()
    asd=pipe.asd
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
                      "chirpH","vertH","eccH","chirpL","vertL","eccL")}
    ta=time.time()
    for snr in NET_SNR_GRID:
        XH=[];XL=[];meta=[]
        for _ in range(N_PER):
            um=int(rng.random()<UM_FRAC); WH,WL,MT=banks["um" if um else "sig"]; k=int(rng.integers(0,len(WH)))
            wH=np.asarray(WH[k],np.float32); wL=np.asarray(WL[k],np.float32); L=len(wH)
            s0=np.sqrt(ip.compute_optimal_snr(wH,asd["H1"])**2+ip.compute_optimal_snr(wL,asd["L1"])**2)
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
            meta.append((snr,float(MT[k]),um,off))
        XH=np.stack(XH); XL=np.stack(XL)
        sH,gH,cH,whH,qtH=score_block(pipe,arms,XH,"H1"); sL,gL,cL,whL,qtL=score_block(pipe,arms,XL,"L1")
        camH=mr.cam_t0_batch(arms[0],qtH,DEV); camL=mr.cam_t0_batch(arms[0],qtL,DEV)   # Grad-CAM ROI centers
        coh=pipe._coherence(whH,whL); net=(sH+sL)/np.sqrt(2.0)
        ninj=len(meta)
        net=net.reshape(ninj,NGRID); sH=sH.reshape(ninj,NGRID); sL=sL.reshape(ninj,NGRID)
        gH=gH.reshape(ninj,NGRID); gL=gL.reshape(ninj,NGRID); cH=cH.reshape(ninj,NGRID)
        cL=cL.reshape(ninj,NGRID); coh=coh.reshape(ninj,NGRID); pk=np.argmax(net,1)
        for i in range(ninj):
            j=pk[i]; sn,mt,um,off=meta[i]; fi_=i*NGRID+j   # flat index of peak grid window
            R["net_snr"].append(sn); R["mtot"].append(mt); R["is_um"].append(um); R["off"].append(off)
            R["sigH"].append(float(sH[i,j])); R["sigL"].append(float(sL[i,j])); R["net"].append(float(net[i,j]))
            R["coh"].append(float(coh[i,j])); R["cenH"].append(float(cH[i,j])); R["cenL"].append(float(cL[i,j]))
            R["gH"].append(float(gH[i,j])); R["gL"].append(float(gL[i,j]))
            csH,vH,eH=mr.morph_one_cam(qtH[fi_],whH[fi_].astype(np.float64),int(camH[fi_])); csL,vL,eL=mr.morph_one_cam(qtL[fi_],whL[fi_].astype(np.float64),int(camL[fi_]))
            R["chirpH"].append(csH); R["vertH"].append(vH); R["eccH"].append(eH)
            R["chirpL"].append(csL); R["vertL"].append(vL); R["eccL"].append(eL)
        print(f"  snr_net={snr}: {ninj} inj, peak-net med {np.median(net.max(1)):.1f} | net>=4 frac {np.mean(net.max(1)>=4.0):.2f} ({(time.time()-ta)/60:.1f}m)",flush=True)
    off=np.array(R["off"]); print(f"[inj] offset uniformity: min {off.min()} max {off.max()} (step {STEP}) mean {off.mean():.0f}~{STEP/2:.0f}",flush=True)
    np.savez(f"{OUT}/{name}_inj.npz",**{k:np.array(v) for k,v in R.items()})
    print(f"[inj] saved {OUT}/{name}_inj.npz ({len(R['net'])} inj, {int(np.sum(R['is_um']))} UM)",flush=True)

if __name__=="__main__": main()

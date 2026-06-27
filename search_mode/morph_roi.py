"""Grad-CAM-localized ROI morphology (shared by inject.py and driver_search_cascade.py).
The ROI is placed at the CNN glitch-arm's Grad-CAM attention peak (the LEARNED signal locator),
NOT the energy-argmax (which collapses onto noise for marginal signals). chirpslope is CLIPPED."""
import numpy as np, torch
import torch.nn.functional as Fn
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
FS=4096; WN=4*4096; CHIRP_CLIP=1500.0; CTX_FS=8192   # build_qt center-crop ~2s -> 128 cols span CTX_FS samples

def cam_t0_batch(arm, qt, dev, chunk=256):
    """Grad-CAM attention peak time-column (clamped interior 14..113) per tile. qt: [n,256,128]."""
    out=np.empty(len(qt),int)
    for c0 in range(0,len(qt),chunk):
        sub=qt[c0:c0+chunk]
        x=torch.from_numpy(sub[:,None]).float().to(dev).requires_grad_(True); feat={}
        def hk(m,i,o): feat['a']=o; o.retain_grad()
        h=arm.blocks[3].register_forward_hook(hk); arm.zero_grad(); g=arm(x); g.sum().backward(); h.remove()
        A=feat['a']; w=A.grad.mean(dim=(2,3),keepdim=True); cam=(w*A).sum(1,keepdim=True).clamp(min=0)
        cam=Fn.interpolate(cam,size=(256,128),mode="bilinear",align_corners=False)[:,0]
        camt=cam.sum(1).detach().cpu().numpy()                       # [m,128] attention per time-col
        for i in range(len(sub)): out[c0+i]=int(np.clip(np.argmax(gaussian_filter1d(camt[i],2)),14,113))
    return out

def morph_one_cam(qt, wl, t0):
    """qt: [256,128] min-max tile; wl: [WN] whitened series; t0: CAM time-col -> [chirpslope,vert,ecc]."""
    P=qt**2; wt=14; a=max(0,t0-wt); b=min(128,t0+wt+1); roiP=P[:,a:b]
    thr=np.percentile(roiP,90); ys,xs=np.where(roiP>=thr); ww=roiP[ys,xs]
    if len(ww)<8: vert,ecc=1.0,0.0
    else:
        xm=(xs*ww).sum()/ww.sum(); ym=(ys*ww).sum()/ww.sum()
        vx=(ww*(xs-xm)**2).sum()/ww.sum(); vy=(ww*(ys-ym)**2).sum()/ww.sum(); vxy=(ww*(xs-xm)*(ys-ym)).sum()/ww.sum()
        l1=0.5*((vx+vy)+np.sqrt((vx-vy)**2+4*vxy**2)); l2=0.5*((vx+vy)-np.sqrt((vx-vy)**2+4*vxy**2))
        vert=float(vy/(vx+1e-9)); ecc=float((l1-l2)/(l1+l2+1e-9))
    pk=WN//2+int((t0-64)*(CTX_FS/128.0))                              # CAM time-col -> sample in whitened window
    pk=int(np.clip(pk,int(0.2*FS),WN-int(0.2*FS)))
    X=np.fft.rfft(wl); f=np.fft.rfftfreq(len(wl),1/FS); X[(f<20)|(f>150)]=0; xb=np.fft.irfft(X,n=len(wl))
    half=int(0.12*FS); s=xb[pk-half:pk+half]; av=hilbert(s); en=np.abs(av)
    fi=np.diff(np.unwrap(np.angle(av)))/(2*np.pi)*FS; w2=en[:-1]**2; m=w2>0.3*w2.max()
    if m.sum()<6: slope=0.0
    else:
        t=np.arange(len(fi))[m]; fw=fi[m]; w2m=w2[m]; tm=(t*w2m).sum()/w2m.sum(); fm=(fw*w2m).sum()/w2m.sum()
        slope=abs(((w2m*(t-tm)*(fw-fm)).sum()/((w2m*(t-tm)**2).sum()+1e-9))*FS)
    return float(min(slope,CHIRP_CLIP)),vert,ecc

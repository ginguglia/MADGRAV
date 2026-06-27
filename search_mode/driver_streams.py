"""C5 search-driver, STAGE A (v2, O4a-ASD + arm g): cache per-detector streams ONCE per detector.
Per window (0.25s stride) per detector, cache: sigma (CAE recon, frozen norm), centroid,
central-1s band-limited whitened series (coherence input), AND g (5-seed deploy-arm ensemble
logit) computed on the SAME QT tile. WHITENING USES THE O4a REFERENCE ASD (the run-matched ASD;
validated: event-centered sigma/centroid/g reproduce the retrospective values exactly).
Self-consistent search-mode features: time-slide background (noise) + injections (signal) +
events are all scored in THIS domain, then a search LR is re-fit (no cross-run leakage).
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn
import multiprocessing as _mp
from scipy.ndimage import zoom
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import improved_pipeline as ip
from massive_pipeline import MassiveEventPipeline

SC=os.path.join(MADGRAV_ROOT,"spectrogram_cascade")
LRD=os.path.join(MADGRAV_ROOT,"lr_cascade")
O4A=os.environ.get("SM_PREP",os.path.join(MADGRAV_ROOT,"data","o3a_search_prep"))
STR=os.environ.get("SM_STRAIN","search_mode/strain"); OUT="search_mode/streams_o4a"; os.makedirs(OUT,exist_ok=True)
SEG=json.load(open(os.environ.get("SM_SEGJSON_EV","search_mode/o3a_segments_event.json")))
FS=4096; WIN=4.0; WN=int(WIN*FS); STRIDE=0.25; DEV=os.environ.get("SM_DEV","cuda:1")   # box convention: default cuda:1; override via SM_DEV
EVENTS=json.load(open(os.environ["SM_EVENTSJSON"])) if os.environ.get("SM_EVENTSJSON") else {"GW231028_153006":1382542224.3,"GW231123_135430":1384782888.7}
_pool=None
# ---- FORK-AFTER-CUDA FIX -------------------------------------------------------------------------
# The QT worker pool used to be a default *fork* Pool created lazily inside build_qt(), i.e. AFTER
# cpipe()/carm() had initialized a CUDA context in this process. fork-after-CUDA-init is undefined
# behavior (torch 1.12 / CUDA 11.2 / driver 580 / consumer Ada) and can deadlock the whole machine.
# Use a "forkserver" context instead: the forkserver daemon is started via a clean exec (NOT a fork of
# this CUDA-poisoned process), so workers are forked from a pristine, CUDA-free server -- safe even when
# pool() is first called after CUDA init. We preload the worker's module (improved_pipeline) in the
# server so each worker has it ready; that import is CUDA-free (verified: importing improved_pipeline
# does not call torch.cuda / .to(cuda) / instantiate any model at module scope), so no GPU contexts are
# created in the workers. Worker output is byte-identical to the old fork pool (same function, same args,
# same numpy/gwpy code path -- forkserver only changes HOW the child process is created, not what it runs).
_QT_CTX=_mp.get_context("forkserver")
_QT_CTX.set_forkserver_preload(["improved_pipeline"])
# Worker count is a knob (SM_QT_WORKERS). The QT-magnitude step is the CPU bottleneck and was previously
# run serially in the parent (~2/64 cores). gwpy q_transform is itself FFT/BLAS-threaded (BLAS thread caps
# unset -> uses many cores per worker), so process-level scaling SATURATES around ~16 workers (measured:
# 16/48/64 all ~31 keys/s); going higher only oversubscribes the shared 64-core box. Default 16 = the
# saturation point with headroom for the parent / GPU feed / other users. Clamped to >=1.
_QT_WORKERS=max(1,int(os.environ.get("SM_QT_WORKERS","16")))
def pool():
    global _pool
    if _pool is None: _pool=_QT_CTX.Pool(_QT_WORKERS)
    return _pool

class GlitchArm(nn.Module):
    def __init__(s):
        super().__init__(); ch=[1,16,32,64,128]
        s.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i],ch[i+1],3,padding=1),nn.BatchNorm2d(ch[i+1]),nn.ReLU(),nn.MaxPool2d(2)) for i in range(4)])
        s.head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x):
        for b in s.blocks: x=b(x)
        return s.head(x).squeeze(-1)

def build_qt(pipe, white):
    qi=ip.center_crop_waveforms(white, sample_rate=FS, context_seconds=pipe.ctx)
    args=[(w,FS,ip.QTRANSFORM_FRANGE,ip.QTRANSFORM_QRANGE,1.0) for w in qi]
    mags=pool().map(ip._compute_qt_image_worker, args)
    return ip.min_max_norm(np.stack([zoom(m,(256/m.shape[0],128/m.shape[1]),order=1) for m in mags]).astype(np.float32)).astype(np.float32)

def sigma_from_qt(pipe, qt, det):
    mu,sd=(pipe.norm["muH"],pipe.norm["sdH"]) if det=="H1" else (pipe.norm["muL"],pipe.norm["sdL"])
    return (pipe._recon(qt)-mu)/sd

def g_from_qt(arms, qt, bs=512):
    out=np.empty(len(qt))
    with torch.no_grad():
        for c0 in range(0,len(qt),bs):
            x=torch.from_numpy(qt[c0:c0+bs,None]).to(DEV)
            out[c0:c0+len(x)]=np.mean([a(x).cpu().numpy() for a in arms],axis=0)
    return out

def coh_series(pipe, white):
    cc=white.shape[1]//2; half=int(pipe.coh_win*FS/2)
    a=white[:,cc-half:cc+half].astype(np.float64); a=pipe._bandlimit(a); a=a-a.mean(1,keepdims=True)
    return a.astype(np.float32)

def run(name, det, pipe, arms, batch=2048):
    t0=SEG[name]["coincident_lock"][0]
    raw=np.load(f"{STR}/{name}_{det}.npz")["strain"]
    step=int(STRIDE*FS); n=(len(raw)-WN)//step+1
    idx=np.arange(n)*step; gps=t0+idx/FS+WIN/2.0
    half=int(pipe.coh_win*FS/2); slen=2*half
    sig=np.empty(n); cen=np.empty(n); g=np.empty(n)
    ser=np.lib.format.open_memmap(f"{OUT}/{name}_{det}_series.npy",mode="w+",dtype=np.float32,shape=(n,slen))
    ta=time.time()
    for c0 in range(0,n,batch):
        cs=idx[c0:c0+batch]
        wb=pipe._whiten(np.stack([raw[i:i+WN] for i in cs]).astype(np.float32), det)
        qt=build_qt(pipe, wb)
        sig[c0:c0+len(cs)]=sigma_from_qt(pipe, qt, det)
        g[c0:c0+len(cs)]=g_from_qt(arms, qt)
        cen[c0:c0+len(cs)]=pipe._centroid(wb)
        ser[c0:c0+len(cs)]=coh_series(pipe, wb)
        if (c0//batch)%5==0: print(f"  {name} {det}: {c0+len(cs)}/{n} ({(time.time()-ta)/60:.1f}m)",flush=True)
    np.savez(f"{OUT}/{name}_{det}_meta.npz", gps=gps, sigma=sig, centroid=cen, g=g, t0=t0,
             stride=STRIDE, n=n, slen=slen, det=det)
    ser.flush()
    print(f"DONE {name} {det}: n={n} in {(time.time()-ta)/60:.1f}m sigma[max]={sig.max():.2f} g[max]={g.max():.2f}",flush=True)

def main():
    pipe=MassiveEventPipeline(calib_path=f"{SC}/massive_calibration_BA.json", prep=O4A, device=DEV)
    arms=[GlitchArm().to(DEV) for _ in range(5)]
    for i,a in enumerate(arms): a.load_state_dict(torch.load(f"{LRD}/p1v42/arm_deploy_seed{i}.pt",map_location=DEV)); a.eval()
    print(f"[streams-o4a] pipe on {pipe.device} ASD=O4a tcoh={pipe.tcoh:.4f} fcut={pipe.f_cut:.1f} +5-seed arm",flush=True)
    for name in EVENTS:
        for det in ("H1","L1"):
            if os.path.exists(f"{OUT}/{name}_{det}_meta.npz"): print(f"skip {name} {det}",flush=True); continue
            run(name, det, pipe, arms)
    open(f"{OUT}/STREAMS_DONE","w").close(); print("[streams-o4a] STREAMS_DONE",flush=True)

if __name__=="__main__": main()

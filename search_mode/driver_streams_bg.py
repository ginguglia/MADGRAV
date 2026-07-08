"""Lean per-segment streams (sigma, centroid, g) for the all-vs-all background segments.
ON-DEMAND coherence policy: NO series cache (strain retained; coherence recomputed for survivors).
Reuses driver_streams (O4a ASD + 5-seed arm). -> streams_bg/<segname>_{det}_meta.npz
"""
import os,sys,json,time
import numpy as np
# --- MUSICA port: MADGRAV_ROOT-relative sys.path (was a hardcoded dev-tree path) ---
MADGRAV_ROOT=os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import driver_streams as DS
import torch
from massive_pipeline import MassiveEventPipeline
STR=os.environ.get("SM_STRAIN","search_mode/strain"); OUT=os.environ.get("SM_STREAMS","search_mode/streams_bg"); os.makedirs(OUT,exist_ok=True)
SEG=json.load(open(os.environ.get("SM_BGJSON","search_mode/bg_segments.json"))); FS=4096; WN=4*FS; STRIDE=1.0   # 1s stride (4x faster; background-appropriate)
VETO=json.load(open(os.environ["SM_VETO"])) if (os.environ.get("SM_VETO") and os.path.exists(os.environ["SM_VETO"])) else None   # CAT2 mask: {segname:{good_segments:[[lo,hi],...]}}
def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"
def cat2_mask(name,gps):                              # True where window center is OUTSIDE all good intervals -> veto
    if not VETO or name not in VETO: return np.zeros(len(gps),bool)
    good=VETO[name].get("good_segments",[]); inside=np.zeros(len(gps),bool)
    for lo,hi in good: inside|=(gps>=lo)&(gps<=hi)
    return ~inside
def run(name,a,det,pipe,arms,batch=2048):
    raw=np.load(f"{STR}/{name}_{det}.npz")["strain"]; step=int(STRIDE*FS); n=(len(raw)-WN)//step+1
    idx=np.arange(n)*step; gps=a+idx/FS+2.0
    sig=np.empty(n); cen=np.empty(n); g=np.empty(n); ta=time.time()
    for c0 in range(0,n,batch):
        cs=idx[c0:c0+batch]
        wb=pipe._whiten(np.stack([raw[i:i+WN] for i in cs]).astype(np.float32),det)
        qt=DS.build_qt(pipe,wb)
        sig[c0:c0+len(cs)]=DS.sigma_from_qt(pipe,qt,det); g[c0:c0+len(cs)]=DS.g_from_qt(arms,qt); cen[c0:c0+len(cs)]=pipe._centroid(wb)
    vmask=cat2_mask(name,gps); sig=np.where(vmask,-1e9,sig)   # CAT2-vetoed windows -> never trigger / enter bg
    _tmp=f"{OUT}/.{name}_{det}_meta.{os.getpid()}.tmp.npz"   # atomic write: tmp -> os.replace, so a killed shard never leaves a truncated .npz that os.path.exists would skip
    np.savez(_tmp,gps=gps,sigma=sig,centroid=cen,g=g,t0=a,stride=STRIDE,n=n,det=det,n_vetoed=int(vmask.sum()))
    os.replace(_tmp,f"{OUT}/{name}_{det}_meta.npz")
    print(f"DONE {name} {det}: n={n} vetoed={int(vmask.sum())} in {(time.time()-ta)/60:.1f}m",flush=True)
def main():
    pipe=MassiveEventPipeline(calib_path=f"{DS.SC}/massive_calibration_BA.json",prep=DS.O4A,device=DS.DEV)
    arms=[DS.GlitchArm().to(DS.DEV) for _ in range(5)]
    for i,arm in enumerate(arms): arm.load_state_dict(torch.load(f"{DS.LRD}/p1v42/arm_deploy_seed{i}.pt",map_location=DS.DEV)); arm.eval()
    print(f"[streams_bg] O4a ASD + arm on {pipe.device}",flush=True)
    for a,b,d,nm in SEG["segments"]:
        name=segname(a,nm)              # build ALL segments (incl event locks) at 1s into streams_bg for a uniform grid
        for det in ("H1","L1"):
            if os.path.exists(f"{OUT}/{name}_{det}_meta.npz"): print(f"skip {name} {det}",flush=True); continue
            if not os.path.exists(f"{STR}/{name}_{det}.npz"): print(f"MISSING strain {name} {det} -- skip",flush=True); continue
            run(name,a,det,pipe,arms)
    open(f"{OUT}/STREAMS_BG_DONE","w").close(); print("[streams_bg] DONE",flush=True)
if __name__=="__main__": main()

"""All-vs-all multi-segment search-mode background (PILOT) -- v3, post validation-panel.
v3 fixes (cross-fit FAR-count audit): (1) a background survivor is counted ONLY against the event
scored by the SAME held-out model (event-fold == survivor-pairing-fold), so loglr is always compared
on a common per-fold scale -- no cross-model mixing; (2) per-event livetime = that event's OWN fold's
background (not the pooled total); (3) injections split BY SOURCE EVENT (each event's injections go to
the fold opposite the model that scores it -> no own-waveform leak); (4) louder survivors persisted
with identities (seg/gps/sigma/loglr) for hardware-injection / real-event screening.
v2 (post code-review): same-fold-only cross-fit, storm-lag slides, balanced noise, nL1 Poisson UL.
Scores GW231028 + GW231123, each against its own-fold background. -> search_out/multi_<event>.json
Run: python driver_search_multi.py
"""
import os,sys,json,time
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import driver_streams as DS
from massive_pipeline import MassiveEventPipeline
SEG=json.load(open(os.environ.get("SM_BGJSON","search_mode/bg_segments.json")))
S_O4A="search_mode/streams_o4a"; S_BG=os.environ.get("SM_STREAMS","search_mode/streams_bg"); STRAIN=os.environ.get("SM_STRAIN","search_mode/strain")
INJ=os.environ.get("SM_INJ","search_mode/inj_out"); OUT=os.environ.get("SM_OUT","search_mode/search_out")
cal=json.load(open("spectrogram_cascade/massive_calibration_BA.json")); TCOH=cal["tcoh"]; LAGSAMP=int(cal["lag_samples"])
FS=4096; WN=4*FS; STRIDE=1.0; YR=3.1557e7; ANALYZED_FRAC=0.5; GCLIP=6.0; MERGE_S=4.0   # 1s stride (4x faster); ANALYZED_FRAC=2s-crop/4s-window (stride-independent, matches single-stretch)
RIDGE=1.0; SAME_SEG_MINLAG_S=3600.0; OFFSET_STEP_S=4.0; MAX_OFFSETS_PER_PAIR=int(os.environ.get("SM_MAX_OFFSETS","400"))   # sampled count per pairing (env-overridable for cheap subsample tests)
EVENTS=json.load(open(os.environ["SM_EVENTSJSON"])) if os.environ.get("SM_EVENTSJSON") else {"GW231028_153006":1382542224.3,"GW231123_135430":1384782888.7}
RNG=np.random.default_rng(20260614)
def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"
def gate(g,s): return np.clip(g,-GCLIP,GCLIP)*np.clip(np.asarray(s)/3.0,0,1)
def have(a,nm):
    n=segname(a,nm); d=S_BG          # all 8 segments built at 1s into streams_bg (uniform grid)
    return os.path.exists(f"{d}/{n}_H1_meta.npz") and os.path.exists(f"{d}/{n}_L1_meta.npz")
def load_seg(a,nm):
    n=segname(a,nm); d=S_BG
    mH=np.load(f"{d}/{n}_H1_meta.npz"); mL=np.load(f"{d}/{n}_L1_meta.npz"); k=min(int(mH["n"]),int(mL["n"]))
    return dict(name=n,t0=float(a),n=k,gps=mH["gps"][:k],sH=mH["sigma"][:k],sL=mL["sigma"][:k],
                cH=mH["centroid"][:k],cL=mL["centroid"][:k],gH=mH["g"][:k],gL=mL["g"][:k])
def feats(sH,sL,coh,cH,cL,gH,gL): return np.column_stack([sH,sL,coh,cH,cL,gate(gH,sH),gate(gL,sL)])
def loglr(mu,sd,beta,F): return beta[0]+((np.asarray(F,float)-mu)/sd)@beta[1:]
def fit_lr(Xn,Xs):                       # per-fold mu/sd computed HERE from this fold's data only
    X=np.vstack([Xn,Xs]); y=np.concatenate([np.zeros(len(Xn)),np.ones(len(Xs))])
    mu=X.mean(0); sd=X.std(0)+1e-9; Z=np.column_stack([np.ones(len(X)),(X-mu)/sd])
    w=np.where(y==1,len(y)/(2*max(1,len(Xs))),len(y)/(2*max(1,len(Xn))))
    def nll(b):
        p=1/(1+np.exp(-(Z@b))); return -np.sum(w*(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)))+RIDGE*np.sum(b[1:]**2)
    def grad(b):
        p=1/(1+np.exp(-(Z@b))); g=Z.T@(w*(p-y)); g[1:]+=2*RIDGE*b[1:]; return g
    bnds=[(None,None)]*Z.shape[1]; bnds[3]=(0,None)
    return mu,sd,minimize(nll,np.zeros(Z.shape[1]),jac=grad,method="L-BFGS-B",bounds=bnds).x
def pair_idx(nA,nB,off):
    if off>=0: iH=np.arange(0,max(0,min(nA,nB-off))); iL=iH+off
    else: iL=np.arange(0,max(0,min(nB,nA+off))); iH=iL-off
    return iH,iL
# ---- on-demand coherence (FFT-vectorized) ----
_pipe=None
def pipe():
    global _pipe
    if _pipe is None: _pipe=MassiveEventPipeline(calib_path=f"{DS.SC}/massive_calibration_BA.json",prep=DS.O4A,device=DS.DEV)
    return _pipe
# ---- whitened coh-series cache (free-win #2): build the stride-1.0 series memmap ONCE per epoch, then
# series_for() is a float64 mmap-slice instead of re-whitening 235MB raw every call (~768x on the read).
# Guarded by a config sidecar so a stride/coh_win/band/ASD change can never silently mis-index. ----
SERIESCACHE=os.environ.get("SM_SERIESCACHE", f"{STRAIN}/_seriescache")
def _series_cfg(p,n):
    return dict(stride=float(STRIDE),coh_win=float(p.coh_win),slen=int(2*int(p.coh_win*FS/2)),
                f_cut=float(p.f_cut),tcoh=float(p.tcoh),prep=str(DS.O4A),wn=int(WN),fs=int(FS),n=int(n))
def _series_paths(name,det): return f"{SERIESCACHE}/{name}_{det}_series.npy", f"{SERIESCACHE}/{name}_{det}_series.json"
def _compute_series(p,raw,ws,det,half):  # SINGLE source of truth -> cached & uncached are bit-identical
    step=int(STRIDE*FS); wb=p._whiten(np.stack([raw[w*step:w*step+WN] for w in ws]).astype(np.float32),det)
    cc=wb.shape[1]//2; a=wb[:,cc-half:cc+half].astype(np.float64); a=p._bandlimit(a); a-=a.mean(1,keepdims=True)
    return a
def build_series_cache(name,det,batch=2048,log=None):
    p=pipe(); raw=np.load(f"{STRAIN}/{name}_{det}.npz")["strain"]; step=int(STRIDE*FS)
    n=(len(raw)-WN)//step+1; half=int(p.coh_win*FS/2); slen=2*half
    os.makedirs(SERIESCACHE,exist_ok=True); npy,js=_series_paths(name,det)
    mm=np.lib.format.open_memmap(npy,mode="w+",dtype=np.float64,shape=(n,slen))
    for c0 in range(0,n,batch):
        ws=np.arange(c0,min(c0+batch,n)); mm[c0:c0+len(ws)]=_compute_series(p,raw,ws,det,half)
        if log and (c0//batch)%10==0: log(f"  {name} {det}: {c0+len(ws)}/{n}")
    mm.flush(); json.dump(_series_cfg(p,n),open(js,"w"))
    return n
def series_for(name,det,widx):
    p=pipe(); half=int(p.coh_win*FS/2); npy,js=_series_paths(name,det)
    if os.path.exists(npy) and os.path.exists(js):
        try:
            cfg=json.load(open(js)); mm=np.load(npy,mmap_mode="r"); want=_series_cfg(p,mm.shape[0])
            # host-safety panel wf_9910fdb7: advise the kernel this read-only memmap is random-access so it does
            # NOT prefetch (readahead amplification) and can drop pages under pressure. RESULT-IDENTICAL (the
            # np.array(mm[w]) below returns a private copy; advice only changes kernel page management). Linux-only,
            # fully guarded.
            try:
                import mmap as _mmaplib
                if getattr(mm,"_mmap",None) is not None and hasattr(mm._mmap,"madvise") and hasattr(_mmaplib,"MADV_RANDOM"):
                    mm._mmap.madvise(_mmaplib.MADV_RANDOM)
            except Exception: pass
            if all(cfg.get(k)==want.get(k) for k in ("stride","coh_win","slen","f_cut","tcoh","prep","wn","fs")):
                return {int(w):np.array(mm[w]) for w in widx}      # bit-identical float64 slice
        except Exception: pass                                     # stale/corrupt cache -> fall through to recompute
    raw=np.load(f"{STRAIN}/{name}_{det}.npz")["strain"]; out={}
    for c0 in range(0,len(widx),1024):
        ws=list(widx[c0:c0+1024]); a=_compute_series(p,raw,ws,det,half)
        for k,w in enumerate(ws): out[int(w)]=a[k]
    return out
# ---- GPU-batched coherence (free-win #1): torch.fft float64 on cuda:1; numpy-identical to ~1e-16.
# numpy fallback when no CUDA preserves the exact original path. batch<=25k (~6.6GB on shared cuda:1). ----
COH_BATCH=int(os.environ.get("SM_COH_BATCH","20000"))
_torch=None
def _torch_mod():
    global _torch
    if _torch is None:
        try: import torch as _t; _torch=_t
        except Exception: _torch=False
    return _torch
def coh_vec(A,B):                        # A,B: [m,L] aligned series -> BA symmetric-norm coherence (FFT)
    if len(A)==0: return np.empty(0)
    th=_torch_mod()
    if th is False or not th.cuda.is_available():        # numpy path (unchanged) when no GPU
        L=A.shape[1]; FA=np.fft.rfft(A,axis=1); FB=np.fft.rfft(B,axis=1)
        cc=np.fft.irfft(FA*np.conj(FB),n=L,axis=1); win=np.concatenate([cc[:,:LAGSAMP+1],cc[:,-LAGSAMP:]],axis=1)
        return (2.0*np.abs(win).max(1)/((A*A).sum(1)+(B*B).sum(1)+1e-30))
    dev=DS.DEV; out=np.empty(len(A))
    for c0 in range(0,len(A),COH_BATCH):
        a=th.from_numpy(np.ascontiguousarray(A[c0:c0+COH_BATCH],dtype=np.float64)).to(dev)
        b=th.from_numpy(np.ascontiguousarray(B[c0:c0+COH_BATCH],dtype=np.float64)).to(dev)
        L=a.shape[1]; cc=th.fft.irfft(th.fft.rfft(a,dim=1)*th.conj(th.fft.rfft(b,dim=1)),n=L,dim=1)
        win=th.cat([cc[:,:LAGSAMP+1],cc[:,-LAGSAMP:]],dim=1)
        out[c0:c0+a.shape[0]]=((2.0*win.abs().amax(dim=1))/((a*a).sum(1)+(b*b).sum(1)+1e-30)).cpu().numpy()
    return out

def main():
    avail=[(a,nm) for a,b,d,nm in SEG["segments"] if have(a,nm)]
    segs=[load_seg(a,nm) for a,nm in avail]; nseg=len(segs); fold=[i%2 for i in range(nseg)]
    assert nseg>=2 and 0 in fold and 1 in fold, "need >=2 segments spanning both folds"
    print(f"[multi] {nseg} segs {[s['name'] for s in segs]} folds={fold}",flush=True)
    step_off=int(round(OFFSET_STEP_S/STRIDE)); same_minoff=int(round(SAME_SEG_MINLAG_S/STRIDE)); evrad=int(2.0/STRIDE)
    evloc={ev:(i,int(np.abs(segs[i]["gps"]-g0).argmin())) for ev,g0 in EVENTS.items() for i,s in enumerate(segs) if s["name"]==ev}
    assert set(evloc)==set(EVENTS), "event segment(s) missing"
    def is_ev(si,idx):
        m=np.zeros(len(idx),bool)
        for ev,(esi,ew) in evloc.items():
            if si==esi: m|=np.abs(idx-ew)<=evrad
        return m
    # injections: split BY SOURCE EVENT. Event in fold g is scored by M[1-g] (fit on sf==1-g);
    # send the event's own injections to sf=g so M[1-g] never trains on the event's own waveform.
    Xs_all=[]; sf_list=[]
    for ev in EVENTS:
        za=np.load(f"{INJ}/{ev}_inj.npz"); X=feats(za["sigH"],za["sigL"],za["coh"],za["cenH"],za["cenL"],za["gH"],za["gL"])
        Xs_all.append(X); sf_list.append(np.full(len(X),fold[evloc[ev][0]],dtype=int))
    Xs_all=np.vstack(Xs_all); sf=np.concatenate(sf_list)
    # SAME-FOLD pairings only (both detector segments in one fold -> scored by other fold = both held out)
    pairings=[]
    for ai in range(nseg):
        for bi in range(nseg):
            if fold[ai]!=fold[bi]: continue
            if ai==bi:
                hi=segs[ai]["n"]-step_off
                cand=np.arange(same_minoff,hi+1,step_off)
                if len(cand)==0: continue
                cand=np.concatenate([cand,-cand])
            else:
                hi=min(segs[ai]["n"],segs[bi]["n"])-step_off
                cand=np.arange(-hi,hi+1,step_off)
            if len(cand)>MAX_OFFSETS_PER_PAIR: cand=RNG.choice(cand,MAX_OFFSETS_PER_PAIR,replace=False)
            for o in cand: pairings.append((ai,bi,int(o)))
    RNG.shuffle(pairings)
    print(f"[multi] {len(pairings)} same-fold pairings (offset step {OFFSET_STEP_S}s, cap {MAX_OFFSETS_PER_PAIR}/pair)",flush=True)
    # ---- noise sample per fold (net>=3, event-excised) with REAL coherence (on-demand, subsampled) ----
    NCAP=15000; npr={0:[],1:[]}
    for ai,bi,off in pairings:
        g=fold[ai]
        if len(npr[g])>=NCAP: continue
        A,B=segs[ai],segs[bi]; iH,iL=pair_idx(A["n"],B["n"],off)
        if not len(iH): continue
        net=(A["sH"][iH]+B["sL"][iL])/np.sqrt(2.); m=np.where((net>=3.0)&~(is_ev(ai,iH)|is_ev(bi,iL)))[0]
        for j in m:
            npr[g].append((ai,int(iH[j]),bi,int(iL[j])))
            if len(npr[g])>=NCAP: break
    def gv(key,si,w): return np.array([segs[s][key][ww] for s,ww in zip(si,w)])
    Xn={}; cohnoise={}
    for g in (0,1):
        P=npr[g]
        if not P: Xn[g]=np.zeros((0,7)); continue
        ai_=np.array([p[0] for p in P]); iH_=np.array([p[1] for p in P]); bi_=np.array([p[2] for p in P]); iL_=np.array([p[3] for p in P])
        nH={}; nL={}
        for a_,h_,b_,l_ in P: nH.setdefault(a_,set()).add(h_); nL.setdefault(b_,set()).add(l_)
        SH={a_:series_for(segs[a_]["name"],"H1",sorted(w)) for a_,w in nH.items()}
        SL={b_:series_for(segs[b_]["name"],"L1",sorted(w)) for b_,w in nL.items()}
        co=coh_vec(np.array([SH[a_][h_] for a_,h_,b_,l_ in P]),np.array([SL[b_][l_] for a_,h_,b_,l_ in P]))  # REAL coherence
        cohnoise[g]=co
        Xn[g]=feats(gv("sH",ai_,iH_),gv("sL",bi_,iL_),co,gv("cH",ai_,iH_),gv("cL",bi_,iL_),gv("gH",ai_,iH_),gv("gL",bi_,iL_))
        print(f"[multi] fold {g} noise: {len(P)} rows, real coh med {np.median(co):.3f}",flush=True)
    # fit model[g] on fold-g noise+signal (own mu/sd); a fold-g coincidence is scored by model[1-g]
    M={}
    for g in (0,1): M[g]=fit_lr(Xn[g],Xs_all[sf==g])
    for g in (0,1):
        mu,sd,be=M[g]; ho=np.median(loglr(mu,sd,be,Xn[1-g])) if len(Xn[1-g]) else np.nan
        sg=np.median(loglr(mu,sd,be,Xs_all[sf==(1-g)]))
        print(f"[multi] model[{g}] noise {len(Xn[g])} beta_coh={be[3]:+.3f} HELD-OUT noise_med={ho:.2f} sig_med={sg:.2f}",flush=True)
    # ---- event loglr (held-out model 1-eventfold); on-demand coherence of event window ----
    evres={}
    for ev,(esi,ew) in evloc.items():
        g=fold[esi]; mu,sd,be=M[1-g]
        sH=series_for(segs[esi]["name"],"H1",[ew])[ew]; sL=series_for(segs[esi]["name"],"L1",[ew])[ew]
        co=float(coh_vec(sH[None,:],sL[None,:])[0])
        F=feats([segs[esi]["sH"][ew]],[segs[esi]["sL"][ew]],[co],[segs[esi]["cH"][ew]],[segs[esi]["cL"][ew]],[segs[esi]["gH"][ew]],[segs[esi]["gL"][ew]])
        evres[ev]=dict(loglr=float(loglr(mu,sd,be,F)[0]),net=float((segs[esi]["sH"][ew]+segs[esi]["sL"][ew])/np.sqrt(2)),coh=co,fold=g)
        print(f"[multi] EVENT {ev}: net={evres[ev]['net']:.2f} coh={co:.3f} loglr={evres[ev]['loglr']:.2f} (held-out {1-g})",flush=True)
    # data-driven coherence ceiling for the prune. Real noise/event coherence never approaches 1.0
    # (median ~0.13); capping the prune's coh upper bound at coh=1.0 kept ~9.5M survivors that can
    # NEVER beat the event once their real coh is plugged in -> intractable whitening front-load.
    # Use max OBSERVED coherence (30k-row noise sample + the events) + 0.15 margin. A survivor needing
    # coh above this to win would have to exceed the loudest of ~30k real coincidences (negligible).
    # The final rescore still uses each survivor's REAL coh, so kept survivors are scored exactly.
    obs_coh_max=max([float(c.max()) for c in cohnoise.values() if len(c)]+[e["coh"] for e in evres.values()])
    COH_CEIL=float(min(1.0,obs_coh_max+0.15)); cohmax={g:COH_CEIL for g in (0,1)}
    print(f"[multi] prune coh ceiling = {COH_CEIL:.3f} (max observed {obs_coh_max:.3f}); rescore uses real coh",flush=True)
    # per-fold prune: a fold-g survivor can only ever count against an event in fold g (same held-out
    # model rule), so prune each fold at ITS event's loglr, not the global min. RESULT-IDENTICAL (a
    # survivor with upper-bound up<=its-fold-event-loglr can never have real ll>that loglr); avoids
    # keeping millions of fold-0 survivors that can never beat GW231028's high loglr. +inf if no event.
    thr_fold={g:min([e["loglr"] for e in evres.values() if e["fold"]==g],default=np.inf) for g in (0,1)}
    # ---- background: loglr-upper prune (held-out by fold), event-excised ----
    surv=[]; face_live={0:0.0,1:0.0}; ta=time.time()
    for k,(ai,bi,off) in enumerate(pairings):
        g=fold[ai]; mu,sd,be=M[1-g]; A,B=segs[ai],segs[bi]; iH,iL=pair_idx(A["n"],B["n"],off)
        if not len(iH): continue
        keep=~(is_ev(ai,iH)|is_ev(bi,iL)); iH,iL=iH[keep],iL[keep]
        if not len(iH): continue
        face_live[g]+=len(iH)*STRIDE
        up=loglr(mu,sd,be,feats(A["sH"][iH],B["sL"][iL],np.full(len(iH),cohmax[1-g]),A["cH"][iH],B["cL"][iL],A["gH"][iH],B["gL"][iL]))
        for si in np.where(up>thr_fold[g])[0]: surv.append((ai,int(iH[si]),bi,int(iL[si]),g))
        if k%1000==0: print(f"  pairing {k}/{len(pairings)} ({(time.time()-ta)/60:.1f}m) surv={len(surv)}",flush=True)
    far_live={g:face_live[g]*ANALYZED_FRAC/YR for g in (0,1)}   # per-fold honest livetime
    print(f"[multi] livetime honest fold0={far_live[0]:.2f}yr fold1={far_live[1]:.2f}yr (face {face_live[0]/YR:.2f}/{face_live[1]/YR:.2f}); {len(surv)} survivors",flush=True)
    # ---- on-demand coherence (vectorized) for survivors ----
    needH={}; needL={}
    for ai,iH,bi,iL,g in surv: needH.setdefault(ai,set()).add(iH); needL.setdefault(bi,set()).add(iL)
    serH={ai:series_for(segs[ai]["name"],"H1",sorted(w)) for ai,w in needH.items()}
    serL={bi:series_for(segs[bi]["name"],"L1",sorted(w)) for bi,w in needL.items()}
    louder={ev:[] for ev in evres}
    CHUNK=200000   # bound peak memory of the coherence array regardless of survivor count
    for c0 in range(0,len(surv),CHUNK):
        sub=surv[c0:c0+CHUNK]
        A_=np.array([serH[ai][iH] for ai,iH,bi,iL,g in sub]); B_=np.array([serL[bi][iL] for ai,iH,bi,iL,g in sub])
        coh=coh_vec(A_,B_)
        for j,(ai,iH,bi,iL,g) in enumerate(sub):
            mu,sd,be=M[1-g]
            ll=float(loglr(mu,sd,be,feats([segs[ai]["sH"][iH]],[segs[bi]["sL"][iL]],[coh[j]],[segs[ai]["cH"][iH]],[segs[bi]["cL"][iL]],[segs[ai]["gH"][iH]],[segs[bi]["gL"][iL]]))[0])
            for ev in evres:
                # count a survivor ONLY against the event scored by the SAME held-out model:
                # event fold == survivor pairing fold g  <=>  both scored by M[1-g] (common loglr scale)
                if fold[evloc[ev][0]]==g and ll>evres[ev]["loglr"]: louder[ev].append((ai,iH,bi,iL,ll))
        print(f"  coherence chunk {c0//CHUNK+1}/{(len(surv)+CHUNK-1)//CHUNK} done ({(time.time()-ta)/60:.1f}m)",flush=True)
    # ---- per-event FAR: HEADLINE = distinct-L1-family rate (offset-double-count-safe); analytic Poisson CI ----
    def segkey(si,w): return (si,int(segs[si]["gps"][w]//MERGE_S))
    for ev in evres:
        gev=fold[evloc[ev][0]]; flt=far_live[gev]   # per-event livetime = its OWN fold's background only
        Lr=louder[ev]; n_raw=len(Lr)
        pairkeys=set((segkey(ai,iH),segkey(bi,iL)) for ai,iH,bi,iL,ll in Lr); n_dd=len(pairkeys)
        l1fams=sorted(set(segkey(bi,iL) for ai,iH,bi,iL,ll in Lr)); nL1=len(l1fams)
        dom=0.0
        if Lr:
            byL1=Counter(kk[1] for kk in pairkeys); a=np.array(list(byL1.values())); dom=float(a.max()/a.sum())
        far_L1=nL1/flt if flt>0 else np.nan          # HEADLINE: louder distinct L1 glitch families / yr
        far_dd=n_dd/flt if flt>0 else np.nan          # diagnostic (double-counts across H1 offsets)
        # analytic Poisson 90% interval on nL1 (Garwood); UL90 for the 0/near-0 case
        lo=float(chi2.ppf(0.05,2*nL1)/2/flt) if (nL1>0 and flt>0) else 0.0
        hi=float(chi2.ppf(0.95,2*(nL1+1))/2/flt) if flt>0 else None
        ul90=float(chi2.ppf(0.90,2*(nL1+1))/2/flt) if flt>0 else None
        # persist louder-survivor identities for hardware-injection / real-event screening
        louder_detail=[dict(segH=segs[ai]["name"],gpsH=float(segs[ai]["gps"][iH]),sigH=float(segs[ai]["sH"][iH]),
                            segL=segs[bi]["name"],gpsL=float(segs[bi]["gps"][iL]),sigL=float(segs[bi]["sL"][iL]),
                            loglr=float(ll)) for ai,iH,bi,iL,ll in Lr]
        res=dict(event_loglr=evres[ev]["loglr"],event_net=evres[ev]["net"],event_coh=evres[ev]["coh"],event_fold=gev,
            far_live_yr=flt,n_louder_raw=n_raw,n_louder_dedup=n_dd,n_distinct_L1fam=nL1,L1_dominance=dom,
            far_per_yr=far_L1,far_dd_diag_per_yr=far_dd,far_ci90=[lo,hi],far_ul90_per_yr=ul90,
            is_upper_limit=(nL1==0 or nL1<=2 or dom>0.5),scoring_model_beta_coh=float(M[1-gev][2][3]),
            louder_events=louder_detail,n_segments=nseg,n_pairings=len(pairings))
        json.dump(res,open(f"{OUT}/multi_{ev}.json","w"),indent=2)
        print(f"\n[multi] {ev}: fold={gev} lt={flt:.2f}yr loglr={evres[ev]['loglr']:.2f} louder raw={n_raw} dedup={n_dd} L1fam={nL1} dom={dom:.2f}",flush=True)
        print(f"   FAR(L1-fam)={far_L1:.3g}/yr CI90=[{lo:.3g},{hi:.3g}] UL90={ul90:.3g} (diag far_dd={far_dd:.3g}) -> {'UL' if res['is_upper_limit'] else 'measured'}",flush=True)

if __name__=="__main__": main()

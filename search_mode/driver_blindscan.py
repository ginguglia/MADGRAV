"""BLIND zero-lag foreground search over ALL built segments (search-mode discovery), TWO-STAGE cascade.
Reuses driver_search_multi's EXACT machinery (feats/loglr/fit_lr/coh_vec/series_for/pair_idx + the same
per-fold cross-fit LR model and time-slid background). Instead of scoring 2 named events it scans EVERY
zero-lag H1[i]^L1[i] coincidence with net sigma > NET_CUT, clusters them, and FARs each trigger.

STAGE 1 (LR cascade, full-band):   loglr ranking -> per-trigger FAR_LR + the trigger's louder bg survivors.
STAGE 2 (HM CNN 20-140 veto):      for CANDIDATE triggers (top by loglr + known-event matches), apply the
                                   demotion-only 2-detector CNN veto to the trigger + its louder survivors
                                   (same band-matching cascade_cnn_far uses) -> per-trigger FAR_CNN (20-140).
Cross-fit preserved: a fold-g foreground trigger is FAR'd ONLY against fold-g background (common loglr scale).
-> search_out/blindscan.json   Device via BLIND_DEV (default cuda:1; resolved by _resolve_dev, degrades gracefully).
"""
import os,sys,json,time
from collections import Counter
import numpy as np
from scipy.stats import chi2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import driver_streams as DS
DS.DEV=os.environ.get("BLIND_DEV","cuda:1")      # box convention: default cuda:1 (override via BLIND_DEV)
import torch, torch.nn as nn
# device resolution (host-safety panel wf_9910fdb7): honor the cuda:1 default but degrade gracefully on a box with
# fewer GPUs so the pipeline stays portable. CPU is allowed ONLY with SM_ALLOW_CPU=1 -- CPU CNN forward is NOT
# byte-identical to the frozen cuda calibration, so a production/FAR run must never silently fall to CPU.
def _resolve_dev(d):
    if not str(d).startswith("cuda"): return d
    if not torch.cuda.is_available():
        if os.environ.get("SM_ALLOW_CPU")=="1":
            sys.stderr.write("[dev] no CUDA -> CPU (SM_ALLOW_CPU=1; NON-CALIBRATION: not byte-identical to frozen GPU)\n"); return "cpu"
        raise RuntimeError("No CUDA device and SM_ALLOW_CPU!=1. Frozen FAR/CNN requires GPU; set BLIND_DEV or SM_ALLOW_CPU=1 for a dev (non-calibration) CPU run.")
    idx=int(str(d).split(":")[1]) if ":" in str(d) else 0
    if idx>=torch.cuda.device_count():
        sys.stderr.write(f"[dev] {d} absent ({torch.cuda.device_count()} GPU(s) visible) -> cuda:0\n"); return "cuda:0"
    return d
DS.DEV=_resolve_dev(DS.DEV)
import driver_search_multi as M                  # identical statistic + helpers
import morph_roi as mr, improved_pipeline as ip
from massive_pipeline import MassiveEventPipeline
import _pb_cnn_precompute as PB                   # STAGE-2 batched CNN precompute (bit-identical lookup)

# ---- FFT-cached coherence (flush free-win): per (seg,det) group, compute the forward RFFT + squared-norm
# of the DISTINCT survivor windows ONCE on GPU (from the SAME cached series rows series_for returns), then
# compute coherence by indexing those spectra per survivor. BIT-IDENTICAL to M.coh_vec (FFT is linear, so a
# cached forward transform of the exact cache rows is exact; verified 0.0 abs diff on real cached rows, and
# byte-identical blindscan/survivors/detections JSON end-to-end). Replaces the per-flush matmap/gser CPU
# series gather + per-call rfft that fed M.coh_vec single-threaded; GPU stays busy. Memory ~ #distinct
# survivor windows per seg-pair (NOT all ~9-14k windows/seg). Distinct-window dedup mirrors series_for. ----
def _fft_distinct(name,det,widx):                 # rfft+nrm of the DISTINCT survivor windows of this (seg,det), on GPU.
    rows=M.series_for(name,det,widx)              # {int(w): cache row} -- the SAME rows the gather path used
    ks=np.fromiter(rows.keys(),np.int64); ks.sort()
    a=torch.from_numpy(np.ascontiguousarray(np.stack([rows[int(k)] for k in ks]),dtype=np.float64)).to(DS.DEV)
    return ks,dict(F=torch.fft.rfft(a,dim=1),nrm=(a*a).sum(1),slen=int(a.shape[1]))
def coh_from_fft(Hname,Lname,iH,iL):              # iH,iL int64 array [m]; returns coh [m] == M.coh_vec exactly
    ksH,cH=_fft_distinct(Hname,"H1",iH); ksL,cL=_fft_distinct(Lname,"L1",iL); SLEN=cH["slen"]
    pH=torch.as_tensor(np.searchsorted(ksH,iH),dtype=torch.long,device=DS.DEV)
    pL=torch.as_tensor(np.searchsorted(ksL,iL),dtype=torch.long,device=DS.DEV)
    out=np.empty(len(iH)); LAGSAMP=M.LAGSAMP
    for c0 in range(0,len(iH),M.COH_BATCH):       # same batch width as M.coh_vec (bounded GPU mem)
        jH=pH[c0:c0+M.COH_BATCH]; jL=pL[c0:c0+M.COH_BATCH]
        cc=torch.fft.irfft(cH["F"][jH]*torch.conj(cL["F"][jL]),n=SLEN,dim=1)
        win=torch.cat([cc[:,:LAGSAMP+1],cc[:,-LAGSAMP:]],dim=1)
        out[c0:c0+len(jH)]=((2.0*win.abs().amax(dim=1))/(cH["nrm"][jH]+cL["nrm"][jL]+1e-30)).cpu().numpy()
    return out

NET_CUT=4.0                                       # STEP 1: foreground trigger cut (net sigma > 4)
CLUSTER_S=4.0                                     # merge zero-lag triggers within this many s (keep max loglr)
GLITCH_THRESH=0.5                                 # STEP 2: HM CNN absolute glitch gate (P(signal)>thresh = astrophysical)
CAND_LOGLR_FLOOR=float(os.environ.get("SM_CAND_FLOOR","4.5"))  # only FAR triggers with loglr>=this (candidates); prune bg to here. env-settable.
                                                  # DEFAULT 4.5: admits IMBH/loud-low-loglr events (GW190521 loglr~4.6). Quieter triggers are
                                                  # obviously background; pruning lower OOMs (~1e9 bg) -> ALWAYS shard (SM_NSHARD>1) at this floor.
                                                  # Revert to the legacy OR-veto baseline with SM_CAND_FLOOR=5.0 SM_NETSIG_FLOOR=0 SM_PERARM=0.
NETSIG_FLOOR=float(os.environ.get("SM_NETSIG_FLOOR","4.0"))  # >0: enable NET-SIGMA OR-admission channel (high-mass specialist).
                                                  # A loud-but-low-loglr trigger (e.g. IMBH GW190521 net~7.9, loglr 4.6) is
                                                  # admitted via net sigma + the SAME OR-combine CNN gate, FAR'd against a
                                                  # net-sigma-ranked CNN-gated background (built free in the prune loop, NO
                                                  # extra coherence). Detection = loglr channel OR net-sigma channel.
N_CNN_CANDIDATES=120                              # cap CNN-gated candidates (loudest by loglr) (+ known-event matches)
SURV_ABORT=int(os.environ.get("SM_SURV_ABORT", 8_000_000))   # safety backstop only (RAM); env-overridable.
                                                  # O3a dense bg needs ~50M survivors at floor 5.0 -> set SM_SURV_ABORT
                                                  # high on a big-RAM box rather than raising the (sensitivity) floor.
OUT=M.OUT; YR=M.YR; STRIDE=M.STRIDE; FS=M.FS; WN=4*FS; FLO,FHI=20.0,140.0; WT=113
DEV=DS.DEV

# ---- HM CNN (20-140 native ROI) setup, mirrors cascade_cnn_far.py ----
_pipe=None
def cpipe():
    global _pipe
    if _pipe is None: _pipe=MassiveEventPipeline(calib_path=f"{DS.SC}/massive_calibration_BA.json",prep=DS.O4A,device=DEV)
    return _pipe
_arm=None
def carm():
    global _arm
    if _arm is None:
        _arm=DS.GlitchArm().to(DEV); _arm.load_state_dict(torch.load(f"{DS.LRD}/p1v42/arm_deploy_seed0.pt",map_location=DEV)); _arm.eval()
    return _arm
class Net2(nn.Module):
    def __init__(s):
        super().__init__(); ch=[2,16,32,64,128]
        s.b=nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i],ch[i+1],3,padding=1),nn.BatchNorm2d(ch[i+1]),nn.ReLU(),nn.MaxPool2d(2)) for i in range(4)])
        s.h=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x):
        for bl in s.b: x=bl(x)
        return s.h(x).squeeze(-1)
LFLO,LFHI=50.0,500.0                              # LM-specialist band (HM is FLO,FHI=20-140)
_net=None
def cnet():                                       # HM specialist (20-140)
    global _net
    if _net is None:
        _net=Net2().to(DEV); _net.load_state_dict(torch.load(os.path.join(MADGRAV_ROOT,"search_mode","hm_native_seed0.pt"),map_location=DEV)); _net.eval()
    return _net
_lmnet=None
def lmnet():                                       # LM specialist (50-500)
    global _lmnet
    if _lmnet is None:
        _lmnet=Net2().to(DEV); _lmnet.load_state_dict(torch.load(os.path.join(MADGRAV_ROOT,"search_mode","lm_native_seed0.pt"),map_location=DEV)); _lmnet.eval()
    return _lmnet
# _STR raw-strain cache: bounded LRU (host-safety panel wf_9910fdb7). Was an unbounded dict that grew with the
# number of distinct segments touched (the real host-RAM leak). RESULT-IDENTICAL: eviction only forces a re-read
# of identical .npz bytes, and _win() returns an astype(float32) COPY so nothing holds an evictable reference
# (cap could be 1 and stay exact). Cap = SM_STRAIN_CACHE_SEGS segments x 2 detectors.
from collections import OrderedDict as _OD
_STRCAP=max(2,int(os.environ.get("SM_STRAIN_CACHE_SEGS","4"))*2)
_STR=_OD()
def _strain(n,d):
    k=(n,d)
    if k in _STR:
        _STR.move_to_end(k); return _STR[k]
    r=np.load(f"{M.STRAIN}/{n}_{d}.npz")["strain"]; _STR[k]=r
    while len(_STR)>_STRCAP: _STR.popitem(last=False)
    return r
def _win(n,d,idx):
    r=_strain(n,d); s=r[idx*FS:idx*FS+WN]
    return (s if len(s)==WN else np.pad(s,(0,WN-len(s)))).astype(np.float32)
def _fullmag(wh):                                  # full QT magnitude + gradcam t0 (compute ONCE, crop both bands)
    t0=int(mr.cam_t0_batch(carm(),DS.build_qt(cpipe(),wh),DEV)[0])
    qi=ip.center_crop_waveforms(wh,sample_rate=FS,context_seconds=cpipe().ctx)
    mag=np.asarray(ip._compute_qt_image_worker((qi[0],FS,ip.QTRANSFORM_FRANGE,ip.QTRANSFORM_QRANGE,1.0)),float)
    return mag,t0,mag.shape[1]
def _crop(mag,a,T,flo,fhi):                        # native band crop at shared time placement a (from H1 gradcam)
    fax=ip.QTRANSFORM_FRANGE[0]+0.5*np.arange(mag.shape[0]); m=mag[(fax>=flo)&(fax<=fhi)]
    out=np.zeros((m.shape[0],WT),dtype=np.float32); sa=max(0,a); sb=min(T,a+WT)
    if sb>sa: out[:,sa-a:sb-a]=m[:,sa:sb]
    return ((out-out.min())/(out.max()-out.min()+1e-9)).astype(np.float32)
def cnn_hm_lm(nH,iH,nL,iL):                        # -> (hm,lm); QT once per leg, both legs use H1 t0 (matches dry test)
    whH=cpipe()._whiten(_win(nH,"H1",iH)[None,:],"H1"); whL=cpipe()._whiten(_win(nL,"L1",iL)[None,:],"L1")
    magH,t0,T=_fullmag(whH); magL,_,_=_fullmag(whL); a=int(round((t0-14)/128*T))
    def stk(flo,fhi): return np.stack([_crop(magH,a,T,flo,fhi),_crop(magL,a,T,flo,fhi)])
    with torch.no_grad():
        hm=float(torch.sigmoid(cnet()(torch.from_numpy(stk(FLO,FHI)[None]).float().to(DEV))).item())
        lm=float(torch.sigmoid(lmnet()(torch.from_numpy(stk(LFLO,LFHI)[None]).float().to(DEV))).item())
    return hm,lm
def cnn_score(nH,iH,nL,iL):                        # OR-combine GATE value = max(HM,LM); branches recorded separately
    hm,lm=cnn_hm_lm(nH,iH,nL,iL); return max(hm,lm)

def main():
    # host-memory budget (host-safety panel wf_9910fdb7): SM_HOST_MEM_GB -> fill any UNSET SM_* memory knob and
    # optionally arm the RLIMIT_DATA backstop. Explicit env knobs (set by the launchers) always win -> existing
    # runs are byte-identical. With nothing set it self-bounds to a conservative default instead of grabbing all RAM.
    try:
        import _budget; _budget.apply()
    except Exception as _e:
        sys.stderr.write(f"[budget] skipped ({_e})\n")
    avail=[(a,nm) for a,b,d,nm in M.SEG["segments"] if M.have(a,nm)]
    segs=[M.load_seg(a,nm) for a,nm in avail]; nseg=len(segs); fold=[i%2 for i in range(nseg)]
    print(f"[blind] {nseg} segs folds={fold}",flush=True)
    step_off=int(round(M.OFFSET_STEP_S/STRIDE)); same_minoff=int(round(M.SAME_SEG_MINLAG_S/STRIDE)); evrad=int(2.0/STRIDE)
    evloc={ev:(i,int(np.abs(segs[i]["gps"]-g0).argmin())) for ev,g0 in M.EVENTS.items() for i,s in enumerate(segs) if s["name"]==ev}
    NO_EXCISION=bool(os.environ.get("SM_NO_EXCISION"))   # TRUE BLIND: excise NOTHING from noise/background (treat all data as unknown).
    if NO_EXCISION: print("[blind] SM_NO_EXCISION=1: NO event excision -> fully blind, conservative FAR (real events stay in background)",flush=True)
    def is_ev(si,idx):
        if NO_EXCISION: return np.zeros(len(idx),bool)
        m=np.zeros(len(idx),bool)
        for ev,(esi,ew) in evloc.items():
            if si==esi: m|=np.abs(idx-ew)<=evrad
        return m
    Xs_all=[]; sf_list=[]
    BOTHFOLDS=bool(os.environ.get("SM_INJ_BOTHFOLDS"))   # single-event runs: broad injection bank -> both folds (no own-template leak)
    for ev in M.EVENTS:
        za=np.load(f"{M.INJ}/{ev}_inj.npz"); X=M.feats(za["sigH"],za["sigL"],za["coh"],za["cenH"],za["cenL"],za["gH"],za["gL"])
        if BOTHFOLDS:
            for fg in (0,1): Xs_all.append(X); sf_list.append(np.full(len(X),fg,dtype=int))
        else:
            Xs_all.append(X); sf_list.append(np.full(len(X),fold[evloc[ev][0]],dtype=int))
    Xs_all=np.vstack(Xs_all); sf=np.concatenate(sf_list)
    pairings=[]
    for ai in range(nseg):
        for bi in range(nseg):
            if fold[ai]!=fold[bi]: continue
            if ai==bi:
                hi=segs[ai]["n"]-step_off; cand=np.arange(same_minoff,hi+1,step_off)
                if len(cand)==0: continue
                cand=np.concatenate([cand,-cand])
            else:
                hi=min(segs[ai]["n"],segs[bi]["n"])-step_off; cand=np.arange(-hi,hi+1,step_off)
            if len(cand)>M.MAX_OFFSETS_PER_PAIR: cand=M.RNG.choice(cand,M.MAX_OFFSETS_PER_PAIR,replace=False)
            for o in cand: pairings.append((ai,bi,int(o)))
    M.RNG.shuffle(pairings)
    print(f"[blind] {len(pairings)} same-fold pairings",flush=True)
    # ---- 4-GPU SHARD selector: SM_NSHARD=N SM_SHARD=k -> the prune loop iterates only `mine` (a strided/
    # contiguous SUBSET of pairings); everything else (segs/fold/pairings/mdl/fgt/floor/COH_CEIL/famN-keys)
    # is computed from the FULL deterministic pairings so every shard agrees. bg=union, famN=per-(fold,L1key)
    # MAX, face_live=SUM across shards. SM_NSHARD=1 (default) -> mine==pairings -> full single-process run
    # end-to-end. SM_NSHARD>1 -> write the partial bg/famN/face_live to SM_SHARD_DIR/shard_k.npz and EXIT
    # after the prune (STAGE-1/2/1b run once in driver_blindscan_merge.py on the merged background). ----
    SHARD=int(os.environ.get("SM_SHARD","0")); NSHARD=int(os.environ.get("SM_NSHARD","1"))
    SHARD_DIR=os.environ.get("SM_SHARD_DIR","search_mode/_pb_shard_parts")
    SHARD_MODE=os.environ.get("SM_SHARD_MODE","stride")   # "stride"=pairings[k::N]; "block"=contiguous slice
    if NSHARD>1:
        if SHARD_MODE=="block":
            _n=len(pairings); _lo=(_n*SHARD)//NSHARD; _hi=(_n*(SHARD+1))//NSHARD; mine=pairings[_lo:_hi]
        else:
            mine=pairings[SHARD::NSHARD]
        print(f"[blind] SHARD {SHARD}/{NSHARD} mode={SHARD_MODE} {len(mine)}/{len(pairings)} pairings (partial-output mode)",flush=True)
    else:
        mine=pairings
    NCAP=15000; npr={0:[],1:[]}
    for ai,bi,off in pairings:
        g=fold[ai]
        if len(npr[g])>=NCAP: continue
        A,B=segs[ai],segs[bi]; iH,iL=M.pair_idx(A["n"],B["n"],off)
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
        nH={};nL={}
        for a_,h_,b_,l_ in P: nH.setdefault(a_,set()).add(h_); nL.setdefault(b_,set()).add(l_)
        SH={a_:M.series_for(segs[a_]["name"],"H1",sorted(w)) for a_,w in nH.items()}
        SL={b_:M.series_for(segs[b_]["name"],"L1",sorted(w)) for b_,w in nL.items()}
        co=M.coh_vec(np.array([SH[a_][h_] for a_,h_,b_,l_ in P]),np.array([SL[b_][l_] for a_,h_,b_,l_ in P]))
        cohnoise[g]=co
        ai_=np.array([p[0] for p in P]);iH_=np.array([p[1] for p in P]);bi_=np.array([p[2] for p in P]);iL_=np.array([p[3] for p in P])
        Xn[g]=M.feats(gv("sH",ai_,iH_),gv("sL",bi_,iL_),co,gv("cH",ai_,iH_),gv("cL",bi_,iL_),gv("gH",ai_,iH_),gv("gL",bi_,iL_))
        print(f"[blind] fold {g} noise {len(P)} rows coh med {np.median(co):.3f}",flush=True)
    mdl={g:M.fit_lr(Xn[g],Xs_all[sf==g]) for g in (0,1)}

    # ===== FOREGROUND: zero-lag triggers, net>NET_CUT, clustered =====
    fg=[]
    for si,s in enumerate(segs):
        net=(s["sH"]+s["sL"])/np.sqrt(2.); hit=np.where(net>NET_CUT)[0]
        if not len(hit): continue
        gap=int(round(CLUSTER_S/STRIDE)); groups=[]; cur=[hit[0]]
        for i in hit[1:]:
            if i-cur[-1]<=gap: cur.append(i)
            else: groups.append(cur); cur=[i]
        groups.append(cur)
        for gr in groups:
            j=gr[int(np.argmax(net[gr]))]; fg.append((si,int(j)))
    print(f"[blind] {len(fg)} clustered zero-lag triggers (net>{NET_CUT})",flush=True)
    needH={};needL={}
    for si,j in fg: needH.setdefault(si,set()).add(j); needL.setdefault(si,set()).add(j)
    serH={si:M.series_for(segs[si]["name"],"H1",sorted(w)) for si,w in needH.items()}
    serL={si:M.series_for(segs[si]["name"],"L1",sorted(w)) for si,w in needL.items()}
    fgt=[]
    for si,j in fg:
        g=fold[si]; mu,sd,be=mdl[1-g]
        co=float(M.coh_vec(serH[si][j][None,:],serL[si][j][None,:])[0])
        F=M.feats([segs[si]["sH"][j]],[segs[si]["sL"][j]],[co],[segs[si]["cH"][j]],[segs[si]["cL"][j]],[segs[si]["gH"][j]],[segs[si]["gL"][j]])
        ll=float(M.loglr(mu,sd,be,F)[0]); net=float((segs[si]["sH"][j]+segs[si]["sL"][j])/np.sqrt(2.))
        fgt.append(dict(seg=segs[si]["name"],gps=float(segs[si]["gps"][j]),si=si,idx=j,fold=g,net=net,coh=co,loglr=ll))
    # prune background down to CAND_LOGLR_FLOOR. Default: floor never above the quietest KNOWN event (so they're
    # always FAR'd). SM_BLIND_FLOOR=1 -> ignore event priors: floor fixed at CAND_LOGLR_FLOOR (fully blind candidate
    # selection; events below it become honest near-misses, not floor-dragged FARs). Avoids the survivor explosion
    # when a known event sits below the floor (e.g. O3a event at loglr 4.6 -> floor 4.6 -> bg survivors blow past abort).
    known_ll=[t["loglr"] for t in fgt for ev,(esi,ew) in evloc.items() if t["si"]==esi and abs(t["idx"]-ew)<=evrad]
    floor=CAND_LOGLR_FLOOR if os.environ.get("SM_BLIND_FLOOR") else min([CAND_LOGLR_FLOOR]+known_ll)
    thr_fold={g:floor for g in (0,1)}
    ncand=sum(1 for t in fgt if t["loglr"]>=floor)
    print(f"[blind] prune floor={floor:.2f} (CAND_LOGLR_FLOOR={CAND_LOGLR_FLOOR}, known {[round(x,2) for x in known_ll]}); {ncand} candidate triggers >= floor",flush=True)
    # PRINCIPLED COHERENCE CEILING (prefilter). The prune keeps every survivor whose loglr at coh=COH_CEIL
    # exceeds floor; loglr is monotone non-decreasing in coh (fit_lr pins beta_coh>=0), so this is an EXACT
    # upper bound and drops NO real survivor PROVIDED COH_CEIL >= every survivor's real coh. The time-slid
    # background is independent noise -> coh tightly concentrated; the legacy +0.15 pad over a ~30k-row
    # subsample max was loose (~50x more survivors than needed at this run's steep beta_coh). Replace with an
    # explicit subsample-max -> population-max correction (exp-tail spacing) + a k-sigma Gumbel margin, floored
    # by the zero-lag trigger max. Bit-identical FAR holds as long as the realized max survivor coh stays below
    # COH_CEIL -- LOGGED by the tripwire (cohmax_seen) at end of the prune. SM_COH_CEIL_LEGACY=1 -> old ceiling.
    sub_max=max([float(c.max()) for c in cohnoise.values() if len(c)]+[0.0])      # noise-subsample max
    trig_max=max([t["coh"] for t in fgt]+[0.0])                                   # zero-lag triggers (genuine coincidence)
    _c=np.concatenate([c for c in cohnoise.values() if len(c)]) if cohnoise else np.array([0.0])
    _thr=float(np.percentile(_c,99.0)); _beta=max(1e-4,float((_c[_c>_thr]-_thr).mean()))   # exp tail scale of indep-noise coh
    _N=float(len(pairings))*float(segs[0]["n"] if segs else 1)                    # >= true survivor count; only its log matters
    KSIG=float(os.environ.get("SM_COH_KSIGMA","6.0"))                            # Gumbel safety margin (higher = safer, looser)
    pop_bound=sub_max+_beta*np.log(max(1.0,_N/max(1,len(_c))))+KSIG*_beta*np.pi/np.sqrt(6.0)
    COH_CEIL=float(min(1.0,max(pop_bound,trig_max+0.02)))
    if os.environ.get("SM_COH_CEIL_LEGACY"): COH_CEIL=float(min(1.0,max(sub_max,trig_max)+0.15))   # opt-out: old loose ceiling
    if os.environ.get("SM_COH_CEIL"): COH_CEIL=float(os.environ["SM_COH_CEIL"])  # explicit override
    cohmax_seen=[0.0]                                                             # TRIPWIRE: max realized survivor coh (must stay < COH_CEIL)
    print(f"[blind] coh ceiling {COH_CEIL:.3f} (sub_max {sub_max:.3f} trig_max {trig_max:.3f} beta {_beta:.4f} KSIG {KSIG:.0f})",flush=True)

    # ===== BACKGROUND: loglr-upper prune, then exact coherence (keep identities for CNN stage) =====
    def L1key(bi,iL): return (bi,int(segs[bi]["gps"][iL]//M.MERGE_S))   # L1 4s-family (FAR counts distinct families)
    famN={}                                          # NET-SIGMA channel: per (fold,L1key) keep loudest-net bg rep, free in this loop
    # SEQUENTIAL BLOCK STREAMING (memory-bounded, RESULT-IDENTICAL): the COH_CEIL upper-bound prefilter is loose
    # (~1e9 survivors at the low floor GW190521 pulls us to), so we never materialize it whole. Process the pairings
    # in blocks: accumulate the prefilter to SM_BLOCK_SURV rows, rescore that block with EXACT coherence, keep ONLY
    # real-loglr>floor into bg[g] (the genuinely-loud set -- small + bounded), then free the block. Downstream FAR
    # only ever counts bg with real loglr > a candidate's loglr (>=floor), so dropping ll<=floor is exact; famN
    # (net-sigma, no coherence) and livetime are additive across blocks. One GPU, peak RAM = one block. Scales to
    # arbitrary livetime (just more flushes) -> same code serves the full O3a search and bigger real searches.
    # 2026-06-26 (host-safety panel wf_9910fdb7): default lowered 40,000,000 -> 1,000,000. The 40M default was the
    # exact term that force-rebooted the box (a hundreds-of-GB np.stack matmap re-faulting the file-backed series
    # cache under swap-off -> global reclaim livelock, no rc=137). RESULT-IDENTICAL: bg[g] accrues by union across
    # blocks and the final partial always flushes, so block size never changes the kept set / FAR. Sized via _budget
    # (SM_HOST_MEM_GB); a launcher may still override SM_BLOCK_SURV explicitly.
    BLOCK_SURV=int(os.environ.get("SM_BLOCK_SURV","1000000"))
    # CH: per-gather row chunk (transient 2*CH*row64 bytes). Was a hardcoded 200000 (~13GB floor); now SM_GATHER_CH.
    # RESULT-IDENTICAL: chunking is per-row; the COH_CEIL tripwire is a chunk-invariant max.
    bg={0:[],1:[]}; face_live={0:0.0,1:0.0}; ta=time.time(); nflush=0; CH=int(os.environ.get("SM_GATHER_CH","60000"))   # bg per fold: (loglr, ai, iH, bi, iL)
    def flush_block(ai_arr,iH_arr,bi_arr,iL_arr,g_arr):
        nonlocal nflush
        n=len(ai_arr)
        if n==0: return
        nflush+=1
        # GATHER coherence (GPU-light: transient coh_vec, no persistent FFT cache -> many shards per GPU).
        needH={int(s):np.unique(iH_arr[ai_arr==s]) for s in np.unique(ai_arr)}
        needL={int(s):np.unique(iL_arr[bi_arr==s]) for s in np.unique(bi_arr)}
        sH={ai:M.series_for(segs[ai]["name"],"H1",w.tolist()) for ai,w in needH.items()}
        sL={bi:M.series_for(segs[bi]["name"],"L1",w.tolist()) for bi,w in needL.items()}
        kept=0
        def matmap(cache):                                    # per seg: (sorted sample-idx array, stacked 2D series matrix) built ONCE
            MM={}
            for s,d in cache.items():
                ks=np.fromiter(d.keys(),np.int64); ks.sort(); MM[s]=(ks,np.stack([d[int(k)] for k in ks]))
            return MM
        HM=matmap(sH); LM=matmap(sL)
        def gser(MM,si,w):                                    # vectorized series gather: searchsorted into the prebuilt matrix
            out=None
            for s in np.unique(si):
                msk=si==s; ks,mat=MM[int(s)]; rows=mat[np.searchsorted(ks,w[msk])]
                if out is None: out=np.empty((len(si),)+rows.shape[1:],rows.dtype)
                out[msk]=rows
            return out
        def gvv(key,si,w):                                    # scalar-feature gather: segs[s][key] numpy array -> per-seg fancy index (vectorized)
            out=np.empty(len(si),float)
            for s in np.unique(si): msk=si==s; out[msk]=segs[int(s)][key][w[msk]]
            return out
        for c0 in range(0,n,CH):
            sl=slice(c0,c0+CH)
            ai_,iH_,bi_,iL_,g_=ai_arr[sl],iH_arr[sl],bi_arr[sl],iL_arr[sl],g_arr[sl]
            A_=gser(HM,ai_,iH_); B_=gser(LM,bi_,iL_)
            coh=M.coh_vec(A_,B_)
            if len(coh): cohmax_seen[0]=max(cohmax_seen[0],float(coh.max()))   # tripwire: realized survivor coh must stay < COH_CEIL
            if len(coh) and float(coh.max())>=COH_CEIL: raise SystemExit(f"[blind] TRIPWIRE ABORT coh {float(coh.max()):.3f} >= COH_CEIL {COH_CEIL:.3f} -- ceiling too tight, FAR would bias; raise SM_COH_KSIGMA and rerun")  # fail-fast (panel wf_d21af1e7): abort in seconds, not at end-of-prune
            for gg in (0,1):                                  # BATCHED rescore per fold (feats/loglr vectorized; cf. noise fit line 161)
                m=np.where(g_==gg)[0]
                if not len(m): continue
                mu,sd,be=mdl[1-gg]
                F=M.feats(gvv("sH",ai_[m],iH_[m]),gvv("sL",bi_[m],iL_[m]),coh[m],gvv("cH",ai_[m],iH_[m]),gvv("cL",bi_[m],iL_[m]),gvv("gH",ai_[m],iH_[m]),gvv("gL",bi_[m],iL_[m]))
                ll=M.loglr(mu,sd,be,F)                        # array over the whole fold-gg subset
                for jj in np.where(ll>floor)[0]:
                    p=m[jj]; bg[gg].append((float(ll[jj]),int(ai_[p]),int(iH_[p]),int(bi_[p]),int(iL_[p]))); kept+=1
        print(f"  [block {nflush}] rescored {n} prefilter -> kept {kept} (ll>{floor:.2f}); bg total {len(bg[0])+len(bg[1])} ({(time.time()-ta)/60:.1f}m)",flush=True)
    # survivors held as per-column numpy buffers (lists of small per-pairing arrays); NO list of tuples.
    col_ai=[]; col_iH=[]; col_bi=[]; col_iL=[]; col_g=[]; surv_n=0
    for k,(ai,bi,off) in enumerate(mine):
        g=fold[ai]; mu,sd,be=mdl[1-g]; A,B=segs[ai],segs[bi]; iH,iL=M.pair_idx(A["n"],B["n"],off)
        if not len(iH): continue
        keep=~(is_ev(ai,iH)|is_ev(bi,iL)); iH,iL=iH[keep],iL[keep]
        if not len(iH): continue
        face_live[g]+=len(iH)*STRIDE
        if NETSIG_FLOOR>0:                            # collect loud bg L1-families for the net-sigma channel (no coherence -> free)
            netbg=(A["sH"][iH]+B["sL"][iL])/np.sqrt(2.)
            for s_ in np.where(netbg>=NETSIG_FLOOR)[0]:
                iLs=int(iL[s_]); kky=(g,L1key(bi,iLs)); v=float(netbg[s_])
                if kky not in famN or v>famN[kky][0]: famN[kky]=(v,ai,int(iH[s_]),bi,iLs)
        up=M.loglr(mu,sd,be,M.feats(A["sH"][iH],B["sL"][iL],np.full(len(iH),COH_CEIL),A["cH"][iH],B["cL"][iL],A["gH"][iH],B["gL"][iL]))
        sel=np.where(up>thr_fold[g])[0]
        if len(sel):                                  # append SMALL numpy arrays (per-pairing), not per-survivor tuples
            col_ai.append(np.full(len(sel),ai,dtype=np.int64)); col_iH.append(iH[sel].astype(np.int64))
            col_bi.append(np.full(len(sel),bi,dtype=np.int64)); col_iL.append(iL[sel].astype(np.int64))
            col_g.append(np.full(len(sel),g,dtype=np.int64));   surv_n+=len(sel)
        if surv_n>=BLOCK_SURV:                         # flush this block: concat ONCE, rescore + keep loud, then free
            flush_block(np.concatenate(col_ai),np.concatenate(col_iH),np.concatenate(col_bi),
                        np.concatenate(col_iL),np.concatenate(col_g))
            col_ai=[]; col_iH=[]; col_bi=[]; col_iL=[]; col_g=[]; surv_n=0
        if k%2000==0: print(f"  prune {k}/{len(mine)} ({(time.time()-ta)/60:.1f}m) surv_block={surv_n} bg={len(bg[0])+len(bg[1])}",flush=True)
    if surv_n:                                         # final partial block
        flush_block(np.concatenate(col_ai),np.concatenate(col_iH),np.concatenate(col_bi),
                    np.concatenate(col_iL),np.concatenate(col_g))
    col_ai=col_iH=col_bi=col_iL=col_g=None
    # ---- SHARD PARTIAL-OUTPUT MODE: write this shard's partial bg/famN/face_live/cohmax and EXIT.
    # The merge driver (driver_blindscan_merge.py) loads all N partials (bg=union, famN=per-(fold,L1key)
    # max, face_live=sum, cohmax=max) and runs STAGE-1/2/1b once. NSHARD=1 falls through to the full run. ----
    if NSHARD>1:
        os.makedirs(SHARD_DIR,exist_ok=True)
        def _bg_arr(g):
            return np.array(bg[g],dtype=np.float64) if bg[g] else np.zeros((0,5),np.float64)  # (ll,ai,iH,bi,iL)
        fam_keys=np.array([[k[0],k[1][0],k[1][1]] for k in famN],dtype=np.int64).reshape(-1,3)   # (g,bi,l1bin)
        fam_vals=np.array([[v[0],v[1],v[2],v[3],v[4]] for v in famN.values()],dtype=np.float64).reshape(-1,5)
        out=f"{SHARD_DIR}/shard_{SHARD}.npz"; tmp=f"{SHARD_DIR}/.shard_{SHARD}.tmp.npz"
        # ATOMIC write (host-safety panel wf_9910fdb7): a crash/OOM mid-savez must NEVER leave a truncated
        # shard_*.npz that the launcher treats as 'done' and the merge consumes as a complete bg-by-union (which
        # would bias FAR LOW -> could move a borderline event e.g. GW230824). Write to a temp file, fsync, then
        # atomically rename. A killed shard leaves NO final file -> the launcher re-runs it from scratch (bounded
        # by block size) -> byte-identical. (Full mid-shard resume at pairing-index k is a deferred efficiency item;
        # re-running one bounded shard is already correct.)
        with open(tmp,"wb") as _fh:
            np.savez(_fh,bg0=_bg_arr(0),bg1=_bg_arr(1),fam_keys=fam_keys,fam_vals=fam_vals,
                     face_live=np.array([face_live[0],face_live[1]],np.float64),
                     cohmax_seen=np.array([cohmax_seen[0]],np.float64),
                     shard=np.array([SHARD,NSHARD],np.int64))
            _fh.flush(); os.fsync(_fh.fileno())
        os.replace(tmp,out)
        print(f"[blind] SHARD {SHARD}/{NSHARD} DONE bg0={len(bg[0])} bg1={len(bg[1])} famN={len(famN)} "
              f"face_live={face_live} cohmax_seen={cohmax_seen[0]:.3f} -> {out}",flush=True)
        return
    far_live={g:face_live[g]*M.ANALYZED_FRAC/YR for g in (0,1)}
    print(f"[blind] livetime fold0={far_live[0]:.3f}yr fold1={far_live[1]:.3f}yr; {len(bg[0])+len(bg[1])} loud bg survivors over {nflush} blocks",flush=True)
    _trip="OK (prefilter provably lossless)" if cohmax_seen[0]<COH_CEIL else "VIOLATED -> ceiling too tight, FAR may be biased; rerun with higher SM_COH_KSIGMA"
    print(f"[blind] PREFILTER TRIPWIRE: max realized survivor coh {cohmax_seen[0]:.3f} vs COH_CEIL {COH_CEIL:.3f} (margin {COH_CEIL-cohmax_seen[0]:+.3f}) -> {_trip}",flush=True)
    for g in (0,1): bg[g].sort(key=lambda r:-r[0])     # loudest first

    # ---- PER-ARM CNN-RANK helpers (panel wf_9019f7d5, GREEN-WITH-FIXES). A louder bg family counts in arm X
    # iff cnn_X >= the event's own cnn_X (a RANK), vs the OR-veto rule cnn>0.5 (absolute). All None-guarded. ----
    PERARM=int(os.environ.get("SM_PERARM","1"))        # DEFAULT 1: per-arm is the headline best_far statistic (SM_PERARM=0 -> legacy OR-veto)
    def _ul90(n,flt): return float(chi2.ppf(0.90,2*(n+1))/2/flt) if (flt and flt>0) else None  # 1-sided 90% Poisson UL
    def _perarm(a,b): return 2*min(a,b) if (a is not None and b is not None) else None          # x2 = arm-choice Bonferroni

    # ===== STAGE-1 FAR_LR (full-band) per trigger =====
    fgt.sort(key=lambda d:-d["loglr"])
    for t in fgt:
        g=t["fold"]; flt=far_live[g]
        if t["loglr"]<floor:                       # below candidate floor: obviously background, no precise FAR
            t["_louder"]=[]; t["n_louder_L1fam"]=None; t["far_lr_per_yr"]=None; t["far_lr_ul90"]=None
            t["below_floor"]=True; continue
        louder=[(ai,iH,bi,iL) for ll,ai,iH,bi,iL in bg[g] if ll>t["loglr"]]
        fams=set(L1key(bi,iL) for ai,iH,bi,iL in louder)
        t["_louder"]=louder; t["n_louder_L1fam"]=len(fams)
        t["far_lr_per_yr"]=len(fams)/flt if flt>0 else None
        t["far_lr_ul90"]=float(chi2.ppf(0.90,2*(len(fams)+1))/2/flt) if flt>0 else None

    # ===== STAGE-2 HM CNN (20-140) veto on CANDIDATE triggers =====
    above=[i for i,t in enumerate(fgt) if t["loglr"]>=floor]      # FAR'd candidates (fgt already sorted by loglr desc)
    cand_idx=set(above[:N_CNN_CANDIDATES])
    if NETSIG_FLOOR>0:                                            # net-sigma OR-channel: admit loud-but-low-loglr triggers too
        for i,t in enumerate(fgt):
            if t["net"]>=NETSIG_FLOOR: cand_idx.add(i)
    for i,t in enumerate(fgt):
        for ev,(esi,ew) in evloc.items():
            if t["si"]==esi and abs(t["idx"]-ew)<=evrad: t["matches_known"]=ev; cand_idx.add(i)
    # ---- STAGE-2 CNN PRECOMPUTE (bit-identical to the lazy per-pair scoring): collect every louder-bg pair
    # that surv_cnn would have scored across BOTH FAR channels (loglr + net-sigma), dedup per-detector QT mags,
    # and score them up-front; surv_cnn then becomes a pure cnn_cache lookup. The per-pair forward is kept
    # VERBATIM (single-sample .item()) inside PB.precompute_cnn so the result is 0.0-diff. ----
    need_pairs=set()
    for i in cand_idx:
        if fgt[i].get("far_lr_per_yr") is not None: need_pairs.update(fgt[i]["_louder"])   # loglr channel louder bg
    if NETSIG_FLOOR>0:                                        # net-sigma channel families >= each netcand's net, same fold
        _fams=sorted(famN.values(),key=lambda r:-r[0])
        for i in (j for j in cand_idx if fgt[j]["net"]>=NETSIG_FLOOR):
            g=fgt[i]["fold"]
            for r in _fams:
                if r[0]>=fgt[i]["net"] and fold[r[1]]==g: need_pairs.add((r[1],r[2],r[3],r[4]))
    cnn_cache=PB.precompute_cnn(need_pairs,segs,cpipe,carm,cnet,lmnet,_win)   # (ai,iH,bi,iL) -> (hm,lm)
    print(f"[blind] STAGE-2 CNN precompute: {len(cnn_cache)} deduped louder-bg pairs scored up-front",flush=True)
    def surv_cnn(ai,iH,bi,iL):                               # pure lookup (precompute covers every needed pair)
        key=(int(ai),int(iH),int(bi),int(iL))
        if key not in cnn_cache: cnn_cache[key]=cnn_hm_lm(segs[ai]["name"],iH,segs[bi]["name"],iL)  # defensive fallback
        hm,lm=cnn_cache[key]; return max(hm,lm)
    _fallback_hits=[0]
    def surv_cnn_pair(ai,iH,bi,iL):                          # per-arm (hm,lm) lookup; reuses the same precompute cache
        key=(int(ai),int(iH),int(bi),int(iL))
        if key not in cnn_cache: _fallback_hits[0]+=1; cnn_cache[key]=cnn_hm_lm(segs[ai]["name"],iH,segs[bi]["name"],iL)
        return cnn_cache[key]                                # (hm,lm)
    def _f(v,fmt):                                            # None-safe numeric formatter
        return format(v,fmt) if v is not None else "NA"
    print(f"[blind] STEP2/3: CNN glitch-gate (>{GLITCH_THRESH}) + lag FAR on {len(cand_idx)} candidates",flush=True)
    for ci,i in enumerate(sorted(cand_idx)):
        t=fgt[i]; g=t["fold"]; flt=far_live[g]
        t["cnn_hm"],t["cnn_lm"]=cnn_hm_lm(t["seg"],t["idx"],t["seg"],t["idx"])   # STEP 2: BOTH branches
        t["cnn"]=max(t["cnn_hm"],t["cnn_lm"])                                    # OR-combine gate
        t["kept_by"]=("HM" if t["cnn_hm"]>GLITCH_THRESH else "")+("+LM" if t["cnn_lm"]>GLITCH_THRESH else "") or "none"
        t["is_glitch"]=bool(t["cnn"]<GLITCH_THRESH)
        # STEP 3: lag FAR with the SAME absolute OR-combine glitch gate on the background (band-matched).
        # Below-floor triggers (far_lr None) have no loglr-pruned bg above them -> loglr channel N/A; the
        # net-sigma channel (STAGE-1b) FARs them instead. Do NOT mint a spurious far_cnn=0 from _louder=[].
        if t.get("far_lr_per_yr") is None:
            t["n_louder_L1fam_cnn"]=None; t["far_cnn_per_yr"]=None; t["far_cnn_ul90"]=None
            t["far_lr_hm"]=t["far_lr_lm"]=t["far_lr_perarm"]=t["far_lr_perarm_ul90"]=None  # sub-floor: no spurious 0
        else:
            kept=[(ai,iH,bi,iL) for (ai,iH,bi,iL) in t["_louder"] if surv_cnn(ai,iH,bi,iL)>GLITCH_THRESH]
            fams=set(L1key(bi,iL) for ai,iH,bi,iL in kept)
            t["n_louder_L1fam_cnn"]=len(fams); t["far_cnn_per_yr"]=len(fams)/flt if flt>0 else None
            t["far_cnn_ul90"]=float(chi2.ppf(0.90,2*(len(fams)+1))/2/flt) if flt>0 else None
            # --- PER-ARM RANK (loglr channel): count louder bg L1-families by EACH arm vs the event's own cnn,
            #     with a self-segment guard (exclude families from the candidate's own seg in either detector). ---
            si=t["si"]; fam_hm=set(); fam_lm=set()
            for (ai,iH,bi,iL) in t["_louder"]:
                if ai==si or bi==si: continue                # self-segment guard (stronger than the is_ev mask)
                hm,lm=surv_cnn_pair(ai,iH,bi,iL); k=L1key(bi,iL)
                if hm>=t["cnn_hm"]: fam_hm.add(k)
                if lm>=t["cnn_lm"]: fam_lm.add(k)
            t["far_lr_hm"]=len(fam_hm)/flt if flt>0 else None; t["far_lr_lm"]=len(fam_lm)/flt if flt>0 else None
            t["far_lr_perarm"]=_perarm(t["far_lr_hm"],t["far_lr_lm"])
            t["far_lr_perarm_ul90"]=_perarm(_ul90(len(fam_hm),flt),_ul90(len(fam_lm),flt))
        print(f"  cand {ci+1}/{len(cand_idx)} {t['seg']} loglr={_f(t.get('loglr'),'.2f')} HM={_f(t.get('cnn_hm'),'.3f')} LM={_f(t.get('cnn_lm'),'.3f')} ({t['kept_by']})"
              f"{' GLITCH' if t['is_glitch'] else ''} FAR_LR={_f(t.get('far_lr_per_yr'),'.3g')} -> FAR_CNN={_f(t.get('far_cnn_per_yr'),'.3g')}/yr {t.get('matches_known','')}",flush=True)

    # ===== STAGE-1b NET-SIGMA OR-CHANNEL (high-mass specialist): FAR by net sigma, SAME OR-combine CNN gate =====
    # Blind-fair: foreground net-candidate kept iff its own CNN passes (is_glitch False); FAR'd against the
    # net-sigma-ranked, CNN-gated background L1-families (famN); same per-fold livetime as the loglr channel.
    if NETSIG_FLOOR>0:
        fams=sorted(famN.values(),key=lambda r:-r[0])            # (maxnet,ai,iH,bi,iL) per L1-family, loudest first
        netcands=[i for i in cand_idx if fgt[i]["net"]>=NETSIG_FLOOR]
        print(f"[blind] NET-SIGMA channel: floor={NETSIG_FLOOR}; {len(famN)} bg L1-families, {len(netcands)} fg net-candidates",flush=True)
        for i in sorted(netcands,key=lambda i:-fgt[i]["net"]):
            t=fgt[i]; g=t["fold"]; flt=far_live[g]
            kept=[r for r in fams if r[0]>=t["net"] and fold[r[1]]==g and surv_cnn(r[1],r[2],r[3],r[4])>GLITCH_THRESH]
            t["n_louder_netfam_cnn"]=len(kept)
            t["far_net_per_yr"]=len(kept)/flt if flt>0 else None
            t["far_net_ul90"]=float(chi2.ppf(0.90,2*(len(kept)+1))/2/flt) if flt>0 else None
            # --- PER-ARM RANK (net-sigma channel): louder-in-net famN families, by EACH arm, self-seg guarded ---
            si=t["si"]; nhm=nlm=0
            for r in fams:
                if not (r[0]>=t["net"] and fold[r[1]]==g): continue
                if r[1]==si or r[3]==si: continue            # self-segment guard
                hm,lm=surv_cnn_pair(r[1],r[2],r[3],r[4])
                if hm>=t["cnn_hm"]: nhm+=1
                if lm>=t["cnn_lm"]: nlm+=1
            t["far_net_hm"]=nhm/flt if flt>0 else None; t["far_net_lm"]=nlm/flt if flt>0 else None
            t["far_net_perarm"]=_perarm(t["far_net_hm"],t["far_net_lm"])
            t["far_net_perarm_ul90"]=_perarm(_ul90(nhm,flt),_ul90(nlm,flt))
            print(f"  net-cand {t['seg']} net={t['net']:.2f} HM={_f(t.get('cnn_hm'),'.3f')} LM={_f(t.get('cnn_lm'),'.3f')}"
                  f"{' GLITCH' if t.get('is_glitch') else ''} -> FAR_net={_f(t.get('far_net_per_yr'),'.3g')}/yr (UL90 {_f(t.get('far_net_ul90'),'.3g')}) {t.get('matches_known','')}",flush=True)

    if PERARM: print(f"[blind] PER-ARM RANK active (SM_PERARM=1); surv_cnn_pair live-fallback hits = {_fallback_hits[0]} (must be 0)",flush=True)
    def best_far(t):                                              # detection statistic = best (lowest) FAR across channels
        if PERARM:                                               # per-arm headline: honest channel trials (per-arm already x2 for arm)
            cps=[c for c in (t.get("far_lr_perarm"),t.get("far_net_perarm")) if c is not None]
            return len(cps)*min(cps) if cps else None            # x n_channels -> total trials = 2(arm) * n_channels
        fs=[f for f in (t.get("far_cnn_per_yr"),t.get("far_net_per_yr")) if f is not None]   # default: unchanged OR-veto
        return min(fs) if fs else None
    # ===== SAVE EVERY SURVIVING BACKGROUND TRIGGER (full provenance, for direct inspection) =====
    # cnn_cache holds every louder-bg family that was CNN-scored across all FAR'd candidates (loglr + net
    # channels) -> exactly the survivors that set the floor. Dump coords+scores so any can be re-plotted via
    # series_for(seg,det,[idx]) without replaying the background search. CNN-survivors (is_glitch False) are
    # the residual-FAR contributors (the morphology-matched triggers the gate cannot reject).
    llmap={}
    for g in (0,1):
        for ll,ai,iH,bi,iL in bg[g]: llmap[(ai,int(iH),bi,int(iL))]=float(ll)
    def _prov(ai,iH,bi,iL,hm,lm):
        return dict(H1_seg=segs[ai]["name"],iH=int(iH),gps_H=float(segs[ai]["gps"][iH]),
                    sigma_H=float(segs[ai]["sH"][iH]),cen_H=float(segs[ai]["cH"][iH]),
                    L1_seg=segs[bi]["name"],iL=int(iL),gps_L=float(segs[bi]["gps"][iL]),
                    sigma_L=float(segs[bi]["sL"][iL]),cen_L=float(segs[bi]["cL"][iL]),
                    lag_s=float((iH-iL)*STRIDE),loglr=llmap.get((ai,int(iH),bi,int(iL))),
                    cnn_hm=float(hm),cnn_lm=float(lm),cnn=float(max(hm,lm)),
                    is_glitch=bool(max(hm,lm)<GLITCH_THRESH))
    bg_surv=[_prov(ai,iH,bi,iL,hm,lm) for (ai,iH,bi,iL),(hm,lm) in cnn_cache.items()]
    bg_surv.sort(key=lambda r:-(r["loglr"] if r["loglr"] is not None else -1e9))
    n_pass=sum(1 for r in bg_surv if not r["is_glitch"])
    json.dump(dict(glitch_thresh=GLITCH_THRESH,n_scored=len(bg_surv),n_cnn_survivors=n_pass,survivors=bg_surv),
              open(f"{OUT}/survivors_bg.json","w"),indent=2)
    print(f"[blind] saved {len(bg_surv)} CNN-scored bg survivors ({n_pass} pass CNN gate -> residual-FAR drivers) -> survivors_bg.json",flush=True)

    for t in fgt: t.pop("_louder",None)
    json.dump(dict(net_cut=NET_CUT,cluster_s=CLUSTER_S,n_segments=nseg,far_live_yr=far_live,netsig_floor=NETSIG_FLOOR,
                   n_triggers=len(fgt),n_cnn_vetoed=len(cand_idx),triggers=fgt),
              open(f"{OUT}/blindscan.json","w"),indent=2)
    # ===== AUTOMATED FINAL PRODUCT: DETECTIONS (candidate kept by the OR-combine, FAR below threshold in EITHER channel) =====
    DET_FAR=float(os.environ.get("DET_FAR","1.0"))   # detection FAR threshold (per yr)
    dets=[t for t in fgt if not t.get("is_glitch") and best_far(t) is not None and best_far(t)<DET_FAR]
    dets.sort(key=lambda d:best_far(d))
    json.dump([dict(seg=d["seg"],gps=d["gps"],net=d["net"],loglr=d["loglr"],cnn_hm=d.get("cnn_hm"),cnn_lm=d.get("cnn_lm"),
                    kept_by=d.get("kept_by"),far_cnn_per_yr=d.get("far_cnn_per_yr"),far_cnn_ul90=d.get("far_cnn_ul90"),
                    far_net_per_yr=d.get("far_net_per_yr"),far_net_ul90=d.get("far_net_ul90"),
                    far_lr_perarm=d.get("far_lr_perarm"),far_lr_perarm_ul90=d.get("far_lr_perarm_ul90"),
                    far_net_perarm=d.get("far_net_perarm"),far_net_perarm_ul90=d.get("far_net_perarm_ul90"),
                    far_lr_hm=d.get("far_lr_hm"),far_lr_lm=d.get("far_lr_lm"),
                    far_net_hm=d.get("far_net_hm"),far_net_lm=d.get("far_net_lm"),best_far=best_far(d),
                    stat=("per-arm" if PERARM else "OR-veto"),
                    channel=("net-sigma" if d.get("far_net_per_yr") is not None and best_far(d)==d.get("far_net_per_yr") else "loglr"),
                    matches_known=d.get("matches_known","")) for d in dets],
              open(f"{OUT}/detections.json","w"),indent=2)
    print(f"\n=== DETECTIONS (OR-combine kept, FAR<{DET_FAR}/yr in loglr OR net-sigma channel): {len(dets)} ===",flush=True)
    print(f"  {'seg/event':22s} {'gps':>12s} {'net':>5s} {'loglr':>6s} {'HM':>6s} {'LM':>6s} {'FAR_LR':>8s} {'FAR_net':>8s} {'best':>8s}",flush=True)
    for d in dets:
        print(f"  {str(d.get('matches_known') or d['seg']):22s} {d['gps']:12.1f} {d['net']:5.2f} {d['loglr']:6.2f} "
              f"{_f(d.get('cnn_hm'),'.3f'):>6} {_f(d.get('cnn_lm'),'.3f'):>6} {_f(d.get('far_cnn_per_yr'),'.3g'):>8} {_f(d.get('far_net_per_yr'),'.3g'):>8} {_f(best_far(d),'.3g'):>8}",flush=True)
    print(f"\n=== BLIND SCAN: {len(fgt)} triggers (net>{NET_CUT}); candidates (loglr>={floor}) ===",flush=True)
    for t in fgt:
        if t.get("far_lr_per_yr") is None: continue          # below floor: not a candidate, skip in summary
        fc=t.get("far_cnn_per_yr"); flag="<<<" if (fc is not None and fc<1.0) else ""
        seg=str(t.get('seg') or '-')
        print(f"  {seg:20s} gps={_f(t.get('gps'),'.1f')} net={_f(t.get('net'),'.2f')} coh={_f(t.get('coh'),'.3f')} loglr={_f(t.get('loglr'),'.2f')} "
              f"FAR_LR={_f(t.get('far_lr_per_yr'),'.3g')}/yr"+(f" FAR_CNN={_f(fc,'.3g')}/yr cnn={_f(t.get('cnn'),'.3f')}{' GLITCH' if t.get('is_glitch') else ''}" if fc is not None else "")
              +f" {t.get('matches_known','')} {flag}",flush=True)

if __name__=="__main__": main()

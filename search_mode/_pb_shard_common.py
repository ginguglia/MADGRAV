"""Shared deterministic setup for the 4-GPU shard launcher + merge.

Everything in build_setup() is a VERBATIM lift of the head of driver_blindscan.main()
up to and including the COH_CEIL/floor computation. It depends ONLY on (segs, RNG seed,
injections) -- NOT on which pairing subset a shard processes -- so EVERY shard and the
merge step reproduce IDENTICAL segs/fold/pairings/mdl/fgt/floor/COH_CEIL/famN-keys.

The ONLY thing a shard does differently is iterate a contiguous/strided SUBSET of `pairings`
in the prune loop; bg is an order-independent union, famN is a per-(fold,L1key) MAX, and
face_live is additive -- so merging the partials is exactly the unsharded result.
"""
import os,sys,json,time
import numpy as np
MADGRAV_ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import driver_streams as DS
DS.DEV=os.environ.get("BLIND_DEV","cuda:1")
import driver_search_multi as M

# mirror driver_blindscan module constants (kept in sync; bit-identical)
NET_CUT=float(os.environ.get("SM_ZL_NET_CUT","4.0"))   # zero-lag candidate net cut for the CNN/merge stages (2026-09-19: env-configurable; default 4.0 = pilot; SM_NET_CUT=3 only reaches the prune background)
ZL_GATES=bool(os.environ.get("SM_ZL_GATES"))            # 2026-09-20: SM_ZL_GATES=1 applies the centroid-v2 + arm gates to the zero-lag trigger windows exactly as driver_blindscan.py:270 does for the prune (foreground == background definition). Default OFF = pilot/coh behaviour (ungated windows).
CLUSTER_S=4.0
GLITCH_THRESH=0.5
CAND_LOGLR_FLOOR=float(os.environ.get("SM_CAND_FLOOR","5.0"))
NETSIG_FLOOR=float(os.environ.get("SM_NETSIG_FLOOR","0"))
OUT=M.OUT; YR=M.YR; STRIDE=M.STRIDE; FS=M.FS

def L1key(segs,bi,iL): return (bi,int(segs[bi]["gps"][iL]//M.MERGE_S))

def build_setup():
    """Returns a dict with every deterministic object the prune loop + merge need."""
    avail=[(a,nm) for a,b,d,nm in M.SEG["segments"] if M.have(a,nm)]
    M.require_complete(avail,M.SEG["segments"],"shard-setup")   # every shard loads the FULL segment set
    segs=[M.load_seg(a,nm) for a,nm in avail]; nseg=len(segs); fold=[i%2 for i in range(nseg)]
    print(f"[shard-setup] {nseg} segs folds={fold}",flush=True)
    step_off=int(round(M.OFFSET_STEP_S/STRIDE)); same_minoff=int(round(M.SAME_SEG_MINLAG_S/STRIDE)); evrad=int(2.0/STRIDE)
    evloc={ev:(i,int(np.abs(segs[i]["gps"]-g0).argmin())) for ev,g0 in M.EVENTS.items() for i,s in enumerate(segs) if s["gps"][0]<=g0<=s["gps"][-1]}   # GPS-containment: match event to the segment whose time range contains it (segments are GPS-named o3a_*, not event-named, so name-equality never matched -> proper cross-fit). SYNCED with driver_blindscan.py:172.
    def is_ev(si,idx):
        m=np.zeros(len(idx),bool)
        for ev,(esi,ew) in evloc.items():
            if si==esi: m|=np.abs(idx-ew)<=evrad
        return m
    Xs_all=[]; sf_list=[]
    BOTHFOLDS=bool(os.environ.get("SM_INJ_BOTHFOLDS"))
    for ev in M.EVENTS:
        if not os.path.exists(f"{M.INJ}/{ev}_inj.npz"):   # SYNCED with driver_blindscan.py:192 — no injection bank for this event:
            print(f"[shard-setup] no {ev}_inj.npz in {M.INJ} -> skipped (inj feats feed the LR REFIT only; harmless under SM_LR_MODEL)",flush=True)
            continue                                       # inj feats feed fit_lr only; under SM_LR_MODEL they are never used
        za=np.load(f"{M.INJ}/{ev}_inj.npz"); _cH,_cL=(za["cenH"],za["cenL"]) if M.CEN_V1 else (za["cenH2"],za["cenL2"]); _gH,_gL=(za["gH2"],za["gL2"]) if M.ARM_KEY=="g2" else (za["gH"],za["gL"])
        _k=M.cen_gate(_cH,_cL)&M.arm_gate(_gH,_gL,za["sigH"],za["sigL"]); X=M.feats(za["sigH"][_k],za["sigL"][_k],za["coh"][_k],_cH[_k],_cL[_k],_gH[_k],_gL[_k],gx=(za["gx"][_k] if M.GX.ON else None))   # MADGRAV2 features + gates (+ gx under SM_GX=1; SYNCED with driver_blindscan.py)
        if BOTHFOLDS:
            for fgi in (0,1): Xs_all.append(X); sf_list.append(np.full(len(X),fgi,dtype=int))
        else:
            Xs_all.append(X); sf_list.append(np.full(len(X),fold[evloc[ev][0]],dtype=int))
    if Xs_all: Xs_all=np.vstack(Xs_all); sf=np.concatenate(sf_list)
    else:      Xs_all=np.zeros((0,0)); sf=np.zeros(0,dtype=int)   # no banks at all: only legal under SM_LR_MODEL (checked below)
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
    print(f"[shard-setup] {len(pairings)} same-fold pairings",flush=True)
    # ---- noise sample per fold (identical to driver_blindscan) ----
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
        if not P: Xn[g]=np.zeros((0,M.NFEAT)); continue
        nH={};nL={}
        for a_,h_,b_,l_ in P: nH.setdefault(a_,set()).add(h_); nL.setdefault(b_,set()).add(l_)
        SH={a_:M.series_for(segs[a_]["name"],"H1",sorted(w)) for a_,w in nH.items()}
        SL={b_:M.series_for(segs[b_]["name"],"L1",sorted(w)) for b_,w in nL.items()}
        _A=np.array([SH[a_][h_] for a_,h_,b_,l_ in P]); _B=np.array([SL[b_][l_] for a_,h_,b_,l_ in P]); co=M.coh_vec(_A,_B); gxn=M.gx_vec(_A,_B)   # MADGRAV2.1: + guided excess
        cohnoise[g]=co
        ai_=np.array([p[0] for p in P]);iH_=np.array([p[1] for p in P]);bi_=np.array([p[2] for p in P]);iL_=np.array([p[3] for p in P])
        Xn[g]=M.feats(gv("sH",ai_,iH_),gv("sL",bi_,iL_),co,gv("cH",ai_,iH_),gv("cL",bi_,iL_),gv("gH",ai_,iH_),gv("gL",bi_,iL_),gx=gxn)
        print(f"[shard-setup] fold {g} noise {len(P)} rows coh med {np.median(co):.3f}"+(f" gx med {np.median(gxn):.2f} p99 {np.percentile(gxn,99):.2f}" if gxn is not None else ""),flush=True)
    _lrm=os.environ.get("SM_LR_MODEL")
    if _lrm:                                          # FROZEN ranking statistic: load the search's (mu,sd,beta) per fold
        _z=np.load(_lrm)                              # (SM_DUMP_MODEL format) instead of refitting on this run's noise draw.
        mdl={g:(_z[f"mu{g}"],_z[f"sd{g}"],_z[f"be{g}"]) for g in (0,1)}
        for g in (0,1):                               # MADGRAV2.1: the frozen model must have been fitted with the same feature set (7, or 8 with SM_GX=1)
            if len(mdl[g][2])!=M.NFEAT+1: raise SystemExit(f"[lr-model] {_lrm}: fold {g} has {len(mdl[g][2])-1} features but this run expects {M.NFEAT} (SM_GX={os.environ.get('SM_GX','0')}); refit the LR with the same SM_GX")
        print(f"[shard-setup] FROZEN LR model loaded from {_lrm} (refit skipped)",flush=True)
    else:
        if len(Xs_all)==0: raise RuntimeError(f"no injection banks found in {M.INJ} and SM_LR_MODEL unset: cannot refit the LR")
        mdl={g:M.fit_lr(Xn[g],Xs_all[sf==g]) for g in (0,1)}

    # ===== FOREGROUND triggers (identical) =====
    fglist=[]
    for si,s in enumerate(segs):
        net=(s["sH"]+s["sL"])/np.sqrt(2.); _hm=net>NET_CUT
        if ZL_GATES: _hm&=M.cen_gate(s["cH"],s["cL"])&M.arm_gate(s["gH"],s["gL"],s["sH"],s["sL"])   # SYNCED with driver_blindscan.py:270
        hit=np.where(_hm)[0]
        if not len(hit): continue
        gap=int(round(CLUSTER_S/STRIDE)); groups=[]; cur=[hit[0]]
        for i in hit[1:]:
            if i-cur[-1]<=gap: cur.append(i)
            else: groups.append(cur); cur=[i]
        groups.append(cur)
        for gr in groups:
            j=gr[int(np.argmax(net[gr]))]; fglist.append((si,int(j)))
    print(f"[shard-setup] {len(fglist)} clustered zero-lag triggers (net>{NET_CUT})",flush=True)
    needH={};needL={}
    for si,j in fglist: needH.setdefault(si,set()).add(j); needL.setdefault(si,set()).add(j)
    serH={si:M.series_for(segs[si]["name"],"H1",sorted(w)) for si,w in needH.items()}
    serL={si:M.series_for(segs[si]["name"],"L1",sorted(w)) for si,w in needL.items()}
    fgt=[]
    for si,j in fglist:
        g=fold[si]; mu,sd,be=mdl[1-g]
        co=float(M.coh_vec(serH[si][j][None,:],serL[si][j][None,:])[0]); gxv=M.gx_vec(serH[si][j][None,:],serL[si][j][None,:])   # MADGRAV2.1: guided excess of the zero-lag pair
        F=M.feats([segs[si]["sH"][j]],[segs[si]["sL"][j]],[co],[segs[si]["cH"][j]],[segs[si]["cL"][j]],[segs[si]["gH"][j]],[segs[si]["gL"][j]],gx=gxv)
        ll=float(M.loglr(mu,sd,be,F)[0]); net=float((segs[si]["sH"][j]+segs[si]["sL"][j])/np.sqrt(2.))
        fgt.append(dict(seg=segs[si]["name"],gps=float(segs[si]["gps"][j]),si=si,idx=j,fold=g,net=net,coh=co,loglr=ll,gx=(float(gxv[0]) if gxv is not None else None)))
    known_ll=[t["loglr"] for t in fgt for ev,(esi,ew) in evloc.items() if t["si"]==esi and abs(t["idx"]-ew)<=evrad]
    floor=CAND_LOGLR_FLOOR if os.environ.get("SM_BLIND_FLOOR") else min([CAND_LOGLR_FLOOR]+known_ll)
    thr_fold={g:floor for g in (0,1)}
    ncand=sum(1 for t in fgt if t["loglr"]>=floor)
    print(f"[shard-setup] prune floor={floor:.2f} (known {[round(x,2) for x in known_ll]}); {ncand} candidate triggers >= floor",flush=True)
    # ===== COH_CEIL (identical) =====
    sub_max=max([float(c.max()) for c in cohnoise.values() if len(c)]+[0.0])
    trig_max=max([t["coh"] for t in fgt]+[0.0])
    _c=np.concatenate([c for c in cohnoise.values() if len(c)]) if cohnoise else np.array([0.0])
    _thr=float(np.percentile(_c,99.0)); _beta=max(1e-4,float((_c[_c>_thr]-_thr).mean()))
    _N=float(len(pairings))*float(segs[0]["n"] if segs else 1)
    KSIG=float(os.environ.get("SM_COH_KSIGMA","6.0"))
    pop_bound=sub_max+_beta*np.log(max(1.0,_N/max(1,len(_c))))+KSIG*_beta*np.pi/np.sqrt(6.0)
    COH_CEIL=float(min(1.0,max(pop_bound,trig_max+0.02)))
    if os.environ.get("SM_COH_CEIL_LEGACY"): COH_CEIL=float(min(1.0,max(sub_max,trig_max)+0.15))
    if os.environ.get("SM_COH_CEIL"): COH_CEIL=float(os.environ["SM_COH_CEIL"])
    print(f"[shard-setup] coh ceiling {COH_CEIL:.3f} (sub_max {sub_max:.3f} trig_max {trig_max:.3f})",flush=True)

    return dict(segs=segs,nseg=nseg,fold=fold,evloc=evloc,evrad=evrad,is_ev=is_ev,
                pairings=pairings,mdl=mdl,fgt=fgt,floor=floor,thr_fold=thr_fold,
                COH_CEIL=COH_CEIL,cohnoise=cohnoise,Xs_all=Xs_all,sf=sf)

#!/usr/bin/env python
"""Stationarity-mask census (2026-10-09, analysis lead go): a TIME mask built from the sigma stream alone (no catalogue, no candidate, no coincidence),
applied to all four runs. Measures, per pre-registered variant: livetime removed (run-wide and in the injection segments = injection cost,
since injections are placed uniformly in time), zero-lag rows removed (26 census rows: 5 withheld + 3 kept net-sigma events + noise rows;
all current detections), net-sigma background families removed (leverage = bg removed / livetime removed), the recomputed net-sigma FAR of
every census row after the mask (clause OFF), the count test with the clause off, and the net event balance (lost - readmitted).
Variants: SEG(q): whole segment masked for a detector if frac(sigma>2) > q, q in {0.2,0.3,0.5};
          LOC(W=64,q): window masked if frac(sigma>2) in [i-W,i+W] excluding i exceeds q, q in {0.2,0.3,0.5}.
Background families from the far shards + cnn_shard cache (exact merge inputs; the survivors json was the wrong source, 08:35).\nUsage: stat_mask_census.py <run>  -> CENSUS_STATMASK_<run>.txt / .json here."""
import sys, os, glob, json, numpy as np
from scipy.stats import poisson, chi2
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

RUN=sys.argv[1]; SC=MADGRAV_SCRATCH + "/madgrav2"; W=MADGRAV_ROOT + "/m2"; HERE=os.path.dirname(os.path.abspath(__file__))
SRC={"o3a":"off5hw","o3b":"off5hw","o4a":"off5","o4b":"off5"}[RUN]; OUT=f"{SC}/search_out_{RUN}_m2_gx_{SRC}"
T_FG={"o3a":0.146,"o3b":0.132,"o4a":0.172,"o4b":0.156}[RUN]; SIG=2.0; QS=(0.2,0.3,0.5); WLOC=64; GL=0.5; FLOOR=3.0
import re, subprocess
_ml=sorted(glob.glob(f"{W}/freeze/logs/merge-v2-{RUN}_*.log"),key=os.path.getmtime)[-1]   # the production _v2 merge: same streams, same arm rule
sd=re.search(r"streams=(\S+)",open(_ml).read()).group(1); _fam_target=re.search(r"net-families (\d+)->(\d+)",open(_ml).read()).groups()
ARM_THR=-2.05   # SM_ARM_RULE max:-2.05 (merge log)
segs=sorted(os.path.basename(f)[:-12] for f in glob.glob(f"{sd}/*_H1_meta.npz")); sidx={s:i for i,s in enumerate(segs)}; fold={s:i%2 for i,s in enumerate(segs)}
print(f"[{RUN}] {len(segs)} segments from {sd}",flush=True)
# ---- masks per segment, detector, variant ----
VAR=[f"SEG{q}" for q in QS]+[f"LOC{q}" for q in QS]
masks={}; t0s={}; nwin={}; SIGMA={}; G2={}
for s in segs:
    t0s[s]=None; masks[s]={}
    for det in ("H1","L1"):
        z=np.load(f"{sd}/{s}_{det}_meta.npz",allow_pickle=True); sg=np.asarray(z["sigma"],float); n=len(sg); t0s[s]=float(z["t0"]); nwin[s]=n
        SIGMA.setdefault(s,{})[det]=sg; G2.setdefault(s,{})[det]=np.asarray(z["g2"],float)
        ex=(sg>SIG).astype(float); fr=ex.mean() if n else 0.0
        c=np.concatenate([[0.0],np.cumsum(ex)]); i=np.arange(n); lo=np.maximum(i-WLOC,0); hi=np.minimum(i+WLOC+1,n)
        cnt=c[hi]-c[lo]-ex; den=(hi-lo-1); loc=np.where(den>0,cnt/np.maximum(den,1),0.0)
        m={}
        for q in QS: m[f"SEG{q}"]=np.full(n,fr>q); m[f"LOC{q}"]=loc>q
        masks[s][det]=m
tot=sum(nwin.values())
def rem_frac(v,segset=None):
    ss=segs if segset is None else [s for s in segs if s in segset]; n=sum(nwin[s] for s in ss)
    h=sum(masks[s]["H1"][v].sum() for s in ss); l=sum(masks[s]["L1"][v].sum() for s in ss); u=sum((masks[s]["H1"][v]|masks[s]["L1"][v]).sum() for s in ss)
    return h/n, l/n, u/n
def masked(s,det,gps,v):
    if s not in masks: return None
    i=int(round(gps-t0s[s])); 
    if i<0 or i>=nwin[s]: return None
    return bool(masks[s][det][v][i])
def row_masked(s,gps,v): 
    a,b=masked(s,"H1",gps,v),masked(s,"L1",gps,v); return None if a is None else (a or b)
# ---- injection segments ----
try:
    inj=np.load(f"{SC}/vt22/inj_scored_{RUN}_m22.npz",allow_pickle=True); injsegs=set(str(e) for e in inj["ev"])
except Exception as e: inj=None; injsegs=set(); print("no inj file",e)
# ---- zero-lag rows ----
cen=[c for c in json.load(open(f"{W}/asdveto/census_rows_26.json")) if c["run"]==RUN]
det=json.load(open(f"{OUT}/detections_v2.json")); bs=json.load(open(f"{OUT}/blindscan_v2.json")); TR={round(t["gps"]):t for t in bs["triggers"]}; TL={int(k):float(v) for k,v in bs["far_live_yr"].items()}
# ---- background families (net-sigma) EXACTLY as the merge builds them: per-(fold, L1 4 s bin) max-net family from the far shards
#      (pairs already pass the centroid + arm gates), CNN scores of the representative pair from the merge's cnn_shard cache,
#      per-arm rank against the candidate's OWN CNN score, self-segment guard, FAR = 2*min(n_hm,n_lm)/livetime(fold) ----
SH=f"{SC}/_{RUN}_m2_gx_{SRC}_far_shards"; K=[];V=[]
for f in sorted(glob.glob(f"{SH}/shard_*.npz")): z=np.load(f); K.append(z["fam_keys"]); V.append(z["fam_vals"])
K=np.concatenate(K); V=np.concatenate(V); print(f"[{RUN}] {len(K)} family rows from {len(glob.glob(f'{SH}/shard_*.npz'))} shards",flush=True)
order=np.lexsort((-V[:,0],K[:,2],K[:,1],K[:,0])); K=K[order]; V=V[order]
first=np.ones(len(K),bool); first[1:]=np.any(K[1:]!=K[:-1],axis=1); K=K[first]; V=V[first]
fnet=V[:,0]; fai=V[:,1].astype(np.int64); fiH=V[:,2].astype(np.int64); fbi=V[:,3].astype(np.int64); fiL=V[:,4].astype(np.int64); fg=K[:,0]
# ---- MERGE-TIME ARM RULE max:-2.05 on the family representatives (merge log: net-families 2.03M -> 35k in o3b) ----
_sH=np.empty(len(fnet)); _sL=np.empty(len(fnet)); _gH=np.empty(len(fnet)); _gL=np.empty(len(fnet))
for s_ in segs:
    i=sidx[s_]; a=np.where(fai==i)[0]; b=np.where(fbi==i)[0]
    if len(a): _sH[a]=SIGMA[s_]["H1"][fiH[a]]; _gH[a]=G2[s_]["H1"][fiH[a]]
    if len(b): _sL[b]=SIGMA[s_]["L1"][fiL[b]]; _gL[b]=G2[s_]["L1"][fiL[b]]
_ok=np.where(_sH>=_sL,_gH,_gL)>=ARM_THR
print(f"[{RUN}] arm rule: families {len(fnet)} -> {int(_ok.sum())} (merge log said {_fam_target[0]} -> {_fam_target[1]})",flush=True)
fnet,fai,fiH,fbi,fiL,fg=fnet[_ok],fai[_ok],fiH[_ok],fbi[_ok],fiL[_ok],fg[_ok]
def pkey(ai,iH,bi,iL): return ((ai*4000+bi)*400000+iH)*400000+iL
P=[];Hs=[];Ls=[]
for f in glob.glob(f"{OUT}/cnn_shard_*.npz"): z=np.load(f); P.append(z["pairs"]); Hs.append(z["hm"]); Ls.append(z["lm"])
P=np.concatenate(P); Hs=np.concatenate(Hs); Ls=np.concatenate(Ls); ck=pkey(P[:,0],P[:,1],P[:,2],P[:,3])
cu,ci=np.unique(ck,return_index=True); fk=pkey(fai,fiH,fbi,fiL); pos=np.searchsorted(cu,fk); pos[pos>=len(cu)]=0; hit=cu[pos]==fk
fhm=np.where(hit,Hs[ci[pos]],np.nan); flm=np.where(hit,Ls[ci[pos]],np.nan)
print(f"[{RUN}] {len(fnet)} families; CNN scores found for {hit.sum()} ({hit.mean():.4f}); among net>=3: {hit[fnet>=3].mean():.4f}; max iH/nwin check: {max(fiH[i]<nwin[segs[fai[i]]] for i in range(0,len(fai),max(1,len(fai)//2000)))}",flush=True)
def fam_masked(v):
    if v is None: return np.zeros(len(fnet),bool)
    m=np.zeros(len(fnet),bool)
    for s in segs:
        i=sidx[s]; selH=np.where(fai==i)[0]; selL=np.where(fbi==i)[0]
        if len(selH): m[selH]|=masks[s]["H1"][v][np.minimum(fiH[selH],nwin[s]-1)]
        if len(selL): m[selL]|=masks[s]["L1"][v][np.minimum(fiL[selL],nwin[s]-1)]
    return m
def families(v): return fam_masked(v)
def far_net(fm,t,Tg):
    """fm = boolean masked-family array; t = zero-lag trigger dict (net, fold, si, cnn_hm, cnn_lm)"""
    g=t["fold"]; si=t["si"]; sel=(~fm)&(fnet>=t["net"])&(fg==g)&(fai!=si)&(fbi!=si)
    nh=int(np.sum(sel&(fhm>=t["cnn_hm"]))); nl=int(np.sum(sel&(flm>=t["cnn_lm"]))); n=min(nh,nl)
    return 2*n/Tg, 2*chi2.ppf(0.90,2*n+2)/2/Tg, n
F0=families(None)
res=dict(run=RUN,n_segments=len(segs),n_windows=tot,variants={},rows=[])
# validation of the FAR reconstruction on unmasked data (stored far_net_perarm of the census rows)
val=[]
for c in cen:
    t=TR.get(round(c["gps"])); 
    if t is None or t.get("far_net_perarm") is None: continue
    g=t["fold"]; f,u,n=far_net(F0,t,TL[g]); val.append((c["name"],t["net"],t["far_net_perarm"],f,n))
res["validation"]=val
lines=[f"# Stationarity-mask census {RUN} (2026-10-09): {len(segs)} segments, {tot} windows, T_fg {T_FG} yr, slide livetime {TL}",
       f"net-sigma background: {int(np.sum(fnet>=3))} families after the arm rule (merge log {_fam_target[0]}->{_fam_target[1]}); CNN scores for {hit[fnet>=3].mean():.4f} of them",
       "FAR reconstruction check (stored far_net_perarm vs recomputed from survivors, unmasked): "+"; ".join(f"{n} {a:.3f}/{b:.3f}" for n,_,a,b,_ in val)]
cat_rows=[(r.get("matches_known") or f"NONCAT@{r['gps']:.0f}",r["seg"],r["gps"],r["channel"],r["best_far"]) for r in det]
for v in VAR:
    fh,fl,fu=rem_frac(v); fi=rem_frac(v,injsegs)[2] if injsegs else float("nan")
    Fv=families(v); nf0=int(np.sum(fnet>=3)); nfv=int(np.sum((fnet>=3)&~Fv)); lev=((nf0-nfv)/nf0)/fu if fu>0 else float("nan")
    lost=[n for n,s,g,ch,f in cat_rows if row_masked(s,g,v)]
    rows=[]; readmit=[]; noise_in=[]
    for c in cen:
        t=TR.get(round(c["gps"])); m=row_masked(c["seg"],c["gps"],v)
        if t is None: rows.append(dict(variant=v,name=c["name"],kind=c["kind"],catalogue=c["name"].startswith("GW"),in_set=False,masked=m,net=c["net"],far0=float("nan"),far_mask=float("nan"),n_louder=-1)); continue
        g=t["fold"]; net=t["net"]; Tg=TL[g]*(1-fu)
        f1,u1,n1=far_net(Fv,t,Tg); f0,u0,n0=far_net(F0,t,TL[g])
        iscat=c["name"].startswith("GW"); kept_now=any(c["gps"]==r["gps"] for r in det)
        rows.append(dict(variant=v,name=c["name"],kind=c["kind"],catalogue=iscat,in_set=kept_now,masked=m,net=net,far0=f0,far_mask=f1,n_louder=n1))
        if iscat and not kept_now and not m and f1<1.0: readmit.append(c["name"])
        if not iscat and not m and f1<1.0: noise_in.append((c["name"],round(f1,3)))
    n_noise=len(noise_in); p=poisson.sf(n_noise-1,T_FG) if n_noise>0 else 1.0
    res["variants"][v]=dict(frac_H1=fh,frac_L1=fl,frac_union=fu,frac_inj_segs=fi,families_before=nf0,families_after=nfv,leverage=lev,
                            detections_lost=lost,readmitted=readmit,noise_rows_surviving=noise_in,count_test_p=p,net_balance=len(readmit)-len(lost))
    res["rows"]+=rows
    lines.append(f"\n## {v}: livetime removed H1 {fh:.4f} L1 {fl:.4f} union {fu:.4f} | in injection segments {fi:.4f} (= injection cost)")
    lines.append(f"   net-sigma families {nf0} -> {nfv} ({(nf0-nfv)/nf0:.3f} removed); leverage {lev:.1f}")
    lines.append(f"   current detections in masked time (LOST): {len(lost)} {lost}")
    lines.append(f"   withheld catalogue rows unmasked with recomputed FAR<1 (READMITTED, clause off): {len(readmit)} {readmit}")
    lines.append(f"   noise rows surviving (clause off, FAR<1): {n_noise} {noise_in}  expected {T_FG:.3f}  P(>=n) {p:.1e}  {'FAIL' if p<0.01 else 'ok'}")
    lines.append(f"   NET BALANCE (readmitted - lost): {len(readmit)-len(lost):+d}")
    for r in rows: lines.append(f"     {r['name']:18s} {'cat' if r['catalogue'] else 'noise':5s} in_set={str(r['in_set']):5s} masked={str(r['masked']):5s} net {r['net']:.2f} FAR {r['far0']:.3f} -> {r['far_mask']:.3f} (N {r['n_louder']})")
open(f"{HERE}/CENSUS_STATMASK_{RUN}.txt","w").write("\n".join(lines)+"\n"); json.dump(res,open(f"{HERE}/CENSUS_STATMASK_{RUN}.json","w"),indent=1,default=float)
print("\n".join(lines))

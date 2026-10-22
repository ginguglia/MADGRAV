"""Price net-sigma guard variants EXACTLY on the FAR-setting background, standalone (2026-10-14), 0 jobs.
Reconstructs the merge's net-sigma channel (famN reps within candidate reach = the cnn_shard cache, LM floor, fold livetime) from the
crop out dir + nf275 far shards, validates it against n_louder_netfam_cnn of the crop _v2 triggers, then re-counts louder families
and the zero-lag detections under each variant. Both placements' CNN scores come from the two existing full caches
(search_out_<run>_m2_gx_off5_crop = coincident crop, search_out_<run>_m2_gx_off5_nf275 = Grad-CAM crop); bank loglr from the crop S3 bank.
Variants:  V0 production (coinc gate & lm>0.0382) | V1 V0 & Grad-CAM gate | V2 V0 & (Grad-CAM gate | loglr>0) | V3 V0 & lm>0.5
           V4 V2 & lm>0.5 (= drafted guard) | V5 V0 & (Grad-CAM gate | loglr>=LRFLOOR)
Usage: python -I price_netsig_guard.py <run> [NETMIN=2.75]  -> chain/PRICE_NSGUARD_<run>.txt"""
import sys, os, json, glob, time, numpy as np
run=sys.argv[1]; NETMIN=float(sys.argv[2]) if len(sys.argv)>2 else 2.75
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M=MADGRAV_SCRATCH + "/madgrav2"; OUT=f"{M}/search_out_{run}_m2_gx_off5_crop"; PROD=f"{M}/search_out_{run}_m2_gx_off5_nf275"
SH=f"{M}/_{run}_m2_gx_off5_nf275_far_shards"; S3=f"{MADGRAV_SCRATCH}/madgrav2_spec/s3_crop/s3_{run}.npz"
REP=f"{MADGRAV_ROOT}/m2/popb_cap/occlusion/chain/PRICE_NSGUARD_{run}.txt"
THR=0.5; LMF=0.0382; LRFLOOR=2.75; t0=time.time(); L=[]
def say(s): print(s,flush=True); L.append(s)
B=json.load(open(f"{OUT}/blindscan_v2.json")); T=B["triggers"]; far_live=B["far_live_yr"]; far_live={int(k):v for k,v in (far_live.items() if isinstance(far_live,dict) else enumerate(far_live))}
say(f"[price] {run}: {len(T)} triggers, far_live {far_live}, netsig_floor {B.get('netsig_floor')}, lm_min {B.get('netsig_lm_min')}")
# segment index -> name (for the S3-bank loglr join), validated on the triggers
si2name={}
for t in T: si2name.setdefault(t["si"],t["seg"])
# caches
def load_cache(d):
    c={}
    for f in sorted(glob.glob(f"{d}/cnn_shard_*.npz")):
        z=np.load(f); P=z["pairs"]; H=z["hm"]; Lm=z["lm"]
        for j in range(len(P)): c[(int(P[j,0]),int(P[j,1]),int(P[j,2]),int(P[j,3]))]=(float(H[j]),float(Lm[j]))
    return c
Cc=load_cache(OUT); Cg=load_cache(PROD); say(f"[price] caches: coinc {len(Cc)} pairs, gradcam {len(Cg)} pairs, common {len(set(Cc)&set(Cg))}  ({time.time()-t0:.0f}s)")
# famN from far shards (per (fold,bi,l1bin) max by net), restricted to net>=NETMIN
famN={}
for f in sorted(glob.glob(f"{SH}/shard_*.npz")):
    z=np.load(f); K=z["fam_keys"]; V=z["fam_vals"]; m=V[:,0]>=NETMIN; K=K[m]; V=V[m]
    for j in range(len(K)):
        k=(int(K[j,0]),int(K[j,1]),int(K[j,2])); net=float(V[j,0])
        if k not in famN or net>famN[k][0]: famN[k]=(net,int(V[j,1]),int(V[j,2]),int(V[j,3]),int(V[j,4]))
say(f"[price] famN (net>={NETMIN}): {len(famN)} families  ({time.time()-t0:.0f}s)")
# pair-level arm rule (merge line 176: SM_ARM_RULE=max:-2.05 on famN reps); segs in SM_BGJSON order filtered by stream presence
ENV=dict(kv.split("=",1) for kv in open(f"{MADGRAV_ROOT}/launchers/logs_m2/{run}_off5_crop_m2_gx_jids.env").read().split("EXP=ALL,")[-1].split("\n")[0].split(",") if "=" in kv)
STR=ENV["SM_STREAMS"]; SEG=json.load(open(ENV["SM_BGJSON"]))["segments"]; ARMTHR=-2.05
names=[nm for a,b,d,nm in SEG if os.path.exists(f"{STR}/{nm}_H1_meta.npz") and os.path.exists(f"{STR}/{nm}_L1_meta.npz")]
badmap=sum(1 for si,nm in si2name.items() if names[si]!=nm); say(f"[price] segs {len(names)} from {ENV['SM_BGJSON']} + {STR}; trigger si->name mismatches {badmap}")
assert badmap==0
segs=[]
for nm in names:
    mH=np.load(f"{STR}/{nm}_H1_meta.npz"); mL=np.load(f"{STR}/{nm}_L1_meta.npz"); k=min(int(mH["n"]),int(mL["n"]))
    segs.append(dict(name=nm,sH=mH["sigma"][:k],sL=mL["sigma"][:k],gH=mH["g2"][:k],gL=mL["g2"][:k]))
si2name={i:nm for i,nm in enumerate(names)}
def pair_ok(ai,iH,bi,iL):
    A,Bs=segs[ai],segs[bi]; sH,sL=float(A["sH"][iH]),float(Bs["sL"][iL]); v=float(A["gH"][iH]) if sH>=sL else float(Bs["gL"][iL]); return v>=ARMTHR
n0=len(famN); famN={k:v for k,v in famN.items() if pair_ok(v[1],v[2],v[3],v[4])}; say(f"[price] ARM RULE max:{ARMTHR}: families {n0} -> {len(famN)}  ({time.time()-t0:.0f}s)")
fams=sorted(famN.values(),key=lambda r:-r[0])
F_net=np.array([r[0] for r in fams]); F_ai=np.array([r[1] for r in fams]); F_bi=np.array([r[3] for r in fams]); F_fold=np.array([r[1]%2 for r in fams])
F_hm=np.full(len(fams),-np.inf); F_lm=np.full(len(fams),-np.inf); G_hm=np.full(len(fams),np.nan); G_lm=np.full(len(fams),np.nan)
for j,r in enumerate(fams):
    k=(r[1],r[2],r[3],r[4])
    if k in Cc: F_hm[j],F_lm[j]=Cc[k]
    if k in Cg: G_hm[j],G_lm[j]=Cg[k]
# bank loglr from the crop S3 bank, keyed by names
z=np.load(S3,allow_pickle=True); kb=z["kind"]=="bank"
s3ll={(str(a),int(b),str(c),int(d)):float(l) for a,b,c,d,l in zip(z["H1_seg"][kb],z["iH"][kb],z["L1_seg"][kb],z["iL"][kb],z["loglr"][kb])}
F_ll=np.full(len(fams),np.nan)
for j,r in enumerate(fams):
    na=si2name.get(r[1]); nb=si2name.get(r[3])
    if na and nb: F_ll[j]=s3ll.get((na,r[2],nb,r[4]),np.nan)
inC=np.isfinite(F_hm); inG=np.isfinite(G_hm); inLL=np.isfinite(F_ll)
say(f"[price] families with coinc score {inC.sum()}, gradcam score {inG.sum()}, both {(inC&inG).sum()}, S3-bank loglr {inLL.sum()}; "
    f"among coinc-gate-open (net>=4): gradcam {int((inC&inG&(F_net>=4)&(np.maximum(F_hm,F_lm)>THR)&(F_lm>LMF)).sum())}/{int((inC&(F_net>=4)&(np.maximum(F_hm,F_lm)>THR)&(F_lm>LMF)).sum())}, "
    f"loglr {int((inC&inLL&(F_net>=4)&(np.maximum(F_hm,F_lm)>THR)&(F_lm>LMF)).sum())}")
Gpass=np.where(inG,np.maximum(G_hm,G_lm)>=THR,False); LLpos=np.where(inLL,F_ll>0,False); LLfloor=np.where(inLL,F_ll>=LRFLOOR,False)
V0=(np.maximum(F_hm,F_lm)>THR)&(F_lm>LMF)
VAR={"V0 prod":V0,"V1 &gradcam":V0&Gpass,"V2 &(gradcam|loglr>0)":V0&(Gpass|LLpos),"V3 &lm>0.5":V0&(F_lm>0.5),"V4 drafted":V0&(Gpass|LLpos)&(F_lm>0.5),"V5 &(gradcam|loglr>=2.75)":V0&(Gpass|LLfloor)}
negnet=-F_net; cs={}
for name,m in VAR.items(): cs[name]={g:np.concatenate([[0],np.cumsum((F_fold==g)&m)]) for g in (0,1)}
say("[price] FAR-setting families by net bin (fold0+fold1), per variant:")
bins=[(4,4.5),(4.5,5),(5,6),(6,8),(8,99)]
say("  bin      "+" ".join(f"{n[:14]:>14s}" for n in VAR))
for lo,hi in bins:
    mb=(F_net>=lo)&(F_net<hi); say(f"  [{lo},{hi:>3}) "+" ".join(f"{int((m&mb).sum()):>14d}" for m in VAR.values()))
# foreground: gradcam scores + raw loglr by gps
Pg=json.load(open(f"{PROD}/blindscan_p205.json"))["triggers"]; pg=np.array([t["gps"] for t in Pg])
Pc=json.load(open(f"{OUT}/blindscan_p205.json"))["triggers"]; pc=np.array([t["gps"] for t in Pc])
def near(arr,lst,g):
    j=int(np.argmin(np.abs(arr-g))); return lst[j] if abs(arr[j]-g)<1.5 else None
net_rows=[t for t in T if t.get("far_net_per_yr") is not None]
say(f"[price] zero-lag net-sigma-open triggers: {len(net_rows)}; validation of the reconstruction (nk vs n_louder_netfam_cnn):")
bad=0
for t in net_rows:
    k=int(np.searchsorted(negnet,-float(t["net"]),side="right")); nk=int(cs["V0 prod"][t["fold"]][k])
    if nk!=t["n_louder_netfam_cnn"]: bad+=1
    t["_k"]=k
say(f"  mismatches {bad}/{len(net_rows)}" + ("  <-- reconstruction NOT exact, numbers below are approximate" if bad else "  EXACT"))
say("[price] zero-lag rows under each variant (far_net only; <1/yr = detection in this channel):")
hdr=f"  {'name':22s} {'gps':>12s} {'net':>5s} {'lmC':>5s} {'gcam':>9s} {'llraw':>6s} | "+" ".join(f"{n[:10]:>10s}" for n in VAR); say(hdr)
tot={n:[0,0] for n in VAR}
for t in sorted(net_rows,key=lambda x:-x["net"]):
    g=near(pg,Pg,t["gps"]); c=near(pc,Pc,t["gps"]); ll=c["loglr"] if c else t["loglr"]
    gp=(max(g["cnn_hm"],g["cnn_lm"])>=THR) if g else False
    fgok={"V0 prod":True,"V1 &gradcam":gp,"V2 &(gradcam|loglr>0)":gp or ll>0,"V3 &lm>0.5":t["cnn_lm"]>0.5,"V4 drafted":(gp or ll>0) and t["cnn_lm"]>0.5,"V5 &(gradcam|loglr>=2.75)":gp or ll>=LRFLOOR}
    cells=[]
    for n in VAR:
        far=cs[n][t["fold"]][t["_k"]]/far_live[t["fold"]]
        det=fgok[n] and far<1.0
        if det: tot[n][0 if t.get("matches_known") else 1]+=1
        cells.append(("%.3f"%far if fgok[n] else "veto")+("*" if det else " "))
    if any(c.endswith("*") for c in cells) or t.get("matches_known"):
        say(f"  {(t.get('matches_known') or '** NON-CAT **'):22s} {t['gps']:12.1f} {t['net']:5.2f} {t['cnn_lm']:5.2f} {('%.2f/%.2f'%(g['cnn_hm'],g['cnn_lm'])) if g else '   none  ':>9s} {ll:6.2f} | "+" ".join(f"{c:>10s}" for c in cells))
say("[price] net-sigma-channel detections (catalogue / non-catalogue) per variant: "+"; ".join(f"{n}: {v[0]}/{v[1]}" for n,v in tot.items()))
say(f"[price] done {time.time()-t0:.0f}s"); open(REP,"w").write("\n".join(L)+"\n")

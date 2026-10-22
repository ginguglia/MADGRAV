"""Fill the LR statistic for net-sigma FAR-setting families that the S3 bank does not carry (loglr < 3.0), 2026-10-14.
Exact pipeline path: C.build_setup() under the run's EXP env (frozen LR model, SM_GX=1, rewhitened coherence), M.series_for ->
M.coh_vec / M.gx_vec -> M.feats -> M.loglr with the cross-fit model mdl[1-fold(ai)] (driver_blindscan.py:437). Families = famN reps
(net >= NETMIN) surviving the arm rule max:-2.05 whose coincident-crop CNN gate is open (max(hm,lm)>0.5 & lm>0.0382).
Output: chain/loglr_fill_<run>.npz (pairs[n,4], loglr[n], net[n]); consumed by price_netsig_guard.py when present.
Run: RUN=o3b python -I loglr_fill.py   (env EXP taken from launchers/logs_m2/<run>_off5_crop_m2_gx_jids.env)"""
import os, sys, glob, json, time, numpy as np
run=os.environ["RUN"]; NETMIN=float(os.environ.get("NETMIN","3.0")); ARM=-2.05; THR=0.5; LMF=0.0382
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

ENVL=open(f"{MADGRAV_ROOT}/launchers/logs_m2/{run}_off5_crop_m2_gx_jids.env").read().split("EXP=ALL,")[-1].split("\n")[0]
for kv in ENVL.split(","):
    if "=" in kv: k,v=kv.split("=",1); os.environ.setdefault(k,v)
os.environ["SM_DEV"]="cpu"; os.environ["BLIND_DEV"]="cpu"; os.environ["SM_ALLOW_CPU"]="1"
ROOT=os.environ["MADGRAV_ROOT"]; sys.path.insert(0,f"{ROOT}/search_mode"); os.chdir(ROOT)
import _pb_shard_common as C, driver_search_multi as M
M_=MADGRAV_SCRATCH + "/madgrav2"; OUT=f"{M_}/search_out_{run}_m2_gx_off5_crop"; SH=f"{M_}/_{run}_m2_gx_off5_nf275_far_shards"
t0=time.time(); S=C.build_setup(); segs=S["segs"]; fold=S["fold"]; mdl=S["mdl"]; print(f"[fill] setup {time.time()-t0:.0f}s",flush=True)
famN={}
for f in sorted(glob.glob(f"{SH}/shard_*.npz")):
    z=np.load(f); K=z["fam_keys"]; V=z["fam_vals"]; m=V[:,0]>=NETMIN; K=K[m]; V=V[m]
    for j in range(len(K)):
        k=(int(K[j,0]),int(K[j,1]),int(K[j,2])); net=float(V[j,0])
        if k not in famN or net>famN[k][0]: famN[k]=(net,int(V[j,1]),int(V[j,2]),int(V[j,3]),int(V[j,4]))
def ok(ai,iH,bi,iL):
    A,B=segs[ai],segs[bi]; sH,sL=float(A["sH"][iH]),float(B["sL"][iL]); return (float(A["gH"][iH]) if sH>=sL else float(B["gL"][iL]))>=ARM
fams=[v for v in famN.values() if ok(v[1],v[2],v[3],v[4])]
cache={}
for f in sorted(glob.glob(f"{OUT}/cnn_shard_*.npz")):
    z=np.load(f)
    for (ai,iH,bi,iL),hm,lm in zip(z["pairs"],z["hm"],z["lm"]): cache[(int(ai),int(iH),int(bi),int(iL))]=(float(hm),float(lm))
z=np.load(f"{MADGRAV_SCRATCH}/madgrav2_spec/s3_crop/s3_{run}.npz",allow_pickle=True); kb=z["kind"]=="bank"
have=set(zip(map(str,z["H1_seg"][kb]),map(int,z["iH"][kb]),map(str,z["L1_seg"][kb]),map(int,z["iL"][kb])))
todo=[]
for net,ai,iH,bi,iL in fams:
    hl=cache.get((ai,iH,bi,iL))
    if hl is None or not (max(hl)>THR and hl[1]>LMF): continue
    if (segs[ai]["name"],iH,segs[bi]["name"],iL) in have: continue
    todo.append((net,ai,iH,bi,iL))
print(f"[fill] famN {len(famN)} -> arm {len(fams)}; coinc-open without S3-bank loglr: {len(todo)}  ({time.time()-t0:.0f}s)",flush=True)
nH={};nL={}
for net,ai,iH,bi,iL in todo: nH.setdefault(ai,set()).add(iH); nL.setdefault(bi,set()).add(iL)
SHs={a:M.series_for(segs[a]["name"],"H1",sorted(w)) for a,w in nH.items()}; SLs={b:M.series_for(segs[b]["name"],"L1",sorted(w)) for b,w in nL.items()}
print(f"[fill] series loaded ({time.time()-t0:.0f}s)",flush=True)
A=np.array([SHs[ai][iH] for _,ai,iH,bi,iL in todo]); B=np.array([SLs[bi][iL] for _,ai,iH,bi,iL in todo])
coh=M.coh_vec(A,B); gx=M.gx_vec(A,B) if M.GX.ON else None
ai_=np.array([t[1] for t in todo]); iH_=np.array([t[2] for t in todo]); bi_=np.array([t[3] for t in todo]); iL_=np.array([t[4] for t in todo])
gv=lambda key,s,i: np.array([float(segs[a][key][j]) for a,j in zip(s,i)])
F=M.feats(gv("sH",ai_,iH_),gv("sL",bi_,iL_),coh,gv("cH",ai_,iH_),gv("cL",bi_,iL_),gv("gH",ai_,iH_),gv("gL",bi_,iL_),gx=gx)
ll=np.full(len(todo),np.nan)
for gg in (0,1):
    m=np.array([fold[a]==gg for a in ai_]); mu,sd,be=mdl[1-gg]
    if m.any(): ll[m]=M.loglr(mu,sd,be,np.asarray(F)[m])
np.savez(f"{MADGRAV_ROOT}/m2/popb_cap/occlusion/chain/loglr_fill_{run}.npz",pairs=np.array([t[1:] for t in todo],dtype=np.int64),loglr=ll,net=np.array([t[0] for t in todo]))
print(f"[fill] {run}: {len(ll)} families; loglr>0: {int((ll>0).sum())}, >=2.75: {int((ll>=2.75).sum())}, >=3: {int((ll>=3).sum())}; max {np.nanmax(ll):.2f}; coh max {coh.max():.3f}  done {time.time()-t0:.0f}s",flush=True)

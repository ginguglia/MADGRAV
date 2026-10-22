"""p1 of the louder-leg (max) rule for arm _off5 (and _def) over TRIGGERING injection pairs (sigma_net>=4), cross-fit, PER RUN of the
positives_off5 set (o3a / o4a / o4b / loud). The event check of 2026-09-29 quoted -2.49 from the first 6000 pairs of the sorted glob
(o3a-dominated); pair_rule.py quoted -1.955 on a 5950-pair sample across all files. Output: models/p1_by_run_off5.txt"""
import os, sys, glob, json, hashlib, numpy as np, torch
H=os.path.dirname(os.path.abspath(__file__)); exec(open(f"{H}/arm_model.py").read())
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH; M=f"{SC}/madgrav2/arm/models"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
def fold_of(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(),16)%2
def load_arm(tag):
    models={0:[],1:[]}
    for js in sorted(glob.glob(f"{M}/arm{tag}_f*_s*.json")):
        S=json.load(open(js)); m=Arm(); m.load_state_dict(torch.load(js.replace(".json",".pt"),map_location="cpu")); m.eval(); models[S["fold"]].append(m)
    return models
def logit(models,fold,tiles):
    x=torch.from_numpy(np.asarray(tiles,np.float32)[:,None]); out=[]
    with torch.no_grad():
        for m in models[1-fold]: p=torch.cat([m.psig(x[i:i+256]) for i in range(0,len(x),256)]).numpy().astype(np.float64); out.append(np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1)))
    return np.mean(out,0)
def main():
    ARMS={t:load_arm(t) for t in ("_def","_off5")}; rng=np.random.default_rng(3); NPER=int(os.environ.get("NPER","3000")); out=[]
    for run in os.environ.get("RUNS","o3a,o4a,o4b,loud").split(","):
        pat = "pos_o4b_off5loud_shard*.npz" if run=="loud" else f"pos_{run}_off5_shard*.npz"
        P=sorted(f for f in glob.glob(f"{SC}/madgrav2/arm/positives_off5/{pat}") if "smoke" not in f)
        if not P: print(f"{run}: no positives files ({pat}), skipped",flush=True); continue
        vals={t:[] for t in ARMS}; n=0
        for f in P:
            z=np.load(f); keys=list(z.files)
            if n==0 and run=="o3a": print("keys:",keys,flush=True)
            sH,sL=z["sigma_H"],z["sigma_L"]; net=z["sigma_net"]; k=np.flatnonzero(net>=4)
            if not len(k): continue
            k=rng.permutation(k)[:max(1,NPER//len(P))]; gH=z["gps_H"]
            fold=np.array([fold_of(int(g//86400)) for g in gH[k]])
            for t in vals:
                v=np.empty(len(k))
                for fo in (0,1):
                    j=np.flatnonzero(fold==fo)
                    if not len(j): continue
                    lH=logit(ARMS[t],fo,z["qt_H"][k[j]]); lL=logit(ARMS[t],fo,z["qt_L"][k[j]]); v[j]=np.where(sH[k[j]]>=sL[k[j]],lH,lL)
                vals[t].append(v)
            n+=len(k)
        line=f"{run:5s} n={n:5d} p1 louder-leg: "+"  ".join(f"{t} {np.percentile(np.concatenate(vals[t]),1):+.3f} (p0.5 {np.percentile(np.concatenate(vals[t]),0.5):+.3f}, p2 {np.percentile(np.concatenate(vals[t]),2):+.3f})" for t in vals)
        print(line,flush=True); out.append(line)
    open(f"{M}/p1_by_run_off5.txt","a").write("\n".join(out)+"\n")
if __name__=="__main__": main()

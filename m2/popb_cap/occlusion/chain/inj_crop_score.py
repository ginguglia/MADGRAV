"""Score net-sigma-channel INJECTION candidates with the June CNN under a given crop placement (SM_CROP=gradcam|coinc), 2026-10-14.
Input: inj_<run>_m2_off5/<run>/*_inj.npz (whitened 1 s series serH/serL, sigma, centroid2, arm g2, coh, gx, net_snr, mtot).
Candidates = net >= NETSIG floor (3.0) & centroid gate & arm gate & pair arm rule max:-2.05 (merge fg rule). CNN via B.cnn_from_wh
(same path as inject.py / cnn_hm_lm). Also computes loglr with the frozen LR model (fold = segment parity in the run's segment list).
Output: chain/injcrop_<run>_<crop>.npz (file, row, net_snr, mtot, net, hm, lm, loglr). Env: RUN, SM_CROP, optional LIMIT (rows per file).
Run: RUN=o4b SM_CROP=coinc python -I inj_crop_score.py"""
import os, sys, glob, time, json, numpy as np
run=os.environ["RUN"]; CROP=os.environ.get("SM_CROP","gradcam"); LIMIT=int(os.environ.get("LIMIT","0")); ARM=-2.05
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

ENVL=open(f"{MADGRAV_ROOT}/launchers/logs_m2/{run}_off5_crop_m2_gx_jids.env").read().split("EXP=ALL,")[-1].split("\n")[0]
for kv in ENVL.split(","):
    if "=" in kv: k,v=kv.split("=",1); os.environ.setdefault(k,v)
os.environ["SM_CROP"]=CROP; os.environ["SM_DEV"]="cpu"; os.environ["BLIND_DEV"]="cpu"; os.environ["SM_ALLOW_CPU"]="1"
ROOT=os.environ["MADGRAV_ROOT"]; sys.path.insert(0,f"{ROOT}/search_mode"); os.chdir(ROOT)
import driver_search_multi as M, driver_blindscan as B
FLOOR=float(os.environ.get("SM_NETSIG_FLOOR_INJ","3.0"))
z=np.load(os.environ["SM_LR_MODEL"]); mdl={g:(z[f"mu{g}"],z[f"sd{g}"],z[f"be{g}"]) for g in (0,1)}
names=[nm for a,b,d,nm in M.SEG["segments"] if M.have(a,nm)]; fold={nm:i%2 for i,nm in enumerate(names)}
files=sorted(glob.glob(f"{MADGRAV_SCRATCH}/madgrav2/inj_{run}_m2_off5/{run}/*_inj.npz")); t0=time.time()
R={k:[] for k in ("file","row","net_snr","mtot","net","hm","lm","loglr","gxv")}
for f in files:
    d=np.load(f); nm=os.path.basename(f).replace("_inj.npz",""); g=fold.get(nm)
    if g is None: print(f"[injcrop] {nm} not in segment list -> skip",flush=True); continue
    sH,sL,cH,cL,gH,gL=d["sigH"],d["sigL"],d["cenH2"],d["cenL2"],d["gH2"],d["gL2"]
    pair=np.where(sH>=sL,gH,gL)>=ARM
    sel=np.where((d["net"]>=FLOOR)&M.cen_gate(cH,cL)&M.arm_gate(gH,gL,sH,sL)&pair)[0]
    if LIMIT: sel=sel[:LIMIT]
    F=M.feats(sH[sel],sL[sel],d["coh"][sel],cH[sel],cL[sel],gH[sel],gL[sel],gx=(d["gx"][sel] if M.GX.ON else None)); mu,sd,be=mdl[1-g]; ll=M.loglr(mu,sd,be,F)
    for n,i in enumerate(sel):
        hm,lm=B.cnn_from_wh(d["serH"][i][None,:].astype(np.float64),d["serL"][i][None,:].astype(np.float64))
        R["file"].append(nm); R["row"].append(int(i)); R["net_snr"].append(float(d["net_snr"][i])); R["mtot"].append(float(d["mtot"][i])); R["net"].append(float(d["net"][i]))
        R["hm"].append(hm); R["lm"].append(lm); R["loglr"].append(float(ll[n])); R["gxv"].append(float(d["gx"][i]))
    print(f"[injcrop] {nm}: {len(sel)} candidates of {len(sH)} ({time.time()-t0:.0f}s, {len(R['hm'])} total)",flush=True)
out=f"{MADGRAV_ROOT}/m2/popb_cap/occlusion/chain/injcrop_{run}_{CROP}{'_lim'+str(LIMIT) if LIMIT else ''}.npz"
np.savez(out,**{k:np.array(v) for k,v in R.items()}); print(f"[injcrop] wrote {out} rows {len(R['hm'])} in {time.time()-t0:.0f}s",flush=True)

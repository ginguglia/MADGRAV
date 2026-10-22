#!/usr/bin/env python
"""Arm _off5 vs _def on the stream windows of EVERY H1+L1-coincident catalogue event (four runs), frozen whitening, CPU.
For each event: every stream window pair (same index, 1 s stride) within +-1.5 s with net >= 3; tiles rebuilt from strain under the run's
frozen prep (driver_streams.build_qt); both arms scored cross-fit (fold of the UTC day never seen); the louder-leg rule 'max' evaluated at
_def's production threshold (-2.05) and at _off5's own p1 threshold over triggering injection pairs (sigma_net >= 4, positives_off5, cross-fit).
Tripwire: _def logit on the rebuilt tile vs the stream meta g2 at the same window. Output: models/event_check_off5.txt + .json"""
import os, sys, glob, json, hashlib, numpy as np, torch
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

ROOT=MADGRAV_ROOT; os.chdir(ROOT)
for p in ("search_mode","improved","spectrogram_cascade","dev_v2/glitch_arm"): sys.path.insert(0,os.path.join(ROOT,p))
import det_flags; det_flags.apply(); import driver_blindscan as B, driver_streams as DS
H=os.path.dirname(os.path.abspath(__file__)); exec(open(f"{H}/arm_model.py").read())
if __name__=="__main__":  # guard: forkserver QT workers re-import __main__ (job 1774505 timed out re-running the script per worker)
    NEW=os.environ.get("ARM_NEW_TAG","_off5")   # 2026-09-30: candidate arm tag (default _off5 = original behaviour); outputs event_check<NEW>.txt/.json
    MG=MADGRAV_ROOT; SC=MADGRAV_SCRATCH; M=f"{SC}/madgrav2/arm/models"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
    sys.path.insert(0,f"{MG}/m2/cae"); from common import gps_map, find_window_file, cut, WIN
    def fold_of(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(),16)%2
    def load_arm(tag):
        models={0:[],1:[]}
        for js in sorted(glob.glob(f"{M}/arm{tag}_f*_s*.json")):
            S=json.load(open(js)); m=Arm(); m.load_state_dict(torch.load(js.replace(".json",".pt"),map_location="cpu")); m.eval(); models[S["fold"]].append(m)
        return models
    ARMS={t:load_arm(t) for t in ("_def",NEW)}; print("models:",{t:{f:len(v) for f,v in a.items()} for t,a in ARMS.items()},flush=True)
    def logit(models,fold,tiles):
        x=torch.from_numpy(np.asarray(tiles,np.float32)[:,None]); out=[]
        with torch.no_grad():
            for m in models[1-fold]: p=torch.cat([m.psig(x[i:i+256]) for i in range(0,len(x),256)]).numpy().astype(np.float64); out.append(np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1)))
        return np.mean(out,0)
    # --- p1 threshold of _off5 (and _def for reference) over triggering injection pairs, cross-fit ---
    rng=np.random.default_rng(3); P=sorted(f for f in glob.glob(f"{SC}/madgrav2/arm/positives_off5/pos_*_shard*.npz") if "smoke" not in f)
    vals={"_def":[],NEW:[]}; n=0
    for f in P:
        z=np.load(f,allow_pickle=True); k=np.flatnonzero(np.asarray(z["pass_net"])==1)
        if not len(k): continue
        k=rng.choice(k,min(len(k),150),replace=False); gH=np.asarray(z["gps_H"])[k]; sH=np.asarray(z["sigma_H"])[k]; sL=np.asarray(z["sigma_L"])[k]
        fold=np.array([fold_of(int(g//86400)) for g in gH])
        for t in vals:
            lH=np.zeros(len(k)); lL=np.zeros(len(k))
            for fo in (0,1):
                m=fold==fo
                if m.any(): lH[m]=logit(ARMS[t],fo,z["qt_H"][k[m]]); lL[m]=logit(ARMS[t],fo,z["qt_L"][k[m]])
            vals[t].extend(np.where(sH>=sL,lH,lL).tolist())
        n+=len(k)
        if n>=6000: break
    THR={t:float(np.percentile(v,1)) for t,v in vals.items()}; print(f"p1 thresholds over {n} triggering off5 injection pairs (louder-leg rule): {THR}  (production _def rule: -2.05)",flush=True)
    # --- events ---
    Hs=json.load(open(f"{MG}/figures/catalog_o3o4/hl_snr_o3o4.json")); G=gps_map()
    STREAMS={"o3a":f"{SC}/madgrav2/streams_o3a_full","o3b":f"{SC}/madgrav2/streams_o3b_full","o4a":f"{SC}/madgrav2/streams_o4a_full","o4b":f"{SC}/madgrav2/streams_o4b_full_cat2"}
    ALIAS={"GW190412":"GW190412_053044","GW190521":"GW190521_030229"}
    det72=set()
    for r in STREAMS: det72|={ALIAS.get(x["matches_known"],x["matches_known"]) for x in json.load(open(f"{SC}/madgrav2/search_out_{r}_m2_gx/detections_s3rank.json")) if x["matches_known"] and x["best_far"]<1 and x["best_ul90"]<1}
    pipes={r:B.MassiveEventPipeline(calib_path=f"{ROOT}/spectrogram_cascade/massive_calibration_BA.json",prep=f"{MG}/data/{r}_search_prep",device="cpu") for r in STREAMS}
    rows=[]; trip=[]
    for r,ST in STREAMS.items():
        segs={os.path.basename(f).rsplit("_",2)[0]:np.load(f)["gps"] for f in glob.glob(f"{ST}/*_H1_meta.npz")}
        ev=[(n,v) for n,v in Hs.items() if v["run"].lower()==r and v.get("coincident")]
        for n,v in ev:
            g=G.get(n)
            if g is None: continue
            s=next((k for k,gg in segs.items() if gg[0]-1<=g<=gg[-1]+1),None)
            if s is None: continue
            m={d:np.load(f"{ST}/{s}_{d}_meta.npz") for d in ("H1","L1")}; k=np.flatnonzero(np.abs(m["H1"]["gps"]-g)<1.5)
            sH=m["H1"]["sigma"][k]; sL=m["L1"]["sigma"][k]; net=(sH+sL)/np.sqrt(2); kk=k[net>=3]
            if not len(kk): rows.append(dict(name=n,run=r,status="no-trigger")); continue
            files={d:find_window_file(r,d,g,WIN/2+3.0) for d in ("H1","L1")}
            if None in files.values(): rows.append(dict(name=n,run=r,status="no-strain")); continue
            fold=fold_of(int(g//86400)); L={}
            for d in ("H1","L1"):
                W=np.stack([cut(files[d],float(m[d]["gps"][i])) for i in kk]).astype(np.float32); qt=DS.build_qt(pipes[r],pipes[r]._whiten(W,d))
                for t in ARMS: L[(t,d)]=logit(ARMS[t],fold,qt)
                trip.extend((L[("_def",d)]-np.asarray(m[d]["g2"][kk],float)).tolist())
            rec=dict(name=n,run=r,status="ok",det72=n in det72,snr=v["snr_hl"],mtot=v["mtot"],windows=[])
            for j,i in enumerate(kk):
                louder="H1" if m["H1"]["sigma"][i]>=m["L1"]["sigma"][i] else "L1"
                rec["windows"].append(dict(dt=float(m["H1"]["gps"][i]-g),net=float((m["H1"]["sigma"][i]+m["L1"]["sigma"][i])/np.sqrt(2)),g2_meta=float(m[louder]["g2"][i]),def_=float(L[("_def",louder)][j]),off5=float(L[(NEW,louder)][j])))
            rec["pass_def"]=any(w["def_"]>=-2.05 for w in rec["windows"]); rec["pass_off5"]=any(w["off5"]>=THR[NEW] for w in rec["windows"]); rec["pass_off5_at_205"]=any(w["off5"]>=-2.05 for w in rec["windows"])
            rows.append(rec)
        print(f"[{r}] done {sum(1 for x in rows if x['run']==r)} events",flush=True)
    trip=np.asarray(trip); out=[f"TRIPWIRE rebuilt-tile _def logit vs stream g2 (same window): n={len(trip)} median|d| {np.median(np.abs(trip)):.3f} p95 {np.percentile(np.abs(trip),95):.3f} max {np.abs(trip).max():.3f}",
         f"thresholds: _def -2.05 (production) | _off5 p1 {THR[NEW]:.3f} (_def p1 on the same off5 pairs {THR['_def']:.3f})",""]
    ok=[x for x in rows if x["status"]=="ok"]
    out.append(f"{'run':4s} {'trig':>4s} {'pass _def':>9s} {'pass _off5':>10s} {'off5@-2.05':>10s} | det72: {'n':>3s} {'pass_def':>8s} {'pass_off5':>9s}")
    for r in STREAMS:
        R=[x for x in ok if x["run"]==r]; D=[x for x in R if x["det72"]]
        out.append(f"{r:4s} {len(R):4d} {sum(x['pass_def'] for x in R):9d} {sum(x['pass_off5'] for x in R):10d} {sum(x['pass_off5_at_205'] for x in R):10d} | {len(D):8d} {sum(x['pass_def'] for x in D):8d} {sum(x['pass_off5'] for x in D):9d}")
    out.append(f"ALL  {len(ok):4d} {sum(x['pass_def'] for x in ok):9d} {sum(x['pass_off5'] for x in ok):10d} {sum(x['pass_off5_at_205'] for x in ok):10d} | {sum(x['det72'] for x in ok):8d} {sum(x['pass_def'] and x['det72'] for x in ok):8d} {sum(x['pass_off5'] and x['det72'] for x in ok):9d}")
    out.append("\nRECOVERED by _off5 (vetoed by _def, pass _off5 at its p1): "+", ".join(f"{x['name']}({x['run']}, snr {x['snr']:.1f}, best off5 {max(w['off5'] for w in x['windows']):+.2f})" for x in ok if not x["pass_def"] and x["pass_off5"]))
    out.append("LOST by _off5 (pass _def, vetoed by _off5): "+", ".join(f"{x['name']}({x['run']}, snr {x['snr']:.1f}, det72={x['det72']}, best off5 {max(w['off5'] for w in x['windows']):+.2f})" for x in ok if x["pass_def"] and not x["pass_off5"]))
    out.append("det72 events vetoed by _off5: "+", ".join(f"{x['name']}({x['run']})" for x in ok if x["det72"] and not x["pass_off5"]))
    txt="\n".join(out); print(txt); open(f"{M}/event_check{NEW}.txt","w").write(txt+"\n"); json.dump(dict(thr=THR,rows=rows),open(f"{M}/event_check{NEW}.json","w"))
    if __name__=="__main__": pass

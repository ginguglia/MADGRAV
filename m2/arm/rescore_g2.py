#!/usr/bin/env python
"""Rescore the arm-v2 logit g2 of an existing streams directory with another arm tag (MADGRAV 2.1 arm _off5, 2026-09-29).
Only windows with sigma > SMIN (default 0) are rescored: every consumer weights g2 by clip(s/3,0,1) (LR feature gate(g2,s)) or reads the
louder leg of a net>=3 pair (s>=2.12), so g2 of the other windows never enters anything; they keep the source value. CAT2-vetoed windows
(sigma=-1e9) are excluded the same way. Tiles are rebuilt exactly as driver_streams_bg.run: strain cut at idx*step, pipe._whiten per batch,
DS.build_qt -- the per-event check of 2026-09-29 reproduced the production g2 to 0.002 through this path. TRIPWIRE per (segment, det): the
SOURCE arm rescored on the first TRIP selected windows must match the stored g2 to TRIP_TOL, else the shard aborts.
Output: DST/<name>_<det>_meta.npz = source meta with g2 replaced, plus DST/<name>_<det>_g2<tag>.json.
Env: MADGRAV_ROOT SRC DST SHARD NSHARD SM_ARM_V2_TAG (new) SRC_ARM_TAG (_def) SM_ARM_V2_DIR SM_STRAIN SM_BGJSON SM_PREP SM_DEV=cpu
     SMIN (0) SMOKE (0 = off; N = first N selected windows of the first (segment, det) only) TRIP (256) TRIP_TOL (0.02)"""
import os, sys, json, time, numpy as np
ROOT=os.environ["MADGRAV_ROOT"]; os.chdir(ROOT)
for p in ("search_mode","improved","spectrogram_cascade"): sys.path.insert(0,os.path.join(ROOT,p))
SRC=os.environ["SRC"]; DST=os.environ["DST"]; os.environ.setdefault("SM_STREAMS",DST)
import torch, driver_streams as DS, driver_streams_bg as DSB, arm_v2
from massive_pipeline import MassiveEventPipeline

def main():
    SHARD=int(os.environ.get("SHARD","0")); NSHARD=int(os.environ.get("NSHARD","1")); SMIN=float(os.environ.get("SMIN","0"))
    SMOKE=int(os.environ.get("SMOKE","0")); TRIP=int(os.environ.get("TRIP","256")); TOL=float(os.environ.get("TRIP_TOL","0.02")); BATCH=int(os.environ.get("BATCH","1024"))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","32"))); os.makedirs(DST,exist_ok=True)
    pipe=MassiveEventPipeline(calib_path=os.environ.get("SM_CALIB",f"{DS.SC}/massive_calibration_BA.json"),prep=DS.O4A,device="cpu")
    NEWTAG=arm_v2.TAG; NEW=arm_v2.ArmV2("cpu")
    SRCTAG=os.environ.get("SRC_ARM_TAG","_def"); arm_v2.TAG=SRCTAG; OLD=arm_v2.ArmV2("cpu"); arm_v2.TAG=NEWTAG
    step=int(DSB.STRIDE*DSB.FS); WN=DSB.WN
    print(f"[{SHARD}/{NSHARD}] new arm {NEWTAG} {NEW.n} | source arm {SRCTAG} {OLD.n} | prep {DS.O4A} | strain {DSB.STR} | SRC {SRC} -> DST {DST} | SMIN {SMIN} SMOKE {SMOKE}",flush=True)
    for idx,(a,b,d,nm) in enumerate(DSB.SEG["segments"]):
        if idx%NSHARD!=SHARD: continue
        name=DSB.segname(a,nm)
        for det in ("H1","L1"):
            out=f"{DST}/{name}_{det}_meta.npz"
            if os.path.exists(out): print(f"[{SHARD}] skip {name} {det}",flush=True); continue
            src=f"{SRC}/{name}_{det}_meta.npz"
            if not os.path.exists(src): print(f"[{SHARD}] MISSING source meta {name} {det}",flush=True); continue
            z=np.load(src); M={k:z[k] for k in z.files}; sig=np.asarray(M["sigma"],float); gps=np.asarray(M["gps"],float); n=int(M["n"])
            sel=np.flatnonzero(sig>SMIN)
            if SMOKE: sel=sel[:SMOKE]
            g2=np.array(M["g2"],float).copy(); t0=time.time(); trip=None; ntrip=0
            if len(sel):
                raw=np.load(f"{DSB.STR}/{name}_{det}.npz")["strain"]; n_chk=(len(raw)-WN)//step+1
                assert n_chk==n,(name,det,n_chk,n)
                for c0 in range(0,len(sel),BATCH):
                    cs=sel[c0:c0+BATCH]; W=np.stack([raw[i*step:i*step+WN] for i in cs]).astype(np.float32)
                    qt=DS.build_qt(pipe,pipe._whiten(W,det)); g=gps[cs]
                    g2[cs]=NEW.g2_from_qt(qt,g)
                    if c0==0:
                        ntrip=min(TRIP,len(cs)); dd=OLD.g2_from_qt(qt[:ntrip],g[:ntrip])-np.asarray(M["g2"],float)[cs[:ntrip]]; trip=float(np.abs(dd).max())
                        if trip>TOL: raise SystemExit(f"[{SHARD}] TRIPWIRE FAIL {name} {det}: max|d|={trip:.4f} > {TOL} on {ntrip} windows")
            M["g2"]=g2; tmp=f"{DST}/.{name}_{det}_meta.{os.getpid()}.tmp.npz"; np.savez(tmp,**M); os.replace(tmp,out)
            json.dump(dict(name=name,det=det,n=n,n_rescored=int(len(sel)),smin=SMIN,tag=NEWTAG,src_tag=SRCTAG,tripwire_max_abs=trip,tripwire_n=ntrip,
                           wall_s=round(time.time()-t0,1),smoke=SMOKE),open(f"{DST}/{name}_{det}_g2{NEWTAG}.json","w"))
            print(f"[{SHARD}] DONE {name} {det}: n={n} rescored={len(sel)} tripwire max|d|={trip} ({ntrip}) in {(time.time()-t0)/60:.1f}m ({len(sel)/max(time.time()-t0,1e-9):.1f}/s)",flush=True)
            if SMOKE: print(f"[{SHARD}] SMOKE done",flush=True); return
    print(f"[{SHARD}] RESCORE_SHARD_DONE",flush=True)
if __name__=="__main__": main()

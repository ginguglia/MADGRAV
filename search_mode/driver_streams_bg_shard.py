"""Sharded GPU stream-build over bg_segments.json. Run N copies with SHARD=0..N-1, NSHARD=N, each pinned to
one GPU via CUDA_VISIBLE_DEVICES (so the only visible device is cuda:0). Processes segments where
idx%NSHARD==SHARD; skips existing _meta.npz. Identical per-segment math to driver_streams_bg.run().

NOTE: ALL execution lives under `if __name__=="__main__"` (via main()). driver_streams builds its QT
worker pool with a *forkserver* context, which re-imports this module in each worker; without the guard
the unguarded top-level code (pipe init + the build loop) re-ran in every worker -> fork-bomb. The guard
makes worker re-imports side-effect-free. (hepgpu2 used a default *fork* pool that masked this.)"""
import os,sys,json,time
import numpy as np
# --- MUSICA port: MADGRAV_ROOT-relative sys.path (was hardcoded hepgpu2 orchestrated-pipeline path) ---
MADGRAV_ROOT=os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
for _p in ("search_mode","improved","spectrogram_cascade"):
    _ap=os.path.join(MADGRAV_ROOT,_p)
    if _ap not in sys.path: sys.path.insert(0,_ap)
import driver_streams as DS
DS.DEV=os.environ.get("SM_DEV","cuda:0")   # hardcoded cuda:0 by default; SM_DEV=cpu for the CPU lane
import driver_streams_bg as DSB
import torch
from massive_pipeline import MassiveEventPipeline

def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"

def main():
    SHARD=int(os.environ.get("SHARD","0")); NSHARD=int(os.environ.get("NSHARD","1"))
    SEG=json.load(open(os.environ.get("SM_BGJSON","search_mode/bg_segments.json"))); OUT=DSB.OUT; STR=DSB.STR
    os.makedirs(OUT,exist_ok=True)
    pipe=MassiveEventPipeline(calib_path=f"{DS.SC}/massive_calibration_BA.json",prep=DS.O4A,device=DS.DEV)
    arms=[DS.GlitchArm().to(DS.DEV) for _ in range(5)]
    for i,arm in enumerate(arms): arm.load_state_dict(torch.load(f"{DS.LRD}/p1v42/arm_deploy_seed{i}.pt",map_location=DS.DEV)); arm.eval()
    print(f"[shard {SHARD}/{NSHARD}] pipe on {pipe.device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}) OUT={OUT} STR={STR}",flush=True)
    for idx,(a,b,d,nm) in enumerate(SEG["segments"]):
        if idx%NSHARD!=SHARD: continue
        name=segname(a,nm)
        for det in ("H1","L1"):
            if os.path.exists(f"{OUT}/{name}_{det}_meta.npz"): print(f"[{SHARD}] skip {name} {det}",flush=True); continue
            if not os.path.exists(f"{STR}/{name}_{det}.npz"): print(f"[{SHARD}] WAIT-MISSING-STRAIN {name} {det}",flush=True); continue
            DSB.run(name,a,det,pipe,arms)
    print(f"[{SHARD}] STREAMS_SHARD_DONE",flush=True)

if __name__=="__main__": main()

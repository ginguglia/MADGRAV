"""Sharded resumable fetch of bg_segments.json strain. Run N copies with SHARD=0..N-1, NSHARD=N.
Each handles segments where idx%NSHARD==SHARD; skips already-fetched files. Same .npz convention as fetch_bg.py."""
import os,json,time
import numpy as np
from gwpy.timeseries import TimeSeries
SHARD=int(os.environ.get("SHARD","0")); NSHARD=int(os.environ.get("NSHARD","1"))
SEG=json.load(open(os.environ.get("SM_BGJSON","search_mode/o3a_bg_segments_56.json"))); OUT=os.environ.get("SM_STRAIN","search_mode/strain_o3a"); os.makedirs(OUT,exist_ok=True)
FS=4096; CHUNK=4096.0
def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"
for idx,(a,b,d,nm) in enumerate(SEG["segments"]):
    if idx%NSHARD!=SHARD: continue
    name=segname(a,nm)
    for det in ("H1","L1"):
        f=f"{OUT}/{name}_{det}.npz"
        if os.path.exists(f): print(f"[{SHARD}] skip {name} {det}",flush=True); continue
        chunks=[]; t=a; ta=time.time()
        while t<b:
            te=min(t+CHUNK,b)
            for at in range(4):
                try: chunks.append(np.asarray(TimeSeries.fetch_open_data(det,t,te,sample_rate=FS,cache=False).value,np.float32)); break
                except Exception as ex:
                    if at==3: print(f"[{SHARD}] FAIL {name} {det} [{t:.0f}]: {str(ex)[:60]}",flush=True); raise
                    time.sleep(15)
            t=te
        np.savez(f,strain=np.concatenate(chunks),gps_start=a,fs=FS,gps_end=b)
        print(f"[{SHARD}] OK {name} {det}: {d/3600:.1f}h in {(time.time()-ta)/60:.1f}min",flush=True)
print(f"[{SHARD}] FETCH_SHARD_DONE",flush=True)

"""Fetch strain for the all-vs-all background segments (bg_segments.json); skip already-fetched event locks."""
import os,json,time
import numpy as np
from gwpy.timeseries import TimeSeries
SEG=json.load(open(os.environ.get("SM_BGJSON","search_mode/o3a_bg_segments_56.json")))
OUT=os.environ.get("SM_STRAIN","search_mode/strain_o3a"); os.makedirs(OUT,exist_ok=True)
FS=4096; CHUNK=4096.0
def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"
for a,b,d,nm in SEG["segments"]:
    name=segname(a,nm)
    for det in ("H1","L1"):
        f=f"{OUT}/{name}_{det}.npz"
        if os.path.exists(f): print(f"skip {name} {det}",flush=True); continue
        chunks=[]; t=a; ta=time.time()
        while t<b:
            te=min(t+CHUNK,b)
            for at in range(4):
                try: chunks.append(np.asarray(TimeSeries.fetch_open_data(det,t,te,sample_rate=FS,cache=False).value,np.float32)); break
                except Exception as ex:
                    if at==3: print(f"FAIL {name} {det} [{t:.0f}]: {str(ex)[:60]}",flush=True); raise
                    time.sleep(15)
            t=te
        np.savez(f,strain=np.concatenate(chunks),gps_start=a,fs=FS,gps_end=b)
        print(f"OK {name} {det}: {d/3600:.1f}h in {(time.time()-ta)/60:.1f}min",flush=True)
open(f"{OUT}/BG_FETCH_DONE","w").close(); print("BG_FETCH_DONE",flush=True)

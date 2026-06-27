"""Resilient sharded fetch of bg_segments.json strain. Survives flaky GWOSC (500/503/timeouts).
Run N copies with SHARD=0..N-1, NSHARD=N. Same .npz convention + same skip-existing behaviour as fetch_bg_par.py,
but: (1) never re-downloads a file already on disk, (2) catches ALL exceptions per chunk with long exponential
backoff, (3) on a chunk that ultimately fails it abandons that SEGMENT (writes nothing partial) and moves on,
(4) loops over the whole shard in repeated PASSES until every assigned file exists or MAXPASS is hit.
Idempotent: a complete segment file is only written via atomic rename after ALL its chunks succeed."""
import os,json,time
import numpy as np
from gwpy.timeseries import TimeSeries

SHARD=int(os.environ.get("SHARD","0")); NSHARD=int(os.environ.get("NSHARD","1"))
MAXPASS=int(os.environ.get("MAXPASS","200"))         # outer passes before giving up for the night
CHUNK_TRIES=int(os.environ.get("CHUNK_TRIES","10"))  # per-chunk attempts within a pass
BG_JSON=os.environ.get("BG_JSON",os.environ.get("SM_BGJSON","search_mode/o3a_bg_segments_56.json"))
OUT=os.environ.get("STRAIN_OUT",os.environ.get("SM_STRAIN","search_mode/strain_o3a"))
SEG=json.load(open(BG_JSON)); os.makedirs(OUT,exist_ok=True)
FS=4096; CHUNK=4096.0
def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"

def fetch_chunk(det,t,te):
    """Fetch one chunk with long exponential backoff. Returns array or raises after CHUNK_TRIES."""
    delay=15
    for at in range(CHUNK_TRIES):
        try:
            return np.asarray(TimeSeries.fetch_open_data(det,t,te,sample_rate=FS,cache=False).value,np.float32)
        except Exception as ex:
            if at==CHUNK_TRIES-1:
                raise
            print(f"[{SHARD}] retry {det} [{t:.0f}] att{at+1}/{CHUNK_TRIES} in {delay}s: {str(ex)[:50]}",flush=True)
            time.sleep(delay); delay=min(delay*2,300)   # 15,30,60,120,240,300,300...

def fetch_segment(name,a,b,det):
    f=f"{OUT}/{name}_{det}.npz"
    if os.path.exists(f): return "skip"
    tmp=f"{OUT}/.{name}_{det}.tmp.npz"                  # ends in .npz so np.savez writes EXACTLY this name
    chunks=[]; t=a; ta=time.time()
    while t<b:
        te=min(t+CHUNK,b)
        try:
            chunks.append(fetch_chunk(det,t,te))
        except Exception as ex:
            print(f"[{SHARD}] DEFER {name} {det} [{t:.0f}]: {str(ex)[:50]} -- abandon segment this pass",flush=True)
            return "defer"
        t=te
    np.savez(tmp,strain=np.concatenate(chunks),gps_start=a,fs=FS,gps_end=b)
    os.replace(tmp,f)                                   # atomic: file appears only when complete
    print(f"[{SHARD}] OK {name} {det}: {(b-a)/3600:.1f}h in {(time.time()-ta)/60:.1f}min",flush=True)
    return "ok"

mine=[(segname(a,nm),a,b) for i,(a,b,d,nm) in enumerate(SEG["segments"]) if i%NSHARD==SHARD]
for p in range(MAXPASS):
    pending=[(n,a,b,det) for (n,a,b) in mine for det in ("H1","L1")
             if not os.path.exists(f"{OUT}/{n}_{det}.npz")]
    if not pending:
        print(f"[{SHARD}] ALL DONE after pass {p}",flush=True); break
    print(f"[{SHARD}] pass {p}: {len(pending)} files pending",flush=True)
    progressed=False
    for n,a,b,det in pending:
        try:
            r=fetch_segment(n,a,b,det)
        except Exception as ex:                          # never let one segment kill the shard
            print(f"[{SHARD}] ERR {n} {det}: {str(ex)[:80]} -- skip this pass",flush=True); r="err"
        if r=="ok": progressed=True
    if not progressed:
        wait=120
        print(f"[{SHARD}] pass {p} made no progress (GWOSC down?) -- sleeping {wait}s",flush=True)
        time.sleep(wait)
print(f"[{SHARD}] FETCH_RESILIENT_EXIT",flush=True)

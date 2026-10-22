"""Split the 725-seg full-O3a coincident config into ~56-seg chunks (SM_BGJSON format) for the
chunked foreground scan, and write a full segments-event map (segname -> coincident_lock) so the
driver never KeyErrors on a gps-named segment. Outputs to $SCRATCH/o3a_chunks/."""
import json, os
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH
full=json.load(open(f"{SC}/o3a_full_coincident.json"))
segs=full["segments"]
OUT=f"{SC}/o3a_chunks"; os.makedirs(OUT,exist_ok=True)
def segname(a,nm): return nm if nm else f"seg_{int(round(a))}"
seg_ev={segname(a,nm):{"coincident_lock":[a,b]} for a,b,d,nm in segs}
json.dump(seg_ev, open(f"{OUT}/o3a_segments_event_full.json","w"))
CHUNK=56; n=0
for k in range(0,len(segs),CHUNK):
    sub=segs[k:k+CHUNK]
    json.dump({"run":f"O3a-chunk{n}","n_segments":len(sub),"segments":sub}, open(f"{OUT}/chunk_{n}.json","w"))
    n+=1
print(f"wrote {n} chunks of <= {CHUNK} segs ({len(segs)} total) -> {OUT}")

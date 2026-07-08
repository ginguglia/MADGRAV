"""Build o4b_full_coincident.json: H1_DATA & L1_DATA coincident segments over O4b, same
convention as O3a/O3b (flags 'H1_DATA & L1_DATA', min_seg_s=16, names o4b_<gps>)."""
import os, json, time
from gwpy.segments import DataQualityFlag
T0, T1 = 1396191618, 1422115218     # O4b
MIN_SEG = 16
OUT = os.environ.get("SM_BGJSON", os.path.join(os.environ.get("SCRATCH", "."), "o4b_full_coincident.json"))
def fr(flag, tries=8):
    delay = 15
    for k in range(tries):
        try: return DataQualityFlag.fetch_open_data(flag, T0, T1)
        except Exception as e:
            print(f"  retry {flag} ({k+1}/{tries}): {str(e)[:60]}", flush=True)
            if k == tries-1: raise
            time.sleep(delay); delay = min(delay*2, 300)
print(f"[o4b] querying H1_DATA & L1_DATA over GPS [{T0},{T1}] ({(T1-T0)/86400:.1f} days span)...", flush=True)
t=time.time()
h1 = fr("H1_DATA").active; print(f"  H1_DATA: {len(h1)} segs, {sum(float(s[1]-s[0]) for s in h1)/86400:.2f} d", flush=True)
l1 = fr("L1_DATA").active; print(f"  L1_DATA: {len(l1)} segs, {sum(float(s[1]-s[0]) for s in l1)/86400:.2f} d", flush=True)
coinc = h1 & l1
segs = [[float(s[0]), float(s[1]), float(s[1]-s[0]), f"o4b_{int(s[0])}"] for s in coinc if float(s[1]-s[0]) >= MIN_SEG]
tot = sum(x[2] for x in segs)
out = dict(run="O4b", flags="H1_DATA & L1_DATA (coincident); CAT2 applied later as veto mask",
           sample_rate=4096, min_seg_s=MIN_SEG, n_segments=len(segs),
           total_coincident_s=tot, total_days=tot/86400.0, segments=segs)
json.dump(out, open(OUT, "w"))
print(f"\n[o4b] WROTE {OUT}: {len(segs)} coincident segments = {tot/86400:.2f} days (query {(time.time()-t)/60:.1f} min)", flush=True)

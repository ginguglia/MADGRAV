"""Build o3b_full_coincident.json: H1_DATA & L1_DATA coincident segments over O3b, same convention as O3a
(flags 'H1_DATA & L1_DATA', min_seg_s=16, names o3b_<gps>). CAT2 applied later as a veto mask (like O3a)."""
import os, json, time
from gwpy.segments import DataQualityFlag
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

T0, T1 = 1256655618, 1269363618     # O3b: 2019-11-01 15:00 UTC -> 2020-03-27 17:00 UTC
MIN_SEG = 16                        # match O3a min_seg_s
OUT = MADGRAV_SCRATCH + "/o3b_full_coincident.json"
def fr(flag, tries=8):
    delay = 15
    for k in range(tries):
        try: return DataQualityFlag.fetch_open_data(flag, T0, T1)
        except Exception as e:
            print(f"  retry {flag} ({k+1}/{tries}): {str(e)[:60]}", flush=True)
            if k == tries-1: raise
            time.sleep(delay); delay = min(delay*2, 300)
print(f"[o3b] querying H1_DATA & L1_DATA over GPS [{T0},{T1}] ({(T1-T0)/86400:.1f} days span)...", flush=True)
t=time.time()
h1 = fr("H1_DATA").active; print(f"  H1_DATA: {len(h1)} segs, {sum(float(s[1]-s[0]) for s in h1)/86400:.2f} d", flush=True)
l1 = fr("L1_DATA").active; print(f"  L1_DATA: {len(l1)} segs, {sum(float(s[1]-s[0]) for s in l1)/86400:.2f} d", flush=True)
coinc = h1 & l1
segs = [[float(s[0]), float(s[1]), float(s[1]-s[0]), f"o3b_{int(s[0])}"] for s in coinc if float(s[1]-s[0]) >= MIN_SEG]
tot = sum(x[2] for x in segs)
out = dict(run="O3b", flags="H1_DATA & L1_DATA (coincident); CAT2 applied later as veto mask",
           sample_rate=4096, min_seg_s=MIN_SEG, n_segments=len(segs),
           total_coincident_s=tot, total_days=tot/86400.0, segments=segs)
json.dump(out, open(OUT, "w"))
print(f"\n[o3b] WROTE {OUT}: {len(segs)} coincident segments = {tot/86400:.2f} days (query {(time.time()-t)/60:.1f} min)", flush=True)

"""Fetch continuous H1+L1 strain over each event's contiguous lock (chunked, resumable)."""
import os, json, time
import numpy as np
from gwpy.timeseries import TimeSeries

SEG = json.load(open("search_mode/segments.json"))
OUT = "search_mode/strain"; os.makedirs(OUT, exist_ok=True)
FS = 4096; CHUNK = 4096.0  # seconds per fetch chunk

for name, d in SEG.items():
    lock = d["coincident_lock"]
    t0, t1 = lock[0], lock[1]
    for det in ("H1", "L1"):
        f = f"{OUT}/{name}_{det}.npz"
        if os.path.exists(f):
            print(f"skip {name} {det} (exists)", flush=True); continue
        chunks = []
        t = t0; ta = time.time()
        while t < t1:
            te = min(t + CHUNK, t1)
            for attempt in range(4):
                try:
                    ts = TimeSeries.fetch_open_data(det, t, te, sample_rate=FS, cache=False)
                    chunks.append(np.asarray(ts.value, np.float64)); break
                except Exception as ex:
                    if attempt == 3:
                        print(f"FAIL {name} {det} [{t:.0f},{te:.0f}]: {str(ex)[:70]}", flush=True)
                        raise
                    time.sleep(15)
            t = te
        strain = np.concatenate(chunks)
        np.savez(f, strain=strain, gps_start=t0, fs=FS, gps_end=t1)
        print(f"OK {name} {det}: {len(strain)} samp ({len(strain)/FS/3600:.2f}h) in {(time.time()-ta)/60:.1f} min -> {f}", flush=True)
print("FETCH_DONE", flush=True)
open(f"{OUT}/FETCH_DONE","w").close()

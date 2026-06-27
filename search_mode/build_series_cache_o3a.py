"""One-time build of the stride-1.0 whitened coh-series memmap cache for all O3a segments (free-win #2).
Series-only -> NO QT pool forked -> no 35GB/worker OOM. After this, series_for() mmap-slices instead of
re-whitening. cuda:1 (for _whiten). Reusable across ALL O3a events/runs in this noise epoch.
"""
import os,time
ROOT=os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.dirname(__file__)+"/.."); os.chdir(ROOT); SM="search_mode"
os.environ.update(dict(SM_PREP=f"{ROOT}/data/o3a_search_prep", SM_STRAIN=f"{SM}/strain_o3a",
    SM_STREAMS=f"{SM}/streams_o3a", SM_INJ=f"{SM}/inj_out_o3a", SM_OUT=f"{SM}/search_out_o3a_f45"))
os.environ.setdefault("SM_DEV","cuda:1")
os.environ.setdefault("SM_BGJSON", f"{SM}/o3a_bg_segments.json")   # override via env (e.g. the 56-seg list)
import sys; sys.path.insert(0,"search_mode"); sys.path.insert(0,"improved")
import driver_search_multi as M
ta=time.time(); log=lambda s: print(f"[{(time.time()-ta)/60:5.1f}m] {s}",flush=True)
avail=[(a,nm) for a,b,d,nm in M.SEG["segments"] if M.have(a,nm)]
log(f"cache dir={M.SERIESCACHE}; {len(avail)} segments x 2 det")
done=0
for a,nm in avail:
    name=M.segname(a,nm)
    for det in ("H1","L1"):
        if not os.path.exists(f"{M.STRAIN}/{name}_{det}.npz"): log(f"  SKIP {name} {det} (no strain)"); continue
        npy,js=M._series_paths(name,det)
        if os.path.exists(npy) and os.path.exists(js): log(f"  exists {name} {det}"); done+=1; continue
        t=time.time(); n=M.build_series_cache(name,det); done+=1
        log(f"  built {name} {det}: n={n} in {(time.time()-t):.0f}s ({done} done)")
log(f"DONE: {done} caches in {M.SERIESCACHE}")

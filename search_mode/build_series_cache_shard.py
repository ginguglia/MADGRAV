"""Sharded CPU build of the whitened coh-series memmap cache (parallel across a Slurm array).
Each task builds flat[(seg,det)] pairs at indices [SM_CACHE_SHARD :: SM_CACHE_NSHARD]; distinct files
per task -> no write collision; 'exists' skip -> resumable. Writes to SM_SERIESCACHE (must match the
search). CPU whitening (SM_DEV=cpu) is float32 -> ~1e-5 from a GPU cache; negligible downstream, but the
frozen/publishable cache should be rebuilt on GPU later.
"""
import os,time
ROOT=os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.dirname(__file__)+"/.."); os.chdir(ROOT); SM="search_mode"
# setdefault (NOT update) so the launcher's SM_STRAIN/SM_STREAMS (e.g. the full-O3a paths) take precedence
for _k,_v in dict(SM_PREP=f"{ROOT}/data/o3a_search_prep", SM_STRAIN=f"{SM}/strain_o3a",
    SM_STREAMS=f"{SM}/streams_o3a", SM_INJ=f"{SM}/inj_out_o3a", SM_OUT=f"{SM}/search_out_o3a_f45").items():
    os.environ.setdefault(_k,_v)
os.environ.setdefault("SM_DEV","cpu"); os.environ.setdefault("SM_ALLOW_CPU","1")
os.environ.setdefault("SM_BGJSON", f"{SM}/o3a_bg_segments_56.json")
import sys; sys.path.insert(0,"search_mode"); sys.path.insert(0,"improved")
import driver_search_multi as M
SHARD=int(os.environ.get("SM_CACHE_SHARD","0")); NSHARD=int(os.environ.get("SM_CACHE_NSHARD","1"))
ta=time.time(); log=lambda s: print(f"[{(time.time()-ta)/60:5.1f}m sh{SHARD}] {s}",flush=True)
# cache only needs STRAIN (it whitens raw strain) -> gate on strain existence, NOT M.have (which requires streams)
def _has_strain(a,nm):
    n=M.segname(a,nm)
    return os.path.exists(f"{M.STRAIN}/{n}_H1.npz") and os.path.exists(f"{M.STRAIN}/{n}_L1.npz")
LENMIN=float(os.environ.get("SM_CACHE_LENMIN","0")); LENMAX=float(os.environ.get("SM_CACHE_LENMAX","1e18"))
avail=[(a,nm) for a,b,d,nm in M.SEG["segments"] if _has_strain(a,nm) and LENMIN < (b-a) <= LENMAX]
if LENMIN>0 or LENMAX<1e17: log(f"length filter: {LENMIN:.0f} < (b-a) <= {LENMAX:.0f}s -> {len(avail)} segs")
flat=[(a,nm,det) for (a,nm) in avail for det in ("H1","L1")]
mine=flat[SHARD::NSHARD]
log(f"cache dir={M.SERIESCACHE}; total {len(flat)} (seg,det); this shard {len(mine)}")
done=0
for a,nm,det in mine:
    name=M.segname(a,nm)
    if not os.path.exists(f"{M.STRAIN}/{name}_{det}.npz"): log(f"  SKIP {name} {det} (no strain)"); continue
    npy,js=M._series_paths(name,det)
    if os.path.exists(npy) and os.path.exists(js): log(f"  exists {name} {det}"); done+=1; continue
    t=time.time(); n=M.build_series_cache(name,det); done+=1
    log(f"  built {name} {det}: n={n} in {(time.time()-t):.0f}s")
log(f"SHARD {SHARD} DONE: {done} caches")

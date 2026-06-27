#!/usr/bin/env python3
"""MADGRAV host-memory budget — ONE knob SM_HOST_MEM_GB bounds peak host RAM on ANY machine.

Designed by the host-safety panel (wf_9910fdb7) after the 2026-06-26 force-reboot, which was a GLOBAL
page-cache / reclaim livelock (NOT a cgroup-anon overrun) driven by the old SM_BLOCK_SURV=40,000,000 default.

PRINCIPLE: the pipeline must respect a user-set host-RAM ceiling by STREAMING finer (more flushes = slower,
NEVER OOM), trading speed for RAM. It does NOT auto-grab MemAvailable (that is exactly how it ate the 503 GB
box) -- MemAvailable is only an UPPER CLAMP and the source of a conservative default.

Usage:
  - As a library at the top of a driver:   import _budget; _budget.apply()     # fills UNSET knobs only
  - As a shell helper:                      eval "$(python search_mode/_budget.py 16)"   # emits export lines

Knobs it derives (only if not already set in the environment -> explicit launcher overrides always win):
  SM_BLOCK_SURV       coherence-matmap flush size      (HOG A -- the reboot term)
  SM_GATHER_CH        per-gather row chunk             (HOG A')
  SM_PRECOMPUTE_MAXGB CNN QT magnitude buffer (merge)  (HOG B)
  SM_QT_WORKERS       forkserver QT pool size          (HOG E -- per-worker RSS x N)
  SM_QT_BATCH         stream full-res mag batch        (HOG F)
  SM_STRAIN_CACHE_SEGS _STR LRU depth                  (HOG C)

Coefficients are CONSERVATIVE (err toward smaller blocks) and individually overridable; calibrate per box
from VmHWM via SM_MATMAP_BPR / SM_WORKER_RSS_MB. RESULT-IDENTICAL: every knob is a streaming/scheduling
control -- bg accrues by union, chunking is per-row, the LRU re-reads identical bytes. FAR is unchanged.
"""
import os, sys

GB = 1024.0**3

# ---- conservative, env-overridable coefficients (bytes unless noted) ----
ROW64        = 4096 * 8                                              # one coh series window, float64
# CALIBRATED 2026-06-26 from the cert run (two real points): block 2,000,000 -> 110 GB peak, block 50,000 -> 16.45 GB
# peak => slope ~48 KB/row, intercept (BASELINE) ~14 GB. The slope is the per-flush matmap + still-alive sH/sL source
# dicts + bg tuples. Re-calibrate per box via SM_MATMAP_BPR / SM_BASELINE_GB from VmHWM (MALLOC_ARENA_MAX=2 lowers BASELINE).
BPR          = float(os.environ.get("SM_MATMAP_BPR", "48000"))      # bytes per surviving row at flush (measured)
BASELINE_GB  = float(os.environ.get("SM_BASELINE_GB", "14"))        # non-block resident: streams+noise model+torch+arena frag
WORKER_RSS   = float(os.environ.get("SM_WORKER_RSS_MB", "500")) * 1e6   # per QT forkserver worker resident
STR_ENTRY    = float(os.environ.get("SM_STR_ENTRY_MB", "650")) * 1e6    # one (seg,det) raw-strain array
MAG_BYTES    = float(os.environ.get("SM_QT_MAG_MB", "8")) * 1e6         # one full-res QT magnitude image (stream stage)
R_BASE       = 3.0 * GB                                              # interp + numpy/torch + CNN/CAE weights
R_BG         = float(os.environ.get("SM_BG_RESERVE_GB", "2")) * GB   # bg[g] loud-survivor lists (grow w/ livetime, not B)


def mem_available_gb():
    """System MemAvailable in GiB (Linux). None if unreadable."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024.0 / GB          # kB -> GiB
    except Exception:
        pass
    return None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def derive(B_gb):
    """Return {knob: str_value} for a host-RAM budget of B_gb GiB. Pure; sets nothing."""
    B = float(B_gb) * GB
    safety   = max(0.10 * B, 2.0 * GB)
    pagecach = max(0.20 * B, 4.0 * GB)                              # HOG D working set (file-backed, not anon)
    base     = BASELINE_GB * GB                                     # non-block resident (anon): streams+noise+torch+frag
    B_eff    = B - base - safety - pagecach                         # room left for the per-flush block matmap
    infeasible = B_eff < 1.0 * GB                                   # too small even for a minimal block on this config
    B_eff    = max(1.0 * GB, B_eff)
    nproc    = os.cpu_count() or 4

    # PRUNE stage (the reboot stage): bound matmap + gather transient + bg reserve.
    ch_bytes = min(0.30 * B_eff, 13e9)
    CH       = int(_clamp(ch_bytes / (2 * ROW64), 5000, 200000))
    block    = int(_clamp((B_eff - 2 * CH * ROW64 - R_BG) / BPR, 50000, 20_000_000))

    # MERGE stage: strain LRU + CNN precompute buffer + workers.
    qt_workers   = int(_clamp((0.20 * B_eff) / WORKER_RSS, 1, min(nproc, 16)))
    strain_segs  = int(_clamp((0.15 * B_eff) / (2 * STR_ENTRY), 1, 8))
    precompute_g = max(2.0, 0.45 * B_eff / GB)

    # STREAM stage: full-res mag batch.
    qt_batch = int(_clamp((0.40 * B_eff) / MAG_BYTES, 64, 2048))

    return {
        "SM_BLOCK_SURV":       str(block),
        "SM_GATHER_CH":        str(CH),
        "SM_PRECOMPUTE_MAXGB": str(round(precompute_g, 1)),
        "SM_QT_WORKERS":       str(qt_workers),
        "SM_QT_BATCH":         str(qt_batch),
        "SM_STRAIN_CACHE_SEGS": str(strain_segs),
        "_SM_BUDGET_B_GB":     str(round(float(B_gb), 1)),
        "_SM_ANON_CEIL_GB":    str(round((B - pagecach) / GB, 1)),
        "_SM_INFEASIBLE":      "1" if infeasible else "0",
    }


def chosen_budget_gb():
    """Resolve B: explicit SM_HOST_MEM_GB, else conservative default min(24, MemAvailable) with a warning."""
    env = os.environ.get("SM_HOST_MEM_GB")
    avail = mem_available_gb()
    if env:
        B = float(env)
        if avail is not None and B > avail:
            sys.stderr.write(f"[budget] WARNING: SM_HOST_MEM_GB={B:g} > MemAvailable={avail:.0f} GiB; "
                             f"clamping to {avail:.0f}.\n")
            B = avail
        return B
    B = min(24.0, avail if avail is not None else 24.0)
    sys.stderr.write(f"[budget] SM_HOST_MEM_GB unset -> conservative default {B:.0f} GiB "
                     f"(MemAvailable={avail if avail is None else round(avail)} GiB). "
                     f"Set SM_HOST_MEM_GB to use more.\n")
    return B


def maybe_setrlimit(anon_ceil_gb):
    """OPT-IN hard backstop (SM_RLIMIT=1): cap ANONYMOUS address space via RLIMIT_DATA so a runaway raises a
    catchable MemoryError instead of taking the host. NOT RLIMIT_AS (that would kill the 137 GB read-only memmap
    reservation). Off by default -- a mis-estimated ceiling could abort a legitimate run; the structural block-size
    bound is the primary protection, this is belt-and-suspenders."""
    if os.environ.get("SM_RLIMIT", "0") != "1":
        return False
    try:
        import resource
        ceil = int(max(4.0, float(anon_ceil_gb)) * GB)
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        newhard = ceil if hard == resource.RLIM_INFINITY else min(ceil, hard)
        resource.setrlimit(resource.RLIMIT_DATA, (ceil, newhard))
        sys.stderr.write(f"[budget] RLIMIT_DATA set to {anon_ceil_gb:g} GiB (anon backstop).\n")
        return True
    except Exception as e:
        sys.stderr.write(f"[budget] RLIMIT_DATA not set ({e}).\n")
        return False


def apply(verbose=True):
    """Fill any UNSET SM_* knob from the derived budget (explicit env always wins), optionally arm RLIMIT_DATA.
    Returns the full derived dict. Safe to call once at driver startup."""
    B = chosen_budget_gb()
    d = derive(B)
    set_here = []
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if os.environ.get(k) in (None, ""):
            os.environ[k] = v
            set_here.append(f"{k}={v}")
    maybe_setrlimit(float(d["_SM_ANON_CEIL_GB"]))
    if d.get("_SM_INFEASIBLE") == "1":
        sys.stderr.write(f"[budget] WARNING: SM_HOST_MEM_GB={B:g} is BELOW the ~{BASELINE_GB:g} GiB non-block baseline "
                         f"for this config -> block forced to its 50k floor; peak may exceed the budget. Raise "
                         f"SM_HOST_MEM_GB (>= ~{BASELINE_GB+8:g}), reduce #segments, or set MALLOC_ARENA_MAX=2.\n")
    if verbose:
        sys.stderr.write(f"[budget] B={B:g} GiB -> derived [{', '.join(set_here) or 'all knobs pre-set'}]\n")
    return d


if __name__ == "__main__":
    B = float(sys.argv[1]) if len(sys.argv) > 1 else chosen_budget_gb()
    d = derive(B)
    # emit shell export lines (only for the real knobs); a launcher can `eval "$(python _budget.py 16)"`.
    for k, v in d.items():
        if not k.startswith("_"):
            print(f"export {k}={v}")
    sys.stderr.write(f"[budget] B={B:g} GiB anon_ceil={d['_SM_ANON_CEIL_GB']} GiB\n")

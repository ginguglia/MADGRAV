"""Identity-B smoke (2026-10-22): run m2/vt96/inject96.py UNCHANGED but with the pre-patch (git HEAD) search_mode/inject.py.
Same sys.path set-up as inject96.py, then m2/vt96p/headinj is put first so `import inject` binds the HEAD copy; inject96.py's own
`import inject` then finds it in sys.modules. Everything else (driver_blindscan, arm_v2, ...) comes from MADGRAV_ROOT as usual."""
import os, sys, runpy
ROOT = os.environ["MADGRAV_ROOT"]
for _p in ("search_mode", "improved", "spectrogram_cascade"):
    _ap = os.path.join(ROOT, _p)
    if _ap not in sys.path: sys.path.insert(0, _ap)
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

HEAD = MADGRAV_ROOT + "/m2/vt96p/headinj"; sys.path.insert(0, HEAD)
import inject
assert inject.__file__.startswith(HEAD), inject.__file__
print(f"[inj96-head] inject module = {inject.__file__}", flush=True)
runpy.run_path(MADGRAV_ROOT + "/m2/vt96/inject96.py", run_name="__main__")

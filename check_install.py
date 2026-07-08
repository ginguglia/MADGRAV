#!/usr/bin/env python
"""MADGRAV install check. Verifies vendored assets exist and the core module closure imports.
Does NOT run the pipeline, fetch data, or require strain. CPU is allowed (set SM_ALLOW_CPU=1).

Run:
  cd <package> && MADGRAV_ROOT=$(pwd) SM_ALLOW_CPU=1 python check_install.py
"""
import os, sys, traceback

ROOT = os.environ.get("MADGRAV_ROOT") or os.path.abspath(os.path.dirname(__file__))
os.environ["MADGRAV_ROOT"] = ROOT
os.environ.setdefault("SM_ALLOW_CPU", "1")
# point the module-level data-env at the vendored JSONs so driver_streams/driver_search_multi import
# without requiring a provisioned run dir (these read a segment list at import time).
os.environ.setdefault("SM_SEGJSON_EV", os.path.join(ROOT, "search_mode", "o3a_segments_event.json"))
os.environ.setdefault("SM_EVENTSJSON", os.path.join(ROOT, "search_mode", "o3a_events.json"))
os.environ.setdefault("SM_BGJSON", os.path.join(ROOT, "search_mode", "o3a_bg_segments_56.json"))
# driver_search_multi reads spectrogram_cascade/massive_calibration_BA.json relative to CWD -> run from ROOT
os.chdir(ROOT)
for _p in ("search_mode", "improved", "spectrogram_cascade"):
    _ap = os.path.join(ROOT, _p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

print(f"MADGRAV_ROOT = {ROOT}\n")

# ---- (1) vendored assets (STEP 2) ----
ASSETS = [
    "search_mode/hm_native_seed0.pt",
    "search_mode/lm_native_seed0.pt",
    "search_mode/o3a_bg_segments_56.json",
    "search_mode/o3a_events.json",
    "search_mode/o3a_segments_event.json",
    "search_mode/veto_mask_o3a_56.json",
    "lr_cascade/p1v42/arm_deploy_seed0.pt",
    "lr_cascade/p1v42/arm_deploy_seed1.pt",
    "lr_cascade/p1v42/arm_deploy_seed2.pt",
    "lr_cascade/p1v42/arm_deploy_seed3.pt",
    "lr_cascade/p1v42/arm_deploy_seed4.pt",
    "spectrogram_cascade/massive_calibration_BA.json",
    "assets/models/baseline_cae_weaksup_best.pt",
    "data/o3a_search_prep/reference_psd_H1.npz",
    "data/o3a_search_prep/reference_psd_L1.npz",
]
print("=== (1) vendored assets ===")
assets_ok = True
for rel in ASSETS:
    p = os.path.join(ROOT, rel)
    if os.path.isfile(p):
        sz = os.path.getsize(p)
        print(f"  OK   {sz:>9,d} B  {rel}")
    else:
        assets_ok = False
        print(f"  MISS            ---  {rel}")

# ---- (2) core module closure ----
MODULES = [
    "utilities",            # vendored (improved/)
    "prepare_o1_data",      # improved/
    "improved_pipeline",    # improved/ (CAE + whitening)
    "massive_pipeline",     # spectrogram_cascade/
    "_budget",              # search_mode/
    "morph_roi",            # search_mode/
    "_pb_cnn_precompute",   # search_mode/
    "driver_search_multi",  # search_mode/
    "driver_streams",       # search_mode/
]
print("\n=== (2) core module imports ===")
mods_ok = True
for m in MODULES:
    try:
        __import__(m)
        print(f"  OK    {m}")
    except Exception as e:
        mods_ok = False
        print(f"  FAIL  {m}: {type(e).__name__}: {e}")
        traceback.print_exc()

# ---- (3) summary ----
print("\n=== summary ===")
print(f"  assets : {'PASS' if assets_ok else 'FAIL'}")
print(f"  imports: {'PASS' if mods_ok else 'FAIL'}")
overall = assets_ok and mods_ok
print(f"\n{'PASS' if overall else 'FAIL'}")
sys.exit(0 if overall else 1)

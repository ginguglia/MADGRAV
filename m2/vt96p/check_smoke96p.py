"""vt96p gate (2026-10-22): A (patched, grid) must equal B (HEAD, grid) key by key and byte by byte; C (patched, prod) must have the same
rows plus place_u with a sane offset distribution. Non-zero exit kills the dependent campaign arrays."""
import numpy as np, sys
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH + "/madgrav2"; ev="o4b_1396417050"
A=np.load(f"{SC}/inj_o4b_m2_96pidA/o4b/{ev}_inj.npz"); B=np.load(f"{SC}/inj_o4b_m2_96pidB/o4b/{ev}_inj.npz"); C=np.load(f"{SC}/inj_o4b_m2_96psmoke/o4b/{ev}_inj.npz")
ok=True
if set(A.files)!=set(B.files): print("FAIL key sets differ", sorted(set(A.files)^set(B.files))); ok=False
for k in sorted(set(A.files)&set(B.files)):
    a,b=A[k],B[k]
    same=a.shape==b.shape and a.dtype==b.dtype and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a,b))
    if not same: print(f"FAIL key {k}: shape {a.shape}/{b.shape} dtype {a.dtype}/{b.dtype} maxdiff {np.nanmax(np.abs(a.astype(float)-b.astype(float))) if a.shape==b.shape and a.dtype.kind in 'fciu' else 'n/a'}"); ok=False
print(f"identity A==B over {len(A.files)} keys, {len(A['net'])} rows: {'PASS' if ok else 'FAIL'}")
extra=set(C.files)-set(A.files); miss=set(A.files)-set(C.files)
if extra!={"place_u"} or miss: print(f"FAIL prod keys: extra {sorted(extra)} missing {sorted(miss)}"); ok=False
if len(C["net"])!=len(A["net"]): print(f"FAIL prod rows {len(C['net'])} != {len(A['net'])}"); ok=False
u=C["place_u"] if "place_u" in C.files else np.array([])
if u.size:
    STEP=1024; q=np.bincount(((u+2*STEP)//STEP).astype(int),minlength=4)
    print(f"prod place_u: min {u.min()} max {u.max()} (expect [-2048,2048)) quarters {q.tolist()}")
    if u.min()<-2*STEP or u.max()>=2*STEP: print("FAIL place_u range"); ok=False
# the injections are paired: same sources/SNR/offset in A and C
for k in ("net_snr","mtot","is_um","off"):
    if not np.array_equal(A[k],C[k]): print(f"FAIL pairing: {k} differs between grid and prod"); ok=False
print(f"paired rows: grid net med {np.median(A['net']):.2f} vs prod net med {np.median(C['net']):.2f} (prod expected <= grid), net>=2.75 frac {np.mean(A['net']>=2.75):.2f} vs {np.mean(C['net']>=2.75):.2f}")
print("GATE", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

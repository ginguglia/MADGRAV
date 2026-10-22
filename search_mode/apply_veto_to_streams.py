"""Apply the CAT2 veto mask to an existing streams directory WITHOUT recomputing anything (2026-09-19).
Reproduces exactly what driver_streams_bg.run() does at build time: vmask = cat2_mask(name, gps); sigma[vmask] = -1e9;
n_vetoed = vmask.sum(). Every other column is copied unchanged, so the output equals a rebuild under SM_VETO.
Usage: SM_VETO=<mask.json> python apply_veto_to_streams.py <src_streams_dir> <dst_streams_dir>"""
import os, sys, glob, json, numpy as np
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
VETO = json.load(open(os.environ["SM_VETO"]))
def cat2_mask(name, gps):
    if name not in VETO: raise KeyError(f"{name} absent from SM_VETO")
    inside = np.zeros(len(gps), bool)
    for lo, hi in VETO[name].get("good_segments", []): inside |= (gps >= lo) & (gps <= hi)
    return ~inside
tot = 0; vet = 0; n = 0
for f in sorted(glob.glob(f"{src}/*_meta.npz")):
    b = os.path.basename(f); name = b[: -len("_H1_meta.npz")]
    z = dict(np.load(f, allow_pickle=True)); vm = cat2_mask(name, z["gps"])
    z["sigma"] = np.where(vm, -1e9, z["sigma"]); z["n_vetoed"] = int(vm.sum())
    tmp = f"{dst}/.{b}.tmp.npz"; np.savez(tmp, **z); os.replace(tmp, f"{dst}/{b}")
    tot += len(vm); vet += int(vm.sum()); n += 1
print(f"[veto-streams] {n} meta files {src} -> {dst}: vetoed {vet}/{tot} windows = {vet/tot*100:.3f}%")

"""Per-injection bank tag for the 2.2 field (2026-10-05): lm (lowmass_bank_v1, mtot 10-22), sig (o1_o3 bank, m1 10-120), um
(ultramassive). Rows aligned with vt22/inj_scored_<run>_m22.npz (matched by exact total_mass, which inject.py copies from the bank).
Writes vt22/bank_tag_<run>_m22.npz {tag}. Used by figures/catalog_o3o4/plot_mtot_snr_smooth_m2.py under MG_BANK_EQUAL=1."""
import sys, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

sys.path.insert(0,MADGRAV_ROOT + "/improved"); import improved_pipeline as ip
SC=MADGRAV_SCRATCH + "/madgrav2/vt22"; MG=MADGRAV_ROOT
lm=set(float(m) for m in ip.load_o1_signal_bank(MADGRAV_ROOT + "/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2")["total_mass"])
sg=set(float(m) for m in ip.load_o1_signal_bank(f"{MG}/data/o1_o3_signal_bank_projected_2s_x10")["total_mass"])
print(f"[bank-tags] lm {len(lm)} masses, sig {len(sg)} masses, overlap {len(lm&sg)}")
for run in sys.argv[1:] or ("o3a","o3b","o4a","o4b"):
    md=np.load(f"{SC}/inj_scored_{run}_m22.npz")["mtot"]; um=np.load(f"{SC}/inj_scored_{run}_m22.npz")["is_um"].astype(bool)
    tag=np.array(["um" if u else ("lm" if float(m) in lm else ("sig" if float(m) in sg else "?")) for m,u in zip(md,um)])
    u,c=np.unique(tag,return_counts=True); print(f"[bank-tags] {run}: {dict(zip(u,c))}")
    np.savez(f"{SC}/bank_tag_{run}_m22.npz",tag=tag)

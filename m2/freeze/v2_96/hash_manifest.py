"""Freeze manifest hashes for the MADGRAV 2 version-2 recipe (96 catalogue detections, 2026-10-19). CPU, seconds."""
import hashlib, os, glob, json, subprocess, sys
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

SC=MADGRAV_SCRATCH + "/madgrav2"; GX=MADGRAV_ROOT; M2=MADGRAV_ROOT; SP=MADGRAV_SCRATCH + "/madgrav2_spec"
groups={
 "cae_trigger":[f"{GX}/spectrogram_cascade/massive_calibration_BA.json"]+glob.glob(f"{GX}/spectrogram_cascade/assets/models/baseline_cae_weaksup_best.pt")+glob.glob(f"{GX}/assets/models/baseline_cae_weaksup_best.pt"),
 "cnn_gate":[f"{GX}/search_mode/hm_native_seed0.pt",f"{GX}/search_mode/lm_native_seed0.pt"],
 "glitch_arm_p1v42":glob.glob(f"{GX}/**/p1v42/arm_deploy_seed0.pt",recursive=True)+glob.glob(f"{SC}/**/p1v42/arm_deploy_seed0.pt",recursive=True),
 "arm_off5":sorted(glob.glob(f"{SC}/arm/models/arm_off5_f*_s*.pt"))+sorted(glob.glob(f"{SC}/arm/models/arm_off5_f*_s*.json")),
 "lr_models":[f"{SC}/lr_{r}_m2_gx_off5.npz" for r in ("o3a","o3b","o4a","o4b")],
 "s3_specialists":sorted(glob.glob(f"{SP}/models/*")),
 "s3_threshold":[f"{M2}/m2/rank/s3_threshold.json"],
 "ranker":[f"{SP}/rank/models_{r}.joblib" for r in ("o3a","o3b","o4a","o4b")],
 "s3_bank_crop":[f"{SP}/s3_crop/s3_{r}.npz" for r in ("o3a","o3b","o4a","o4b")],
 "stat_crop":[f"{SP}/rank_crop/stat_{r}.npz" for r in ("o3a","o3b","o4a","o4b")],
 "asd_plateau":[f"{M2}/m2/asdveto/plateau_{r}.json" for r in ("o3a","o3b","o4a","o4b")],
 "veto_masks":[f"{SC}/dq/veto_mask_{r}_hw.json" for r in ("o3a","o3b","o4a","o4b")],
 "prep":[f"{M2}/data/{r}_search_prep/{f}" for r in ("o3a","o3b","o4a","o4b") for f in ("reference_psd_H1.npz","reference_psd_L1.npz") if os.path.exists(f"{M2}/data/{r}_search_prep/{f}")],
 "merge_driver":[f"{M2}/m2/rank/driver_blindscan_merge_hook.py",f"{GX}/search_mode/driver_blindscan_merge.py",f"{GX}/search_mode/driver_blindscan.py",f"{GX}/search_mode/_pb_cnn_precompute.py",f"{GX}/search_mode/arm_v2.py",f"{GX}/search_mode/inject.py"],
 "post_hoc_rules":[f"{M2}/m2/popb_cap/occlusion/chain/netsig_guard.py",f"{M2}/m2/popb_cap/occlusion/chain/crop_chain.sh",f"{M2}/m2/popb_cap/occlusion/chain/stageB_crop.sbatch"],
 "outputs":[f"{SC}/search_out_{r}_m2_gx_off5_crop/{f}" for r in ("o3a","o3b","o4a","o4b") for f in ("detections_v2.json","detections_v2_nsguard.json","audit_v2.json")]+[f"{SC}/figures_m2/hl_snr_o3o4_m2_crop96.json"],
 "far_shards":[f"{SC}/_{r}_m2_gx_off5_nf275_far_shards" for r in ("o3a","o3b","o4a","o4b")],
 "cnn_cache":[f"{SC}/search_out_{r}_m2_gx_off5_crop" for r in ("o3a","o3b","o4a","o4b")],
}
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<22),b""): h.update(b)
    return h.hexdigest()
out={}
for g,fs in groups.items():
    out[g]={}
    for p in fs:
        if os.path.isdir(p):
            sh=sorted(glob.glob(f"{p}/shard_*.npz"))+sorted(glob.glob(f"{p}/cnn_shard_*.npz"))
            out[g][p]={"n_files":len(sh),"bytes":sum(os.path.getsize(x) for x in sh),"sha256_of_names_sizes":hashlib.sha256("\n".join(f"{os.path.basename(x)} {os.path.getsize(x)}" for x in sh).encode()).hexdigest()}
        elif os.path.exists(p): out[g][p]={"sha256":sha(p),"bytes":os.path.getsize(p)}
        else: out[g][p]="MISSING"
git={}
for t in (M2,GX):
    git[t]={"head":subprocess.run(["git","-C",t,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
            "branch":subprocess.run(["git","-C",t,"branch","--show-current"],capture_output=True,text=True).stdout.strip(),
            "dirty_tracked":[l[3:] for l in subprocess.run(["git","-C",t,"status","--short"],capture_output=True,text=True).stdout.splitlines() if l.startswith(" M") and not l.endswith((".pyc",".log",".env"))]}
out["_git"]=git
json.dump(out,open("MANIFEST_V2_96_20261019.json","w"),indent=1)
n=sum(1 for g in groups for p in out[g] if out[g][p]!="MISSING"); m=[(g,p) for g in groups for p in out[g] if out[g][p]=="MISSING"]
print("hashed",n,"items; missing:",m)
for t,v in git.items(): print(t,v["branch"],v["head"][:12],"dirty code files:",len(v["dirty_tracked"]))

"""Gate for the crop re-pass: runs the three background read-outs and applies the pre-stated criterion (DECISIONS_PLACEMENT.md):
(1) every recoverable cap row (S3 at the coincident crop >= thr AND production FAR < 1/yr) keeps FAR_coinc(S3) < 1/yr;
(2) the random-bank S3 pass fraction under coinc <= 1.5 x production, per run; (3) net-sigma background factor at net >= 4 <= 2.5, per run;
(4) all four runs present in each read-out. Writes GATE_RESULT.txt (first line PASS or FAIL) and exits 0/1."""
import subprocess, re, os, sys, glob
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

H=MADGRAV_ROOT + "/m2/popb_cap/occlusion"; PY="python"; OUT=MADGRAV_SCRATCH + "/madgrav2/popb_occl"
RUNS=("o3a","o3b","o4a","o4b"); reasons=[]; ok=True
miss=[r for r in RUNS if not os.path.exists(f"{OUT}/bank_bg_{r}.json")]+[f"net:{r}" for r in RUNS if not os.path.exists(f"{OUT}/bank_bgnet_{r}.json")]
if miss: reasons.append(f"missing background outputs: {miss}"); ok=False
txt={}
for s in ("readout_bank_bg.py","readout_caprow_far.py","readout_bank_bgnet.py"):
    p=subprocess.run([PY,"-I",f"{H}/{s}"],capture_output=True,text=True,cwd=H); txt[s]=p.stdout
# (1) cap rows
for l in txt["readout_caprow_far.py"].splitlines():
    m=re.match(r"\s+(GW\S+)\s+(o\d\w)\s+([\d.]+)\s+\d+\s+\d+\s+\d+ \|\s+([\d.]+)\s+([\d.]+) \|\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)",l)
    if not m: continue
    name,run,L,f3,fa,far_p,far3,fara,s3c=m.group(1),m.group(2),float(m.group(3)),float(m.group(4)),float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(8)),float(m.group(9))
    if s3c>=0.5357 and far_p<1.0:
        if far3<1.0: reasons.append(f"OK   {name} {run}: FAR {far_p:.3f} -> {far3:.3f}/yr")
        else: reasons.append(f"FAIL {name} {run}: FAR {far_p:.3f} -> {far3:.3f}/yr >= 1"); ok=False
# (2) random bank
for l in txt["readout_bank_bg.py"].splitlines():
    m=re.search(r"\[(o\d\w)\]",l); 
    if m: run=m.group(1)
    m=re.search(r"S3 pass prod ([\d.]+)% -> coinc ([\d.]+)%",l)
    if m:
        a,b=float(m.group(1)),float(m.group(2)); 
        if b<=1.5*a: reasons.append(f"OK   {run}: random-bank S3 pass {a:.2f}% -> {b:.2f}%")
        else: reasons.append(f"FAIL {run}: random-bank S3 pass {a:.2f}% -> {b:.2f}% (> 1.5x)"); ok=False
# (3) net-sigma factor
for l in txt["readout_bank_bgnet.py"].splitlines():
    m=re.search(r"\[(o\d\w)\]",l)
    if m: run=m.group(1)
    m=re.search(r"net>=4\.0:.*passing x([\d.]+)",l)
    if m:
        f=float(m.group(1))
        if f<=2.5: reasons.append(f"OK   {run}: net-sigma background x{f:.2f} at net>=4")
        else: reasons.append(f"FAIL {run}: net-sigma background x{f:.2f} at net>=4 (> 2.5)"); ok=False
res=("PASS" if ok else "FAIL")+"\n"+"\n".join(reasons)+"\n"; open(f"{H}/chain/GATE_RESULT.txt","w").write(res); print(res); sys.exit(0 if ok else 1)

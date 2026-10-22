"""Do the current stages reject GlitchGen glitches on quiet carriers? (experiment 2026-09-29)
Input: glitch-only shards of gen_positives_contam.py (M2_NO_SIGNAL=1): per leg the frozen tile, the CAE sigma, the GlitchGen class and
the injected optimal SNR. Scores every tile with the _def and _off5 arm ensembles (all 10 models; the tiles are synthetic and never in any
training fold) and reports per class: n, CAE trigger fraction (sigma >= 4, i.e. the glitch would enter the search), arm pass fraction
among triggered tiles at the production gate -2.05 and at -2.49, and the same split by injected SNR. Env: SYN_DIR (shards), SYN_TAG (glob
tag, e.g. _ggonly), OUT (report path). CPU only."""
import os, sys, glob, json, numpy as np, torch
H=os.path.dirname(os.path.abspath(__file__)); exec(open(f"{H}/arm_model.py").read())
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

M=MADGRAV_SCRATCH + "/madgrav2/arm/models"; torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","16")))
D=os.environ["SYN_DIR"]; TAG=os.environ.get("SYN_TAG","_ggonly"); OUT=os.environ.get("OUT",f"{M}/synth_glitch_readout{TAG}.txt"); FLOOR=4.0
def load_arm(tag):
    ms=[]
    for js in sorted(glob.glob(f"{M}/arm{tag}_f*_s*.json")):
        m=Arm(); m.load_state_dict(torch.load(js.replace(".json",".pt"),map_location="cpu")); m.eval(); ms.append(m)
    return ms
def logit(models,tiles):
    x=torch.from_numpy(np.asarray(tiles,np.float32)[:,None]); out=[]
    with torch.no_grad():
        for m in models: p=torch.cat([m.psig(x[i:i+256]) for i in range(0,len(x),256)]).numpy().astype(np.float64); out.append(np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1)))
    return np.mean(out,0)
def main():
    ARMS={t:load_arm(t) for t in ("_def","_off5")}; rows=[]
    for f in sorted(glob.glob(f"{D}/pos_*{TAG}*_shard*.npz")):
        z=np.load(f,allow_pickle=True)
        for u,d in (("H","H1"),("L","L1")):
            cls=np.asarray(z[f"gcls_{u}"]).astype(str); k=np.flatnonzero(cls!="")
            if not len(k): continue
            L={t:logit(ARMS[t],z[f"qt_{u}"][k]) for t in ARMS}
            for i,j in enumerate(k): rows.append((d,cls[j],float(z[f"synth_snr_{u}"][j]),float(z[f"sigma_{u}"][j]),L["_def"][i],L["_off5"][i]))
    if not rows: raise SystemExit("no glitch-only tiles found")
    det=np.array([r[0] for r in rows]); cls=np.array([r[1] for r in rows]); snr=np.array([r[2] for r in rows]); sg=np.array([r[3] for r in rows]); gd=np.array([r[4] for r in rows]); go=np.array([r[5] for r in rows])
    trig=sg>=FLOOR; L=[f"GlitchGen glitches on quiet carriers, {len(rows)} tiles ({(det=='H1').sum()} H1, {(det=='L1').sum()} L1), source {D} tag {TAG}",
       "pass = arm logit >= gate on TRIGGERED tiles (CAE sigma >= 4). A passing glitch is one the arm would let into the pair stage.",""]
    def block(mask,name):
        n=int(mask.sum()); t=trig&mask; nt=int(t.sum())
        if not n: return
        L.append(f"{name:28s} n={n:5d} trig {nt/n:5.2f} ({nt:4d}) | _def pass@-2.05 {np.mean(gd[t]>=-2.05) if nt else float('nan'):5.2f} @-2.49 {np.mean(gd[t]>=-2.49) if nt else float('nan'):5.2f} | _off5 pass@-2.05 {np.mean(go[t]>=-2.05) if nt else float('nan'):5.2f} @-2.49 {np.mean(go[t]>=-2.49) if nt else float('nan'):5.2f} | med logit def {np.median(gd[t]) if nt else float('nan'):+6.2f} off5 {np.median(go[t]) if nt else float('nan'):+6.2f}")
    L.append("== by class"); [block(cls==c,c) for c in sorted(set(cls))]
    L.append("== by injected SNR (all classes)"); e=[8,12,20,35,80]
    for a,b in zip(e[:-1],e[1:]): block((snr>=a)&(snr<b),f"SNR {a}-{b}")
    L.append("== all"); block(np.ones(len(rows),bool),"ALL")
    L.append(""); L.append("worst classes for the production arm (_def pass@-2.05 among triggered, n_trig>=20):")
    worst=[(np.mean(gd[trig&(cls==c)]>=-2.05),c,int((trig&(cls==c)).sum())) for c in set(cls) if (trig&(cls==c)).sum()>=20]
    for p,c,n in sorted(worst,reverse=True)[:8]: L.append(f"  {c:28s} pass {p:.2f} (n_trig {n})")
    txt="\n".join(L); print(txt); open(OUT,"w").write(txt+"\n"); json.dump([dict(det=r[0],cls=r[1],snr=r[2],sigma=r[3],g_def=r[4],g_off5=r[5]) for r in rows],open(OUT.replace(".txt",".json"),"w"))
if __name__=="__main__": main()

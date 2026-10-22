"""MADGRAV2 heavy bank v2 (2026-09-23, PREREG_SPECIALISTS_20260923 §2): BYTE-IDENTICAL RECIPE to MADGRAV1
dev_v2/glitch_arm/bank/gen_heavy_bank.py except the default seed (20260923, so no row repeats v1) and chunks 40-399
(source_id continues after v1's 0-9999). Purpose: 90k new heavy waveforms so the 200k specialist positives need no reuse.
Convention kept: per-detector peak at index 4096, physical delay LOGGED not applied (the renderer applies it).
--- original docstring ---
Amended positive class — HEAVY-TIER projected bank (Step 3 ruling 2026-08-19).
Replaces the UM bank where it is structurally absent (fmin=20 Hz -> 10-20 Hz half of the QT band had zero positive content) and
extends above its 400 Msun ceiling. Same on-disk format as data/ultramassive_bank (signals_XXXX.npz: H1,L1 (n,8192) float32 @4096 Hz,
peak at index 4096; signals_XXXX.csv with the parameter table) so improved_pipeline.load_o1_signal_bank reads it unchanged.
Design (pre-registered): Mtot DETECTOR-FRAME log-flat on [MTOT_LO, MTOT_HI] (default 150-800); q uniform [1, QMAX]; f_min = 10 Hz
(non-negotiable); IMRPhenomXPHM; precessing spins (|chi| uniform [0,0.99], isotropic tilts); extrinsics volumetric (d ~ d^2 on
[D_LO,D_HI] Mpc, isotropic sky, cos(inc) uniform, psi uniform, phase uniform); GPS uniform in the O3a span. Confuser-band
oversampling is NOT baked in here (it is a training-draw weight fixed after the quota freeze) — this bank is the flat parent.
Usage: python gen_heavy_bank.py --out DIR --chunk K --n 250 [--seed S]   (one chunk per job; K-dependent seed)"""
import argparse, os, csv, numpy as np
ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--chunk",type=int,required=True); ap.add_argument("--n",type=int,default=250)
ap.add_argument("--mtot-lo",type=float,default=150.); ap.add_argument("--mtot-hi",type=float,default=800.); ap.add_argument("--qmax",type=float,default=4.)
ap.add_argument("--dlo",type=float,default=100.); ap.add_argument("--dhi",type=float,default=8000.); ap.add_argument("--fmin",type=float,default=10.)
ap.add_argument("--apx",default="IMRPhenomXPHM"); ap.add_argument("--seed",type=int,default=20260923); a=ap.parse_args()
FS=4096; NS=2*FS; PK=NS//2; rng=np.random.default_rng([a.seed,a.chunk])
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.types import TimeSeries
dets={d:Detector(d) for d in ("H1","L1")}
GPS_LO,GPS_HI=1238166018,1253977218   # O3a
rows=[]; H={"H1":[], "L1":[]}; i=0; tries=0
while i<a.n and tries<20*a.n:
    tries+=1
    mt=float(np.exp(rng.uniform(np.log(a.mtot_lo),np.log(a.mtot_hi)))); q=float(rng.uniform(1.,a.qmax)); m1=mt*q/(1+q); m2=mt/(1+q)
    c1,c2=rng.uniform(0,0.99,2); t1,t2=np.arccos(rng.uniform(-1,1,2)); p1,p2=rng.uniform(0,2*np.pi,2)
    s1=(c1*np.sin(t1)*np.cos(p1),c1*np.sin(t1)*np.sin(p1),c1*np.cos(t1)); s2=(c2*np.sin(t2)*np.cos(p2),c2*np.sin(t2)*np.sin(p2),c2*np.cos(t2))
    d=float((rng.uniform(a.dlo**3,a.dhi**3))**(1/3)); inc=float(np.arccos(rng.uniform(-1,1))); phi=float(rng.uniform(0,2*np.pi))
    ra=float(rng.uniform(0,2*np.pi)); dec=float(np.arcsin(rng.uniform(-1,1))); psi=float(rng.uniform(0,np.pi)); gps=float(rng.uniform(GPS_LO,GPS_HI))
    try:
        hp,hc=get_td_waveform(approximant=a.apx,mass1=m1,mass2=m2,spin1x=s1[0],spin1y=s1[1],spin1z=s1[2],spin2x=s2[0],spin2y=s2[1],spin2z=s2[2],
                              distance=d,inclination=inc,coa_phase=phi,delta_t=1./FS,f_lower=a.fmin)
    except Exception as e:
        continue
    out={}; ok=True
    for dn,det in dets.items():
        fp,fc=det.antenna_pattern(ra,dec,psi,gps); dt=det.time_delay_from_earth_center(ra,dec,gps)
        h=(fp*hp.numpy()+fc*hc.numpy()).astype(np.float64)
        # place the |h| peak at index PK of a 2-s window (same convention as the existing banks); apply the detector delay as a sample shift
        pk=int(np.argmax(np.abs(h))); w=np.zeros(NS); src0=pk-PK   # peak at index PK per detector (existing-bank convention; delay logged, not applied)
        lo=max(0,src0); hi=min(len(h),src0+NS); 
        if hi<=lo: ok=False; break
        w[lo-src0:hi-src0]=h[lo:hi]; out[dn]=(w.astype(np.float32),fp,fc,int(np.argmax(np.abs(w))),float(dt))
    if not ok or not np.all(np.isfinite(out["H1"][0])) or not np.all(np.isfinite(out["L1"][0])): continue
    rows.append(dict(source_id=a.chunk*a.n+i,projection_mode="projected",approximant=a.apx,mass1=m1,mass2=m2,mass_ratio=q,total_mass=mt,
        chi1=c1,chi2=c2,tilt1=t1,tilt2=t2,spin1x=s1[0],spin1y=s1[1],spin1z=s1[2],spin2x=s2[0],spin2y=s2[1],spin2z=s2[2],
        distance_mpc=d,inclination_rad=inc,coa_phase_rad=phi,ra_rad=ra,dec_rad=dec,polarization_rad=psi,reference_time_gps=gps,snr_target=0.0,
        H1_f_plus=out["H1"][1],H1_f_cross=out["H1"][2],H1_peak_index=out["H1"][3],L1_f_plus=out["L1"][1],L1_f_cross=out["L1"][2],L1_peak_index=out["L1"][3],H1_delay_s=out["H1"][4],L1_delay_s=out["L1"][4],
        f_min=a.fmin,mtot_det_frame=1))
    for dn in H: H[dn].append(out[dn][0])
    i+=1
os.makedirs(a.out,exist_ok=True); base=os.path.join(a.out,f"signals_{a.chunk:04d}")
np.savez(base+".npz",H1=np.stack(H["H1"]),L1=np.stack(H["L1"]))
with open(base+".csv","w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
print(f"[heavy-bank] chunk {a.chunk}: {i} waveforms ({tries} tries) -> {base}.npz  Mtot {min(r['total_mass'] for r in rows):.0f}-{max(r['total_mass'] for r in rows):.0f}")

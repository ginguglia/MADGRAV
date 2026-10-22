"""Bank of GAN blips (gengli 0.2.0, Lopez et al. 2022, arXiv:2205.09204) per detector at 4096 Hz: unit-peak whitened-domain excess
power, Tukey alpha 0.25, no low-pass. Stored with the sum of squares so any SNR (lalsuite white-noise normalisation: sum x^2 =
snr^2 * srate/2 on a unit-variance whitened series) can be set at use. Env: N (3000), OUT dir, SEED (20260929)."""
import os, numpy as np, time, gengli
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

N=int(os.environ.get("N","3000")); OUT=os.environ.get("OUT",MADGRAV_SCRATCH + "/madgrav2/arm/synth"); SEED=int(os.environ.get("SEED","20260929")); FS=4096.
for det in ("H1","L1"):
    g=gengli.glitch_generator(det); t=time.time(); X=[]
    for b in range(0,N,100): X.append(np.asarray(g.get_glitch(min(100,N-b), srate=FS, seed=SEED+b), np.float32))
    X=np.concatenate(X); pk=np.abs(X).max(1); X=X/pk[:,None]; ss=(X**2).sum(1)
    np.savez(f"{OUT}/gengli_blips_{det}.npz", x=X, sumsq=ss, srate=FS, seed=SEED, gengli="0.2.0", note="unit-peak whitened-domain blips; scale a*x so that a^2*sumsq = snr^2*srate/2 on a unit-variance whitened series")
    print(f"{det}: {X.shape} in {time.time()-t:.0f}s, len {X.shape[1]} samples ({X.shape[1]/FS:.3f} s), sumsq median {np.median(ss):.1f}", flush=True)

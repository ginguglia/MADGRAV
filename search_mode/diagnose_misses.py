"""Diagnosis: are the CAE's real-event misses information-limited (low network SNR / unfavorable
detector split) or representational (decent-SNR events missed anyway)?

For each O3a/O3b catalog event (GWOSC network SNR + strain): score CAE net sigma = (sH+sL)/sqrt2 under
(a) the frozen reference ASD (reproduces the retrospective eval) and (b) a local +/-64s ASD (clean).
Correlate misses with network SNR, total mass, and per-detector CAE split. If low-sigma == low-netSNR
-> info-limited (a bottleneck cannot help). If decent-netSNR events are missed -> representational.

Run from MADGRAV_ROOT: SM_ALLOW_CPU=1 BLIND_DEV=cpu python diagnose_misses.py
"""
import os, sys, json, glob, numpy as np, warnings; warnings.filterwarnings("ignore")
ROOT = os.environ["MADGRAV_ROOT"]
for p in ("search_mode", "improved", "spectrogram_cascade"):
    sys.path.insert(0, os.path.join(ROOT, p))
import driver_blindscan as B
import improved_pipeline as ip
from gwpy.timeseries import TimeSeries
from scipy.ndimage import zoom
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

FS = 4096; WIN = 4.0; WN = int(WIN * FS)

# GWTC catalog: persistent copy in the MADGRAV tree (was login-local /tmp; moved so this runs on compute nodes).
_CAT_PATH = os.path.join(ROOT, "data", "gwtc_test.json")
if not os.path.exists(_CAT_PATH):
    _CAT_PATH = "/tmp/gwtc_test.json"  # legacy fallback
CAT = json.load(open(_CAT_PATH))["events"]
BYNAME = {(v.get("commonName") or k): v for k, v in CAT.items()}
RUNS = [("O3a", ["search_mode/o3a_events_full.json", "search_mode/o3a_events.json"], MADGRAV_SCRATCH + "/strain_o3a_full"),
        ("O3b", ["search_mode/o3b_events.json"], MADGRAV_SCRATCH + "/strain_o3b_full")]


def build_qt(pipe, wh):
    qi = ip.center_crop_waveforms(wh, sample_rate=FS, context_seconds=pipe.ctx)
    mags = [ip._compute_qt_image_worker((w, FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0)) for w in qi]
    return ip.min_max_norm(np.stack([zoom(m, (256 / m.shape[0], 128 / m.shape[1]), order=1)
                                     for m in mags]).astype(np.float32)).astype(np.float32)


def sig(pipe, win, det):
    mu, sd = (pipe.norm["muH"], pipe.norm["sdH"]) if det == "H1" else (pipe.norm["muL"], pipe.norm["sdL"])
    wh = pipe._whiten(win[None, :].astype(np.float32), det)
    return float((pipe._recon(build_qt(pipe, wh)).reshape(-1)[0] - mu) / sd)


def seg_index(strdir):
    segs = []
    for f in sorted(glob.glob(strdir + "/*_H1.npz")):
        d = np.load(f); segs.append((float(d["gps_start"]), float(d["gps_end"]), os.path.basename(f)[:-7]))
    return segs


def run():
    pipe = B.cpipe()   # frozen reference ASD = o3a_search_prep
    ref = {d: pipe.asd[d] for d in ("H1", "L1")}
    rows = []
    for run, evfs, strdir in RUNS:
        ev = {}
        for f in evfs:
            if os.path.exists(f): ev.update(json.load(open(f)))
        segs = seg_index(strdir)
        for name, gps in ev.items():
            meta = BYNAME.get(name, {})
            snr = meta.get("network_matched_filter_snr"); mtot = meta.get("total_mass_source")
            if snr is None: continue
            hit = [s for s in segs if s[0] <= gps <= s[1]]
            if not hit: continue
            t0, _, sname = hit[0]
            try:
                strH = np.load(f"{strdir}/{sname}_H1.npz")["strain"]; strL = np.load(f"{strdir}/{sname}_L1.npz")["strain"]
            except Exception: continue
            i0 = int(round((gps - WIN / 2.0 - t0) * FS))
            if i0 < 64 * FS or i0 + WN > len(strH) - 64 * FS: continue
            wH = strH[i0:i0 + WN]; wL = strL[i0:i0 + WN]
            # reference ASD
            pipe.asd["H1"], pipe.asd["L1"] = ref["H1"], ref["L1"]
            sHr, sLr = sig(pipe, wH, "H1"), sig(pipe, wL, "L1")
            # local ASD
            laH = TimeSeries(strH[i0 - 64 * FS:i0 + 64 * FS].astype(np.float64), sample_rate=FS).asd(fftlength=4, overlap=2, method="median")
            laL = TimeSeries(strL[i0 - 64 * FS:i0 + 64 * FS].astype(np.float64), sample_rate=FS).asd(fftlength=4, overlap=2, method="median")
            pipe.asd["H1"], pipe.asd["L1"] = laH, laL
            sHl, sLl = sig(pipe, wH, "H1"), sig(pipe, wL, "L1")
            pipe.asd["H1"], pipe.asd["L1"] = ref["H1"], ref["L1"]
            rows.append(dict(name=name, run=run, gps=gps, netSNR=float(snr), mtot=(float(mtot) if mtot else None),
                             net_ref=(sHr + sLr) / np.sqrt(2), net_loc=(sHl + sLl) / np.sqrt(2),
                             sH_loc=sHl, sL_loc=sLl,
                             split=abs(sHl - sLl) / (abs(sHl) + abs(sLl) + 1e-9)))
            r = rows[-1]
            print(f"  {name:20s} {run} SNR={snr:5.1f} Mtot={str(round(mtot) if mtot else '?'):>4} | "
                  f"netσ ref={r['net_ref']:6.2f} loc={r['net_loc']:6.2f} (H {sHl:5.1f} L {sLl:5.1f})", flush=True)
    json.dump(rows, open(os.path.join(ROOT, "search_mode", "diagnose_misses.json"), "w"), indent=2)
    # ---- analysis ----
    a = rows
    def frac(cond):
        s = [r for r in a if cond(r)]; return len(s)
    print(f"\n=== {len(a)} events scored ===", flush=True)
    for cut in (3.0, 5.0):
        miss = [r for r in a if r["net_loc"] < cut]
        det = [r for r in a if r["net_loc"] >= cut]
        print(f"\n-- threshold net σ(local) ≥ {cut}: {len(det)}/{len(a)} detected, {len(miss)} missed --", flush=True)
        if miss:
            snrs = [r["netSNR"] for r in miss]
            print(f"   MISSED netSNR: min {min(snrs):.1f} med {np.median(snrs):.1f} max {max(snrs):.1f}", flush=True)
            hi = [r for r in miss if r["netSNR"] >= 12]
            print(f"   MISSED with netSNR≥12 (representational-miss candidates): {len(hi)}", flush=True)
            for r in sorted(hi, key=lambda x: -x["netSNR"]):
                print(f"      {r['name']:20s} SNR={r['netSNR']:.1f} Mtot={r['mtot']} netσloc={r['net_loc']:.2f} split={r['split']:.2f}", flush=True)
    print("\nDONE -> search_mode/diagnose_misses.json", flush=True)


if __name__ == "__main__":
    run()

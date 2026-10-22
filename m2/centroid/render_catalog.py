"""Render the frozen-geometry tile (gps-centred, same recipe as madgrav_qt_build/tiles.py) for EVERY catalog event in the
frozen search footprint (HL coincident, Mtot>10, rho_HL>10 -> the 138-event set of recon_select_test/score_events.py),
compute all centroid definitions, save feat_cat.npz with name, run, det, mtot, catalog per-detector SNR, adopted flag,
miss cause. Read-only use of MADGRAV1 strain caches and prep. CPU."""
import os, sys, json, time, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

os.environ.setdefault("RS_CAMPAIGN", "margin2"); os.environ.setdefault("CALIB_DIR", MADGRAV_SCRATCH + "/recon_select/calib_margin2")
M1 = MADGRAV_SCRATCH + "/MADGRAV1"; os.environ["MADGRAV_ROOT"] = M1
for p in ("search_mode", "improved", "spectrogram_cascade", "dev_v2/recon_select_test", "figures/catalog_o3o4"):
    sys.path.insert(0, f"{M1}/{p}")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gps_map, find_window_file, cut, prep, BA, WIN, FS
from massive_pipeline import MassiveEventPipeline
from scipy.ndimage import zoom
import improved_pipeline as ip
import adopted_set
from centroid_defs import compute_all
OUT = MADGRAV_SCRATCH + "/madgrav2/centroid"; SMOKE = "--smoke" in sys.argv

H = json.load(open(f"{M1}/figures/catalog_o3o4/hl_snr_o3o4.json"))
miss = {e["name"]: e for e in json.load(open(f"{M1}/figures/catalog_o3o4/misses_decomposition.json"))}
adopted = {d["name"] for d in adopted_set.load()}
G = gps_map()
import re
from gwpy.time import to_gps
def gps_of(n):
    if n in G: return G[n]
    base = n.split("_")[0]
    if base in G: return G[base]
    m = re.match(r"GW(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$", n)
    return float(to_gps(f"20{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}")) if m else None
ev = sorted([(n, v) for n, v in H.items() if v.get("plotted", True) and v.get("coincident") and (v.get("mtot") or 0) > 10 and (v.get("snr_hl") or 0) > 10], key=lambda x: (x[1]["run"], x[0]))
print(f"{len(ev)} catalog events in footprint; {sum(n in adopted for n,_ in ev)} adopted", flush=True)
if SMOKE: ev = ev[:3]

def tile(pipe, det, raw):
    wh = pipe._whiten(raw[None, :], det)
    qi = ip.center_crop_waveforms(wh, sample_rate=FS, context_seconds=pipe.ctx)
    mags = [ip._compute_qt_image_worker((w, FS, ip.QTRANSFORM_FRANGE, ip.QTRANSFORM_QRANGE, 1.0)) for w in qi]
    st = np.stack([zoom(m, (256 / m.shape[0], 128 / m.shape[1]), order=1) for m in mags]).astype(np.float32)
    return ip.min_max_norm(st).astype(np.float32)[0]

tiles, meta = [], {k: [] for k in ("name", "run", "det", "mtot", "snr_det", "snr_hl", "adopted", "miss_cause", "status")}
t0 = time.time(); pipes = {}
for i, (n, v) in enumerate(ev, 1):
    run = v["run"].lower(); status = "ok"
    g = gps_of(n)
    if g is None: status = "no_gps"
    else:
        files = {d: find_window_file(run, d, g, WIN / 2 + 0.5) for d in ("H1", "L1")}
        if None in files.values(): status = "no_strain"
    if status != "ok":
        print(f"[{i}/{len(ev)}] {n}: {status}", flush=True); continue
    if run not in pipes: pipes[run] = MassiveEventPipeline(calib_path=BA, prep=prep(run), device="cpu")
    for d in ("H1", "L1"):
        try: t = tile(pipes[run], d, cut(files[d], g))
        except Exception as e: print(f"[{i}] {n} {d}: {e}", flush=True); continue
        tiles.append(t)
        for k, val in dict(name=n, run=v["run"], det=d, mtot=v["mtot"], snr_det=v["per"].get(d, np.nan), snr_hl=v["snr_hl"],
                           adopted=int(n in adopted), miss_cause=miss.get(n, {}).get("cause", ""), status=status).items():
            meta[k].append(val)
    if i % 10 == 0: print(f"[{i}/{len(ev)}] {(time.time()-t0)/60:.1f} min", flush=True)
tiles = np.stack(tiles); feats = compute_all(tiles)
sfx = "_smoke" if SMOKE else ""
np.savez(f"{OUT}/feat_cat{sfx}.npz", **feats, **{k: np.array(v) for k, v in meta.items()})
np.savez(f"{OUT}/cat_tiles{sfx}.npz", qt=tiles.astype(np.float16), name=np.array(meta["name"]), det=np.array(meta["det"]))
print(f"saved {len(tiles)} legs -> feat_cat{sfx}.npz, cat_tiles{sfx}.npz in {(time.time()-t0)/60:.1f} min", flush=True)

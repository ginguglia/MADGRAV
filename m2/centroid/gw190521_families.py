"""Which background families set GW190521's (IMBH) FAR in the adopted MADGRAV1 statistic (LR-only axis, netmax 10.6,
A5 g_net gate), and how many of them the new cluster centroid (clu98 < 200 Hz per leg) would remove.
Reproduces inclusive_exclusive_far.counts() on the O3a Background, then renders the family rep-pair legs."""
import os, sys, csv, json, time, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

MG = MADGRAV_SCRATCH + "/MADGRAV1"; DET = f"{MG}/details/successor_statistic"; SC = MADGRAV_SCRATCH
thr = json.load(open(f"{DET}/gnet_threshold.json")); GNET = float(thr["threshold"] if isinstance(thr, dict) and "threshold" in thr else thr.get("gnet", -4.02))
os.environ["SM_GNET"] = str(GNET); NETMAX = 10.6
sys.path.insert(0, f"{MG}/search_mode"); sys.path.insert(0, "$MADGRAV_SCRATCH/madgrav_qt_build"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import successor_stat as S
from tiles import tile, WIN, FS, WN
from centroid_defs import compute_all
NAME = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "GW190521"
CUT = 200.0
rows = {r["name"]: r for r in csv.DictReader(open(f"{MG}/figures/catalog_o3o4/madgrav_far_final_x1.csv"))}
r = rows[NAME]; run = r["run"]; print(f"{NAME}: run {run} seg {r['seg']} gps {r['gps']} loglr {float(r['loglr']):.3f} net {float(r['net']):.2f} hm {float(r['cnn_hm']):.4f} lm {float(r['cnn_lm']):.4f}; GNET {GNET}, NETMAX {NETMAX}", flush=True)
bg = S.Background(run, f"{DET}/bg_veto_{run.lower()}.npz", verbose=False)
dr = [x for x in rows.values() if x["run"] == run]
touch = np.zeros(len(bg.ll), bool)
for x in dr:
    Sx = bg.seg_ix[x["seg"]]; g = float(x["gps"])
    touch |= (bg.hseg == Sx) & (np.abs(bg.gpsH - g) <= 4.0); touch |= (bg.lseg == Sx) & (np.abs(bg.gpsL - g) <= 4.0)
Sx = bg.seg_ix[r["seg"]]; f = int(bg.seg_fold[Sx])
keep = (bg.hseg != Sx) & (bg.lseg != Sx) & (bg.net < NETMAX) & ~touch & bg.gpass & (bg.fold == f)
mm = keep & (bg.ll > float(r["loglr"]))
hm, lm = float(r["cnn_hm"]), float(r["cnn_lm"])
fam_hm = np.unique(bg.fam[mm & (bg.hm >= hm)]); fam_lm = np.unique(bg.fam[mm & (bg.lm >= lm)])
side, fams = ("hm", fam_hm) if len(fam_hm) <= len(fam_lm) else ("lm", fam_lm)
print(f"N_lr = min(|fam hm>={hm:.3f}|={len(fam_hm)}, |fam lm>={lm:.3f}|={len(fam_lm)}) = {len(fams)} families (csv N_excl expected 175 for GW190521); T_fold {bg.F[f]['T']:.1f} yr", flush=True)
# rep pair per family: max lnLambda among qualifying pairs
q = mm & ((bg.hm >= hm) if side == "hm" else (bg.lm >= lm))
idx = np.where(q)[0]; best = {}
for i in idx:
    fk = bg.fam[i]
    if fk not in best or bg.ll[i] > bg.ll[best[fk]]: best[fk] = i
reps = np.array([best[fk] for fk in fams])
print(f"rep pairs: lnL {bg.ll[reps].min():.2f}-{bg.ll[reps].max():.2f}, sigma_net {bg.net[reps].min():.1f}-{bg.net[reps].max():.1f}; distinct H1 windows {len(set(zip(bg.hseg[reps], bg.gpsH[reps])))}, L1 windows {len(set(zip(bg.lseg[reps], bg.gpsL[reps])))}", flush=True)
def strain_path(seg, det):
    for d in (f"{SC}/strain_{run.lower()}_full", f"{SC}/strain_{run.lower()}_56"):
        p = f"{d}/{seg}_{det}.npz"
        if os.path.exists(p): return p
def window_at(path, gps):
    z = np.load(path, allow_pickle=True); gs = float(z["gps_start"]); i0 = int(round((gps - WIN / 2 - gs) * FS)); w = np.asarray(z["strain"][i0:i0 + WN], dtype=np.float32); assert len(w) == WN; return w
cache = {}; t0 = time.time()
def cen_of(seg, det, gps):
    k = (seg, det, round(gps, 1))
    if k not in cache:
        p = strain_path(seg, det); cache[k] = np.nan if p is None else float(compute_all(tile(run.lower(), det, window_at(p, gps))[None])["clu98"][0])
    return cache[k]
out = []
for n, i in enumerate(reps, 1):
    hs, ls = bg.seg_names[bg.hseg[i]], bg.seg_names[bg.lseg[i]]
    cH = cen_of(hs, "H1", float(bg.gpsH[i])); cL = cen_of(ls, "L1", float(bg.gpsL[i]))
    out.append(dict(fam=int(bg.fam[i]), ll=float(bg.ll[i]), net=float(bg.net[i]), hm=float(bg.hm[i]), lm=float(bg.lm[i]), gH=float(bg.gnet[i]) if bg.gnet is not None else np.nan,
                    H1_seg=hs, gpsH=float(bg.gpsH[i]), L1_seg=ls, gpsL=float(bg.gpsL[i]), cH=cH, cL=cL, removed=bool((np.isfinite(cH) and cH >= CUT) or (np.isfinite(cL) and cL >= CUT))))
    if n % 25 == 0: print(f"  [{n}/{len(reps)}] {(time.time()-t0)/60:.1f} min", flush=True)
rem = sum(o["removed"] for o in out); nan = sum(not (np.isfinite(o["cH"]) and np.isfinite(o["cL"])) for o in out)
print(f"\nRESULT {NAME}: {rem}/{len(out)} FAR-setting families removed by clu98 < {CUT} Hz on either leg ({nan} with an unrenderable leg); remaining N = {len(out)-rem} -> FAR would scale {len(out)} -> {len(out)-rem}", flush=True)
for c in (180, 190, 200, 210, 220, 250):
    print(f"  cut {c}: removed {sum((np.isfinite(o['cH']) and o['cH']>=c) or (np.isfinite(o['cL']) and o['cL']>=c) for o in out)}/{len(out)}")
json.dump(out, open(f"{SC}/madgrav2/centroid/{NAME}_families_clu98.json", "w"), indent=1)
print("top families by lnL (lnL, net, cH, cL, removed):"); [print(f"  {o['ll']:.2f} {o['net']:.1f} {o['cH']:.0f}/{o['cL']:.0f} {'X' if o['removed'] else '.'}") for o in sorted(out, key=lambda o: -o["ll"])[:15]]

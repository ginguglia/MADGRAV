"""MADGRAV2 glitch-arm dataset assembly (2026-09-17 plan, [[madgrav2-arm-retrain-plan]]).
Classes (frozen 256x128 model tile, fp16):
  sig    : legs of the MADGRAV2 injection population (m2/positives): ARM_SIG_RULE=leg4 (sigma_det>=4) or pair4 (both legs of
           pairs with sigma_net>=4, i.e. every leg the pipeline would admit, including faint ones)
  glitch : mined real single-detector glitch tiles, all four runs (relabel-v2 morphology for O3a)
  murky  : CAE-passing time-slid background tiles (dev_v2/g8/murky), the population that sets the FAR
Rules: TIME partition by UTC day, 2 folds (sha1(day) % 2), same day -> same fold for every class and both detectors;
inside a fold, days with sha1(day)%10==0 form the calibration split (temperature scaling). Quarantine: any tile within
+-64 s of a catalog event (GWTC + O4 names, both detectors), of an adopted detection, or of a counted FAR-family leg
(same detector) is excluded from ALL splits. Output: /scratch/.../madgrav2/arm/data/{cls}_f{fold}.npy + _meta.npz."""
import os, sys, glob, csv, json, re, hashlib, time, numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)

M1 = MADGRAV_SCRATCH + "/MADGRAV1"; SC = MADGRAV_SCRATCH; OUT = os.environ.get("ARM_DATA", f"{SC}/madgrav2/arm/data"); os.makedirs(OUT, exist_ok=True)
SIG_RULE = os.environ.get("ARM_SIG_RULE", "leg4")   # leg4: per-leg sigma>=4 (v1) | pair4: both legs of pairs with sigma_net>=4 (v2, matches the pipeline admission)
SMOKE = "--smoke" in sys.argv; QUAR_S = 64.0
def fold_of(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(), 16) % 2
def calib_of(day): return int(hashlib.sha1(("c" + str(int(day))).encode()).hexdigest(), 16) % 10 == 0
# ---------------- quarantine table: per detector sorted gps arrays ----------------
from gwpy.time import to_gps
q = {"H1": [], "L1": []}
for fn in glob.glob(MADGRAV_EXTDATA + "/gwosc_eventapi/GWTC-*.json"):
    for v in json.load(open(fn)).get("events", {}).values():
        for d in q: q[d].append(float(v["GPS"]))
H = json.load(open(f"{M1}/figures/catalog_o3o4/hl_snr_o3o4.json"))
for n in H:
    m = re.match(r"GW(\d{2})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$", n)
    if m:
        g = float(to_gps(f"20{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}"))
        for d in q: q[d].append(g)
for r in csv.DictReader(open(f"{M1}/figures/catalog_o3o4/madgrav_far_final_x1.csv")):
    for d in q: q[d].append(float(r["gps"]))
nfam = 0
for r in csv.DictReader(open("$MADGRAV_SCRATCH/madgrav_qt_build/bg_top50/bg_families_all.csv")):
    q["H1"].append(float(r["gps_H"])); q["L1"].append(float(r["gps_L"])); nfam += 1
Q = {d: np.unique(np.array(v)) for d, v in q.items()}
print(f"[quarantine] H1 {len(Q['H1'])} / L1 {len(Q['L1'])} anchors ({nfam} family legs per det, catalog + detections both dets); +-{QUAR_S} s", flush=True)
def quarantined(det, gps):
    a = Q[det]; i = np.searchsorted(a, gps); lo = np.clip(i - 1, 0, len(a) - 1); hi = np.clip(i, 0, len(a) - 1)
    return (np.abs(a[lo] - gps) <= QUAR_S) | (np.abs(a[hi] - gps) <= QUAR_S)
# ---------------- sources ----------------
def src_glitch():
    lab = {}
    for r in csv.DictReader(open(f"{M1}/dev_v2/glitch_arm/retrain_v2/relabel/mined_labels.csv")): lab[(r["det"], r["seg"], int(r["j"]))] = r["label"]
    for d in ("mined", "mined_o3b", "mined_o4a", "mined_o4b"):
        for f in sorted(glob.glob(f"{M1}/dev_v2/glitch_arm/retrain_v2/{d}/*shard*.npz"))[: (1 if SMOKE else None)]:
            z = np.load(f, allow_pickle=True); det = str(z["det"]); n = len(z["qt"])
            yield z["qt"], dict(det=np.full(n, det), gps=z["gps"].astype(float), run=np.array([s.split("_")[0] for s in z["seg"]]), seg=z["seg"],
                                label=np.array([lab.get((det, s, int(j)), "glitch") for s, j in zip(z["seg"], z["j"])]), sigma=z["sigma"].astype(float), src=np.full(n, os.path.basename(f)))
def src_murky():
    for f in sorted(glob.glob(f"{M1}/dev_v2/g8/murky/murky_*_shard*.npz"))[: (1 if SMOKE else None)]:
        z = np.load(f, mmap_mode="r"); b = os.path.basename(f).split("_"); det = b[2]; n = z["qt"].shape[0]
        t0 = np.array([float(s.split("_")[1]) for s in z["seg"]]); gps = t0 + np.asarray(z["j"], float) * 1.0 + 2.0
        yield np.asarray(z["qt"]), dict(det=np.full(n, det), gps=gps, run=np.full(n, b[1]), seg=np.asarray(z["seg"]), label=np.full(n, "murky"), sigma=np.asarray(z["sigma"], float), src=np.full(n, os.path.basename(f)))
def src_sig():
    for f in sorted(f for f in glob.glob(f"{os.environ.get('ARM_POS', f'{SC}/madgrav2/positives')}/pos_*_shard*.npz") if "smoke" not in f)[: (1 if SMOKE else None)]:   # 2026-09-28: ARM_POS = alternative positives dir (off5 retrain)
        z = np.load(f, allow_pickle=True); run = os.path.basename(f).split("_")[1]
        for d, u in (("H1", "H"), ("L1", "L")):
            keep = (np.asarray(z["pass_net"]) == 1) if SIG_RULE == "pair4" else (np.asarray(z[f"sigma_{u}"]) >= 4.0); n = int(keep.sum())
            if not n: continue
            yield np.asarray(z[f"qt_{u}"])[keep], dict(det=np.full(n, d), gps=np.asarray(z[f"gps_{u}"], float)[keep], run=np.full(n, run), seg=np.asarray(z[f"seg_{u}"])[keep],
                                                    label=np.array([f"sig_mt{m:.0f}" for m in np.asarray(z["mtot"])[keep]]), sigma=np.asarray(z[f"sigma_{u}"], float)[keep], src=np.full(n, os.path.basename(f) + d),
                                                    mtot=np.asarray(z["mtot"], float)[keep], snr=np.asarray(z["snr_target"], float)[keep])
# ---------------- assemble: two passes (count, then write memmaps) ----------------
def src_synth_glitch():
    """2026-09-29: synthetic glitch-only tiles (gen_positives_contam.py with M2_NO_SIGNAL=1: GlitchGen glitch on a quiet carrier, no chirp)
    added to the GLITCH class when ARM_SYNTH_GLITCH=<dir>; label = glitch_synth_<GlitchGen class>. Both legs of every pair are tiles."""
    d = os.environ.get("ARM_SYNTH_GLITCH")
    if not d: return
    for f in sorted(f for f in glob.glob(f"{d}/pos_*_shard*.npz") if "smoke" not in f)[: (1 if SMOKE else None)]:
        z = np.load(f, allow_pickle=True); run = os.path.basename(f).split("_")[1]; assert int(z["no_signal"]) == 1, f
        for dd, u in (("H1", "H"), ("L1", "L")):
            keep = np.asarray(z[f"gcls_{u}"]).astype(str) != ""; n = int(keep.sum())
            if not n: continue
            yield np.asarray(z[f"qt_{u}"])[keep], dict(det=np.full(n, dd), gps=np.asarray(z[f"gps_{u}"], float)[keep], run=np.full(n, run), seg=np.asarray(z[f"seg_{u}"])[keep],
                                                    label=np.array(["glitch_synth_" + c for c in np.asarray(z[f"gcls_{u}"]).astype(str)[keep]]), sigma=np.asarray(z[f"sigma_{u}"], float)[keep], src=np.full(n, os.path.basename(f) + dd))
import itertools
SRCS = {"sig": src_sig, "glitch": lambda: itertools.chain(src_glitch(), src_synth_glitch()), "murky": src_murky}   # synth glitches join the glitch class when ARM_SYNTH_GLITCH is set
sfx = "_smoke" if SMOKE else ""; report = []
for cls, gen in SRCS.items():
    t0 = time.time(); metas = {0: [], 1: []}; total = quar = 0
    # pass 1: metadata, quarantine, fold
    chunks = []
    for qt, m in gen():
        day = (m["gps"] // 86400).astype(int); fold = np.array([fold_of(d) for d in day]); cal = np.array([calib_of(d) for d in day])
        qm = quarantined(m["det"][0], m["gps"]) if len(set(m["det"])) == 1 else np.array([quarantined(d, g) for d, g in zip(m["det"], m["gps"])])
        total += len(day); quar += int(qm.sum()); keep = ~qm
        chunks.append((m["src"][0], keep, fold, cal, day, {k: v for k, v in m.items()}))
    counts = {f: int(sum((c[1] & (c[2] == f)).sum() for c in chunks)) for f in (0, 1)}
    print(f"[{cls}] {total} tiles, quarantined {quar}, fold0 {counts[0]} fold1 {counts[1]} (meta pass {(time.time()-t0)/60:.1f} min)", flush=True)
    mm = {f: np.lib.format.open_memmap(f"{OUT}/{cls}_f{f}{sfx}.npy", mode="w+", dtype=np.float16, shape=(counts[f], 256, 128)) for f in (0, 1)}
    pos = {0: 0, 1: 0}; meta_out = {f: {k: [] for k in ("det", "gps", "day", "run", "seg", "label", "sigma", "src", "calib", "mtot", "snr")} for f in (0, 1)}
    # pass 2: write tiles (re-read the source shards in the same order)
    gi = gen()
    for (src, keep, fold, cal, day, m), (qt, m2) in zip(chunks, gi):
        assert m2["src"][0] == src
        for f in (0, 1):
            sel = keep & (fold == f); n = int(sel.sum())
            if not n: continue
            mm[f][pos[f]:pos[f] + n] = np.asarray(qt[sel], np.float16); pos[f] += n
            for k in ("det", "gps", "run", "seg", "label", "sigma", "src"): meta_out[f][k].append(np.asarray(m[k])[sel])
            meta_out[f]["day"].append(day[sel]); meta_out[f]["calib"].append(cal[sel])
            meta_out[f]["mtot"].append(np.asarray(m.get("mtot", np.full(len(sel), np.nan)))[sel]); meta_out[f]["snr"].append(np.asarray(m.get("snr", np.full(len(sel), np.nan)))[sel])
    for f in (0, 1):
        assert pos[f] == counts[f]; mm[f].flush(); del mm[f]
        np.savez(f"{OUT}/{cls}_f{f}{sfx}_meta.npz", **{k: np.concatenate(v) if v else np.array([]) for k, v in meta_out[f].items()})
        md = meta_out[f]; det = np.concatenate(md["det"]); run = np.concatenate(md["run"]); cal = np.concatenate(md["calib"])
        report.append(f"{cls} fold{f}: n={counts[f]} H1 {int((det=='H1').sum())} L1 {int((det=='L1').sum())} | " + " ".join(f"{r}:{int((run==r).sum())}" for r in np.unique(run)) + f" | calib {int(cal.sum())}")
    print(f"[{cls}] written in {(time.time()-t0)/60:.1f} min", flush=True)
open(f"{OUT}/ASSEMBLY{sfx}.txt", "w").write("\n".join(report) + "\n"); print("\n".join(report), flush=True)

"""Cross-fit scoring of the FAR-setting families and the catalog events with the MADGRAV2 arm.
Each leg is scored by the fold model(s) that never saw its UTC day (fold_of(day) == F -> use models trained on 1-F),
averaged over seeds. Threshold per model = its own held-out t99 (1% signal loss). A leg passes if mean P(sig) >= mean t99;
a family (pair) is removed if either leg fails. Reports: 532 families removed (alone / with centroid 220 Hz), per run,
per detection N (via the 'outranks' column), and the watchlist: catalog adopted events all-legs-pass.
Usage: python score_families.py [ARM_TAG]   (default _pilot)"""
import os, sys, glob, json, re, hashlib, collections, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("ARM_SMOKE", "0")
from gwpy.time import to_gps
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")

TAG = sys.argv[1] if len(sys.argv) > 1 else "_pilot"; MOD = MADGRAV_SCRATCH + "/madgrav2/arm/models"; CEN = MADGRAV_SCRATCH + "/madgrav2/centroid"
def fold_of(day): return int(hashlib.sha1(str(int(day)).encode()).hexdigest(), 16) % 2
# --- models ---
import importlib.util

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_model.py")).read())
models = {0: [], 1: []}; t99 = {0: [], 1: []}
for js in sorted(glob.glob(f"{MOD}/arm{TAG}_f*_s*.json")):
    if "smoke" in js: continue
    S = json.load(open(js)); m = Arm(); m.load_state_dict(torch.load(js.replace(".json", ".pt"), map_location="cpu")); m.eval()
    models[S["fold"]].append(m); t99[S["fold"]].append(S["t99"])
print("models per training fold:", {f: len(v) for f, v in models.items()}, "t99:", {f: np.round(v, 4).tolist() for f, v in t99.items()})
assert models[0] and models[1], "need at least one model per fold for cross-fitting"
def psig(fold_of_leg, tiles):
    """score tiles whose day-fold is fold_of_leg with the models trained on the other fold; return mean P(sig), mean t99"""
    F = 1 - fold_of_leg; x = torch.from_numpy(np.asarray(tiles, np.float32)[:, None]); out = []
    with torch.no_grad():
        for m in models[F]: out.append(torch.cat([m.psig(x[i:i + 256]) for i in range(0, len(x), 256)]).numpy())
    return np.mean(out, 0), float(np.mean(t99[F]))
# --- families ---
ft = np.load(f"{CEN}/fam532_tiles.npz", allow_pickle=True); fm = np.load(f"{CEN}/feat_fam532.npz", allow_pickle=True)
assert np.array_equal(ft["fam"], fm["fam"]) and np.array_equal(ft["det"], fm["det"])
day = (fm["gps"] // 86400).astype(int); folds = np.array([fold_of(d) for d in day]); P = np.empty(len(day)); T = np.empty(len(day))
for f in (0, 1):
    m = folds == f; P[m], T[m] = psig(f, ft["qt"][m])
leg_pass = P >= T; cen_pass = ~(np.isfinite(fm["clu98"]) & (fm["clu98"] >= 220.0))
fam = collections.OrderedDict()
for i, k in enumerate(fm["fam"]):
    e = fam.setdefault(k, dict(run=fm["run"][i], out=str(fm["outranks"][i]), arm=True, cen=True, legs=[]))
    e["arm"] &= bool(leg_pass[i]); e["cen"] &= bool(cen_pass[i]); e["legs"].append((fm["det"][i], float(P[i]), float(fm["clu98"][i])))
n = len(fam); ra = sum(not e["arm"] for e in fam.values()); rc = sum(not e["cen"] for e in fam.values()); rb = sum(not (e["arm"] and e["cen"]) for e in fam.values())
print(f"\n532 FAR-setting families: removed by arm{TAG} (1% loss cross-fit) {ra}/{n}, by centroid<220 {rc}/{n}, by either {rb}/{n}")
for r in ("O3a", "O3b", "O4a", "O4b"):
    es = [e for e in fam.values() if e["run"] == r]; print(f"  {r}: n={len(es)} arm {sum(not e['arm'] for e in es)} cen {sum(not e['cen'] for e in es)} either {sum(not (e['arm'] and e['cen']) for e in es)}")
det = collections.defaultdict(list)
for e in fam.values():
    for tok in e["out"].split(";"):
        mm = re.match(r"\s*(GW[0-9_]+)", tok)
        if mm: det[mm.group(1)].append(e)
tot = {k: 0 for k in ("none", "arm", "cen", "both")}; zero = {k: 0 for k in tot}
print("\n| detection | run | N | after arm | after centroid | after both |"); print("|---|---|---|---|---|---|")
for name, es in sorted(det.items(), key=lambda x: (x[1][0]["run"], x[0])):
    a = sum(e["arm"] for e in es); c = sum(e["cen"] for e in es); b = sum(e["arm"] and e["cen"] for e in es)
    for k, v in (("none", len(es)), ("arm", a), ("cen", c), ("both", b)): tot[k] += v; zero[k] += (v == 0)
    print(f"| {name} | {es[0]['run']} | {len(es)} | {a} | {c} | {b} |")
print(f"\nsummed N over detections: none {tot['none']} | arm {tot['arm']} | centroid {tot['cen']} | both {tot['both']}; detections at N=0: arm {zero['arm']} cen {zero['cen']} both {zero['both']} of {len(det)} (+7 already)")
# --- catalog events (watchlist) ---
ct = np.load(f"{CEN}/cat_tiles.npz", allow_pickle=True); cf = np.load(f"{CEN}/feat_cat.npz", allow_pickle=True)
assert np.array_equal(ct["name"], cf["name"]) and np.array_equal(ct["det"], cf["det"])
def day_of_name(nm):
    m = re.match(r"GW(\d{2})(\d{2})(\d{2})", nm); return int(float(to_gps(f"20{m[1]}-{m[2]}-{m[3]}")) // 86400)
cday = np.array([day_of_name(nm) for nm in cf["name"]]); cfold = np.array([fold_of(d) for d in cday]); CP = np.empty(len(cday)); CT = np.empty(len(cday))
for f in (0, 1):
    m = cfold == f; CP[m], CT[m] = psig(f, ct["qt"][m])
cpass = CP >= CT; ad = cf["adopted"].astype(bool); ev = collections.OrderedDict()
for i, nm in enumerate(cf["name"]): e = ev.setdefault(nm, dict(ad=bool(ad[i]), ok=True, legs=[])); e["ok"] &= bool(cpass[i]); e["legs"].append((cf["det"][i], float(CP[i]), float(CT[i]), float(cf["snr_det"][i])))
n_ad = sum(e["ad"] for e in ev.values()); ok_ad = sum(e["ad"] and e["ok"] for e in ev.values()); ok_ms = sum((not e["ad"]) and e["ok"] for e in ev.values()); n_ms = sum(not e["ad"] for e in ev.values())
print(f"\nWATCHLIST: adopted events with both legs passing the arm: {ok_ad}/{n_ad}; missed catalog events passing: {ok_ms}/{n_ms}")
for nm, e in ev.items():
    if e["ad"] and not e["ok"]: print("  FAIL adopted:", nm, ", ".join(f"{d} P={p:.3f} (t99 {t:.3f}, snr {s:.1f})" for d, p, t, s in e["legs"]))
lo = sorted([(min(p for _, p, _, _ in e["legs"]), nm) for nm, e in ev.items() if e["ad"]])[:6]; print("  lowest adopted leg P(sig):", ", ".join(f"{nm} {p:.3f}" for p, nm in lo))
json.dump(dict(tag=TAG, fam_removed=dict(arm=ra, cen=rc, either=rb, n=n), summed_N=tot, watchlist=dict(adopted_pass=ok_ad, adopted=n_ad, missed_pass=ok_ms, missed=n_ms)), open(f"{MOD}/score_families{TAG}.json", "w"), indent=1)

#!/usr/bin/env python
"""Real-event CROSS-RECOVERY MATRIX - the external arbiter for the
four-epoch VT comparison (audit item 3, design decision 2026-08-12).

Per catalog event in our mass range (Mtot >= 20) that lies inside our
SCANNED H1L1-coincident segments: found/missed at FAR < 1/yr per pipeline
vs MADGRAV. If the VT ratios are right, the per-pipeline recovered counts
on this SHARED event set must order the same way.

Sources (all local):
  events        figures/catalog_o3o4/cat_<run>_v2.csv (GPS parsed from the
                GWyymmdd_hhmmss name; +-1 s is ample for segment membership)
  segments      <run>_full_coincident.json restricted to the scanned set
                (bg_cache seg_names, as in vt_search T_obs)
  MADGRAV       search_out_<run>_far{_f40}/detections.json (gps +-4 s)
  cWB           cwb_far column of the CSVs (GWTC tables; per-year units)
  best-pipeline far_min column (catalog minimum FAR, per-year)
  O4 pipelines  GWTC-4.1 (O4a) / GWTC-5.0 (O4b) SearchSummaryTable.hdf5
                per-pipeline groups, matched by gw_name; combined_far in Hz
                (verified: max == 2/day == 2.31e-5 Hz)
  O3 pipelines  GWTC-2.1 / GWTC-3 search-data tarballs: per-candidate
                LIGOLW xmls in pycbc_all_sky / pycbc_highmass /
                gstlal_all(sky|_sky) / mbta_all_sky, matched by filename
                GPS +-2 s; coinc_inspiral combined_far parsed from the xml
                (Hz). cWB has no O3 xmls -> covered by the CSV column.

Out: cross_recovery_matrix.{json,csv} + summary table on stdout.
Run: madgrav-venv python build_cross_recovery.py   (login-light, ~minutes)
"""
import csv
import glob
import json
import os
import re
import tarfile
from datetime import datetime, timezone

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)


MG = MADGRAV_ROOT
SC = MADGRAV_SCRATCH
EA = MADGRAV_EXTDATA + "/gwosc_eventapi"
HERE = f"{MG}/search_mode/pastro_final"
RUNS = ["o3a", "o3b", "o4a", "o4b"]
MASS_EDGES = np.array([20., 40., 60., 80., 100., 130., 160., 200., 260., 330., 400.])
FAR_YR = 1.0
FAR_HZ = FAR_YR / (365.25 * 86400.0)
GPS_UNIX = 315964800
LEAP = 18
O3_TARBALL = {"o3a": f"{EA}/search_data_GWTC2p1.tar.gz",
              "o3b": f"{EA}/search_data_GWTC3.tar.gz"}
O4_TABLE = {"o4a": f"{EA}/GWTC4p1_SearchSummaryTable.hdf5",
            "o4b": f"{EA}/GWTC5p0_SearchSummaryTable.hdf5"}
O3_PIPES = {"pycbc_all_sky": "PyCBC-broad", "pycbc_highmass": "PyCBC-BBH",
            "gstlal_all_sky": "GstLAL", "gstlal_allsky": "GstLAL",
            "mbta_all_sky": "MBTA"}
O4_PIPES = {"pycbc": "PyCBC", "gstlal": "GstLAL", "MBTA": "MBTA", "CWB": "cWB"}


def name_to_gps(name):
    try:
        d = datetime.strptime(name.strip()[2:], "%y%m%d_%H%M%S")
        return d.replace(tzinfo=timezone.utc).timestamp() - GPS_UNIX + LEAP
    except ValueError:
        return None


def ligolw_combined_far(xml_bytes):
    """Minimal LIGOLW extraction: combined_far column of coinc_inspiral."""
    txt = xml_bytes.decode("utf-8", "replace")
    m = re.search(r'<Table Name="coinc_inspiral[^"]*"(.*?)</Table>', txt, re.S)
    if not m:
        return None
    blk = m.group(1)
    cols = re.findall(r'<Column[^>]*Name="([^"]+)"', blk)
    idx = [i for i, c in enumerate(cols) if c.endswith("combined_far")]
    if not idx:
        return None
    s = re.search(r"<Stream[^>]*>(.*?)</Stream>", blk, re.S)
    if not s:
        return None
    fars = []
    for row in s.group(1).strip().splitlines():
        row = row.strip().rstrip(",")
        if not row:
            continue
        # quoted fields (e.g. ifos "H1,L1,V1") carry embedded commas ->
        # must split CSV-aware, not on bare commas
        parts = next(csv.reader([row]))
        if len(parts) == len(cols):
            try:
                fars.append(float(parts[idx[0]].strip()))
            except ValueError:
                pass
    return min(fars) if fars else None


def o3_pipeline_fars(run, ev_gps_list):
    """{event_gps: {pipe: min_far_hz}} by tarball scan (filename GPS +-2 s)."""
    want = np.array(sorted(ev_gps_list))
    out = {g: {} for g in ev_gps_list}
    pat = re.compile(r"-(1[23]\d{8})-")
    with tarfile.open(O3_TARBALL[run], "r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".xml")):
                continue
            parts = m.name.split("/")
            if len(parts) != 3:          # skip skymaps_ligolw subdirs
                continue
            pipe = O3_PIPES.get(parts[1])
            if pipe is None:
                continue
            hit = pat.search(parts[-1])
            if not hit:
                continue
            fgps = float(hit.group(1))
            j = np.searchsorted(want, fgps)
            near = [g for g in want[max(0, j - 1):j + 2] if abs(g - fgps) <= 2.0]
            if not near:
                continue
            far = ligolw_combined_far(tf.extractfile(m).read())
            if far is None:
                continue
            for g in near:
                prev = out[float(g)].get(pipe)
                out[float(g)][pipe] = far if prev is None else min(prev, far)
    return out


def o4_pipeline_fars(run):
    """{gw_name: {pipe: min_far_hz}} from the SearchSummaryTable."""
    import h5py
    out = {}
    with h5py.File(O4_TABLE[run], "r") as f:
        for key, pipe in O4_PIPES.items():
            if key not in f:
                continue
            t = f[key][()]
            far_max = float(np.max(np.asarray(t["combined_far"], float)))
            assert far_max < 3e-5, f"{run}/{key}: far units not Hz? max={far_max}"
            for r in t:
                name = r["gw_name"]
                name = name.decode() if isinstance(name, bytes) else name
                if not name:
                    continue
                d = out.setdefault(name, {})
                v = float(r["combined_far"])
                d[pipe] = min(d.get(pipe, np.inf), v)
    return out


def main():
    rows = []
    summary = {}
    for run in RUNS:
        segj = json.load(open(f"{SC}/{run}_full_coincident.json"))
        names = set(np.load(
            f"{SC}/search_out_{run}_far"
            f"{'_f40' if run in ('o3a', 'o3b') else ''}/bg_cache_{run}.npz")["seg_names"])
        scanned = [(s[0], s[1]) for s in segj["segments"] if s[3] in names]
        dets = json.load(open(
            f"{SC}/search_out_{run}_far"
            f"{'_f40' if run in ('o3a', 'o3b') else ''}/detections.json"))
        if isinstance(dets, dict):
            dets = dets.get("detections", dets.get("dets", []))
        det_gps = np.array([d["gps"] for d in dets])

        events = []
        n_below, n_nomass, n_unscanned = 0, 0, 0
        for r in csv.DictReader(open(f"{MG}/figures/catalog_o3o4/cat_{run}_v2.csv")):
            g = name_to_gps(r["name"])
            mt = float(r["total_mass_source"]) if r["total_mass_source"] else None
            if mt is None:
                n_nomass += 1
                continue
            if mt < MASS_EDGES[0]:
                n_below += 1
                continue
            if g is None or not any(t0 <= g <= t1 for t0, t1 in scanned):
                n_unscanned += 1
                continue
            events.append(dict(name=r["name"], gps=g, mtot=mt,
                               snr=float(r["network_snr"]) if r["network_snr"] else None,
                               far_min=float(r["far_min"]) if r["far_min"] else None,
                               cwb_far=float(r["cwb_far"]) if r["cwb_far"] else None))
        if run in O3_TARBALL:
            pf = o3_pipeline_fars(run, [e["gps"] for e in events])
            for e in events:
                e["pipe_far_hz"] = pf.get(e["gps"], {})
        else:
            byname = o4_pipeline_fars(run)
            for e in events:
                e["pipe_far_hz"] = byname.get(e["name"], {})

        pipes = sorted({p for e in events for p in e["pipe_far_hz"]})
        cnt = {p: 0 for p in pipes}
        cnt.update(MADGRAV=0, cWB_csv=0, best_catalog=0)
        for e in events:
            e["run"] = run
            e["madgrav"] = bool(np.any(np.abs(det_gps - e["gps"]) <= 4.0)) \
                if len(det_gps) else False
            e["cwb_1yr"] = (e["cwb_far"] is not None and e["cwb_far"] < FAR_YR)
            e["best_1yr"] = (e["far_min"] is not None and e["far_min"] < FAR_YR)
            e["pipe_1yr"] = {p: (f < FAR_HZ) for p, f in e["pipe_far_hz"].items()}
            cnt["MADGRAV"] += e["madgrav"]
            cnt["cWB_csv"] += e["cwb_1yr"]
            cnt["best_catalog"] += e["best_1yr"]
            for p, hit in e["pipe_1yr"].items():
                cnt[p] += hit
            rows.append(e)
        summary[run] = dict(n_events=len(events), counts=cnt,
                            skipped=dict(no_mass=n_nomass, below_20=n_below,
                                         not_in_scanned_segments=n_unscanned))
        print(f"[{run}] {len(events)} events in range+scanned "
              f"(skipped: {n_nomass} no-mass, {n_below} Mtot<20, "
              f"{n_unscanned} outside scanned segs)", flush=True)
        print(f"  found@1/yr: {cnt}", flush=True)

    # per-mass-bin arbiter table
    print("\nPer-mass-bin found@1/yr (all runs pooled):", flush=True)
    hdr = ["bin", "N", "MADGRAV", "cWB_csv", "best_catalog",
           "PyCBC-BBH", "PyCBC-broad", "PyCBC", "GstLAL", "MBTA", "cWB"]
    print("  " + "  ".join(f"{h:>12s}" for h in hdr), flush=True)
    binrows = []
    for k in range(len(MASS_EDGES) - 1):
        sel = [e for e in rows if MASS_EDGES[k] <= e["mtot"] < MASS_EDGES[k + 1]]
        rec = {"bin": f"{MASS_EDGES[k]:.0f}-{MASS_EDGES[k+1]:.0f}",
               "N": len(sel),
               "MADGRAV": sum(e["madgrav"] for e in sel),
               "cWB_csv": sum(e["cwb_1yr"] for e in sel),
               "best_catalog": sum(e["best_1yr"] for e in sel)}
        for p in ("PyCBC-BBH", "PyCBC-broad", "PyCBC", "GstLAL", "MBTA", "cWB"):
            rec[p] = sum(e["pipe_1yr"].get(p, False) for e in sel)
        binrows.append(rec)
        print("  " + "  ".join(f"{str(rec[h]):>12s}" for h in hdr), flush=True)

    json.dump({"far_threshold_yr": FAR_YR, "events": rows, "summary": summary,
               "per_bin": binrows},
              open(f"{HERE}/cross_recovery_matrix.json", "w"), indent=1,
              default=bool)
    with open(f"{HERE}/cross_recovery_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "name", "mtot", "snr", "madgrav", "cwb_1yr",
                    "best_1yr", "pipe_1yr", "far_min", "cwb_far"])
        for e in rows:
            w.writerow([e["run"], e["name"], e["mtot"], e["snr"],
                        int(e["madgrav"]), int(e["cwb_1yr"]), int(e["best_1yr"]),
                        ";".join(f"{p}:{int(v)}" for p, v in
                                 sorted(e["pipe_1yr"].items())),
                        e["far_min"], e["cwb_far"]])
    print(f"\n[matrix] -> {HERE}/cross_recovery_matrix.json/.csv", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the STEP-3d exclusion catalog (design decision 2026-08-12):
UNION of the GWTC releases' candidate lists at the 2/day threshold PLUS our
own MADGRAV detections.

Sources (all local under $MADGRAV_EXTDATA/gwosc_eventapi/ + scratch):
  * GWTC-4.0 / 4.1 / 5.0 Candidate data releases: SearchSummaryTable.hdf5
    /search_summary gps_time (the releases' candidate lists ARE the FAR<2/day
    sets; a defensive far <= 730.5/yr filter is applied where the field
    exists).
  * GWTC-2.1 / GWTC-3 search-data tarballs (search_data_GWTC2p1.tar.gz,
    search_data_GWTC3.tar.gz): per-pipeline subthreshold candidate tables at
    FAR < 2/day, GPS extracted by walk (hdf5/csv/txt members).
  * GWOSC eventapi jsons: GWTC-2.1-confident/-marginal/-auxiliary,
    GWTC-3-confident/-marginal, O3_IMBH_marginal, GWTC-4.0/4.1/5.0,
    O3/O4_Discovery_Papers (GPS field).
  * OUR detections: search_out_<run>_far{_f40}/detections.json gps (9/7/14/17
    = superset of the quoted 44; the 3 extra O3b entries are pre-prune
    triggers - excluding them too is conservative-safe).

Output: search_mode/exclusion_2perday.json {gps: sorted unique, provenance}.
Run: madgrav-venv python build_exclusion_2perday.py [--no-tarballs]
(--no-tarballs validates everything except the O3 archives while they
download; the final run must include them.)
"""
import glob
import io
import json
import os
import sys
import tarfile

import numpy as np
import os as _os
MADGRAV_ROOT = _os.environ.get("MADGRAV_ROOT") or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
MADGRAV_SCRATCH = _os.environ.get("MADGRAV_SCRATCH") or _os.path.join(MADGRAV_ROOT, "scratch")
MADGRAV_EXTDATA = _os.environ.get("MADGRAV_EXTDATA") or _os.path.dirname(MADGRAV_ROOT)


EA = MADGRAV_EXTDATA + "/gwosc_eventapi"
SC = MADGRAV_SCRATCH
SM = MADGRAV_ROOT + "/search_mode"
FAR_2PDAY_YR = 2.0 * 365.25

prov = {}
gps_all = []


def add(tag, vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    prov[tag] = len(vals)
    gps_all.extend(vals)
    print(f"  {tag}: {len(vals)}", flush=True)


def eventapi():
    for cat in ("GWTC-2.1-confident", "GWTC-2.1-marginal", "GWTC-2.1-auxiliary",
                "GWTC-3-confident", "GWTC-3-marginal", "O3_IMBH_marginal",
                "GWTC-4.0", "GWTC-4.1", "GWTC-5.0",
                "O3_Discovery_Papers", "O4_Discovery_Papers"):
        p = f"{EA}/{cat}.json"
        ev = json.load(open(p)).get("events", {})
        add(f"eventapi:{cat}", [v.get("GPS") for v in ev.values()])


def o4_tables():
    import h5py
    for tag in ("GWTC4p0", "GWTC4p1", "GWTC5p0"):
        with h5py.File(f"{EA}/{tag}_SearchSummaryTable.hdf5", "r") as f:
            t = f["search_summary"][()]
            far = np.asarray(t["far"], float)
            g = np.asarray(t["gps_time"], float)
            keep = ~np.isfinite(far) | (far <= FAR_2PDAY_YR)
            add(f"table:{tag}/search_summary", g[keep])


def o3_tarballs():
    """The O3 search-data archives hold one LIGOLW xml (+skymap fits/json)
    PER CANDIDATE, with the integer trigger GPS in the member name (e.g.
    search_data_products/pycbc_all_sky/H1L1-PYCBC_AllSky-1253537236-1.xml).
    Second precision is ample for +-8 s segment exclusion, so GPS is taken
    from member names - no member decompression needed."""
    import re
    pat = re.compile(r"-(1[23]\d{8})-")
    for tag, tb in (("GWTC2p1", f"{EA}/search_data_GWTC2p1.tar.gz"),
                    ("GWTC3", f"{EA}/search_data_GWTC3.tar.gz")):
        got = []
        with tarfile.open(tb, "r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.lower().endswith(".xml"):
                    hit = pat.search(os.path.basename(m.name))
                    if hit:
                        got.append(float(hit.group(1)))
        if not got:
            raise SystemExit(f"FATAL: no candidate times found in {tb} - "
                             "inspect archive structure manually")
        g = np.asarray(got)
        # O3 valid GPS window sanity (1238166018..1269363618 +margin)
        g = g[(g > 1.2e9) & (g < 1.3e9)]
        add(f"tarball:{tag}", g)


def ours():
    for run, sub in (("o3a", "_f40"), ("o3b", "_f40"), ("o4a", ""), ("o4b", "")):
        d = json.load(open(f"{SC}/search_out_{run}_far{sub}/detections.json"))
        if isinstance(d, dict):
            d = d.get("detections", d.get("dets", []))
        add(f"madgrav:{run}", [r["gps"] for r in d])


def main():
    print("[exclusion] building 2/day union ...", flush=True)
    eventapi()
    o4_tables()
    ours()
    if "--no-tarballs" not in sys.argv:
        o3_tarballs()
    else:
        print("  [SKIP] O3 tarballs (--no-tarballs; NOT final)", flush=True)
    g = np.unique(np.round(np.asarray(gps_all, float), 3))
    out = {"n": len(g), "provenance": prov,
           "final": "--no-tarballs" not in sys.argv,
           "gps": g.tolist()}
    json.dump(out, open(f"{SM}/exclusion_2perday.json", "w"))
    print(f"[exclusion] {len(g)} unique times -> "
          f"{SM}/exclusion_2perday.json (final={out['final']})", flush=True)


if __name__ == "__main__":
    main()

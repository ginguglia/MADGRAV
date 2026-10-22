#!/usr/bin/env python
"""SM_PREP guard — resolves and ENFORCES the per-run whitening-prep convention.

Called by launchers/templates/_common.sh in place of the old name-derived default
    SM_PREP="${SM_PREP:-$MADGRAV_ROOT/data/${SM_RUN}_search_prep}"
which silently mis-prepped o3b twice on 2026-08-22 (decision the same day: "refuse to run o3b jobs
with SM_PREP unset rather than defaulting ... remove the trap").

Contract:
  argv[1] = SM_RUN, argv[2] = SM_PREP from the environment ("" if unset)
  stdout  = the resolved absolute SM_PREP (the ONLY thing on stdout, so the caller can capture it)
  stderr  = the audit line, plus any warning
  exit 0  = resolved and consistent with the convention;  exit 1 = refuse, with the reason on stderr

Rules:
  - run marked strict and SM_PREP unset            -> REFUSE (no default is safe for this run)
  - SM_PREP set but != the convention for this run -> REFUSE (wrong prep beats no prep for silence)
  - run absent from the convention file            -> name-derived default, audited, non-fatal
  - authority document hash drifted                -> WARN loudly; do not block (the prep check itself
    is the safety property; a doc edit should not halt every pipeline)
"""
import os, sys, json, hashlib

ROOT = os.environ.get("MADGRAV_ROOT", "")
CONV = os.path.join(ROOT, "dev_v2/glitch_arm/PREP_CONVENTION.json")

def die(msg):
    sys.stderr.write("[prep] REFUSED — " + msg + "\n"); sys.exit(1)

def main():
    run = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    env = (sys.argv[2] if len(sys.argv) > 2 else "").strip().rstrip("/")
    if not run:
        die("SM_RUN empty")
    if not os.path.isfile(CONV):
        # No convention file: fall back to the old behaviour but say so, loudly.
        res = env or os.path.join(ROOT, f"data/{run}_search_prep")
        sys.stderr.write(f"[prep] WARNING convention file missing ({CONV}); name-derived fallback in use\n")
        sys.stderr.write(f"[prep] SM_RUN={run} SM_PREP={res} (UNCHECKED)\n"); print(res); return
    d = json.load(open(CONV))
    # authority-drift check (warn only)
    auth = os.path.join(ROOT, d.get("_authority", ""))
    want = d.get("_authority_sha256", "")
    if want and os.path.isfile(auth):
        got = hashlib.sha256(open(auth, "rb").read()).hexdigest()
        if got != want:
            sys.stderr.write(f"[prep] WARNING authority {d['_authority']} has changed since this "
                             f"convention was derived (sha {got[:12]} != {want[:12]}) — re-derive "
                             f"PREP_CONVENTION.json\n")
    runs = d.get("runs", {})
    if run not in runs:
        res = env or os.path.join(ROOT, f"data/{run}_search_prep")
        sys.stderr.write(f"[prep] SM_RUN={run} not in convention; name-derived {res} (UNCHECKED)\n")
        print(res); return
    spec = runs[run]
    if spec.get("quarantined") or spec.get("prep") is None:
        # QUARANTINE (2026-08-22): no prep is currently adopted for this run. Refuse both candidates
        # rather than silently enforce the one the same day's evidence voided.
        if os.environ.get("PREP_GUARD_ALLOW_MISMATCH") == "1":
            res = env
            if not res:
                die(f"SM_RUN={run} is QUARANTINED and SM_PREP is unset; the override cannot invent one.")
            sys.stderr.write(f"[prep] *** {run} IS QUARANTINED — no adopted prep. Running under SM_PREP={res} "
                             f"by explicit override. Output is DIAGNOSTIC ONLY and must not be quoted. ***\n"
                             f"[prep] *** reason: {spec.get('reason','')} ***\n")
            print(res); return
        die(f"SM_RUN={run} is QUARANTINED — no prep is currently adopted for it.\n"
            f"           {spec.get('reason','')}\n"
            f"           See {d['_authority']} and PREP_CONVENTION.json _amendment_20260822.\n"
            f"           A deliberate diagnostic must set BOTH SM_PREP and PREP_GUARD_ALLOW_MISMATCH=1.")
    conv = os.path.join(ROOT, spec["prep"].lstrip("/")) if not spec["prep"].startswith("/") else spec["prep"]
    conv = conv.rstrip("/")
    if spec.get("strict") and not env:
        die(f"SM_RUN={run} requires SM_PREP to be set EXPLICITLY — no default is safe for this run.\n"
            f"           Convention ({d['_authority']}): SM_PREP={conv}\n"
            f"           Submit via the run's env file (e.g. dev_v2/glitch_arm/o3b_washout.env), or "
            f"export SM_PREP={conv} before sbatch.")
    res = env or conv
    if os.path.realpath(res) != os.path.realpath(conv) and os.environ.get("PREP_GUARD_ALLOW_MISMATCH") == "1":
        # Deliberate cross-prep DIAGNOSTIC (e.g. the o3b sigma recompute, whose whole purpose is to
        # compare the frozen convention against the name-derived one). Must be set explicitly per job;
        # never set it in a site file or a production template. Logged loudly so the mismatch cannot
        # reach a result page unnoticed.
        sys.stderr.write(f"[prep] *** MISMATCH ALLOWED BY PREP_GUARD_ALLOW_MISMATCH=1 ***\n"
                         f"[prep] *** SM_RUN={run} running under SM_PREP={res}, NOT the convention "
                         f"{conv}. Output is DIAGNOSTIC ONLY and must not be quoted. ***\n")
        print(res); return
    if os.path.realpath(res) != os.path.realpath(conv):
        die(f"SM_PREP={res} contradicts the convention for SM_RUN={run}.\n"
            f"           Convention ({d['_authority']}): SM_PREP={conv}\n"
            f"           Refusing rather than building under a prep the frozen configuration rejects.")
    if not os.path.isdir(res):
        die(f"SM_PREP={res} does not exist as a directory")
    sys.stderr.write(f"[prep] SM_RUN={run} SM_PREP={res} — matches {d['_authority']} "
                     f"(strict={bool(spec.get('strict'))})\n")
    print(res)

main()

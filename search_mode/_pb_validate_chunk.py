"""Validate the two-sided block-nested-loop precompute is BIT-IDENTICAL to the fast (retain-small) path.
Runs precompute_cnn on real bg pairs twice: big budget -> fast path; tiny budget+small stream -> block path.
Requires equal keys and max|hm/lm diff| == 0.0. Uses the SAME setup as the merge (build_setup).
NOTE: all work under `if __name__=='__main__'` so the forkserver QT workers don't re-run build_setup."""
import os, glob, numpy as np


def main():
    import driver_blindscan as B
    import _pb_shard_common as C
    import _pb_cnn_precompute as PB

    S = C.build_setup(); segs = S["segs"]
    SD = os.environ["SM_SHARD_DIR"]
    fs = sorted(glob.glob(f"{SD}/shard_*.npz"))
    pairs = []
    for f in fs[:3]:
        z = np.load(f, allow_pickle=True)
        for r in z["fam_vals"][:1500]:
            pairs.append((int(r[1]), int(r[2]), int(r[3]), int(r[4])))
        for g in (0, 1):
            for r in z[f"bg{g}"][:800]:
                pairs.append((int(r[1]), int(r[2]), int(r[3]), int(r[4])))
    pairs = list(dict.fromkeys(pairs))
    print(f"[validate] {len(pairs)} real pairs", flush=True)

    os.environ["SM_PRECOMPUTE_MAXGB"] = "100000"; os.environ.pop("SM_PRECOMPUTE_STREAMMIN", None)
    c1 = PB.precompute_cnn(pairs, segs, B.cpipe, B.carm, B.cnet, B.lmnet, B._win)   # fast path
    os.environ["SM_PRECOMPUTE_MAXGB"] = "10"; os.environ["SM_PRECOMPUTE_STREAMMIN"] = "8"
    c2 = PB.precompute_cnn(pairs, segs, B.cpipe, B.carm, B.cnet, B.lmnet, B._win)   # two-sided block path

    assert set(c1) == set(c2), f"KEY MISMATCH: {len(set(c1) ^ set(c2))} differ"
    md = max(max(abs(c1[k][0] - c2[k][0]), abs(c1[k][1] - c2[k][1])) for k in c1) if c1 else 0.0
    print(f"[validate] {len(c1)} pairs scored both ways; max|hm/lm diff| = {md}", flush=True)
    print("[validate] RESULT: PASS (bit-identical, 0.0-diff)" if md == 0.0 else f"[validate] RESULT: FAIL diff={md}", flush=True)


if __name__ == "__main__":
    main()

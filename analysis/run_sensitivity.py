"""Run the registered sensitivity analyses S1 and S3 after D1, then decode and summarise them.

Waits for the post pipeline (D3, analyses, D1) to finish, so no two heavy jobs overlap. Then, per
session, extract_sensitivity.py (GPU exponents); then decode within-session -- S1: L2 in each of the
five extra settings; S3: L1, L2 and L5 on raw voltage -- and write results/sensitivity.json comparing
each with its primary counterpart (paired, hierarchical bootstrap). S2 was run separately
(results/s2). Within-session is used because H1-H5 are stated for it (PREREGISTRATION.md section 8).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SRC = os.path.join(HERE, "src")
SESSIONS = [(1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1), (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1)]
S1 = [(128, 1), (512, 1), (128, 2), (256, 2), (512, 2)]


def run(args, log, threads=8):
    env = dict(os.environ, IEEG07_THREADS=str(threads))
    print(f"run {os.path.basename(args[0])} {' '.join(map(str, args[1:3]))}", flush=True)
    with open(os.path.join(HERE, "logs", log), "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, "-u"] + args, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise SystemExit(f"failed ({rc}): {args[0]}; see logs/{log}")


def post_done():
    """D1 is the last heavy job before this one. Its own log is the signal: the post pipeline that
    first launched it failed on 2026-10-01 (an out-of-memory in D1 scoring) and D1 was relaunched
    on its own, so the post pipeline's completion line will never be written."""
    log = os.path.join(HERE, "logs", "run_d1.log")
    return (os.path.exists(log) and "=== D1 done ===" in open(log, encoding="utf-8").read()
            and not os.path.exists("D:/ieeg07/d1.lock"))


def summarise():
    sys.path.insert(0, SRC)
    # isort: off
    import resources  # noqa: F401  MUST load before numpy: caps BLAS threads
    # isort: on
    from analyze import hier_boot, load_decoded, overall
    import pandas as pd
    prim = load_decoded(os.path.join(HERE, "results", "decode"))
    out = {"S1": {}, "S3": {}}
    for smax, order in S1:
        c = pd.concat([prim, load_decoded(os.path.join(HERE, "results", "s1", f"decode_s{smax}_o{order}"))
                       .assign(model=lambda d: d.model + "_alt")])
        a, b = overall(c, "L2_alt", "within"), overall(c, "L2", "within")
        m, lo, hi = hier_boot(a - b, f"S1_{smax}_{order}")
        out["S1"][f"smax{smax}_order{order}"] = dict(alt=float(a.mean()), primary=float(b.mean()), diff=m, ci_lo=lo, ci_hi=hi)
    raw = load_decoded(os.path.join(HERE, "results", "s3", "decode")).assign(model=lambda d: d.model + "_raw")
    c = pd.concat([prim, raw])
    for r in ("L1", "L2", "L5"):
        a, b = overall(c, r + "_raw", "within"), overall(c, r, "within")
        m, lo, hi = hier_boot(a - b, f"S3_{r}")
        out["S3"][r] = dict(raw=float(a.mean()), laplacian=float(b.mean()), diff=m, ci_lo=lo, ci_hi=hi)
    a, b = overall(c, "L2_raw", "within"), overall(c, "L1_raw", "within")
    m, lo, hi = hier_boot(a - b, "S3_L2raw_minus_L1raw")
    out["S3"]["L2_minus_L1_on_raw"] = dict(diff=m, ci_lo=lo, ci_hi=hi)
    with open(os.path.join(HERE, "results", "sensitivity.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)


def main():
    waited = 0
    while not post_done():
        if waited % 1800 == 0:
            print(f"waiting for the post pipeline (D1) ({waited} s)", flush=True)
        time.sleep(60)
        waited += 60
    for s, t in SESSIONS:
        if not os.path.exists(f"D:/ieeg07/features_S3_raw/s{s}_t{t}_meta.json"):
            run([os.path.join(SRC, "extract_sensitivity.py"), "--subject", str(s), "--trial", str(t), "--batch", "256"],
                "sensitivity_extract.log")
    for smax, order in S1:
        run([os.path.join(SRC, "decode.py"), "--rungs", "L2", "--splits", "within",
             "--features", f"D:/ieeg07/features_S1_s{smax}_o{order}",
             "--out", os.path.join(HERE, "results", "s1", f"decode_s{smax}_o{order}")], "sensitivity_decode.log")
    run([os.path.join(SRC, "decode.py"), "--rungs", "L1", "L2", "L5", "--splits", "within",
         "--features", "D:/ieeg07/features_S3_raw", "--out", os.path.join(HERE, "results", "s3", "decode"),
         "--exclude-constant"],                                       # D5: 6 constant raw electrode-windows
        "sensitivity_decode.log", threads=12)
    summarise()
    print("=== sensitivity done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

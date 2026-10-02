"""Run deviation D1 end to end, after the primary decode, never alongside another heavy job.

Waits for the primary decode to finish (decode lock absent and its log's completion line present),
then: score every electrode of every session (d1_scores.py), build the circular and cross-fitted
subsets (d1_build_subsets.py), extract within-session features for both (extract_features.py with
--electrodes-json), decode L1 L2 L4 L5 within-session for both (decode.py), and analyse
(d1_analyze.py). Each step skips work already on disk, so a restart resumes.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SESSIONS = [(1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1), (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1)]
D1 = os.path.join(HERE, "results", "d1")
LOCK = "D:/ieeg07/d1.lock"


def run(args, log, threads):
    env = dict(os.environ, IEEG07_THREADS=str(threads))
    with open(os.path.join(HERE, "logs", log), "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, "-u"] + args, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise SystemExit(f"step failed ({rc}): {' '.join(args)}; see logs/{log}")


def primary_done():
    log = os.path.join(HERE, "logs", "decode_all.log")
    return (not os.path.exists("D:/ieeg07/decode.lock") and os.path.exists(log)
            and "=== all rungs decoded ===" in open(log, encoding="utf-8").read())


def main():
    waited = 0
    # D4 (convergence) runs between D3 and D1 and holds this lock while it waits and runs.
    while os.path.exists("D:/ieeg07/d4.lock"):
        if waited % 1800 == 0:
            print(f"waiting for D4 to finish ({waited} s)", flush=True)
        time.sleep(60)
        waited += 60
    while not primary_done():
        if waited % 1800 == 0:
            print(f"waiting for the primary decode ({waited} s)", flush=True)
        time.sleep(60)
        waited += 60
    if os.path.exists(LOCK):
        raise SystemExit(f"{LOCK} exists")
    open(LOCK, "w").write(str(os.getpid()))
    try:
        for s, t in SESSIONS:
            if not os.path.exists(os.path.join(D1, f"electrode_mean_s{s}_t{t}.csv")):
                print(f"score s{s}_t{t}", flush=True)
                run([os.path.join(HERE, "src", "d1_scores.py"), "--subject", str(s), "--trial", str(t),
                     "--workers", "6"], "d1_scores.log", threads=1)
        print("build subsets", flush=True)
        run([os.path.join(HERE, "src", "d1_build_subsets.py")], "d1_subsets.log", threads=4)
        for kind in ("circular", "crossfit"):
            feat = f"D:/ieeg07/features_d1_{kind}"
            for s, t in SESSIONS:
                if not os.path.exists(os.path.join(feat, f"s{s}_t{t}_meta.json")):
                    print(f"extract {kind} s{s}_t{t}", flush=True)
                    run([os.path.join(HERE, "src", "extract_features.py"), "--subject", str(s), "--trial", str(t),
                         "--electrodes-json", os.path.join(D1, f"subsets_{kind}.json"), "--splits", "within",
                         "--out", feat, "--scratch", "D:/ieeg07/scratch"], f"d1_extract_{kind}.log", threads=12)
            print(f"decode {kind}", flush=True)
            run([os.path.join(HERE, "src", "decode.py"), "--rungs", "L1", "L2", "L4", "L5", "--splits", "within",
                 "--features", feat, "--out", os.path.join(D1, f"decode_{kind}")], f"d1_decode_{kind}.log", threads=12)
        print("analyse", flush=True)
        run([os.path.join(HERE, "src", "d1_analyze.py")], "d1_analyze.log", threads=4)
    finally:
        os.remove(LOCK)
    print("=== D1 done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

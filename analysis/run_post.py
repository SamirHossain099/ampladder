"""After the primary decode: D3's heavy rungs, the analyses and figures, then D1 -- strictly in order.

One heavy job at a time (00-SHARED-CONTEXT.md lesson 8). Waits for the primary decode's completion
line and the absence of its lock, then:
  1. D3 (POST HOC): decode L4 and L5 within-session on the common-time index;
  2. d3_analyze.py on all D3 rungs;
  3. analyze.py (H1-H5, D2 context, reproduction table) and figures.py on the full primary results;
  4. run_d1.py (D1 end to end).
Each step's log is in logs/. Steps that already have output are cheap to repeat (decode.py resumes).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(HERE, ".venv", "Scripts", "python.exe")
SRC = os.path.join(HERE, "src")


def run(args, log, threads=12):
    env = dict(os.environ, IEEG07_THREADS=str(threads))
    print(f"run {os.path.basename(args[0])} -> logs/{log}", flush=True)
    with open(os.path.join(HERE, "logs", log), "a", encoding="utf-8") as fh:
        rc = subprocess.run([PY, "-u"] + args, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        raise SystemExit(f"step failed ({rc}): {args[0]}; see logs/{log}")


def primary_done():
    log = os.path.join(HERE, "logs", "decode_all.log")
    return (not os.path.exists("D:/ieeg07/decode.lock") and os.path.exists(log)
            and "=== all rungs decoded ===" in open(log, encoding="utf-8").read())


def main():
    waited = 0
    while not primary_done():
        if waited % 1800 == 0:
            print(f"waiting for the primary decode ({waited} s)", flush=True)
        time.sleep(60)
        waited += 60
    run([os.path.join(SRC, "decode.py"), "--index", os.path.join(HERE, "results", "d3", "index_common_time.csv.gz"),
         "--splits", "within", "--rungs", "L4", "L5", "--out", os.path.join(HERE, "results", "d3", "decode")],
        "d3_decode_heavy.log")
    run([os.path.join(SRC, "d3_analyze.py")], "d3_analyze.log", threads=4)
    run([os.path.join(SRC, "analyze.py")], "analyze.log", threads=4)
    run([os.path.join(SRC, "figures.py")], "figures.log", threads=4)
    print("=== primary analysis and D3 done ===", flush=True)
    run([os.path.join(SRC, "run_d1.py")], "run_d1.log", threads=4)
    print("=== post pipeline done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

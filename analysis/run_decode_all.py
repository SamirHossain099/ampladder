"""Decode every rung on every split once all 12 sessions are extracted -- never alongside extraction.

Waits until (a) the extraction lock is gone and (b) every session's `_meta.json` exists, so the two
heavy jobs never overlap (00-SHARED-CONTEXT.md lesson 8). Then runs `decode.py` one rung at a time,
cheapest first, so the light rungs' results exist even if a heavy one is interrupted; `decode.py`
appends per cell and skips cells already written, so a restart resumes rather than repeats.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS = [(1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1), (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="D:/ieeg07/features")
    ap.add_argument("--extract-lock", default="D:/ieeg07/extract.lock")
    ap.add_argument("--lock", default="D:/ieeg07/decode.lock")
    ap.add_argument("--rungs", nargs="+", default=["L0", "L1", "L2", "L3", "L4", "L5"])
    ap.add_argument("--threads", default="12")
    a = ap.parse_args()

    waited = 0
    while True:
        missing = [f"s{s}_t{t}" for s, t in SESSIONS
                   if not os.path.exists(os.path.join(a.features, f"s{s}_t{t}_meta.json"))]
        if not missing and not os.path.exists(a.extract_lock):
            break
        if waited % 900 == 0:
            print(f"waiting: {len(missing)} sessions not extracted; extract lock "
                  f"{'present' if os.path.exists(a.extract_lock) else 'absent'} ({waited} s)", flush=True)
        time.sleep(60)
        waited += 60

    if os.path.exists(a.lock):
        raise SystemExit(f"{a.lock} exists: another decode is running")
    with open(a.lock, "w") as fh:
        fh.write(str(os.getpid()))
    try:
        py = os.path.join(HERE, ".venv", "Scripts", "python.exe")
        env = dict(os.environ, IEEG07_THREADS=a.threads)
        for rung in a.rungs:
            log = os.path.join(HERE, "logs", f"decode_{rung}.log")
            print(f"{rung}: decoding", flush=True)
            with open(log, "a", encoding="utf-8") as fh:
                rc = subprocess.run([py, "-u", os.path.join(HERE, "src", "decode.py"), "--rungs", rung],
                                    stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
            if rc != 0:
                raise SystemExit(f"{rung}: decode failed ({rc}); see {log}")
            print(f"{rung}: done", flush=True)
    finally:
        os.remove(a.lock)
    print("=== all rungs decoded ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

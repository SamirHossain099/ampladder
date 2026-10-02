"""Extract features for all 12 Lite sessions, one at a time, as each download lands.

Project rule (00-SHARED-CONTEXT.md, lesson 8): never run two heavy jobs concurrently; run steps
sequentially behind a lock and a RAM gate. So this driver runs exactly one `extract_features.py` at a
time. It gives that one job more BLAS threads (default 12 of 20 cores) because the batched estimator
is matmul-bound; RAM use does not grow with threads.

For each session, in download order:
  1. wait until MANIFEST.jsonl records its zip as hashed and extracted (polls; never guesses);
  2. skip it if its `_meta.json` already exists (resumable);
  3. run the extractor under the lock, logging to logs/extract_<tag>.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = [(1, 1), (1, 2), (10, 0), (10, 1), (2, 0), (2, 4), (3, 0), (3, 1), (4, 0), (4, 1), (7, 0), (7, 1)]


def extracted(manifest, name):
    if not os.path.exists(manifest):
        return False
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r["name"] == name and r.get("extracted"):
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="D:/ieeg07/braintreebank")
    ap.add_argument("--features", default="D:/ieeg07/features")
    ap.add_argument("--threads", default="12")
    ap.add_argument("--lock", default="D:/ieeg07/extract.lock")
    a = ap.parse_args()

    if os.path.exists(a.lock):
        raise SystemExit(f"lock {a.lock} exists: another extraction is running (or crashed; check, then delete)")
    with open(a.lock, "w") as fh:
        fh.write(str(os.getpid()))
    try:
        manifest = os.path.join(a.data_root, "MANIFEST.jsonl")
        py = os.path.join(HERE, ".venv", "Scripts", "python.exe")
        env = dict(os.environ, IEEG07_THREADS=a.threads)
        for s, t in ORDER:
            tag = f"s{s}_t{t}"
            if os.path.exists(os.path.join(a.features, f"{tag}_meta.json")):
                print(f"{tag}: already extracted", flush=True)
                continue
            name = f"sub_{s}_trial{t:03}.h5.zip"
            waited = 0
            while not extracted(manifest, name):
                if waited % 600 == 0:
                    print(f"{tag}: waiting for {name} ({waited} s)", flush=True)
                time.sleep(30)
                waited += 30
            print(f"{tag}: extracting", flush=True)
            log = os.path.join(HERE, "logs", f"extract_{tag}.log")
            with open(log, "w", encoding="utf-8") as fh:
                cmd = [py, "-u", os.path.join(HERE, "src", "extract_features.py"),
                       "--subject", str(s), "--trial", str(t), "--out", a.features]
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
            if rc != 0:
                raise SystemExit(f"{tag}: extractor failed with code {rc}; see {log}")
            print(f"{tag}: done", flush=True)
    finally:
        os.remove(a.lock)
    print("=== all sessions extracted ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

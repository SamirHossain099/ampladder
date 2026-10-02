"""Run D4 (POST HOC convergence sensitivity) between D3's heavy decode and D1, never alongside them.

Takes D:/ieeg07/d4.lock at once (run_d1.py waits for it to clear), waits for D3's heavy decode to
finish, then runs d4_convergence.py and releases the lock.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = "D:/ieeg07/d4.lock"


def d3_heavy_done():
    log = os.path.join(HERE, "logs", "d3_decode_heavy.log")
    return os.path.exists(log) and open(log, encoding="utf-8").read().count("=== done ===") >= 1


def main():
    if os.path.exists(LOCK):
        raise SystemExit(f"{LOCK} exists")
    open(LOCK, "w").write(str(os.getpid()))
    try:
        waited = 0
        while not d3_heavy_done():
            if waited % 1800 == 0:
                print(f"waiting for D3's heavy decode ({waited} s)", flush=True)
            time.sleep(60)
            waited += 60
        print("D4: running", flush=True)
        env = dict(os.environ, IEEG07_THREADS="12")
        with open(os.path.join(HERE, "logs", "d4.log"), "a", encoding="utf-8") as fh:
            rc = subprocess.run([os.path.join(HERE, ".venv", "Scripts", "python.exe"), "-u",
                                 os.path.join(HERE, "src", "d4_convergence.py")],
                                stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0:
            raise SystemExit(f"D4 failed ({rc}); see logs/d4.log")
        print("=== D4 done ===", flush=True)
    finally:
        os.remove(LOCK)
    return 0


if __name__ == "__main__":
    sys.exit(main())

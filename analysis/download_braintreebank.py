"""Download the Brain Treebank files Neuroprobe Lite needs, hash them, then extract.

Why not the upstream `braintreebank_download_extract.py`
-------------------------------------------------------
It deletes each zip immediately after extracting it and records nothing, so afterwards there is no
way to say which bytes were analysed. Project 08 hit exactly this with PRMI: the archive was
deleted after extraction and its published SHA-256 could never be checked again. Brain Treebank
publishes no checksums at all, so the only defensible record is one we make ourselves, at download
time, before anything is unpacked.

What this does, per file
------------------------
1. Resumable streamed download to `<name>.part` (HTTP Range), so a dropped connection over a
   75 GB pull costs one chunk, not one file.
2. SHA-256 computed from the bytes on disk once complete, and the size checked against the
   server's Content-Length. A mismatch raises; it never "degrades" to a warning.
3. One row appended to `MANIFEST.jsonl`: name, URL, bytes, sha256, UTC time.
4. Extraction to the data root, then the zip is deleted **only if** `--keep-zips` is not given and
   the manifest row was written. The manifest is the durable record; the zip is not needed to
   reproduce the analysis, and D: does not have room for both copies of the full set.

Data lives on D:, not in the project tree: N: is at 96% and the Lite set is 75.7 GB zipped.

Usage:
    python src/download_braintreebank.py --root D:/ieeg07/braintreebank --lite
    python src/download_braintreebank.py --root ... --lite --dry-run      # list and size only
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from urllib.parse import urljoin

import requests

BASE = "https://braintreebank.dev/"
SKIP = {"brain_treebank_code_release", "2411.08343"}

# Neuroprobe Lite: Neuroprobe paper, Supplementary Section B, and the upstream download script.
LITE_SUBJECT_TRIALS = [(1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                       (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1)]
LITE_FILES = {f"sub_{s}_trial{t:03}.h5.zip" for s, t in LITE_SUBJECT_TRIALS}

CHUNK = 8 * 1024 * 1024


def list_remote():
    html = requests.get(BASE, timeout=60).text
    out = []
    for href in sorted(set(re.findall(r'href="([^"]+)"', html))):
        url = urljoin(BASE, href)
        name = os.path.basename(url)
        if not name or name in SKIP:
            continue
        if not name.endswith((".zip", ".json", ".csv")):
            continue
        out.append((name, url))
    return out


def wanted(name, lite):
    return (not lite) or (not name.startswith("sub_")) or (name in LITE_FILES)


def remote_size(url):
    r = requests.head(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["content-length"])


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, expected):
    """Resumable download to dest; returns once dest exists at exactly `expected` bytes."""
    part = dest + ".part"
    for attempt in range(1, 21):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have == expected:
            break
        if have > expected:
            os.remove(part)
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code != 206:
                    # Server ignored the Range header: restart cleanly rather than append garbage.
                    os.remove(part)
                    continue
                r.raise_for_status()
                with open(part, "ab") as fh:
                    t0, done = time.time(), have
                    for block in r.iter_content(CHUNK):
                        fh.write(block)
                        done += len(block)
                        if time.time() - t0 > 30:
                            print(f"    {os.path.basename(dest)}: {done/1e9:.2f}/{expected/1e9:.2f} GB",
                                  flush=True)
                            t0 = time.time()
        except (requests.RequestException, OSError) as e:
            print(f"    attempt {attempt} failed ({e}); resuming", flush=True)
            time.sleep(min(60, 5 * attempt))
    final = os.path.getsize(part)
    if final != expected:
        raise IOError(f"{os.path.basename(dest)}: got {final} bytes, server says {expected}")
    os.replace(part, dest)


def already_recorded(manifest_path):
    if not os.path.exists(manifest_path):
        return {}
    rows = {}
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["name"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data root, e.g. D:/ieeg07/braintreebank")
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these file names (e.g. to fetch small metadata first)")
    a = ap.parse_args()

    zdir = os.path.join(a.root, "_zip")
    os.makedirs(zdir, exist_ok=True)
    manifest = os.path.join(a.root, "MANIFEST.jsonl")
    done = already_recorded(manifest)

    files = [(n, u) for n, u in list_remote() if wanted(n, a.lite)]
    if a.only:
        files = [(n, u) for n, u in files if n in set(a.only)]
    sizes = {n: remote_size(u) for n, u in files}
    total = sum(sizes.values())
    print(f"{len(files)} files, {total/1e9:.1f} GB ({'Lite' if a.lite else 'full'})", flush=True)
    if a.dry_run:
        for n, _ in files:
            print(f"  {sizes[n]/1e9:7.2f} GB  {n}{'  [recorded]' if n in done else ''}")
        return 0

    for name, url in files:
        # Re-read every iteration: a second invocation (e.g. --only for metadata) may have recorded
        # this file since startup, and re-downloading it would duplicate the manifest row.
        done = already_recorded(manifest)
        if name in done and done[name].get("extracted"):
            print(f"skip {name}: recorded and extracted", flush=True)
            continue
        dest = os.path.join(zdir, name)
        print(f"get  {name} ({sizes[name]/1e9:.2f} GB)", flush=True)
        if not (os.path.exists(dest) and os.path.getsize(dest) == sizes[name]):
            download(url, dest, sizes[name])
        digest = sha256_of(dest)
        row = dict(name=name, url=url, bytes=sizes[name], sha256=digest,
                   downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   extracted=False)

        if name.endswith(".zip"):
            with zipfile.ZipFile(dest) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise IOError(f"{name}: corrupt member {bad}")
                row["members"] = [(i.filename, i.file_size) for i in zf.infolist()]
                zf.extractall(a.root)
        else:
            os.replace(dest, os.path.join(a.root, name))
            row["members"] = [(name, sizes[name])]
        row["extracted"] = True
        with open(manifest, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if name.endswith(".zip") and not a.keep_zips and os.path.exists(dest):
            os.remove(dest)
        print(f"ok   {name}  sha256={digest[:16]}...", flush=True)

    print("=== done ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

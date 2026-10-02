"""Only the package and the code and results behind the paper are public.

The project this repository is cut from also holds working notes, the manuscript, submission material
and an unpublished pre-registration. None of them belongs here. This test checks the directory itself
and, when it is a git checkout, what git would publish: a check on git alone passes while an ignored
file sits in the folder, and a zip of the folder is also a real way to publish it.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN_NAMES = {"CLAUDE.md", "FINDINGS.md", "CORRECTIONS.md", "PREREGISTRATION.md", "VENUE.md",
                   "MANUSCRIPT.md", "MANUSCRIPT.docx", "MANUSCRIPT.pdf", "references.bib", "references.ris",
                   "references-missing.ris", "zotero_keys.json", "make_docx.py", "make_refs.py",
                   "verify_refs.py", "prose_scan.py", "register_scan.py", "sync_zotero.py", "build_release.py"}
FORBIDDEN_DIRS = {"submission", "logs", "package", "release_template", ".venv", "features"}
TEXT = (".py", ".md", ".toml", ".cff", ".yml", ".txt", ".json", ".jsonl", ".csv")
PRIVATE = re.compile(r"N:[\\/]|neuro_projects|EB-2|Dhanasar")


def directory_files():
    out = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", ".pytest_cache", ".ruff_cache", "__pycache__"}
                   and not d.endswith(".egg-info")]
        out += [os.path.relpath(os.path.join(root, f), ROOT) for f in files]
    return out


def git_files():
    try:
        r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [f for f in r.stdout.splitlines() if f]


def test_no_private_file_in_the_directory_or_in_git():
    for view in (directory_files(), git_files()):
        bad = [f for f in view if os.path.basename(f) in FORBIDDEN_NAMES
               or set(re.split(r"[\\/]", f)[:-1]) & FORBIDDEN_DIRS]
        assert not bad, bad


def test_no_em_dash_and_no_private_reference_in_published_text():
    for f in git_files() or directory_files():
        if os.path.basename(f) == "test_packaging.py":      # it holds the patterns it searches for
            continue
        if f.endswith(TEXT) and os.path.exists(os.path.join(ROOT, f)):
            s = open(os.path.join(ROOT, f), encoding="utf-8", errors="replace").read()
            assert chr(0x2014) not in s, f"em dash in {f}"
            m = PRIVATE.search(s)
            assert not m, f"private reference {m.group(0)!r} in {f}"

"""Privacy scan of everything that would be published (SPEC 1.4, 13.4).

Scans every git-tracked or not-ignored file (the files a commit would publish), the fixtures and a
report generated from the fixtures for: e-mail addresses (only GitHub noreply addresses allowed),
IPv4 addresses (except 127.0.0.1 and 0.0.0.0), absolute home-directory paths, and token-like
strings. Maintainers can add a git-ignored `.privacy_denylist` (one word per line; or point
AGBENCH_PRIVACY_DENYLIST at a file) with private words that must never appear (case-insensitive,
whole word). This file contains no private words itself.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 8_000_000
SKIP_DIRS = {".git", ".venv", "venv", "cache", "results", "__pycache__", ".pytest_cache", ".mypy_cache",
             ".ruff_cache", "node_modules", "build", "dist"}
SKIP_FILES = {".privacy_denylist"}
# uv.lock is generated from public package metadata: four-part package versions look like IPv4
# addresses and short hash directory names in its URLs can collide with short denylist words.
# It is still scanned for e-mail addresses, home paths and tokens.
GENERATED_EXEMPT = {"uv.lock"}

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
EMAIL_OK = re.compile(r"@users\.noreply\.github\.com$", re.I)
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4 = re.compile(rf"(?<![\w.=-]){_OCTET}(?:\.{_OCTET}){{3}}(?![\w.-])")
IPV4_OK = {"127.0.0.1", "0.0.0.0"}
_NAME = r"([A-Za-z0-9._-]+)"
HOME = [
    re.compile(r"/home/" + _NAME + r"/"),
    re.compile(r"/Users/" + _NAME + r"/"),
    re.compile(r"\b[A-Za-z]:\\{1,2}Users\\{1,2}" + _NAME + r"\\"),
]
# Generic placeholder names (docs) and the fictitious names used by the sanitizer unit tests are
# fine; anything else looks like a real account.
HOME_OK = {"user", "username", "name", "you", "me", "runner", "public", "default",
           "alice", "bob", "dev", "example"}
TOKENS = [
    re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"),
]


def scan_text(text: str, *, check_ipv4: bool = True, denylist: list[str] | None = None) -> list[tuple[int, str]]:
    """[(line number, problem)] for one file's text."""
    problems = []
    deny = [re.compile(rf"(?<![A-Za-z0-9]){re.escape(w)}(?![A-Za-z0-9])", re.I) for w in denylist or []]
    for no, line in enumerate(text.splitlines(), 1):
        for m in EMAIL.finditer(line):
            if not EMAIL_OK.search(m.group(0)):
                problems.append((no, f"e-mail address {m.group(0)!r}"))
        if check_ipv4:
            for m in IPV4.finditer(line):
                if m.group(0) not in IPV4_OK:
                    problems.append((no, f"IPv4 address {m.group(0)!r}"))
        for rx in HOME:
            for m in rx.finditer(line):
                if m.group(1).lower() not in HOME_OK:
                    problems.append((no, f"home-directory path {m.group(0)!r}"))
        for rx in TOKENS:
            for m in rx.finditer(line):
                problems.append((no, f"token-like string {m.group(0)[:8]}..."))
        for i, rx in enumerate(deny, 1):
            if rx.search(line):
                problems.append((no, f"denylisted word (entry #{i} of the denylist)"))
    return problems


def published_files() -> list[Path]:
    """Files a commit would publish: tracked + untracked-but-not-ignored (falls back to a tree walk)."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, check=True, timeout=60).stdout
        files = [ROOT / p for p in out.decode("utf-8", "replace").split("\0") if p]
    except (OSError, subprocess.SubprocessError):
        files = []
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            files += [Path(dirpath) / f for f in filenames]
    return sorted({f for f in files if f.is_file() and f.name not in SKIP_FILES
                   and not (set(f.relative_to(ROOT).parts[:-1]) & SKIP_DIRS)})


def load_denylist() -> list[str]:
    path = Path(os.environ.get("AGBENCH_PRIVACY_DENYLIST") or ROOT / ".privacy_denylist")
    if not path.is_file():
        return []
    words = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        w = line.strip()
        if w and not w.startswith("#"):
            words.append(w)
    return words


def read_text_file(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None  # binary (PNG and the like)
    return data.decode("utf-8", errors="replace")


def _scan_files(files: list[Path], denylist: list[str]) -> list[str]:
    found = []
    for f in files:
        text = read_text_file(f)
        if text is None:
            continue
        rel = f.relative_to(ROOT).as_posix() if f.is_relative_to(ROOT) else f.name
        generated = f.name in GENERATED_EXEMPT
        for no, what in scan_text(text, check_ipv4=not generated, denylist=[] if generated else denylist):
            found.append(f"{rel}:{no}: {what}")
    return found


def test_scanner_catches_the_patterns():
    at, dot = "@", "."
    assert scan_text("write to someone" + at + "example" + dot + "com")
    assert not scan_text("Co-author: 123+someone" + at + "users.noreply.github.com")
    assert scan_text("host 10" + ".1.2" + ".3 answered")
    assert not scan_text("loopback 127.0.0.1 and 0.0.0.0, version 1.2.3, cudnn==9.10.2.21, os 10.0.26100")
    assert scan_text("cd /" + "home/" + "carol/project")
    assert scan_text("C:" + "\\Users\\" + "carol\\project")
    assert not scan_text("cd /home/<user>/project or ~/project or C:\\Users\\<you>\\x or /home/user/x")
    assert scan_text("token " + "hf_" + "a1B2" * 9)
    assert scan_text("token " + "ghp_" + "Z9y8" * 9)
    assert scan_text("we ran it on server Bigiron-7 yesterday", denylist=["bigiron-7"])
    assert not scan_text("the bigiron-70 cluster", denylist=["bigiron-7"])


def test_published_files_are_clean():
    files = published_files()
    assert any(f.name == "README.md" for f in files) or not (ROOT / "README.md").exists()
    found = _scan_files(files, load_denylist())
    assert not found, "privacy problems:\n" + "\n".join(found[:50])


def test_generated_report_is_clean(tmp_path):
    from agbench.report import build_report

    fixtures = sorted((ROOT / "tests" / "fixtures").glob("*.jsonl"))
    summary = build_report(fixtures, tmp_path / "rep", references_path=ROOT / "references" / "reference_gpus.json")
    outputs = [summary, summary.parent / "comparison.csv", *fixtures]
    found = []
    for f in outputs:
        text = read_text_file(f) or ""
        found += [f"{f.name}:{no}: {what}" for no, what in scan_text(text, denylist=load_denylist())]
    home = str(Path.home())
    if len(home) > 3:
        for f in outputs:
            if home in (read_text_file(f) or ""):
                found.append(f"{f.name}: contains this machine's home directory")
    assert not found, "privacy problems in generated output:\n" + "\n".join(found[:50])


@pytest.mark.skipif(not load_denylist(), reason="no maintainer .privacy_denylist present")
def test_denylist_is_active():
    # Sanity: the maintainer denylist is loaded and matches its own first entry.
    words = load_denylist()
    assert scan_text(f"x {words[0]} y", denylist=words)

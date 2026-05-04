"""Release-hygiene checks for public package polish.

These tests guard against accidental inclusion of cache/build debris and
internal drafting language in public-facing source files.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".cfg", ".cff", ".yml", ".yaml"}
FORBIDDEN_TEXT = (
    "Chat" + "GPT",
    "Clau" + "de",
    "CL" + "AW",
    "v5 " + "audit",
    "transformative" + "-alpha",
    "rlichten" + "berg",
)
FORBIDDEN_PATH_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}


def _source_files():
    for path in ROOT.rglob("*"):
        if any(part in FORBIDDEN_PATH_PARTS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def test_no_internal_drafting_language_in_public_text_files():
    hits = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_TEXT:
            if token in text:
                hits.append(f"{path.relative_to(ROOT)}: {token}")
    assert hits == []



def test_manifest_excludes_cache_and_build_debris():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required = [
        "prune .ruff_cache",
        "prune .pytest_cache",
        "prune .mypy_cache",
        "prune build",
        "prune dist",
        "global-exclude __pycache__",
        "global-exclude *.py[cod]",
        "global-exclude *.egg-info",
    ]
    missing = [line for line in required if line not in manifest]
    assert missing == []

#!/usr/bin/env python3
"""Verify the handoff SHA-256 manifest without trusting file metadata."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST_SHA256.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    obj = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not obj:
        raise AssertionError("manifest must be a nonempty flat JSON object")
    for rel, expected in obj.items():
        if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")):
            raise AssertionError(f"invalid manifest path: {rel!r}")
        parts = Path(rel).parts
        if ".." in parts or Path(rel).is_absolute():
            raise AssertionError(f"unsafe manifest path: {rel}")
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise AssertionError(f"invalid SHA-256 for {rel}")
        p = ROOT / rel
        if not p.is_file():
            raise AssertionError(f"manifest file missing: {rel}")
        got = sha256(p)
        if got != expected:
            raise AssertionError(f"manifest hash mismatch: {rel}\nexpected {expected}\nactual   {got}")
    print(f"PASS manifest: {len(obj)} files")


if __name__ == "__main__":
    main()

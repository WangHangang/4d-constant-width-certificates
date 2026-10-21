#!/usr/bin/env python3
"""One-command preflight for the supervisor-facing handoff package."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from bl4.amo_ray_cache import load_cache
from verify_certificate import verify

ROOT = Path(__file__).resolve().parent
FROZEN_U3 = ROOT / "reference_methods/raised_u3/raised_u3_negative_inner_validated_lower.py"
FROZEN_U3_SHA256 = "e8d7b5b8e9a9feb60b6b6ac65c04f0694a768bd50d0d34736cd0e29c01b495ef"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if sha256(FROZEN_U3) != FROZEN_U3_SHA256:
        raise AssertionError("frozen Raised-U3 dependency hash mismatch")
    print("PASS frozen Raised-U3 dependency hash")

    manifest = ROOT / "MANIFEST_SHA256.json"
    if manifest.exists():
        subprocess.run([sys.executable, str(ROOT/"verify_manifest.py")], check=True, cwd=ROOT)

    tests = [
        "tests/test_fan.py",
        "tests/test_adaptive_fan.py",
        "tests/test_oracles.py",
        "tests/test_raised_u3.py",
        "tests/test_amo_witness.py",
        "tests/test_amo_ray_cache.py",
        "tests/test_parallel_predicates.py",
    ]
    for rel in tests:
        subprocess.run([sys.executable, str(ROOT/rel)], check=True, cwd=ROOT)

    certs = [
        "certificates/M4_k2.json",
        "certificates/M4_k3.json",
        "certificates/RaisedMeissnerFace_k2.json",
        "certificates/RaisedMeissnerFace_k3.json",
        "certificates/Body5_k2.json",
        "certificates/Body5_k3.json",
        "certificates/RaisedU3_k2.json",
        "certificates/RaisedU3_k3.json",
        "certificates/AMO_k2.json",
        "certificates/AMO_k3.json",
        "certificates/AMO_adaptive_smoke_b2_r24.json",
    ]
    for rel in certs:
        r = verify(ROOT/rel)
        print(f"PASS certificate {rel}: lower_bound={r['lower_bound']} above_threshold={r['above_threshold']}")

    cache = load_cache(ROOT/"cache/AMO_shared_ray_cache_r24.json", 1<<24)
    print(f"PASS AMO shared-cache schema: records={len(cache['records'])} radius_den={cache['radius_den']}")
    print("HANDOFF PREFLIGHT PASS")


if __name__ == "__main__":
    main()

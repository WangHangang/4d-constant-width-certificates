#!/usr/bin/env python3
"""Regression: parallel predicate-ray certification matches serial exactly."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bl4.fan import build_fan
from bl4.radial_parallel import certified_radii_parallel


def main():
    vertices, _, scale = build_fan(1)
    den = 1 << 16
    for body in ("M4", "Body5"):
        a = certified_radii_parallel(vertices, scale, den, body, 4.0, 1)
        b = certified_radii_parallel(vertices, scale, den, body, 4.0, 2)
        if a != b:
            raise AssertionError(f"parallel/serial radius mismatch for {body}")
    print("PASS parallel predicate radii = serial radii (M4, Body5, level 1)")


if __name__ == "__main__":
    main()

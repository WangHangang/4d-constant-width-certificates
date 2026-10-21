#!/usr/bin/env python3
"""Seed/merge the shared AMO ray cache from independently verified certificates.

This utility is generator-side only.  Each input certificate is first checked by
``verify_certificate.verify``.  If the requested cache radial denominator differs
from the certificate denominator, the endpoint numerator is rescaled only when
this is exact; the AMO witness is then reverified at the new denominator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bl4.amo_ray_cache import load_cache, atomic_write_cache, put_record
from bl4.certificate import read_certificate
from bl4.fan import build_fan, build_adaptive_fan
from bl4.oracles import REGISTRY
from verify_certificate import verify


def fan_for_certificate(cert):
    if cert.get("fan_type", "uniform") == "adaptive":
        return build_adaptive_fan(int(cert["base_level"]), cert["adaptive_schedule"])
    return build_fan(int(cert["fan_level"]))


def exact_rescale_radius(r, source_den, target_den):
    num = int(r) * int(target_den)
    den = int(source_den)
    if num % den:
        raise AssertionError("endpoint is not exactly representable on target radial grid")
    return num // den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("certificates", nargs="+", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--radius-exp", type=int, default=24)
    args = ap.parse_args()

    target_den = 1 << int(args.radius_exp)
    cache = load_cache(args.cache, target_den)
    oracle = REGISTRY["AMO"]
    before = len(cache["records"])
    inserted = replaced = 0

    for path in args.certificates:
        result = verify(path)
        if result["body"] != "AMO":
            raise AssertionError(f"not an AMO certificate: {path}")
        cert = read_certificate(path)
        vertices, cells, scale = fan_for_certificate(cert)
        source_den = int(cert["radius_den"])
        radii = [int(x) for x in cert["radius_num"]]
        witnesses = cert["endpoint_witnesses"]
        if len(vertices) != len(radii) or len(vertices) != len(witnesses):
            raise AssertionError("certificate ray/witness lengths disagree")
        for u, r, w in zip(vertices, radii, witnesses):
            rr = exact_rescale_radius(r, source_den, target_den)
            point_den = target_den * scale
            if not oracle.verify_witness(tuple(rr*x for x in u), point_den, w):
                raise AssertionError("rescaled cached endpoint failed exact witness verification")
            from bl4.amo_ray_cache import ray_key
            key = ray_key(u, scale)
            old = cache["records"].get(key)
            diag = {"seed_certificate": path.name,
                    "source_radius_den": source_den,
                    "rescaled_to_radius_den": target_den}
            put_record(cache, u, scale, rr, w, diag)
            if old is None:
                inserted += 1
            else:
                replaced += 1
        print(f"PASS seed source {path}: rays={len(vertices)} lower_bound={result['lower_bound']}")

    atomic_write_cache(args.cache, cache)
    print(f"shared cache: before={before} after={len(cache['records'])} inserted={inserted} replaced={replaced}")
    print(f"radius_den={target_den} path={args.cache}")


if __name__ == "__main__":
    main()

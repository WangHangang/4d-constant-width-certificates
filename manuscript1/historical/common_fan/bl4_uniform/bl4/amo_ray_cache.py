"""Shared exact AMO ray-witness cache for V1/V2 generators.

This is generator infrastructure, not a proof assumption.  Records are keyed by
an exact primitive rational direction ``u/scale``.  Before a cached record is
reused for a new fan, the current generator re-runs the exact AMO endpoint
witness verifier for the current integer direction and point denominator.

The cache therefore saves numerical proposal work across uniform levels and
between V1 and V2 without weakening the independent final certificate.
"""
from __future__ import annotations

import json
import os
from math import gcd
from pathlib import Path

SCHEMA = "amo_shared_ray_cache_v1"
INDEXING = "primitive signed integer direction plus positive scale denominator"


def ray_key(u, scale):
    """Canonical string key for the rational direction vector ``u/scale``.

    Opposite rays remain distinct.  Positive common factors in ``u`` and
    ``scale`` are removed, so inherited rays have the same key after fan
    refinement.
    """
    vals = tuple(int(x) for x in u)
    scale = int(scale)
    if len(vals) != 4 or scale <= 0 or not any(vals):
        raise ValueError("invalid AMO ray direction")
    g = scale
    for x in vals:
        g = gcd(g, abs(x))
    prim = tuple(x // g for x in vals)
    den = scale // g
    return ",".join(map(str, prim + (den,)))


def parse_ray_key(key):
    if not isinstance(key, str):
        raise ValueError("ray-cache key is not a string")
    parts = key.split(",")
    if len(parts) != 5:
        raise ValueError("ray-cache key must have five integers")
    vals = tuple(int(x) for x in parts)
    u, den = vals[:4], vals[4]
    if den <= 0 or not any(u):
        raise ValueError("invalid ray-cache key direction")
    if ray_key(u, den) != key:
        raise ValueError("ray-cache key is not primitive/canonical")
    return u, den


def new_cache(radius_den):
    radius_den = int(radius_den)
    if radius_den <= 0:
        raise ValueError("radius_den must be positive")
    return {
        "schema": SCHEMA,
        "body": "AMO",
        "indexing": INDEXING,
        "radius_den": radius_den,
        "records": {},
    }


def preflight_cache_obj(obj, radius_den):
    """Strict schema/range/sentinel preflight before cache reuse."""
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA:
        raise AssertionError("AMO shared-cache schema mismatch")
    if obj.get("body") != "AMO" or obj.get("indexing") != INDEXING:
        raise AssertionError("AMO shared-cache identity/indexing mismatch")
    if int(obj.get("radius_den")) != int(radius_den):
        raise AssertionError("AMO shared-cache radius denominator mismatch")
    records = obj.get("records")
    if not isinstance(records, dict):
        raise AssertionError("AMO shared-cache records must be a dictionary")
    for key, rec in records.items():
        parse_ray_key(key)
        if not isinstance(rec, dict):
            raise AssertionError("AMO shared-cache record is not an object")
        r = rec.get("radius_num")
        if not isinstance(r, int) or r < 0:
            raise AssertionError("AMO shared-cache radius_num must be nonnegative int")
        witness = rec.get("witness")
        # The current AMO generator stores a witness even for r=0; there is no
        # null/sentinel witness in the shared cache.
        if not isinstance(witness, dict):
            raise AssertionError("AMO shared-cache witness must be an object")
        if "diagnostic" in rec and not isinstance(rec["diagnostic"], dict):
            raise AssertionError("AMO shared-cache diagnostic must be an object")
    return obj


def load_cache(path, radius_den):
    path = Path(path)
    if not path.exists():
        return new_cache(radius_den)
    obj = json.loads(path.read_text(encoding="utf-8"))
    return preflight_cache_obj(obj, radius_den)


def atomic_write_cache(path, obj):
    path = Path(path)
    preflight_cache_obj(obj, obj["radius_den"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def get_record(cache, u, scale):
    return cache["records"].get(ray_key(u, scale))


def put_record(cache, u, scale, radius_num, witness, diagnostic=None):
    key = ray_key(u, scale)
    rec = {"radius_num": int(radius_num), "witness": witness}
    if diagnostic is not None:
        rec["diagnostic"] = diagnostic
    cache["records"][key] = rec
    return key

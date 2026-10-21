"""Parallel exact endpoint certification for proof-ready predicate oracles.

Each ray is independent. Floating arithmetic is used only to propose a radius;
the returned integer radius is accepted by the exact body oracle before it is
returned. The parent process preserves input-ray order, so worker scheduling
cannot change the certificate. A parent-side cache may reuse already certified
radii for identical rational directions within one adaptive run.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
from math import gcd


def _direction_key(u, scale):
    g = int(scale)
    for x in u:
        g = gcd(g, abs(int(x)))
    return tuple(int(x)//g for x in u) + (int(scale)//g,)


def _one_radius(task):
    body, u, direction_scale, radius_den, max_radius = task
    # Import inside the worker so Windows ``spawn`` reconstructs the same
    # read-only oracle registry instead of trying to pickle callable closures.
    from bl4.oracles import REGISTRY

    oracle = REGISTRY[body]
    if oracle.verify_witness is not None:
        raise ValueError(f"witness oracle {body} requires its dedicated generator")
    if direction_scale <= 0 or radius_den <= 0:
        raise ValueError("direction_scale and radius_den must be positive")
    if len(u) != 4 or not all(isinstance(x, int) for x in u):
        raise ValueError("ray must be a four-integer tuple")

    point_den = radius_den * direction_scale
    max_num = int(math.ceil(max_radius * radius_den))
    direction = tuple(x / direction_scale for x in u)
    rho = oracle.radial_float(direction, max_radius)
    candidate = min(max_num, max(0, int(math.floor(rho * radius_den)) - 4))

    def accepted(r):
        return oracle.contains(tuple(r * x for x in u), point_den)

    found = None
    for r in range(candidate, max(-1, candidate - 64), -1):
        if accepted(r):
            found = r
            break
    if found is None:
        lo, hi = 0, candidate + 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if accepted(mid):
                lo = mid
            else:
                hi = mid
        found = lo
    for r in range(found + 1, min(max_num, found + 8) + 1):
        if not accepted(r):
            break
        found = r
    if not accepted(found):
        raise AssertionError("uncertified radius escaped final exact check")
    return found


def certified_radii_parallel(vertices, direction_scale, radius_den, body,
                              max_radius=4.0, workers=1, cache=None):
    """Certify one rational endpoint per ray, optionally in worker processes.

    The origin is rechecked exactly in the parent before any workers are
    launched. Results are returned in exactly the same order as ``vertices``.
    ``cache`` is an optional mutable mapping keyed by primitive rational
    directions; cached radii are mathematical data from the same current run,
    not a trust shortcut, because all newly generated radii are exact-checked
    before insertion.
    """
    from bl4.oracles import REGISTRY

    if body not in REGISTRY:
        raise ValueError(f"unknown body {body!r}")
    oracle = REGISTRY[body]
    if oracle.verify_witness is not None:
        raise ValueError(f"witness oracle {body} requires its dedicated generator")
    point_den = int(radius_den) * int(direction_scale)
    if not oracle.contains((0, 0, 0, 0), point_den):
        raise AssertionError("the declared chart origin is not certified inside")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if cache is None:
        cache = {}

    result = [None] * len(vertices)
    missing_indices = []
    tasks = []
    for i, u0 in enumerate(vertices):
        u = tuple(int(x) for x in u0)
        key = _direction_key(u, direction_scale)
        if key in cache:
            result[i] = int(cache[key])
        else:
            missing_indices.append(i)
            tasks.append((body, u, int(direction_scale), int(radius_den), float(max_radius)))

    if tasks:
        if workers == 1:
            new_radii = [_one_radius(t) for t in tasks]
        else:
            # executor.map is ordered: scheduling affects wall time only, not proof data.
            with ProcessPoolExecutor(max_workers=workers) as ex:
                new_radii = list(ex.map(_one_radius, tasks, chunksize=4))
        for i, r in zip(missing_indices, new_radii):
            u = tuple(int(x) for x in vertices[i])
            key = _direction_key(u, direction_scale)
            result[i] = int(r)
            cache[key] = int(r)

    if any(r is None for r in result):
        raise AssertionError("parallel radius assembly left an unset ray")
    return [int(r) for r in result]

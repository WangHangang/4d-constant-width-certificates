"""Float-guided search followed by an exact, proof-producing endpoint check."""
import math
from math import gcd


def _direction_key(u,scale):
    g=scale
    for x in u: g=gcd(g,abs(x))
    return tuple(x//g for x in u)+(scale//g,)


def certified_radii(vertices, direction_scale, radius_den, oracle, max_radius=4.0, cache=None):
    """Return certified lower radii.

    Floating arithmetic chooses a candidate only.  Every returned point is
    accepted again by ``oracle.contains``.  If the predictor is inaccurate,
    the fallback exact bisection preserves correctness.
    """
    point_den = radius_den * direction_scale
    if not oracle.contains((0, 0, 0, 0), point_den):
        raise AssertionError("the declared chart origin is not certified inside")
    max_num = int(math.ceil(max_radius * radius_den))
    radii = []
    if cache is None: cache={}
    for u in vertices:
        key=_direction_key(u,direction_scale)
        if key in cache:
            radii.append(cache[key]); continue
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
        # A few exact upward steps recover predictor rounding loss cheaply.
        for r in range(found + 1, min(max_num, found + 8) + 1):
            if not accepted(r):
                break
            found = r
        if not accepted(found):
            raise AssertionError("uncertified radius escaped final check")
        radii.append(found); cache[key]=found
    return radii

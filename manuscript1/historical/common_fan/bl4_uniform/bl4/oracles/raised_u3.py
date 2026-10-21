"""Exact conservative adapter for the canonical raise of width-one U_3.

Coordinates
-----------
The horizontal coordinates are the tetrahedral coefficient chart used by the
raised-Meissner adapter.  Thus the four tetrahedron vertices are

    A=(1,0,0), B=(0,1,0), C=(0,0,1), D=(-1,-1,-1),

and the Euclidean quadratic form is qform3(x)/8.  The fourth coordinate is
ordinary Cartesian height.

Positive half
-------------
The preserved boundary of U_3 consists of the six complete elliptical arcs

    gamma(q) = (A(1-q^2)+2 B q)/(1+q^2),   0 <= q <= 1,

one for each unordered pair of tetrahedron vertices.  Here q=tan(t/2) and
0<=t<=pi/2, so [0,1] is the whole A-to-B arc, not a subarc.

For a rational point (y,h)=n/D, membership in the positive half is implied by

    ||y-gamma(q)||^2 + h^2 <= 1

for every q in [0,1] and every one of the six arcs.  Clearing the positive
denominator 8 D^2 (1+q^2)^2 gives an integer quartic P(q).  Nonnegativity of
P on [0,1] is certified with exact degree-4 Bernstein coefficients and dyadic
de Casteljau subdivision.  The verifier therefore uses Python integers only.

Negative half
-------------
We use a finite rational subfamily of the rigorous inner model

    L_- = {h<=0} intersect union_{m in T} B_4(m, eta(m)),
    eta(m)=1-sqrt(5/8+||m||^2).

Every witness centre belongs exactly to the tetrahedron T.  The ball test is
reduced to integer inequalities by two safe squarings.  This may reject true
points, but every accepted point is rigorous.

Floating functions below are advisory only.  Every stored endpoint is checked
again by ``contains`` in both generator and independent verifier.
"""
from math import gcd, sqrt
from .base import Oracle, qform3

VERT = {
    "A": (1, 0, 0),
    "B": (0, 1, 0),
    "C": (0, 0, 1),
    "D": (-1, -1, -1),
}
PAIRS = (
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "C"), ("B", "D"), ("C", "D"),
)

# A moderate barycentric grid is a rigorous finite subfamily of the frozen
# L_- ball union.  Refining this integer only changes tightness, not soundness.
WITNESS_DEN = 18


def _witnesses(den=WITNESS_DEN):
    out = []
    for ia in range(den + 1):
        for ib in range(den + 1 - ia):
            for ic in range(den + 1 - ia - ib):
                id_ = den - ia - ib - ic
                m = (ia - id_, ib - id_, ic - id_)
                # Boundary centres have eta=0 and never improve a strict inner
                # endpoint, so omit them.
                if qform3(m) < 3 * den * den:
                    out.append(m)
    return tuple(out)


WITNESSES = _witnesses()


def _poly_add(a, b):
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        out[i] = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)
    return out


def _poly_scale(c, a):
    return [int(c) * int(x) for x in a]


def _poly_mul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += int(x) * int(y)
    return out


def _quartic_for_arc(nh, t, D, A, B):
    # g(q) = D(1+q^2) * ( y - gamma(q) )
    gs = []
    for i in range(3):
        gs.append((
            int(nh[i]) - D * int(A[i]),
            -2 * D * int(B[i]),
            int(nh[i]) + D * int(A[i]),
        ))

    qf = [0] * 5
    for i in range(3):
        qf = _poly_add(qf, _poly_scale(3, _poly_mul(gs[i], gs[i])))
    for i in range(3):
        for j in range(i):
            qf = _poly_add(qf, _poly_scale(-2, _poly_mul(gs[i], gs[j])))

    # 8 D^2 (1+q^2)^2 * (1-h^2) - qform3(g(q)).
    leading = 8 * (D * D - int(t) * int(t))
    base = [leading, 0, 2 * leading, 0, leading]
    out = _poly_add(base, _poly_scale(-1, qf))
    if len(out) < 5:
        out += [0] * (5 - len(out))
    return tuple(out[:5])


def _power_to_bernstein_scaled12(c):
    # Degree-4 Bernstein coefficients b_k, multiplied by 12.
    c0, c1, c2, c3, c4 = map(int, c)
    return (
        12 * c0,
        12 * c0 + 3 * c1,
        12 * c0 + 6 * c1 + 2 * c2,
        12 * c0 + 9 * c1 + 6 * c2 + 3 * c3,
        12 * (c0 + c1 + c2 + c3 + c4),
    )


def _split_bernstein_half(b):
    # Exact de Casteljau split at 1/2.  The returned tuples are multiplied by
    # a common positive factor 16 relative to the input scaling, so signs are
    # unchanged and all entries remain integers.
    b0, b1, b2, b3, b4 = map(int, b)
    s10, s11, s12, s13 = b0 + b1, b1 + b2, b2 + b3, b3 + b4
    s20, s21, s22 = s10 + s11, s11 + s12, s12 + s13
    s30, s31 = s20 + s21, s21 + s22
    s40 = s30 + s31
    left = (16*b0, 8*s10, 4*s20, 2*s30, s40)
    right = (s40, 2*s31, 4*s22, 8*s13, 16*b4)
    return left, right


def _quartic_nonnegative_unit_interval(c, max_depth=24):
    """Exact sufficient proof that a quartic is nonnegative on [0,1].

    Returns False conservatively if the finite subdivision budget cannot
    certify the claim.  A negative Bernstein coefficient alone is not a
    disproof; it triggers subdivision.
    """
    first = _power_to_bernstein_scaled12(c)
    stack = [(first, 0)]
    while stack:
        b, depth = stack.pop()
        if min(b) >= 0:
            continue
        # Endpoint values are Bernstein endpoints.  A negative one is an exact
        # counterexample on this subinterval.
        if b[0] < 0 or b[-1] < 0:
            return False
        if depth >= max_depth:
            return False
        left, right = _split_bernstein_half(b)
        stack.append((right, depth + 1))
        stack.append((left, depth + 1))
    return True


def _positive(n, D):
    if n[3] < 0:
        return False
    nh = tuple(map(int, n[:3]))
    t = int(n[3])
    for X, Y in PAIRS:
        c = _quartic_for_arc(nh, t, int(D), VERT[X], VERT[Y])
        if not _quartic_nonnegative_unit_interval(c):
            return False
    return True


def _negative_one(n, D, m, W=WITNESS_DEN):
    # Same exact algebra as the rigorous L_- ball certificate.  Bring x=n/D
    # and m/W to L=lcm(D,W), with qform3/8 as the horizontal metric.
    g = gcd(D, W)
    L = D // g * W
    xs, ms = L // D, L // W
    xn = tuple(xs * int(v) for v in n)
    mn = tuple(ms * int(v) for v in m)
    diff = tuple(xn[i] - mn[i] for i in range(3))

    # A = 8 L^2 ||x-(m,0)||^2.
    A = qform3(diff) + 8 * xn[3] * xn[3]
    # B = 8 L^2 (5/8 + ||m||^2).
    B = 5 * L * L + qform3(mn)
    # eta = 1 - sqrt(B/(8L^2)).
    # A/(8L^2) <= eta^2 is equivalent, after safe sign checks, to
    # C := 8L^2 + B - A >= 0 and C^2 >= 32 B L^2.
    C = 8 * L * L + B - A
    return B < 8 * L * L and C >= 0 and C * C >= 32 * B * L * L


def contains(point_num, point_den):
    if point_den <= 0 or len(point_num) != 4:
        raise ValueError("invalid rational point")
    n = tuple(map(int, point_num))
    D = int(point_den)
    if n[3] >= 0:
        return _positive(n, D)
    return any(_negative_one(n, D, m) for m in WITNESSES)


def _qf(a):
    x, y, z = a
    return 3*(x*x+y*y+z*z) - 2*(x*y+x*z+y*z)


def _ellipse_point(A, B, q):
    den = 1.0 + q*q
    return tuple(((1.0-q*q)*A[i] + 2.0*q*B[i]) / den for i in range(3))


WF = tuple(tuple(v / WITNESS_DEN for v in m) for m in WITNESSES)


def contains_float(x):
    """Advisory float predicate; exact membership is always rechecked.

    The positive branch deliberately samples the arc family coarsely.  This
    tends to overestimate the admissible radial endpoint, after which the
    common generator backs off/bisects with the exact predicate.

    The negative branch tests the same finite rigorous witness-ball family.
    """
    h = tuple(float(v) for v in x[:3])
    t = float(x[3])
    if t < 0.0:
        for mf in WF:
            mn2 = _qf(mf) / 8.0
            eta = 1.0 - sqrt(5.0 / 8.0 + mn2)
            diff = tuple(h[i] - mf[i] for i in range(3))
            if _qf(diff) / 8.0 + t*t <= eta*eta:
                return True
        return False

    threshold = 1.0 - t*t
    if threshold < 0.0:
        return False
    # Predictor only: 17 nodes per complete A-to-B arc.
    for X, Y in PAIRS:
        A, B = VERT[X], VERT[Y]
        best = 0.0
        for j in range(17):
            q = j / 16.0
            e = _ellipse_point(A, B, q)
            d = tuple(h[i] - e[i] for i in range(3))
            value = _qf(d) / 8.0
            if value > best:
                best = value
        if best > threshold:
            return False
    return True


def _bil_float(a, b):
    return (
        3*(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])
        - (a[0]*b[1] + a[1]*b[0] + a[0]*b[2] + a[2]*b[0]
           + a[1]*b[2] + a[2]*b[1])
    )


def _negative_radial_float(direction, max_radius):
    """Fast advisory radius for the finite rigorous negative witness family."""
    dh = tuple(float(v) for v in direction[:3])
    t = float(direction[3])
    A = _qf(dh) / 8.0 + t*t
    if A <= 0.0:
        return 0.0
    best = 0.0
    for mf in WF:
        mn2 = _qf(mf) / 8.0
        eta = 1.0 - sqrt(5.0 / 8.0 + mn2)
        B = -_bil_float(dh, mf) / 8.0
        C = mn2 - eta*eta
        disc = B*B - A*C
        if disc >= 0.0:
            r = (-B + sqrt(disc)) / A
            if r > best:
                best = r
    return min(float(max_radius), max(0.0, best))


def _positive_radial_float(direction, max_radius):
    """Fast outer predictor from 17 sample nodes on each complete ellipse arc.

    Sampling only a subset of each arc can make this radius too large, never a
    proof of membership.  The common generator subsequently backs off or uses
    exact bisection with ``contains``.
    """
    dh = tuple(float(v) for v in direction[:3])
    t = float(direction[3])
    Acoef = _qf(dh) / 8.0 + t*t
    if Acoef <= 0.0:
        return 0.0
    upper = float(max_radius)
    for X, Y in PAIRS:
        A, B = VERT[X], VERT[Y]
        for j in range(17):
            q = j / 16.0
            e = _ellipse_point(A, B, q)
            dot = _bil_float(dh, e) / 8.0
            C = _qf(e) / 8.0 - 1.0
            disc = dot*dot - Acoef*C
            if disc < 0.0:
                return 0.0
            root = (dot + sqrt(disc)) / Acoef
            if root < upper:
                upper = root
    return max(0.0, upper)


def radial_float(direction, max_radius):
    if float(direction[3]) < 0.0:
        return _negative_radial_float(direction, max_radius)
    return _positive_radial_float(direction, max_radius)


ORACLE = Oracle(
    name="RaisedU3",
    contains=contains,
    contains_float=contains_float,
    description=("Canonical raise of width-one U_3 in tetrahedral coefficient "
                 "coordinates. Positive-half six-ellipse membership is certified "
                 "by integer quartics and exact Bernstein/de Casteljau subdivision; "
                 "negative half uses a finite rational subfamily of the frozen "
                 "rigorous L_- witness-ball inner model."),
    chart_factor={"num": 1, "den": 8, "sqrt_radicand": 2},
    radial_float_fn=radial_float,
)

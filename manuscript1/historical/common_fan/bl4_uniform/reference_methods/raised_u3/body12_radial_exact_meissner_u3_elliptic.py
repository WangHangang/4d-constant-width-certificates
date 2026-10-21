"""Radial Monte-Carlo computation for dimension-raised constant-width bodies.

Body 1 is the canonical Lachand--Robert--Oudet raise of the two
classical Meissner bodies M_F and M_V.  Body 2 is the canonical raise
of U_3.

Both three-dimensional inputs are evaluated by continuous generating-
arc oracles:

* M_F and M_V use their three preserved circular arcs;
* U_3 uses the six explicit elliptical arcs from Mercado-Flores,
  Raggi, and Roldan-Pensado (Mathematika, 2026).

No finite Meissner arc grid and no finite U_3 support-direction grid is
used.  The remaining geometric approximation is the finite sampling of
the continuous four-dimensional upper graph used as ball centers for
the lower half of each raised body.

Dependencies: NumPy and Numba.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import numpy as np
from numba import njit, prange

ALPHA = np.arccos(1.0 / 3.0)
VOL_B4 = np.pi**2 / 2.0
DEFAULT_SEED = 20260713
DEFAULT_N_PAIRS = 20_000
DEFAULT_BODIES = ("M_F", "M_V", "U3")
MEISSNER_BOUNDARY_BISECTION_ITERATIONS = 45
U3_BOUNDARY_BISECTION_ITERATIONS = 45
U3_ARC_ROOT_BISECTION_ITERATIONS = 55
N_BASE_MEISSNER = 200_000
N_BASE_U3 = 100_000
LAYERS = np.linspace(0.0, 1.0, 9)
CACHE_DIR = Path("body12_radial_cache")


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def random_sphere(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x


def layer_cache_tag(layers: np.ndarray) -> str:
    data = np.asarray(layers, dtype=np.float64).tobytes()
    return sha256(data).hexdigest()[:12]


@njit(parallel=True, fastmath=True)
def radial_intersection_unit_balls(
    directions: np.ndarray, centers: np.ndarray
) -> np.ndarray:
    """Radial function of a finite intersection of unit balls."""
    n, dim = directions.shape
    m = centers.shape[0]
    center_norm2 = np.empty(m)
    for j in range(m):
        value = 0.0
        for k in range(dim):
            value += centers[j, k] * centers[j, k]
        center_norm2[j] = value

    result = np.empty(n)
    for i in prange(n):
        best = 1.0e100
        for j in range(m):
            dot = 0.0
            for k in range(dim):
                dot += directions[i, k] * centers[j, k]
            radicand = dot * dot + 1.0 - center_norm2[j]
            if radicand < 0.0:
                radicand = 0.0
            value = dot + np.sqrt(radicand)
            if value < best:
                best = value
        result[i] = best
    return result


# ---------------------------------------------------------------------
# Meissner input bodies: exact continuous circular-arc oracle
# ---------------------------------------------------------------------
def tetrahedron_vertices() -> dict[str, np.ndarray]:
    s = 1.0 / (2.0 * np.sqrt(2.0))
    return {
        "A": s * np.array([1.0, 1.0, 1.0]),
        "B": s * np.array([1.0, -1.0, -1.0]),
        "C": s * np.array([-1.0, 1.0, -1.0]),
        "D": s * np.array([-1.0, -1.0, 1.0]),
    }


def arc_data(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    rho = np.sqrt(3.0) / 2.0
    midpoint = 0.5 * (z + w)
    e1 = (x - midpoint) / rho
    e2 = ((y - midpoint) / rho - np.cos(ALPHA) * e1) / np.sin(ALPHA)
    return midpoint, rho, e1, e2


def meissner_arcs(kind: str):
    vertices = tetrahedron_vertices()
    if kind == "face":
        data = [
            ("A", "B", "C", "D"),
            ("A", "C", "B", "D"),
            ("A", "D", "B", "C"),
        ]
    elif kind == "vertex":
        data = [
            ("B", "C", "A", "D"),
            ("B", "D", "A", "C"),
            ("C", "D", "A", "B"),
        ]
    else:
        raise ValueError("kind must be 'face' or 'vertex'")
    arcs = [
        arc_data(vertices[x], vertices[y], vertices[z], vertices[w])
        for x, y, z, w in data
    ]
    return vertices, arcs


def max_squared_distance_to_arc(points: np.ndarray, arc) -> np.ndarray:
    midpoint, rho, e1, e2 = arc
    z = points - midpoint
    coefficient_a = z @ e1
    coefficient_b = z @ e2

    minimum = np.minimum(
        coefficient_a,
        coefficient_a * np.cos(ALPHA) + coefficient_b * np.sin(ALPHA),
    )
    interior_angle = np.mod(
        np.arctan2(coefficient_b, coefficient_a) + np.pi, 2.0 * np.pi
    )
    amplitude = np.hypot(coefficient_a, coefficient_b)
    minimum = np.where(
        interior_angle <= ALPHA, np.minimum(minimum, -amplitude), minimum
    )
    return np.sum(z * z, axis=1) + rho * rho - 2.0 * rho * minimum


def meissner_raw_depth(points: np.ndarray, kind: str) -> np.ndarray:
    vertices, arcs = meissner_arcs(kind)
    max_distance2 = np.full(points.shape[0], -np.inf)
    for arc in arcs:
        max_distance2 = np.maximum(
            max_distance2, max_squared_distance_to_arc(points, arc)
        )
    if kind == "vertex":
        max_distance2 = np.maximum(
            max_distance2,
            np.sum((points - vertices["A"]) ** 2, axis=1),
        )
    return 1.0 - np.sqrt(np.maximum(max_distance2, 0.0))


def meissner_arc_arrays(kind: str):
    vertices, arcs = meissner_arcs(kind)
    midpoints = np.stack([arc[0] for arc in arcs])
    radii = np.array([arc[1] for arc in arcs], dtype=float)
    e1 = np.stack([arc[2] for arc in arcs])
    e2 = np.stack([arc[3] for arc in arcs])
    extra_vertex = vertices["A"].copy()
    has_extra_vertex = kind == "vertex"
    return midpoints, radii, e1, e2, extra_vertex, has_extra_vertex


@njit(inline="always", fastmath=True)
def meissner_raw_depth_scalar(
    x0: float,
    x1: float,
    x2: float,
    midpoints: np.ndarray,
    radii: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    extra_vertex: np.ndarray,
    has_extra_vertex: bool,
) -> float:
    max_distance2 = -1.0e300
    cos_alpha = np.cos(ALPHA)
    sin_alpha = np.sin(ALPHA)

    for j in range(midpoints.shape[0]):
        z0 = x0 - midpoints[j, 0]
        z1 = x1 - midpoints[j, 1]
        z2 = x2 - midpoints[j, 2]
        coefficient_a = z0 * e1[j, 0] + z1 * e1[j, 1] + z2 * e1[j, 2]
        coefficient_b = z0 * e2[j, 0] + z1 * e2[j, 1] + z2 * e2[j, 2]

        endpoint_0 = coefficient_a
        endpoint_1 = coefficient_a * cos_alpha + coefficient_b * sin_alpha
        minimum = endpoint_0 if endpoint_0 < endpoint_1 else endpoint_1

        interior_angle = np.arctan2(coefficient_b, coefficient_a) + np.pi
        amplitude = np.sqrt(
            coefficient_a * coefficient_a + coefficient_b * coefficient_b
        )
        if interior_angle <= ALPHA and -amplitude < minimum:
            minimum = -amplitude

        rho = radii[j]
        distance2 = (
            z0 * z0 + z1 * z1 + z2 * z2
            + rho * rho - 2.0 * rho * minimum
        )
        if distance2 > max_distance2:
            max_distance2 = distance2

    if has_extra_vertex:
        z0 = x0 - extra_vertex[0]
        z1 = x1 - extra_vertex[1]
        z2 = x2 - extra_vertex[2]
        distance2 = z0 * z0 + z1 * z1 + z2 * z2
        if distance2 > max_distance2:
            max_distance2 = distance2

    if max_distance2 < 0.0:
        max_distance2 = 0.0
    return 1.0 - np.sqrt(max_distance2)


@njit(parallel=True, fastmath=True)
def meissner_radial_exact_kernel(
    directions: np.ndarray,
    midpoints: np.ndarray,
    radii: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    extra_vertex: np.ndarray,
    has_extra_vertex: bool,
    iterations: int,
) -> np.ndarray:
    n = directions.shape[0]
    result = np.empty(n)

    for i in prange(n):
        d0, d1, d2 = directions[i, 0], directions[i, 1], directions[i, 2]
        norm = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if norm == 0.0:
            result[i] = np.nan
            continue
        u0, u1, u2 = d0 / norm, d1 / norm, d2 / norm

        low, high = 0.0, 1.0
        bracketed = False
        for _ in range(4):
            depth_high = meissner_raw_depth_scalar(
                high * u0, high * u1, high * u2,
                midpoints, radii, e1, e2,
                extra_vertex, has_extra_vertex,
            )
            if depth_high < 0.0:
                bracketed = True
                break
            high *= 2.0
        if not bracketed:
            result[i] = np.nan
            continue

        for _ in range(iterations):
            middle = 0.5 * (low + high)
            depth_middle = meissner_raw_depth_scalar(
                middle * u0, middle * u1, middle * u2,
                midpoints, radii, e1, e2,
                extra_vertex, has_extra_vertex,
            )
            if depth_middle >= 0.0:
                low = middle
            else:
                high = middle
        result[i] = low / norm
    return result


def meissner_radial_exact(
    directions: np.ndarray,
    kind: str,
    iterations: int = MEISSNER_BOUNDARY_BISECTION_ITERATIONS,
) -> np.ndarray:
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (n, 3)")
    if np.any(np.linalg.norm(directions, axis=1) == 0.0):
        raise ValueError("directions must be nonzero")

    arc_arrays = meissner_arc_arrays(kind)
    result = meissner_radial_exact_kernel(
        directions,
        arc_arrays[0], arc_arrays[1], arc_arrays[2], arc_arrays[3],
        arc_arrays[4], arc_arrays[5], iterations,
    )
    if np.any(~np.isfinite(result)):
        raise RuntimeError("Could not bracket every Meissner boundary ray")
    return result


def build_meissner_upper_graph(kind: str) -> np.ndarray:
    cache = CACHE_DIR / (
        f"raised_{kind}_exact_arcs_v3_B{N_BASE_MEISSNER}_"
        f"L{len(LAYERS)}_{layer_cache_tag(LAYERS)}.npy"
    )
    if cache.exists():
        return np.load(cache, mmap_mode="r")

    directions = fibonacci_sphere(N_BASE_MEISSNER)
    base_radial = meissner_radial_exact(directions, kind)
    boundary = base_radial[:, None] * directions
    points = np.concatenate([lam * boundary for lam in LAYERS], axis=0)
    depth = np.maximum(0.0, meissner_raw_depth(points, kind))
    upper_height = np.sqrt(np.maximum(0.0, 2.0 * depth - depth * depth))
    graph = np.column_stack((points, upper_height))
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache, graph)
    return np.load(cache, mmap_mode="r")


def positive_radial_meissner(
    directions4: np.ndarray, kind: str, iterations: int = 45
) -> np.ndarray:
    low = np.zeros(len(directions4))
    high = np.ones(len(directions4))
    horizontal = directions4[:, :3]
    vertical = directions4[:, 3]

    for _ in range(iterations):
        middle = 0.5 * (low + high)
        projected = middle[:, None] * horizontal
        depth = meissner_raw_depth(projected, kind)
        inside = (
            (depth >= -1.0e-13)
            & (
                middle * middle * vertical * vertical
                + (1.0 - depth) ** 2
                <= 1.0 + 1.0e-13
            )
        )
        low = np.where(inside, middle, low)
        high = np.where(inside, high, middle)
    return low


# ---------------------------------------------------------------------
# U_3: exact continuous six-elliptical-arc oracle
# ---------------------------------------------------------------------
def u3_elliptical_arc_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return A_i,B_i for gamma_i(t)=A_i cos(t)+B_i sin(t).

    The six arcs are indexed by the six unordered pairs of vertices of
    the regular tetrahedron used throughout this program.  This is the
    reflected version of the symmetric coordinates in Section 5 of the
    Mathematika paper and represents the same U_3 up to an isometry.
    """
    vertices_dict = tetrahedron_vertices()
    vertices = np.stack([
        vertices_dict["A"], vertices_dict["B"],
        vertices_dict["C"], vertices_dict["D"],
    ])
    pairs = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=np.int64,
    )
    return vertices[pairs[:, 0]].copy(), vertices[pairs[:, 1]].copy()


@njit(inline="always")
def _append_unique_root(
    roots: np.ndarray, n_roots: int, value: float, lo: float, hi: float
) -> int:
    tol = 2.0e-11
    if value < lo - tol or value > hi + tol:
        return n_roots
    if value < lo:
        value = lo
    elif value > hi:
        value = hi
    for i in range(n_roots):
        if abs(roots[i] - value) <= tol:
            return n_roots
    roots[n_roots] = value
    return n_roots + 1


@njit(inline="always")
def _sort_small(values: np.ndarray, n: int) -> None:
    for i in range(1, n):
        x = values[i]
        j = i - 1
        while j >= 0 and values[j] > x:
            values[j + 1] = values[j]
            j -= 1
        values[j + 1] = x


@njit(inline="always", fastmath=True)
def _eval_quadratic(a: float, b: float, c: float, x: float) -> float:
    return (a * x + b) * x + c


@njit(inline="always", fastmath=True)
def _eval_cubic(a: float, b: float, c: float, d: float, x: float) -> float:
    return ((a * x + b) * x + c) * x + d


@njit(inline="always", fastmath=True)
def _eval_quartic(
    a: float, b: float, c: float, d: float, e: float, x: float
) -> float:
    return (((a * x + b) * x + c) * x + d) * x + e


@njit
def _quadratic_roots_interval(
    a: float, b: float, c: float, lo: float, hi: float, roots: np.ndarray
) -> int:
    scale = 1.0 + abs(a) + abs(b) + abs(c)
    eps = 1.0e-14 * scale
    n_roots = 0
    if abs(a) <= eps:
        if abs(b) > eps:
            n_roots = _append_unique_root(roots, n_roots, -c / b, lo, hi)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= -eps:
            if discriminant < 0.0:
                discriminant = 0.0
            root_disc = np.sqrt(discriminant)
            n_roots = _append_unique_root(
                roots, n_roots, (-b - root_disc) / (2.0 * a), lo, hi
            )
            n_roots = _append_unique_root(
                roots, n_roots, (-b + root_disc) / (2.0 * a), lo, hi
            )
    _sort_small(roots, n_roots)
    return n_roots


@njit
def _bisect_cubic(
    a: float, b: float, c: float, d: float,
    lo: float, hi: float, iterations: int,
) -> float:
    f_lo = _eval_cubic(a, b, c, d, lo)
    for _ in range(iterations):
        middle = 0.5 * (lo + hi)
        f_mid = _eval_cubic(a, b, c, d, middle)
        if f_lo * f_mid <= 0.0:
            hi = middle
        else:
            lo = middle
            f_lo = f_mid
    return 0.5 * (lo + hi)


@njit
def _cubic_roots_interval(
    a: float, b: float, c: float, d: float,
    lo: float, hi: float, roots: np.ndarray, iterations: int,
) -> int:
    scale = 1.0 + abs(a) + abs(b) + abs(c) + abs(d)
    eps = 1.0e-13 * scale
    if abs(a) <= 1.0e-14 * scale:
        return _quadratic_roots_interval(b, c, d, lo, hi, roots)

    critical = np.empty(2)
    n_critical = _quadratic_roots_interval(
        3.0 * a, 2.0 * b, c, lo, hi, critical
    )
    bounds = np.empty(4)
    n_bounds = 1
    bounds[0] = lo
    for i in range(n_critical):
        if critical[i] > lo + 2.0e-11 and critical[i] < hi - 2.0e-11:
            bounds[n_bounds] = critical[i]
            n_bounds += 1
    bounds[n_bounds] = hi
    n_bounds += 1
    _sort_small(bounds, n_bounds)

    n_roots = 0
    for i in range(n_bounds):
        x = bounds[i]
        if abs(_eval_cubic(a, b, c, d, x)) <= eps:
            n_roots = _append_unique_root(roots, n_roots, x, lo, hi)
    for i in range(n_bounds - 1):
        left, right = bounds[i], bounds[i + 1]
        f_left = _eval_cubic(a, b, c, d, left)
        f_right = _eval_cubic(a, b, c, d, right)
        if f_left * f_right < 0.0:
            root = _bisect_cubic(a, b, c, d, left, right, iterations)
            n_roots = _append_unique_root(roots, n_roots, root, lo, hi)
    _sort_small(roots, n_roots)
    return n_roots


@njit
def _bisect_quartic(
    a: float, b: float, c: float, d: float, e: float,
    lo: float, hi: float, iterations: int,
) -> float:
    f_lo = _eval_quartic(a, b, c, d, e, lo)
    for _ in range(iterations):
        middle = 0.5 * (lo + hi)
        f_mid = _eval_quartic(a, b, c, d, e, middle)
        if f_lo * f_mid <= 0.0:
            hi = middle
        else:
            lo = middle
            f_lo = f_mid
    return 0.5 * (lo + hi)


@njit
def _quartic_roots_unit_interval(
    a: float, b: float, c: float, d: float, e: float,
    roots: np.ndarray, iterations: int,
) -> int:
    lo, hi = 0.0, 1.0
    scale = 1.0 + abs(a) + abs(b) + abs(c) + abs(d) + abs(e)
    eps = 1.0e-13 * scale
    if abs(a) <= 1.0e-14 * scale:
        return _cubic_roots_interval(b, c, d, e, lo, hi, roots, iterations)

    critical = np.empty(3)
    n_critical = _cubic_roots_interval(
        4.0 * a, 3.0 * b, 2.0 * c, d,
        lo, hi, critical, iterations,
    )
    bounds = np.empty(5)
    n_bounds = 1
    bounds[0] = lo
    for i in range(n_critical):
        if critical[i] > lo + 2.0e-11 and critical[i] < hi - 2.0e-11:
            bounds[n_bounds] = critical[i]
            n_bounds += 1
    bounds[n_bounds] = hi
    n_bounds += 1
    _sort_small(bounds, n_bounds)

    n_roots = 0
    for i in range(n_bounds):
        x = bounds[i]
        if abs(_eval_quartic(a, b, c, d, e, x)) <= eps:
            n_roots = _append_unique_root(roots, n_roots, x, lo, hi)
    for i in range(n_bounds - 1):
        left, right = bounds[i], bounds[i + 1]
        f_left = _eval_quartic(a, b, c, d, e, left)
        f_right = _eval_quartic(a, b, c, d, e, right)
        if f_left * f_right < 0.0:
            root = _bisect_quartic(
                a, b, c, d, e, left, right, iterations
            )
            n_roots = _append_unique_root(roots, n_roots, root, lo, hi)
    _sort_small(roots, n_roots)
    return n_roots


@njit(inline="always", fastmath=True)
def _squared_distance_to_ellipse_parameter(
    x0: float, x1: float, x2: float,
    a0: float, a1: float, a2: float,
    b0: float, b1: float, b2: float,
    cosine: float, sine: float,
) -> float:
    p0 = a0 * cosine + b0 * sine
    p1 = a1 * cosine + b1 * sine
    p2 = a2 * cosine + b2 * sine
    z0, z1, z2 = x0 - p0, x1 - p1, x2 - p2
    return z0 * z0 + z1 * z1 + z2 * z2


@njit(fastmath=True)
def u3_max_distance2_scalar(
    x0: float, x1: float, x2: float,
    arc_a: np.ndarray, arc_b: np.ndarray,
    root_iterations: int,
) -> float:
    """Maximum squared distance to the six continuous elliptical arcs."""
    global_max = -1.0e300
    roots = np.empty(4)

    for j in range(arc_a.shape[0]):
        a0, a1, a2 = arc_a[j, 0], arc_a[j, 1], arc_a[j, 2]
        b0, b1, b2 = arc_b[j, 0], arc_b[j, 1], arc_b[j, 2]

        aa = a0 * a0 + a1 * a1 + a2 * a2
        bb = b0 * b0 + b1 * b1 + b2 * b2
        ab = a0 * b0 + a1 * b1 + a2 * b2
        ya = x0 * a0 + x1 * a1 + x2 * a2
        yb = x0 * b0 + x1 * b1 + x2 * b2
        delta = bb - aa

        # With q=tan(t/2), the stationary equation becomes
        # c4*q^4+c3*q^3+c2*q^2+c1*q+c0=0 on q in [0,1].
        c4 = ab + yb
        c3 = -2.0 * delta + 2.0 * ya
        c2 = -6.0 * ab
        c1 = 2.0 * delta + 2.0 * ya
        c0 = ab - yb
        n_roots = _quartic_roots_unit_interval(
            c4, c3, c2, c1, c0, roots, root_iterations
        )

        best = _squared_distance_to_ellipse_parameter(
            x0, x1, x2, a0, a1, a2, b0, b1, b2, 1.0, 0.0
        )
        endpoint = _squared_distance_to_ellipse_parameter(
            x0, x1, x2, a0, a1, a2, b0, b1, b2, 0.0, 1.0
        )
        if endpoint > best:
            best = endpoint

        for k in range(n_roots):
            q = roots[k]
            denominator = 1.0 + q * q
            cosine = (1.0 - q * q) / denominator
            sine = 2.0 * q / denominator
            value = _squared_distance_to_ellipse_parameter(
                x0, x1, x2,
                a0, a1, a2, b0, b1, b2,
                cosine, sine,
            )
            if value > best:
                best = value

        if best > global_max:
            global_max = best
    return global_max


@njit(inline="always", fastmath=True)
def u3_raw_depth_scalar(
    x0: float, x1: float, x2: float,
    arc_a: np.ndarray, arc_b: np.ndarray,
    root_iterations: int,
) -> float:
    max_distance2 = u3_max_distance2_scalar(
        x0, x1, x2, arc_a, arc_b, root_iterations
    )
    if max_distance2 < 0.0:
        max_distance2 = 0.0
    return 1.0 - np.sqrt(max_distance2)


@njit(parallel=True, fastmath=True)
def u3_raw_depth_kernel(
    points: np.ndarray,
    arc_a: np.ndarray,
    arc_b: np.ndarray,
    root_iterations: int,
) -> np.ndarray:
    result = np.empty(points.shape[0])
    for i in prange(points.shape[0]):
        result[i] = u3_raw_depth_scalar(
            points[i, 0], points[i, 1], points[i, 2],
            arc_a, arc_b, root_iterations,
        )
    return result


def u3_raw_depth(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    arc_a, arc_b = u3_elliptical_arc_arrays()
    return u3_raw_depth_kernel(
        points, arc_a, arc_b, U3_ARC_ROOT_BISECTION_ITERATIONS
    )


@njit(parallel=True, fastmath=True)
def u3_radial_exact_kernel(
    directions: np.ndarray,
    arc_a: np.ndarray,
    arc_b: np.ndarray,
    boundary_iterations: int,
    root_iterations: int,
) -> np.ndarray:
    result = np.empty(directions.shape[0])
    for i in prange(directions.shape[0]):
        d0, d1, d2 = directions[i, 0], directions[i, 1], directions[i, 2]
        norm = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if norm == 0.0:
            result[i] = np.nan
            continue
        u0, u1, u2 = d0 / norm, d1 / norm, d2 / norm
        low, high = 0.0, 1.0
        bracketed = False
        for _ in range(4):
            if u3_raw_depth_scalar(
                high * u0, high * u1, high * u2,
                arc_a, arc_b, root_iterations,
            ) < 0.0:
                bracketed = True
                break
            high *= 2.0
        if not bracketed:
            result[i] = np.nan
            continue

        for _ in range(boundary_iterations):
            middle = 0.5 * (low + high)
            depth = u3_raw_depth_scalar(
                middle * u0, middle * u1, middle * u2,
                arc_a, arc_b, root_iterations,
            )
            if depth >= 0.0:
                low = middle
            else:
                high = middle
        result[i] = low / norm
    return result


def u3_radial_exact(
    directions: np.ndarray,
    iterations: int = U3_BOUNDARY_BISECTION_ITERATIONS,
) -> np.ndarray:
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (n, 3)")
    if np.any(np.linalg.norm(directions, axis=1) == 0.0):
        raise ValueError("directions must be nonzero")
    arc_a, arc_b = u3_elliptical_arc_arrays()
    result = u3_radial_exact_kernel(
        directions, arc_a, arc_b, iterations,
        U3_ARC_ROOT_BISECTION_ITERATIONS,
    )
    if np.any(~np.isfinite(result)):
        raise RuntimeError("Could not bracket every U_3 boundary ray")
    return result


def build_u3_upper_graph() -> np.ndarray:
    cache = CACHE_DIR / (
        f"raised_U3_exact_elliptical_arcs_v1_B{N_BASE_U3}_"
        f"L{len(LAYERS)}_{layer_cache_tag(LAYERS)}.npy"
    )
    if cache.exists():
        return np.load(cache, mmap_mode="r")

    directions = fibonacci_sphere(N_BASE_U3)
    base_radial = u3_radial_exact(directions)
    boundary = base_radial[:, None] * directions
    points = np.concatenate([lam * boundary for lam in LAYERS], axis=0)
    depth = np.maximum(0.0, u3_raw_depth(points))
    upper_height = np.sqrt(np.maximum(0.0, 2.0 * depth - depth * depth))
    graph = np.column_stack((points, upper_height))
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache, graph)
    return np.load(cache, mmap_mode="r")


@njit(parallel=True, fastmath=True)
def positive_radial_u3_kernel(
    directions4: np.ndarray,
    arc_a: np.ndarray,
    arc_b: np.ndarray,
    boundary_iterations: int,
    root_iterations: int,
) -> np.ndarray:
    result = np.empty(directions4.shape[0])
    for i in prange(directions4.shape[0]):
        h0, h1, h2 = directions4[i, 0], directions4[i, 1], directions4[i, 2]
        vertical = directions4[i, 3]
        low, high = 0.0, 1.0
        for _ in range(boundary_iterations):
            middle = 0.5 * (low + high)
            depth = u3_raw_depth_scalar(
                middle * h0, middle * h1, middle * h2,
                arc_a, arc_b, root_iterations,
            )
            farthest = 1.0 - depth
            inside = (
                depth >= -1.0e-13
                and middle * middle * vertical * vertical
                + farthest * farthest <= 1.0 + 1.0e-13
            )
            if inside:
                low = middle
            else:
                high = middle
        result[i] = low
    return result


def positive_radial_u3(
    directions4: np.ndarray,
    iterations: int = U3_BOUNDARY_BISECTION_ITERATIONS,
) -> np.ndarray:
    directions4 = np.asarray(directions4, dtype=float)
    arc_a, arc_b = u3_elliptical_arc_arrays()
    return positive_radial_u3_kernel(
        directions4, arc_a, arc_b, iterations,
        U3_ARC_ROOT_BISECTION_ITERATIONS,
    )


# ---------------------------------------------------------------------
# Four-dimensional radial estimators
# ---------------------------------------------------------------------
def radial_raised_meissner(
    directions4: np.ndarray, kind: str, upper_graph: np.ndarray
) -> np.ndarray:
    result = np.empty(len(directions4))
    positive = directions4[:, 3] >= 0.0
    if np.any(positive):
        result[positive] = positive_radial_meissner(
            directions4[positive], kind
        )
    if np.any(~positive):
        result[~positive] = radial_intersection_unit_balls(
            directions4[~positive], upper_graph
        )
    return result


def radial_raised_u3(
    directions4: np.ndarray, upper_graph: np.ndarray
) -> np.ndarray:
    result = np.empty(len(directions4))
    positive = directions4[:, 3] >= 0.0
    if np.any(positive):
        result[positive] = positive_radial_u3(directions4[positive])
    if np.any(~positive):
        result[~positive] = radial_intersection_unit_balls(
            directions4[~positive], upper_graph
        )
    return result


def antithetic_contributions(
    base_directions: np.ndarray, radial_function
) -> np.ndarray:
    radial_positive = radial_function(base_directions)
    radial_negative = radial_function(-base_directions)
    return 0.5 * VOL_B4 * (
        radial_positive**4 + radial_negative**4
    )


def summarize(values: np.ndarray):
    estimate = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return estimate, standard_error, (
        estimate - 1.96 * standard_error,
        estimate + 1.96 * standard_error,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone radial Monte-Carlo reference computation for "
            "dimension-raised M_F, M_V, and U3 bodies."
        )
    )
    parser.add_argument(
        "--bodies",
        nargs="+",
        choices=("M_F", "M_V", "U3"),
        default=list(DEFAULT_BODIES),
        help="Bodies to compute. The default is all three.",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=DEFAULT_N_PAIRS,
        help="Number of antipodal S^3 direction pairs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the common S^3 directions.",
    )
    arguments = parser.parse_args()
    if arguments.n_pairs < 2:
        parser.error("--n-pairs must be at least 2")
    arguments.bodies = tuple(dict.fromkeys(arguments.bodies))
    return arguments


def main() -> None:
    arguments = parse_arguments()
    start = perf_counter()
    directions = random_sphere(arguments.n_pairs, 4, arguments.seed)
    contributions = {}

    meissner_kinds = {"M_F": "face", "M_V": "vertex"}
    for label in arguments.bodies:
        if label not in meissner_kinds:
            continue
        kind = meissner_kinds[label]
        graph = build_meissner_upper_graph(kind)
        contributions[label] = antithetic_contributions(
            directions,
            lambda u, k=kind, g=graph: radial_raised_meissner(u, k, g),
        )
        estimate, se, interval = summarize(contributions[label])
        print(
            f"{label}: {estimate:.12f}  SE={se:.12f}  "
            f"95% CI=[{interval[0]:.12f}, {interval[1]:.12f}]"
        )
        del graph

    if "U3" in arguments.bodies:
        graph = build_u3_upper_graph()
        contributions["U3"] = antithetic_contributions(
            directions, lambda u: radial_raised_u3(u, graph)
        )
        estimate, se, interval = summarize(contributions["U3"])
        print(
            f"U3:  {estimate:.12f}  SE={se:.12f}  "
            f"95% CI=[{interval[0]:.12f}, {interval[1]:.12f}]"
        )
        del graph

    paired_comparisons = [
        ("U3", "M_F"),
        ("U3", "M_V"),
        ("M_F", "M_V"),
    ]
    available = [
        pair for pair in paired_comparisons
        if pair[0] in contributions and pair[1] in contributions
    ]
    if available:
        print("\nPaired differences using the same antipodal directions:")
        for first, second in available:
            difference = contributions[first] - contributions[second]
            estimate, se, interval = summarize(difference)
            print(
                f"{first}-{second}: {estimate:.12f}  SE={se:.12f}  "
                f"95% CI=[{interval[0]:.12f}, {interval[1]:.12f}]"
            )

    print(f"\nElapsed time: {perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()
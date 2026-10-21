#!/usr/bin/env python3
"""Adaptive rational-polytope inner approximation for a raised Meissner body.

This is a longer-run adaptive pilot, not yet the final rigorous certificate.

Version 4 retains the coordinatewise rationalization fix and adds:
* wall-clock and per-stage timing with ETA reporting;
* cached negative-witness geometry;
* continuation from a checkpoint with additional Sobol directions;
* a facet budget for cheaper adaptive refinement.

Mathematical inclusion used by the pilot
----------------------------------------
* Positive vertices are chosen inside the exact positive half K_{M,+}, using
  the continuous three-circular-arc Meissner oracle (M_F by default).
* Negative vertices are chosen inside the universal inner model

      L_- = {x_4 <= 0} cap union_{m in T} B_4(m, eta(m)),
      eta(m) = 1 - sqrt(5/8 + ||m||^2),

  with rational tetrahedral witnesses m.
* Therefore the convex hull of all accepted vertices is contained in the
  canonical raised Meissner body.

Coordinates
-----------
Horizontal coordinates are coefficients in the basis (A,B,C), where A,B,C,D
are the centered vertices of a unit-edge regular tetrahedron and D=-A-B-C.
The horizontal Gram matrix is rational:

      G3 = (1/8) * [[3,-1,-1],[-1,3,-1],[-1,-1,3]].

Every stored vertex has rational coefficient coordinates.  The current hull
volume is evaluated by scipy.spatial.ConvexHull in floating point; a later
certificate version will recompute the accepted hull triangulation using exact
integer determinants and the fixed factor sqrt(det G3).
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.special import ndtri
from scipy.stats import qmc

ALPHA = np.arccos(1.0 / 3.0)
G3 = np.array([[3.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 3.0]]) / 8.0
SQRT_DET_G3 = float(np.sqrt(np.linalg.det(G3)))


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TimingBook:
    def __init__(self) -> None:
        self.started_wall = datetime.now().astimezone()
        self.started_perf = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        print(f"[timing] start {name}", flush=True)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed
            print(f"[timing] end   {name}: {format_seconds(elapsed)}", flush=True)

    def total(self) -> float:
        return time.perf_counter() - self.started_perf

    def payload(self, args: argparse.Namespace) -> dict:
        end = datetime.now().astimezone()
        return {
            "start": self.started_wall.isoformat(),
            "end": end.isoformat(),
            "elapsed_seconds": self.total(),
            "elapsed": format_seconds(self.total()),
            "stages_seconds": self.stages,
            "arguments": vars(args),
        }


def tetrahedron_vertices_cart() -> dict[str, np.ndarray]:
    s = 1.0 / (2.0 * np.sqrt(2.0))
    return {
        "A": s * np.array([1.0, 1.0, 1.0]),
        "B": s * np.array([1.0, -1.0, -1.0]),
        "C": s * np.array([-1.0, 1.0, -1.0]),
        "D": s * np.array([-1.0, -1.0, 1.0]),
    }


VERT = tetrahedron_vertices_cart()
BASIS = np.column_stack((VERT["A"], VERT["B"], VERT["C"]))
BASIS_INV = np.linalg.inv(BASIS)


def coeff_to_cart(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    out = np.empty_like(points)
    out[:, :3] = points[:, :3] @ BASIS.T
    out[:, 3] = points[:, 3]
    return out


def primitive(q: Iterable[int]) -> tuple[int, int, int, int]:
    values = tuple(int(v) for v in q)
    g = 0
    for v in values:
        g = gcd(g, abs(v))
    if g == 0:
        raise ValueError("zero direction")
    return tuple(v // g for v in values)


def lattice_rays(radius: int) -> list[tuple[int, int, int, int]]:
    """All primitive integer rays with max norm at most ``radius``."""
    rays: set[tuple[int, int, int, int]] = set()
    for q in product(range(-radius, radius + 1), repeat=4):
        if q == (0, 0, 0, 0):
            continue
        rays.add(primitive(q))
    return sorted(rays)



def sobol_rays(exponent: int, scale: int, seed: int) -> list[tuple[int, int, int, int]]:
    """Quantized nearly uniform Euclidean directions from a scrambled Sobol net.

    The final rays are primitive integer vectors in tetrahedral coefficient
    coordinates, so every accepted boundary point remains rational.  Sobol is
    used only to choose directions more uniformly than a small lattice box.
    """
    if exponent <= 0:
        return []
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    points = sampler.random_base2(exponent)
    # Avoid infinities in the inverse normal transform.
    eps = np.finfo(float).eps
    normals = ndtri(np.clip(points, eps, 1.0 - eps))
    norms = np.linalg.norm(normals, axis=1)
    normals = normals[norms > 0] / norms[norms > 0, None]
    rays: set[tuple[int, int, int, int]] = set()
    for normal in normals:
        qdir = direction_from_cart_normal(normal, scale)
        if qdir is not None:
            rays.add(qdir)
            rays.add(tuple(-v for v in qdir))
    return sorted(rays)


def arc_data(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray):
    rho = np.sqrt(3.0) / 2.0
    midpoint = 0.5 * (z + w)
    e1 = (x - midpoint) / rho
    e2 = ((y - midpoint) / rho - np.cos(ALPHA) * e1) / np.sin(ALPHA)
    return midpoint, rho, e1, e2


def meissner_arcs(kind: str = "face"):
    if kind == "face":
        data = [("A", "B", "C", "D"), ("A", "C", "B", "D"), ("A", "D", "B", "C")]
    elif kind == "vertex":
        data = [("B", "C", "A", "D"), ("B", "D", "A", "C"), ("C", "D", "A", "B")]
    else:
        raise ValueError("kind must be face or vertex")
    return [arc_data(VERT[x], VERT[y], VERT[z], VERT[w]) for x, y, z, w in data]


ARCS_FACE = meissner_arcs("face")
ARCS_VERTEX = meissner_arcs("vertex")


def max_squared_distance_to_arc_single(y: np.ndarray, arc) -> float:
    midpoint, rho, e1, e2 = arc
    z = y - midpoint
    a = float(z @ e1)
    b = float(z @ e2)
    minimum = min(a, a * np.cos(ALPHA) + b * np.sin(ALPHA))
    angle = float(np.mod(np.arctan2(b, a) + np.pi, 2.0 * np.pi))
    if angle <= ALPHA:
        minimum = min(minimum, -float(np.hypot(a, b)))
    return float(z @ z + rho * rho - 2.0 * rho * minimum)


def meissner_max_distance2(y: np.ndarray, kind: str = "face") -> float:
    arcs = ARCS_FACE if kind == "face" else ARCS_VERTEX
    value = max(max_squared_distance_to_arc_single(y, arc) for arc in arcs)
    if kind == "vertex":
        value = max(value, float(np.sum((y - VERT["A"]) ** 2)))
    return value


def positive_inside_coeff(x: np.ndarray, kind: str = "face") -> tuple[bool, float]:
    y = BASIS @ x[:3]
    d2 = meissner_max_distance2(y, kind)
    residual = 1.0 - (d2 + x[3] * x[3])
    return bool(x[3] >= -1e-15 and residual >= 0.0), float(residual)


def positive_radial(q: np.ndarray, kind: str = "face", iterations: int = 55) -> float:
    assert q[3] >= 0
    low, high = 0.0, 1.0
    while positive_inside_coeff(high * q, kind)[0]:
        high *= 2.0
        if high > 16:
            raise RuntimeError("failed to bracket positive radial root")
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if positive_inside_coeff(mid * q, kind)[0]:
            low = mid
        else:
            high = mid
    return low


def tetra_witness_grid(den: int):
    coeffs = []
    bary = []
    # barycentric weights (a,b,c,d)/den; D has coefficient (-1,-1,-1)
    for ia in range(den + 1):
        for ib in range(den + 1 - ia):
            for ic in range(den + 1 - ia - ib):
                id_ = den - ia - ib - ic
                lam = np.array([ia, ib, ic, id_], dtype=float) / den
                m = np.array([lam[0] - lam[3], lam[1] - lam[3], lam[2] - lam[3]])
                norm2 = float(m @ G3 @ m)
                eta = 1.0 - np.sqrt(5.0 / 8.0 + norm2)
                if eta > 0:
                    coeffs.append(m)
                    bary.append((ia, ib, ic, id_))
    return np.asarray(coeffs), np.asarray(bary, dtype=np.int64)


@dataclass(frozen=True)
class WitnessGrid:
    coeff: np.ndarray
    bary: np.ndarray
    metric_coeff: np.ndarray
    norm2: np.ndarray
    eta: np.ndarray


def build_witness_grid(den: int) -> WitnessGrid:
    coeff, bary = tetra_witness_grid(den)
    metric_coeff = coeff @ G3
    norm2 = np.einsum("ni,ij,nj->n", coeff, G3, coeff)
    eta = 1.0 - np.sqrt(5.0 / 8.0 + norm2)
    return WitnessGrid(coeff, bary, metric_coeff, norm2, eta)


@dataclass
class NegativeResult:
    radial: float
    witness_index: int
    residual: float


def negative_inner_radial(q: np.ndarray, witness: WitnessGrid) -> NegativeResult:
    assert q[3] < 0
    qh = q[:3]
    qnorm2 = float(qh @ G3 @ qh + q[3] * q[3])
    mq = witness.metric_coeff @ qh
    disc = mq * mq + qnorm2 * (witness.eta * witness.eta - witness.norm2)
    valid = disc >= 0.0
    roots = np.full(len(witness.coeff), -np.inf)
    roots[valid] = (mq[valid] + np.sqrt(np.maximum(disc[valid], 0.0))) / qnorm2
    j = int(np.argmax(roots))
    radial = float(roots[j])
    if not np.isfinite(radial) or radial <= 0:
        return NegativeResult(0.0, j, -np.inf)
    x = radial * q
    diff = x[:3] - witness.coeff[j]
    dist2 = float(diff @ G3 @ diff + x[3] * x[3])
    residual = float(witness.eta[j] * witness.eta[j] - dist2)
    return NegativeResult(radial, j, residual)


def negative_inside_with_witness(x: np.ndarray, m: np.ndarray) -> tuple[bool, float]:
    mnorm2 = float(m @ G3 @ m)
    eta = 1.0 - np.sqrt(5.0 / 8.0 + mnorm2)
    diff = x[:3] - m
    residual = float(eta * eta - (diff @ G3 @ diff + x[3] * x[3]))
    return bool(x[3] <= 1e-15 and residual >= 0.0), residual


def nearest_rational_point(
    q: np.ndarray,
    radial: float,
    denominator: int,
    inward: float,
) -> np.ndarray:
    """Round the final point coordinatewise.

    Version 2 rounded the scalar ray parameter.  For a quantized Sobol ray
    whose integer coefficients have size about ``ray_scale``, that creates a
    physical rounding loss of order ``ray_scale / denominator``.  We instead
    round the final four coordinates, so the perturbation is
    ``O(1 / denominator)`` independently of the integer ray size.
    """
    target = radial * (1.0 - inward) * q
    return np.rint(target * denominator) / denominator


def contract_rational_numerator(
    x: np.ndarray,
    denominator: int,
    steps: int = 1,
) -> np.ndarray:
    """Move a rational point toward the origin by integer grid steps."""
    numerator = np.rint(x * denominator).astype(np.int64)
    for _ in range(steps):
        numerator = (
            np.sign(numerator)
            * np.maximum(np.abs(numerator) - 1, 0)
        )
    return numerator.astype(float) / denominator


def direction_from_cart_normal(normal: np.ndarray, scale: int) -> tuple[int, int, int, int] | None:
    coeff = np.empty(4)
    coeff[:3] = BASIS_INV @ normal[:3]
    coeff[3] = normal[3]
    maxabs = float(np.max(np.abs(coeff)))
    if maxabs == 0:
        return None
    q = np.rint(scale * coeff / maxabs).astype(int)
    if np.all(q == 0):
        return None
    return primitive(q)


@dataclass
class PointRecord:
    direction: tuple[int, int, int, int]
    point: np.ndarray
    side: str
    residual: float
    witness_index: int = -1


def load_direction_set(path: Path) -> set[tuple[int, int, int, int]]:
    """Load only primitive ray directions from an earlier checkpoint.

    Unlike ``--resume``, this option recomputes every boundary point with the
    current rationalization and witness settings.
    """
    data = np.load(path, allow_pickle=False)
    return {
        primitive(tuple(int(v) for v in row))
        for row in data["directions"]
    }


def load_records(path: Path, denominator: int) -> list[PointRecord]:
    data = np.load(path, allow_pickle=False)
    stored_den = int(data["denominator"])
    if stored_den != denominator:
        raise ValueError(
            f"resume denominator {stored_den} does not match requested {denominator}"
        )
    directions = data["directions"]
    numerators = data["numerators"]
    sides = data["sides"]
    residuals = data["residuals"]
    witnesses = data["witness_indices"]
    records: list[PointRecord] = []
    for qv, nv, sv, rv, wv in zip(directions, numerators, sides, residuals, witnesses):
        records.append(
            PointRecord(
                tuple(int(x) for x in qv),
                np.asarray(nv, dtype=float) / denominator,
                "positive" if int(sv) > 0 else "negative",
                float(rv),
                int(wv),
            )
        )
    return records


def propose_record(
    direction: tuple[int, int, int, int],
    denominator: int,
    inward: float,
    kind: str,
    witness: WitnessGrid,
) -> PointRecord | None:
    q = np.asarray(direction, dtype=float)
    if q[3] >= 0:
        radial = positive_radial(q, kind)
        x = nearest_rational_point(q, radial, denominator, inward)
        ok, residual = positive_inside_coeff(x, kind)
        for _ in range(64):
            if ok:
                break
            x = contract_rational_numerator(x, denominator)
            ok, residual = positive_inside_coeff(x, kind)
        if not ok:
            return None
        return PointRecord(direction, x, "positive", residual)

    neg = negative_inner_radial(q, witness)
    if neg.radial <= 0:
        return None
    x = nearest_rational_point(q, neg.radial, denominator, inward)
    ok, residual = negative_inside_with_witness(
        x, witness.coeff[neg.witness_index]
    )
    for _ in range(64):
        if ok:
            break
        x = contract_rational_numerator(x, denominator)
        ok, residual = negative_inside_with_witness(
            x, witness.coeff[neg.witness_index]
        )
    if not ok:
        return None
    return PointRecord(
        direction,
        x,
        "negative",
        residual,
        neg.witness_index,
    )


def hull_from_records(records: list[PointRecord]):
    coeff_points = np.vstack([np.zeros(4)] + [r.point for r in records])
    cart_points = coeff_to_cart(coeff_points)
    hull = ConvexHull(cart_points)
    return hull, coeff_points, cart_points


def adaptive_round(
    records: list[PointRecord],
    seen: set[tuple[int, int, int, int]],
    normal_scale: int,
    max_candidates: int,
    denominator: int,
    inward: float,
    kind: str,
    witness: WitnessGrid,
    facet_pair_rays: bool = False,
    facet_budget: int = 0,
):
    """Refine through rational rays through current facet barycentres.

    A facet of an inscribed star polytope approximates a curved radial patch.
    The ray through its barycentre is therefore a more useful refinement
    direction than the Euclidean facet normal.  Because every stored vertex
    has denominator ``denominator``, the sum of the facet numerators gives an
    exact integer direction.  Normal-derived directions are also included as
    a secondary safeguard.
    """
    hull, coeff_points, cart_points = hull_from_records(records)
    proposal_map: dict[tuple[int, int, int, int], tuple[np.ndarray, float]] = {}

    facet_indices = np.arange(len(hull.simplices))
    if facet_budget > 0 and len(facet_indices) > facet_budget:
        simplices = hull.simplices
        pts = cart_points[simplices]
        # Maximum squared edge length is a cheap curvature-error proxy.
        edge_proxy = np.zeros(len(simplices))
        for aa in range(4):
            for bb in range(aa + 1, 4):
                edge_proxy = np.maximum(
                    edge_proxy,
                    np.sum((pts[:, aa] - pts[:, bb]) ** 2, axis=1),
                )
        facet_indices = np.argpartition(edge_proxy, -facet_budget)[-facet_budget:]

    for facet_index in facet_indices:
        equation = hull.equations[facet_index]
        normal, offset = equation[:4], equation[4]
        simplex = hull.simplices[facet_index]

        # Primary candidate: ray through the rational facet barycentre.
        summed = np.rint(denominator * np.sum(coeff_points[simplex], axis=0)).astype(int)
        if np.any(summed):
            q = primitive(summed)
            proposal_map.setdefault(q, (normal, offset))

        if facet_pair_rays:
            # Optional secondary refinement: rays through the six edge
            # midpoints of the tetrahedral facet.
            for aa in range(len(simplex)):
                for bb in range(aa + 1, len(simplex)):
                    pair_sum = np.rint(
                        denominator
                        * (coeff_points[simplex[aa]] + coeff_points[simplex[bb]])
                    ).astype(int)
                    if np.any(pair_sum):
                        qp = primitive(pair_sum)
                        proposal_map.setdefault(qp, (normal, offset))

        # Secondary candidate: rationalized Euclidean normal.
        qn = direction_from_cart_normal(normal, normal_scale)
        if qn is not None:
            proposal_map.setdefault(qn, (normal, offset))

    candidates = []
    for q, (normal, offset) in proposal_map.items():
        if q in seen:
            continue
        rec = propose_record(q, denominator, inward, kind, witness)
        seen.add(q)
        if rec is None:
            continue
        point_cart = coeff_to_cart(rec.point[None, :])[0]
        violation = float(normal @ point_cart + offset)
        if violation > 1e-12:
            candidates.append((violation, rec))

    candidates.sort(key=lambda item: item[0], reverse=True)
    accepted = [rec for _, rec in candidates[:max_candidates]]
    records.extend(accepted)
    return len(accepted), (candidates[0][0] if candidates else 0.0)



def det4_bareiss(matrix: np.ndarray) -> int:
    """Exact determinant of a 4x4 integer matrix."""
    a = [[int(x) for x in row] for row in np.asarray(matrix)]
    sign = 1
    previous = 1
    for k in range(3):
        if a[k][k] == 0:
            swap = next((i for i in range(k + 1, 4) if a[i][k] != 0), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, 4):
            for j in range(k + 1, 4):
                a[i][j] = (a[i][j] * pivot - a[i][k] * a[k][j]) // previous
        previous = pivot
    return sign * a[3][3]


def exact_hull_volume_coefficient(records: list[PointRecord], hull: ConvexHull, denominator: int):
    """Return S with Vol(P)=S*sqrt(2)/(192*denominator**4).

    This exactly evaluates the triangulation returned by Qhull.  A final
    certificate must additionally verify the hull combinatorics and every
    vertex-membership inequality with outward-rounded arithmetic.
    """
    numerators = np.vstack([
        np.zeros((1, 4), dtype=np.int64),
        np.rint(np.vstack([r.point for r in records]) * denominator).astype(np.int64),
    ])
    total = 0
    for simplex in hull.simplices:
        if 0 in simplex:
            raise RuntimeError("origin unexpectedly lies on a hull facet")
        total += abs(det4_bareiss(numerators[simplex]))
    return total

def write_npz(path: Path, records: list[PointRecord], hull: ConvexHull, denominator: int, witness_den: int):
    directions = np.asarray([r.direction for r in records], dtype=np.int64)
    numerators = np.rint(np.vstack([r.point for r in records]) * denominator).astype(np.int64)
    sides = np.asarray([1 if r.side == "positive" else -1 for r in records], dtype=np.int8)
    residuals = np.asarray([r.residual for r in records])
    witnesses = np.asarray([r.witness_index for r in records], dtype=np.int64)
    np.savez_compressed(
        path,
        directions=directions,
        numerators=numerators,
        denominator=np.int64(denominator),
        sides=sides,
        residuals=residuals,
        witness_indices=witnesses,
        witness_denominator=np.int64(witness_den),
        hull_simplices=hull.simplices,
        hull_equations=hull.equations,
        float_volume=np.float64(hull.volume),
    )


def diagnostic_inner_model_volume(
    witness: WitnessGrid,
    exponent: int,
    seed: int,
    kind: str,
) -> tuple[float, float, float]:
    """Randomized-QMC diagnostic for the continuous radial inner model.

    This estimates the volume approached by infinitely dense boundary rays
    for the *current finite witness grid*.  It is not a rigorous enclosure.
    """
    if exponent <= 0:
        return float("nan"), float("nan"), float("nan")

    sampler = qmc.Sobol(d=4, scramble=True, seed=seed)
    points = sampler.random_base2(exponent)
    eps = np.finfo(float).eps
    normals = ndtri(np.clip(points, eps, 1.0 - eps))
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    directions = np.empty_like(normals)
    directions[:, :3] = normals[:, :3] @ BASIS_INV.T
    directions[:, 3] = normals[:, 3]

    positive_fourth = []
    negative_fourth = []
    for direction in directions:
        if direction[3] >= 0:
            radial = positive_radial(direction, kind, iterations=45)
            positive_fourth.append(radial**4)
        else:
            radial = negative_inner_radial(direction, witness).radial
            negative_fourth.append(radial**4)

    ball_volume = np.pi**2 / 2.0
    positive = 0.5 * ball_volume * float(np.mean(positive_fourth))
    negative = 0.5 * ball_volume * float(np.mean(negative_fourth))
    return positive + negative, positive, negative


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("face", "vertex"), default="face")
    ap.add_argument("--lattice-radius", type=int, default=3)
    ap.add_argument("--sobol-m", type=int, default=0,
                    help="add 2^m scrambled Sobol directions before quantization")
    ap.add_argument("--ray-scale", type=int, default=4096,
                    help="integer scale used to quantize Sobol directions")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--denominator", type=int, default=65536)
    ap.add_argument("--witness-denominator", type=int, default=18)
    ap.add_argument("--inward", type=float, default=2e-5)
    ap.add_argument("--adaptive-rounds", type=int, default=2)
    ap.add_argument("--normal-scale", type=int, default=48)
    ap.add_argument("--max-add-per-round", type=int, default=1500)
    ap.add_argument("--facet-pair-rays", action="store_true",
                    help="also refine through pairwise facet-vertex midpoints")
    ap.add_argument(
        "--adaptive-facet-budget",
        type=int,
        default=50000,
        help=(
            "maximum number of large facets inspected per adaptive round; "
            "0 scans every facet"
        ),
    )
    ap.add_argument("--resume", default="",
                    help="resume accepted rational vertices from an earlier npz")
    ap.add_argument(
        "--seed-direction-file",
        default="",
        help=(
            "reuse the ray directions from an earlier npz, but recompute all "
            "points with the present denominator, rationalization, and "
            "negative-witness grid"
        ),
    )
    ap.add_argument(
        "--diagnostic-m",
        type=int,
        default=0,
        help=(
            "before constructing the hull, estimate the continuous inner-model "
            "volume with 2^m scrambled Sobol directions; diagnostic only"
        ),
    )
    ap.add_argument("--skip-exact-volume", action="store_true")
    ap.add_argument("--output", default="raised_meissner_rational_polytope_pilot.npz")
    ap.add_argument(
        "--timing-output",
        default="",
        help="JSON timing log; default is <output-stem>_timing.json",
    )
    args = ap.parse_args()
    timing = TimingBook()
    print(f"run start: {timing.started_wall.isoformat()}", flush=True)

    with timing.stage("build negative witness grid"):
        witness = build_witness_grid(args.witness_denominator)
    print(f"negative witnesses: {len(witness.coeff)}")

    if args.diagnostic_m > 0:
        total, positive, negative = diagnostic_inner_model_volume(
            witness,
            args.diagnostic_m,
            args.seed + 991,
            args.kind,
        )
        print(
            "continuous inner-model QMC diagnostic: "
            f"total={total:.15f} positive={positive:.15f} "
            f"negative={negative:.15f}"
        )
        print("WARNING: this diagnostic is not a rigorous enclosure.")

    with timing.stage("load/generate rational directions"):
        if args.resume:
            records = load_records(Path(args.resume), args.denominator)
            seen = {r.direction for r in records}
            print(f"resumed accepted records: {len(records)}")
            ray_set: set[tuple[int, int, int, int]] = set()
        else:
            records = []
            seen = set()
            ray_set = set(lattice_rays(args.lattice_radius))
            if args.seed_direction_file:
                recovered = load_direction_set(Path(args.seed_direction_file))
                ray_set.update(recovered)
                print(f"recovered checkpoint directions: {len(recovered)}")

        if args.sobol_m > 0:
            srays = sobol_rays(args.sobol_m, args.ray_scale, args.seed)
            ray_set.update(srays)
            print(f"quantized Sobol rays: {len(srays)}")

        rays = sorted(q for q in ray_set if q not in seen)
        print(f"new rational rays to process: {len(rays)}")

    if rays:
        with timing.stage("propose rational boundary points"):
            proposal_start = time.perf_counter()
            accepted_before = len(records)
            for i, q in enumerate(rays, 1):
                rec = propose_record(
                    q, args.denominator, args.inward, args.kind, witness
                )
                seen.add(q)
                if rec is not None and np.linalg.norm(rec.point) > 0:
                    records.append(rec)
                if i % 1000 == 0 or i == len(rays):
                    elapsed = time.perf_counter() - proposal_start
                    rate = i / elapsed if elapsed > 0 else float("inf")
                    eta = (len(rays) - i) / rate if rate > 0 else float("inf")
                    accepted_now = len(records) - accepted_before
                    print(
                        f"  processed {i}/{len(rays)}, accepted {accepted_now}; "
                        f"elapsed={format_seconds(elapsed)}, "
                        f"ETA={format_seconds(eta)}",
                        flush=True,
                    )

    try:
        with timing.stage("initial ConvexHull"):
            hull, _, _ = hull_from_records(records)
    except QhullError as exc:
        raise SystemExit(f"initial hull failed: {exc}")
    print(
        f"initial hull: vertices={len(hull.vertices)} facets={len(hull.simplices)} "
        f"volume={hull.volume:.15f}"
    )

    for round_index in range(args.adaptive_rounds):
        with timing.stage(f"adaptive round {round_index + 1} proposals"):
            added, gap = adaptive_round(
                records,
                seen,
                args.normal_scale,
                args.max_add_per_round,
                args.denominator,
                args.inward,
                args.kind,
                witness,
                facet_pair_rays=args.facet_pair_rays,
                facet_budget=args.adaptive_facet_budget,
            )
        with timing.stage(f"adaptive round {round_index + 1} ConvexHull"):
            hull, _, _ = hull_from_records(records)
        print(
            f"adaptive {round_index+1}: added={added} max_support_violation={gap:.3e} "
            f"vertices={len(hull.vertices)} facets={len(hull.simplices)} "
            f"volume={hull.volume:.15f}"
        )
        checkpoint = Path(args.output)
        write_npz(checkpoint, records, hull, args.denominator, args.witness_denominator)
        np.save(checkpoint.with_name(checkpoint.stem + "_witness_bary.npy"), witness.bary)
        print(f"checkpointed {checkpoint}")
        if added == 0:
            break

    positive_res = [r.residual for r in records if r.side == "positive"]
    negative_res = [r.residual for r in records if r.side == "negative"]
    print(f"records: positive={len(positive_res)} negative={len(negative_res)}")
    if positive_res:
        print(f"minimum positive residual: {min(positive_res):.3e}")
    if negative_res:
        print(f"minimum negative witness residual: {min(negative_res):.3e}")
    print(f"sqrt(det G3) conversion factor: {SQRT_DET_G3:.15f}")
    if not args.skip_exact_volume:
        with timing.stage("exact determinant volume sum"):
            exact_sum = exact_hull_volume_coefficient(records, hull, args.denominator)
        exact_float = exact_sum * np.sqrt(2.0) / (192.0 * args.denominator**4)
        print(
            "algebraic hull evaluation: "
            f"Vol = {exact_sum}*sqrt(2)/(192*{args.denominator}^4) "
            f"= {exact_float:.15f}"
        )

    output = Path(args.output)
    write_npz(output, records, hull, args.denominator, args.witness_denominator)
    np.save(output.with_name(output.stem + "_witness_bary.npy"), witness.bary)
    print(f"saved {output}")
    timing_path = (
        Path(args.timing_output)
        if args.timing_output
        else output.with_name(output.stem + "_timing.json")
    )
    payload = timing.payload(args)
    timing_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"run end: {payload['end']}")
    print(f"total elapsed: {payload['elapsed']}")
    print(f"timing log: {timing_path}")


if __name__ == "__main__":
    main()

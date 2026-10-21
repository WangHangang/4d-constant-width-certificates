#!/usr/bin/env python3
r"""Interval certification of finite upper-graph centres for raised U3.

This program turns the 435 floating seed centres into a rigorously certified,
S4-invariant finite subset of the true positive half K_+ of the canonical
raise of U3.

Proof mechanism
---------------
For a seed centre c=(y,h), membership in K_+ is equivalent to

    h >= 0,    max_{six arcs gamma} ||y-gamma(t)||^2 + h^2 <= 1.

The six arcs are gamma(t)=A cos(t)+B sin(t), 0<=t<=pi/2, where A,B
are vertices of the unit-edge regular tetrahedron.  With q=tan(t/2),
q in [0,1], the squared distance is an explicit rational function F(q).
Its derivative has the sign of the quartic

 P(q)=(yb-1/8)q^4+2 ya q^3+(3/4)q^2+2 ya q-(yb+1/8),

where ya=<y,A>, yb=<y,B>.  Interval arithmetic recursively covers
[0,1].  If P has a strict sign on a box, F is monotone there and only
the appropriate endpoint is needed.  Otherwise the box is bisected;
at the terminal width the entire interval image F(I) is used.  Thus no
stationary point can be missed.

The input seed centres are moved inward by a tiny floating scale factor
before certification.  The actual stored binary64 coordinates—not an
idealized decimal scale—are certified.  Exact signed permutation matrices
(product of signs +1) generate the 24-element tetrahedral symmetry group,
so membership of all orbit points follows from membership of the seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SEEDS = HERE / "raised_u3_s4_active_centers.npz"
DEFAULT_OUTPUT = HERE / "raised_u3_s4_certified_centers.npz"
DEFAULT_CERTIFICATE = HERE / "raised_u3_s4_centres_certificate.json"

VERTEX_SIGNS = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    dtype=np.int8,
)
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_float_interval(x: float):
    """Return a point interval containing exactly the binary64 x."""
    n, d = float(x).as_integer_ratio()
    return mp.iv.mpf(n) / mp.iv.mpf(d)


def tetrahedral_signed_permutations() -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact 3D signed permutations preserving xyz-sign product +1."""
    group = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            if signs[0] * signs[1] * signs[2] != 1:
                continue
            group.append((np.asarray(perm, dtype=np.int64), np.asarray(signs, dtype=np.float64)))
    assert len(group) == 24
    return group


def build_exact_orbit_centres(seed_centres: np.ndarray) -> np.ndarray:
    rows = []
    for perm, signs in tetrahedral_signed_permutations():
        image = seed_centres.copy()
        image[:, :3] = seed_centres[:, perm] * signs
        rows.append(image)
    all_rows = np.concatenate(rows, axis=0)
    # Since only exact sign changes and coordinate permutations are used,
    # exact binary64 row equality is the correct deduplication relation.
    unique = np.unique(all_rows, axis=0)
    return unique


def interval_upper(x):
    return x.b


def interval_lower(x):
    return x.a


def max_distance2_to_one_arc(y: np.ndarray, a_sign: np.ndarray, b_sign: np.ndarray,
                             terminal_depth: int):
    yi = [exact_float_interval(float(v)) for v in y]
    zero = mp.iv.mpf(0)
    one = mp.iv.mpf(1)
    eighth = mp.iv.mpf(1) / 8
    s0 = one / (2 * mp.iv.sqrt(2))

    y2 = sum((v * v for v in yi), zero)
    ya = s0 * sum((int(a_sign[i]) * yi[i] for i in range(3)), zero)
    yb = s0 * sum((int(b_sign[i]) * yi[i] for i in range(3)), zero)

    def derivative_numerator(q):
        # Horner evaluation of P(q).
        return ((((yb - eighth) * q + 2 * ya) * q + mp.iv.mpf(3) / 4) * q
                + 2 * ya) * q - (yb + eighth)

    def distance2(q):
        q2 = q * q
        denominator = one + q2
        cosine = (one - q2) / denominator
        sine = 2 * q / denominator
        return (y2 + mp.iv.mpf(3) / 8 - mp.iv.mpf(1) / 4 * cosine * sine
                - 2 * ya * cosine - 2 * yb * sine)

    stack = [(mp.mpf(0), mp.mpf(1), 0)]
    candidate_values = []
    visited = 0
    terminal_boxes = 0
    while stack:
        left, right, depth = stack.pop()
        visited += 1
        q_box = mp.iv.mpf([left, right])
        p_box = derivative_numerator(q_box)
        if p_box > 0:
            candidate_values.append(distance2(mp.iv.mpf([right, right])))
        elif p_box < 0:
            candidate_values.append(distance2(mp.iv.mpf([left, left])))
        elif depth >= terminal_depth:
            terminal_boxes += 1
            candidate_values.append(distance2(q_box))
        else:
            middle = (left + right) / 2
            stack.append((left, middle, depth + 1))
            stack.append((middle, right, depth + 1))

    best = candidate_values[0]
    for value in candidate_values[1:]:
        # The upper endpoints are point intervals, so this comparison is definite.
        if value.b > best.b:
            best = value
    return best, visited, terminal_boxes


def certify_seed(seed: np.ndarray, terminal_depth: int):
    y = seed[:3]
    h = exact_float_interval(float(seed[3]))
    if not (h >= 0):
        return False, None, 0, 0
    threshold = mp.iv.mpf(1) - h * h
    arc_bounds = []
    visited = 0
    terminal = 0
    for i, j in PAIRS:
        bound, count, term = max_distance2_to_one_arc(
            y, VERTEX_SIGNS[i], VERTEX_SIGNS[j], terminal_depth
        )
        arc_bounds.append(bound)
        visited += count
        terminal += term
    maximum = arc_bounds[0]
    for value in arc_bounds[1:]:
        if value.b > maximum.b:
            maximum = value
    passed = bool(maximum < threshold)
    slack = threshold - maximum
    return passed, slack, visited, terminal


def point_interval_to_string(x, digits: int = 30) -> str:
    # For reporting only; proof decisions above are made in interval arithmetic.
    return mp.nstr(mp.mpf(x), digits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify raised-U3 upper-graph centres by interval arithmetic.")
    parser.add_argument("--seeds-file", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--output-centers", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--inward-scale", type=float, default=1.0 - 1.0e-8)
    parser.add_argument("--interval-dps", type=int, default=60)
    parser.add_argument("--terminal-depth", type=int, default=48)
    parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args()

    mp.mp.dps = args.interval_dps
    mp.iv.dps = args.interval_dps

    seeds_path = args.seeds_file.resolve()
    if not seeds_path.exists():
        raise FileNotFoundError(seeds_path)
    raw = np.load(seeds_path)["centers"]
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("The seed array must have shape (N,4).")
    # Certify the actual rounded binary64 points produced here.
    seeds = np.asarray(raw * np.float64(args.inward_scale), dtype=np.float64)

    print("script SHA-256:", sha256_file(Path(__file__).resolve()))
    print("seed file SHA-256:", sha256_file(seeds_path))
    print("seed centres:", len(seeds))
    print("stored binary64 inward scale:", format(np.float64(args.inward_scale), ".17g"))
    print("interval dps:", args.interval_dps)
    print("terminal bisection depth:", args.terminal_depth)

    start = time.time()
    worst_slack = None
    worst_index = -1
    total_visited = 0
    total_terminal = 0
    for index, seed in enumerate(seeds):
        passed, slack, visited, terminal = certify_seed(seed, args.terminal_depth)
        total_visited += visited
        total_terminal += terminal
        if not passed:
            raise RuntimeError(f"Centre {index} failed interval membership certification: slack={slack}")
        if worst_slack is None or slack.a < worst_slack.a:
            worst_slack = slack
            worst_index = index
        if args.report_every > 0 and ((index + 1) % args.report_every == 0 or index + 1 == len(seeds)):
            print(
                f"certified {index+1:4d}/{len(seeds)}; "
                f"current worst lower slack={point_interval_to_string(worst_slack.a, 16)}",
                flush=True,
            )

    orbit_centres = build_exact_orbit_centres(seeds)
    args.output_centers.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_centers, centers=orbit_centres, seed_centers=seeds)

    certificate = {
        "statement": "Every stored centre belongs to the true positive half K_+ of the canonical raised U3.",
        "proof_type": "computer-assisted interval proof",
        "seed_file": seeds_path.name,
        "seed_file_sha256": sha256_file(seeds_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "output_centers_file": args.output_centers.name,
        "output_centers_sha256": sha256_file(args.output_centers.resolve()),
        "seed_count": int(len(seeds)),
        "orbit_center_count": int(len(orbit_centres)),
        "inward_scale_binary64": format(np.float64(args.inward_scale), ".17g"),
        "interval_dps": int(args.interval_dps),
        "terminal_depth": int(args.terminal_depth),
        "arc_checks": int(len(seeds) * 6),
        "total_interval_boxes_visited": int(total_visited),
        "total_terminal_ambiguous_boxes": int(total_terminal),
        "worst_seed_index": int(worst_index),
        "worst_certified_slack_lower": point_interval_to_string(worst_slack.a, 40),
        "symmetry_generation": "exact coordinate permutations and sign changes with product of signs +1",
    }
    args.certificate.write_text(json.dumps(certificate, indent=2), encoding="utf-8")

    print("\n=== interval centre-membership certificate ===")
    print("all seed centres certified: YES")
    print("seed count:", len(seeds))
    print("exact S4-orbit centre count:", len(orbit_centres))
    print("worst certified slack lower:", certificate["worst_certified_slack_lower"])
    print("interval boxes visited:", total_visited)
    print("terminal ambiguous boxes:", total_terminal)
    print("output centres SHA-256:", certificate["output_centers_sha256"])
    print("certificate:", args.certificate)


if __name__ == "__main__":
    main()

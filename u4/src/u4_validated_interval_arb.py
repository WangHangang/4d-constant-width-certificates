#!/usr/bin/env python3
"""
Rigorous Python verifier for the low-dimensional U_4 volume formula.

This program certifies the exact decomposition

    Vol_4(U_4^(1)) = I1 + I2 + I3

from the manuscript by rigorous real-ball arithmetic (Arb via python-flint).
It deliberately uses a very simple proof kernel: after exact piecewise
splitting and endpoint-regularising changes of variables, each transformed
unit interval/square is partitioned into boxes.  On every box B, Arb encloses
the complete range F(B), hence

    integral_B F in |B| * F(B).

Summing the ten piece enclosures gives a theorem-level enclosure.  No Monte
Carlo, no statistical assumption, and no C++ source code are used.

The ten pieces are exactly:
  I1: 3 A4 branches,
  I2: 5 (z,theta) domain pieces,
  I3: 2 (z,theta) domain pieces.

Recommended first serious run (Windows PowerShell):

    D:\\桌面搬运\\python\\python.exe -m pip install python-flint
    D:\\桌面搬运\\python\\python.exe u4_validated_interval_arb.py \
        --n1 2048 --n2 256 --dps 60 --outdir u4_arb_n256

If the final rigorous upper endpoint is not yet below 0.2219, double --n2
(and, preferably, --n1) and rerun.  The proof remains valid at every level;
refinement only tightens the enclosure.

Dependencies
------------
    python-flint >= 0.8 recommended

Important implementation note
-----------------------------
The project-level script is Python.  python-flint is the Python interface to
FLINT/Arb and supplies rigorous arbitrary-precision ball arithmetic,
including sqrt/asin/acos.  The certificate is therefore not based on the old
Boost/C++ verifier described in an earlier draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

try:
    import flint
    from flint import arb, ctx
except Exception as exc:  # pragma: no cover - user-facing import diagnostic
    raise SystemExit(
        "python-flint is required. Install it into THIS Python environment with:\n"
        "  python -m pip install python-flint\n"
        f"Original import error: {exc}"
    )


# ---------------------------------------------------------------------------
# Basic rigorous-ball helpers
# ---------------------------------------------------------------------------


def A(x: int | str) -> arb:
    """Construct an Arb number from an exact integer/rational/decimal string."""
    return arb(x)


ZERO = A(0)
ONE = A(1)
TWO = A(2)
THREE = A(3)
FOUR = A(4)
FIVE = A(5)
SIX = A(6)
EIGHT = A(8)
TEN = A(10)
SIXTY_FOUR = A(64)

# Domain clamping for functions whose argument has already been proved (by
# the manuscript's exact branch geometry) to lie in [0,1].  A ball built via
# the (mid, rad) constructor always carries Arb's intrinsic outward mag
# padding (~2**-29 relative), so a "[0,1]" ball is never exactly [0,1] and
# intersecting against it can leave the argument's upper bound a hair above
# 1 -- which makes acos()/asin() correctly return an indeterminate (nan)
# result, since part of the input ball then lies outside the real domain.
#
# x.lower() and x.upper() are exact zero-radius points (no mag padding), so
# clamping them against the exact integers 0/1 is exact.  Since sqrt/asin
# are increasing and acos is decreasing, evaluating at the two clamped
# endpoints and taking the union gives a valid, domain-safe enclosure of the
# function on the (already-proved) sub-interval of x lying in [0,1].


def _clamp01(x: arb) -> tuple[arb, arb]:
    lo = x.lower()
    hi = x.upper()
    if lo < ZERO:
        lo = ZERO
    if hi > ONE:
        hi = ONE
    return lo, hi


def safe_sqrt01(x: arb) -> arb:
    lo, hi = _clamp01(x)
    return lo.sqrt().union(hi.sqrt())


def safe_acos01(x: arb) -> arb:
    lo, hi = _clamp01(x)
    return hi.acos().union(lo.acos())


def safe_asin01(x: arb) -> arb:
    lo, hi = _clamp01(x)
    return lo.asin().union(hi.asin())


def safe_sqrt_nonneg(x: arb) -> arb:
    """sqrt(x) where x is proved nonnegative; only the lower bound needs
    clamping (sqrt has no finite upper-domain restriction, so the upper
    bound is used unchanged)."""
    lo = x.lower()
    hi = x.upper()
    if lo < ZERO:
        lo = ZERO
    return lo.sqrt().union(hi.sqrt())


def dyadic_cell(i: int, n: int) -> arb:
    """Closed cell [i/n,(i+1)/n] represented as an Arb ball."""
    if not (0 <= i < n):
        raise ValueError("bad cell index")
    return arb(f"{2*i+1}/{2*n}", f"1/{2*n}")


def lower_str(x: arb, digits: int = 40) -> str:
    return x.lower().str(digits, radius=False)


def upper_str(x: arb, digits: int = 40) -> str:
    return x.upper().str(digits, radius=False)


def ball_record(x: arb, digits: int = 40) -> Dict[str, str]:
    return {
        "lower": lower_str(x, digits),
        "upper": upper_str(x, digits),
        "ball": x.str(digits),
    }


# ---------------------------------------------------------------------------
# Exact constants and branch boundaries
# ---------------------------------------------------------------------------


def init_constants() -> Dict[str, arb]:
    pi = arb.pi()
    sqrt2 = TWO.sqrt()
    sqrt3 = THREE.sqrt()
    sqrt5 = FIVE.sqrt()
    sqrt15 = A(15).sqrt()

    # z_A=sqrt(5/12), z_B=sqrt(5/9), z_D=sqrt(5/8),
    # z_C=sqrt(5/6), z_E=sqrt(15/16).
    zA = (FIVE / A(12)).sqrt()
    zB = (FIVE / A(9)).sqrt()
    zD = (FIVE / EIGHT).sqrt()
    zC = (FIVE / SIX).sqrt()
    zE = (A(15) / A(16)).sqrt()
    return locals()


C = init_constants()
PI = C["pi"]
SQRT2 = C["sqrt2"]
SQRT3 = C["sqrt3"]
SQRT5 = C["sqrt5"]
SQRT15 = C["sqrt15"]
zA, zB, zD, zC, zE = C["zA"], C["zB"], C["zD"], C["zC"], C["zE"]
HALF_PI = PI / TWO


# ---------------------------------------------------------------------------
# Endpoint-regularising maps
# ---------------------------------------------------------------------------


def sinsq01(u: arb) -> Tuple[arb, arb]:
    """
    s(u)=sin^2(pi u/2), ds/du=(pi/2) sin(pi u), u in [0,1].
    Returns (s, ds).
    """
    s = ((PI * u / TWO).sin()) ** 2
    ds = (PI / TWO) * (PI * u).sin()
    return s, ds


def map_interval(u: arb, a: arb, b: arb) -> Tuple[arb, arb]:
    """Map u in [0,1] monotonically to [a,b] by the sine-square map."""
    s, ds = sinsq01(u)
    z = a + (b - a) * s
    jac = (b - a) * ds
    return z, jac


# ---------------------------------------------------------------------------
# Angular boundary functions
# ---------------------------------------------------------------------------


def qfun(z: arb) -> arb:
    return z * z / (FIVE * (ONE - z * z))


def ell2(z: arb) -> arb:
    return safe_acos01(safe_sqrt01(THREE * qfun(z)))


def nu2(z: arb) -> arb:
    return safe_asin01(safe_sqrt01(qfun(z)))


def r2_bound(z: arb) -> arb:
    x = (FIVE - SIX * z * z) / (TEN * (ONE - z * z))
    return safe_acos01(safe_sqrt01(x))


def ell3(z: arb) -> arb:
    return safe_acos01(TWO * safe_sqrt01(qfun(z)))


def r3_bound(z: arb) -> arb:
    x = (FIVE - FOUR * z * z) / (TEN * (ONE - z * z))
    return safe_acos01(safe_sqrt01(x))


def theta_map(t: arb, z: arb, lo: Callable[[arb], arb], hi: Callable[[arb], arb]) -> Tuple[arb, arb]:
    """Map t in [0,1] to [lo(z),hi(z)] by a sine-square map."""
    L = lo(z)
    H = hi(z)
    s, ds = sinsq01(t)
    theta = L + (H - L) * s
    jac = (H - L) * ds
    return theta, jac


def zero_angle(_: arb) -> arb:
    return ZERO


def halfpi_angle(_: arb) -> arb:
    return HALF_PI


# ---------------------------------------------------------------------------
# A_3, A_4 and the low-dimensional U4 integrands
# ---------------------------------------------------------------------------


def A3_middle(t: arb) -> arb:
    """A_3(t)=2pi-6 acos(sqrt(3/2)t) on its middle branch."""
    arg = (THREE / TWO).sqrt() * t
    return TWO * PI - SIX * safe_acos01(arg)


def c_from_z(z: arb) -> arb:
    # c = z / (sqrt(15)*sqrt(1-z^2)).  Only called for z <= z_E < 1.
    return z / (SQRT15 * safe_sqrt_nonneg(ONE - z * z))


def A4_lens_from_z(z: arb) -> arb:
    """A_4 on c in [1/3,1/sqrt(3)]."""
    c = c_from_z(z)
    den = ONE - c * c
    arg1 = (c * c + ONE / THREE) / den
    arg2 = SQRT2 * c / safe_sqrt_nonneg(den)
    # Exact branch geometry proves both arguments lie in [0,1].
    a1 = safe_acos01(arg1)
    a2 = safe_acos01(arg2)
    L = TWO * a1 - FOUR * c * a2
    return EIGHT * PI * c - FOUR * PI + SIX * L


def A4_linear_from_z(z: arb) -> arb:
    """A_4 on c in [1/sqrt(3),1]."""
    c = c_from_z(z)
    return EIGHT * PI * c - FOUR * PI


def A4_full_from_z(_: arb) -> arb:
    return FOUR * PI


def I1_base(z: arb, A4_value: arb) -> arb:
    # (5/64)*8*(2 - sqrt(2)*2z/sqrt(5))*sqrt(1-z^2)*A4
    return (
        (FIVE / SIXTY_FOUR)
        * EIGHT
        * (TWO - SQRT2 * TWO * z / SQRT5)
        * safe_sqrt_nonneg(ONE - z * z)
        * A4_value
    )


def I1_transformed(w: arb, za: arb, zb: arb, branch: str) -> arb:
    z, jz = map_interval(w, za, zb)
    if branch == "lens":
        a4 = A4_lens_from_z(z)
    elif branch == "linear":
        a4 = A4_linear_from_z(z)
    elif branch == "full":
        a4 = A4_full_from_z(z)
    else:
        raise ValueError(branch)
    return I1_base(z, a4) * jz


def common_xy_m_r(z: arb, theta: arb, k: int) -> Tuple[arb, arb, arb, arb, arb]:
    if k not in (2, 3):
        raise ValueError(k)
    one_minus_z2 = ONE - z * z
    root = safe_sqrt_nonneg(one_minus_z2)
    ct = theta.cos()
    st = theta.sin()
    x = root * ct
    y = root * st
    m = z * (A(k * (5 - k)) / FIVE).sqrt()
    r2 = A(5 - k) / FIVE * z * z + one_minus_z2 * ct * ct
    r = safe_sqrt_nonneg(r2)
    return x, y, m, r2, r


def G2(z: arb, theta: arb, a3_mode: str) -> arb:
    x, y, m, r2, r = common_xy_m_r(z, theta, 2)
    del x  # x^0 and A2=2 on the certified domain pieces.
    if a3_mode == "middle":
        a3 = A3_middle(m / (THREE * y))
    elif a3_mode == "full":
        a3 = TWO * PI
    else:
        raise ValueError(a3_mode)
    delta = FOUR * (
        TWO - SQRT2 * (THREE * r2 + m * m) / (FIVE * r * r * r)
    )
    # binom(5,2)/64 = 10/64
    return (
        (TEN / SIXTY_FOUR)
        * y
        * TWO
        * a3
        * (TWO - SQRT2 * r)
        * delta
    )


def G3(z: arb, theta: arb) -> arb:
    x, y, m, r2, r = common_xy_m_r(z, theta, 3)
    del y  # y^0 and A2=2 on the certified domain pieces.
    a3 = A3_middle(m / (THREE * x))
    delta = TWO * (TWO - SQRT2 / r) * (
        TWO - SQRT2 * (TWO * r2 + m * m) / (FIVE * r * r * r)
    )
    # binom(5,3)/64 = 10/64
    return (
        (TEN / SIXTY_FOUR)
        * x
        * a3
        * TWO
        * (TWO - SQRT2 * r)
        * delta
    )


def I2_transformed(
    w: arb,
    t: arb,
    za: arb,
    zb: arb,
    lo: Callable[[arb], arb],
    hi: Callable[[arb], arb],
    mode: str,
) -> arb:
    z, jz = map_interval(w, za, zb)
    theta, jt = theta_map(t, z, lo, hi)
    return G2(z, theta, mode) * jz * jt


def I3_transformed(
    w: arb,
    t: arb,
    za: arb,
    zb: arb,
    lo: Callable[[arb], arb],
    hi: Callable[[arb], arb],
) -> arb:
    z, jz = map_interval(w, za, zb)
    theta, jt = theta_map(t, z, lo, hi)
    return G3(z, theta) * jz * jt


# ---------------------------------------------------------------------------
# Ten proof pieces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    name: str
    component: str
    dimension: int
    evaluator: Callable[..., arb]


PIECES: List[Piece] = [
    Piece("I1_01_lens_zD_zC", "I1", 1, lambda w: I1_transformed(w, zD, zC, "lens")),
    Piece("I1_02_linear_zC_zE", "I1", 1, lambda w: I1_transformed(w, zC, zE, "linear")),
    Piece("I1_03_full_zE_1", "I1", 1, lambda w: I1_transformed(w, zE, ONE, "full")),

    Piece("I2_01_mid_zA_zB_ell2_r2", "I2", 2,
          lambda w, t: I2_transformed(w, t, zA, zB, ell2, r2_bound, "middle")),
    Piece("I2_02_full_zB_zD_ell2_nu", "I2", 2,
          lambda w, t: I2_transformed(w, t, zB, zD, ell2, nu2, "full")),
    Piece("I2_03_full_zD_zC_0_nu", "I2", 2,
          lambda w, t: I2_transformed(w, t, zD, zC, zero_angle, nu2, "full")),
    Piece("I2_04_mid_zB_zC_nu_r2", "I2", 2,
          lambda w, t: I2_transformed(w, t, zB, zC, nu2, r2_bound, "middle")),
    Piece("I2_05_full_zC_1_0_pi2", "I2", 2,
          lambda w, t: I2_transformed(w, t, zC, ONE, zero_angle, halfpi_angle, "full")),

    Piece("I3_01_mid_zA_zB_ell3_r3", "I3", 2,
          lambda w, t: I3_transformed(w, t, zA, zB, ell3, r3_bound)),
    Piece("I3_02_mid_zB_zC_0_r3", "I3", 2,
          lambda w, t: I3_transformed(w, t, zB, zC, zero_angle, r3_bound)),
]


# ---------------------------------------------------------------------------
# Rigorous box integration
# ---------------------------------------------------------------------------


def integrate_1d(evaluator: Callable[[arb], arb], n: int, report_every: int = 0) -> arb:
    total = ZERO
    cell_width = ONE / A(n)
    for i in range(n):
        w = dyadic_cell(i, n)
        total += evaluator(w) * cell_width
        if report_every and (i + 1) % report_every == 0:
            print(f"    {i+1}/{n} cells")
    return total


def integrate_2d(evaluator: Callable[[arb, arb], arb], n: int, report_every: int = 0) -> arb:
    total = ZERO
    cell_area = ONE / A(n * n)
    count = 0
    all_t = [dyadic_cell(j, n) for j in range(n)]
    for i in range(n):
        w = dyadic_cell(i, n)
        for t in all_t:
            total += evaluator(w, t) * cell_area
            count += 1
        if report_every and (i + 1) % report_every == 0:
            print(f"    rows {i+1}/{n}; boxes {count}/{n*n}")
    return total


# ---------------------------------------------------------------------------
# Archival output
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def selected_pieces(names: Iterable[str]) -> List[Piece]:
    names = list(names)
    if not names or names == ["all"]:
        return PIECES
    lookup = {p.name: p for p in PIECES}
    bad = [n for n in names if n not in lookup]
    if bad:
        raise SystemExit(f"Unknown piece(s): {bad}. Use --list-pieces.")
    return [lookup[n] for n in names]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rigorous Arb interval verifier for Vol_4(U4^(1))."
    )
    parser.add_argument("--n1", type=int, default=2048,
                        help="number of boxes for each 1D piece (default: 2048)")
    parser.add_argument("--n2", type=int, default=256,
                        help="grid size n2 x n2 for each 2D piece (default: 256)")
    parser.add_argument("--dps", type=int, default=60,
                        help="Arb working decimal precision (default: 60)")
    parser.add_argument("--target", default="0.2219",
                        help="strict upper target for the combined run (default: 0.2219)")
    parser.add_argument("--outdir", default="u4_validated_arb_output",
                        help="directory for the 10 piece JSON files and manifest")
    parser.add_argument("--piece", nargs="*", default=["all"],
                        help="run selected exact piece names instead of all")
    parser.add_argument("--list-pieces", action="store_true")
    parser.add_argument("--report-every", type=int, default=0,
                        help="progress report every this many rows/cells (0 disables)")
    parser.add_argument("--print-digits", type=int, default=40)
    args = parser.parse_args()

    if args.list_pieces:
        for p in PIECES:
            print(f"{p.name:34s} {p.component} {p.dimension}D")
        return 0

    if args.n1 <= 0 or args.n2 <= 0:
        raise SystemExit("n1 and n2 must be positive")
    if args.dps < 30:
        raise SystemExit("Use at least 30 decimal digits for a certification run")

    ctx.dps = args.dps

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)

    run_pieces = selected_pieces(args.piece)
    print("=== U4 rigorous Python/Arb interval verifier ===")
    print(f"python       : {sys.version.split()[0]}")
    print(f"python-flint : {getattr(flint, '__version__', 'unknown')}")
    print(f"platform     : {platform.platform()}")
    print(f"Arb dps      : {args.dps}")
    print(f"n1           : {args.n1}")
    print(f"n2           : {args.n2}  ({args.n2**2} boxes per 2D piece)")
    print(f"script SHA256: {script_hash}")
    print()

    results: Dict[str, arb] = {}
    piece_files: Dict[str, str] = {}
    piece_hashes: Dict[str, str] = {}
    t0 = time.time()

    for idx, piece in enumerate(run_pieces, start=1):
        print(f"[{idx}/{len(run_pieces)}] {piece.name}")
        ts = time.time()
        if piece.dimension == 1:
            val = integrate_1d(piece.evaluator, args.n1, args.report_every)
            nboxes = args.n1
        else:
            val = integrate_2d(piece.evaluator, args.n2, args.report_every)
            nboxes = args.n2 * args.n2
        elapsed = time.time() - ts
        results[piece.name] = val
        print(f"    enclosure : {val.str(args.print_digits)}")
        print(f"    lower     : {lower_str(val, args.print_digits)}")
        print(f"    upper     : {upper_str(val, args.print_digits)}")
        print(f"    elapsed   : {elapsed:.3f} s")

        record = {
            "piece": piece.name,
            "component": piece.component,
            "dimension": piece.dimension,
            "boxes": nboxes,
            "n1": args.n1,
            "n2": args.n2,
            "dps": args.dps,
            "elapsed_seconds": elapsed,
            "enclosure": ball_record(val, args.print_digits),
            "script_sha256": script_hash,
            "python": sys.version,
            "python_flint": getattr(flint, "__version__", "unknown"),
            "platform": platform.platform(),
        }
        pfile = outdir / f"{piece.name}.json"
        write_json(pfile, record)
        piece_files[piece.name] = str(pfile.name)
        piece_hashes[piece.name] = sha256_file(pfile)

    # Only an all-piece run can form the theorem-level sum.
    all_names = {p.name for p in PIECES}
    if set(results) == all_names:
        comp: Dict[str, arb] = {"I1": ZERO, "I2": ZERO, "I3": ZERO}
        for p in PIECES:
            comp[p.component] += results[p.name]
        total = comp["I1"] + comp["I2"] + comp["I3"]
        target = arb(args.target)
        proved = total.upper() < target

        print("\n=== component enclosures ===")
        for key in ("I1", "I2", "I3"):
            print(f"{key} = {comp[key].str(args.print_digits)}")
            print(f"  lower = {lower_str(comp[key], args.print_digits)}")
            print(f"  upper = {upper_str(comp[key], args.print_digits)}")
        print("\n=== final enclosure ===")
        print(f"Vol_4(U4^(1)) in {total.str(args.print_digits)}")
        print(f"lower = {lower_str(total, args.print_digits)}")
        print(f"upper = {upper_str(total, args.print_digits)}")
        print(f"target = {args.target}")
        print(f"STRICT UPPER TARGET PROVED: {'YES' if proved else 'NO'}")
        if not proved:
            print("Refine by doubling --n2 (and preferably --n1).")

        manifest = {
            "certificate": "U4 width-one low-dimensional rigorous interval enclosure",
            "proof_kernel": "piecewise exact domain split + sine-square maps + Arb box range integration",
            "n1": args.n1,
            "n2": args.n2,
            "dps": args.dps,
            "target": args.target,
            "strict_target_proved": bool(proved),
            "components": {k: ball_record(v, args.print_digits) for k, v in comp.items()},
            "total": ball_record(total, args.print_digits),
            "piece_files": piece_files,
            "piece_sha256": piece_hashes,
            "script": script_path.name,
            "script_sha256": script_hash,
            "python": sys.version,
            "python_flint": getattr(flint, "__version__", "unknown"),
            "platform": platform.platform(),
            "elapsed_seconds": time.time() - t0,
        }
        manifest_path = outdir / "u4_validated_arb_manifest.json"
        write_json(manifest_path, manifest)
        print(f"\nmanifest: {manifest_path}")
        print(f"manifest SHA256: {sha256_file(manifest_path)}")
        return 0 if proved else 2

    # Partial run: useful for parallel/debug execution, but not a complete proof.
    partial_manifest = {
        "certificate": "partial U4 Arb piece run",
        "pieces": {k: ball_record(v, args.print_digits) for k, v in results.items()},
        "piece_files": piece_files,
        "piece_sha256": piece_hashes,
        "script_sha256": script_hash,
        "n1": args.n1,
        "n2": args.n2,
        "dps": args.dps,
    }
    path = outdir / "u4_validated_arb_partial_manifest.json"
    write_json(path, partial_manifest)
    print(f"\npartial manifest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Hybrid rigorous 5-decimal certificate for Vol_4(U_4^(1)).

Goal
----
Certify the correctly rounded value

    Vol_4(U_4^(1)) = 0.22184   (5 decimal places)

without rerunning all seven expensive 4096^2 two-dimensional grids.

Proof architecture
------------------
1. Exact ten-piece reduction from the latest manuscript.
2. Fresh FLINT/Arb validated 1D integration for
      I1_01, I1_02, I1_03,
      I2_02, I2_03, I2_05.
   The last three use the manuscript's exact theta-antiderivative.
3. Reuse the already-established outward Arb enclosures for the four
   genuinely middle-A3 two-dimensional pieces
      I2_01, I2_04, I3_01, I3_02
   from the final 4096^2 human-checkable certificate.
4. Directed Arb summation and an exact decimal rounding-cell test.

Trust boundary
--------------
EXACT:
  exact ten-piece domain split and exact theta-antiderivative for the
  three full-A3 I2 pieces.

FRESHLY VALIDATED HERE:
  six 1D pieces through python-flint / Arb acb.integral.

IMPORTED RIGOROUS INPUT:
  four middle-A3 2D intervals already certified in the manuscript's
  4096^2 human-checkable Arb run.  Their decimal endpoints were printed
  outward in the theorem table.

This is therefore a rigorous *hybrid corollary* of the existing 2D
certificate plus a fresh sharper 1D recomputation.  It is not a brand-new
independent proof of the four imported 2D bounds.

Dependencies
------------
    python -m pip install python-flint

Run
---
    python u4_certify_5dp_semianalytic.py

Optional:
    python u4_certify_5dp_semianalytic.py --dps 45
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

try:
    import flint
    from flint import arb, acb, ctx
except Exception as exc:
    raise SystemExit(
        "python-flint is required in this Python environment.\n"
        "Install it with:\n"
        "  python -m pip install python-flint\n"
        f"Original import error: {exc}"
    )


# ---------------------------------------------------------------------------
# Frozen provenance from the latest manuscript / established certificate
# ---------------------------------------------------------------------------

LATEST_MANUSCRIPT_SHA256 = (
    "3a57f90963a056f91d37d955b9ccd76e0042f7354a99774e981504c5f37e7547"
)

ESTABLISHED_CERTIFICATE_HASHES = {
    "u4_human_checkable_checkpoint_arb.py":
        "fed5cc032a218f8785d0be6d971af35858d5b04b4e9bfe7e8c730ea0b7d9fe5c",
    "u4_validated_interval_arb.py":
        "a28bd08c9b4b5b6934216b5fde3aa28a87c1c3e00f23893033c82a646f5d4c75",
    "u4_i2_04_centered_arb.py":
        "6e098be65efaa67746e0093887ccd49d4f013345c67f2309b56ec3e497906b2b",
    "u4_human_checkable_manifest.json":
        "afd2a18eef221b14670394147173ba4092f1d7eb2994c8eeebf751d737c951fb",
    "u4_human_checkable_checkpoint.json":
        "ada18ab3c7a1eea2e2ce880d2fe91a1439ca70b09006643282d54b9765bc4de3",
}

# Outward-rounded decimal endpoints printed in the final human-checkable
# 4096^2 table.  These are rigorous but slightly coarser than the archived
# binary Arb endpoints, hence safe to reuse.
IMPORTED_MIDDLE_2D = {
    "I2_01": (
        "0.005305474445252067648",
        "0.005305524269440644134",
    ),
    "I2_04": (
        "0.02026090383070103131",
        "0.02026294671750725316",
    ),
    "I3_01": (
        "0.0001180335765627251934",
        "0.0001180371686580424704",
    ),
    "I3_02": (
        "0.002053385336922711151",
        "0.002053408608216891725",
    ),
}

NUMERICAL_SEED = Decimal("0.22183605428592399841723298498165852073108297765535")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def prevent_windows_sleep():
    if sys.platform != "win32":
        yield
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040
    k32 = ctypes.windll.kernel32
    old = k32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )
    if old == 0:
        print("WARNING: Windows sleep prevention could not be enabled.")
    try:
        yield
    finally:
        k32.SetThreadExecutionState(ES_CONTINUOUS)


def width(x: arb) -> arb:
    return x.upper() - x.lower()


def finite_real(x: arb) -> bool:
    s = x.str(30).lower()
    return "nan" not in s and "inf" not in s


def imported_interval(lo: str, hi: str) -> arb:
    L, U = arb(lo), arb(hi)
    if not (L.upper() <= U.lower()):
        raise AssertionError(f"invalid imported interval [{lo}, {hi}]")
    return L.union(U)


# ---------------------------------------------------------------------------
# Branch-safe analytic helpers
# ---------------------------------------------------------------------------

ZERO = acb(0)
ONE = acb(1)
TWO = acb(2)
THREE = acb(3)
FIVE = acb(5)
SIX = acb(6)
EIGHT = acb(8)
TEN = acb(10)
FIFTEEN = acb(15)
SIXTY_FOUR = acb(64)
I = acb(0, 1)

PI = acb.pi()
HALF_PI = PI / TWO
SQRT2 = TWO.sqrt()
SQRT5 = FIVE.sqrt()
SQRT15 = FIFTEEN.sqrt()

zB = (FIVE / acb(9)).sqrt()
zD = (FIVE / EIGHT).sqrt()
zC = (FIVE / SIX).sqrt()
zE = (FIFTEEN / acb(16)).sqrt()


def imag_exact_zero(z: acb) -> bool:
    return z.imag.lower() == 0 and z.imag.upper() == 0


def sqrt_nonneg(z: acb, analytic: bool) -> acb:
    """
    Validated sqrt on a branch already proved nonnegative on the real path.
    For order-0 real interval evaluation, clamp only tiny outward negative
    overreach to 0; this implements the exact domain proof, not a feasibility
    assumption.
    """
    if not analytic and imag_exact_zero(z):
        lo = z.real.lower()
        hi = z.real.upper()
        if lo < 0:
            lo = arb(0)
        return acb(lo.sqrt().union(hi.sqrt()))
    return z.real_sqrt(analytic=analytic)


def asin01(z: acb, analytic: bool) -> acb:
    if not analytic and imag_exact_zero(z):
        lo, hi = z.real.lower(), z.real.upper()
        if lo < 0:
            lo = arb(0)
        if hi > 1:
            hi = arb(1)
        return acb(lo.asin().union(hi.asin()))
    s = (ONE - z*z).sqrt(analytic=True)
    return -I * (I*z + s).log(analytic=True)


def acos01(z: acb, analytic: bool) -> acb:
    if not analytic and imag_exact_zero(z):
        lo, hi = z.real.lower(), z.real.upper()
        if lo < 0:
            lo = arb(0)
        if hi > 1:
            hi = arb(1)
        return acb(hi.acos().union(lo.acos()))
    return HALF_PI - asin01(z, True)


def acb_atan_pos(z: acb, analytic: bool) -> acb:
    """
    Principal atan on the positive-real branch:
        atan z = (log(1+i z) - log(1-i z))/(2i).
    """
    return ((ONE + I*z).log(analytic=analytic)
            - (ONE - I*z).log(analytic=analytic)) / (TWO*I)


def acb_asinh_pos(z: acb, analytic: bool) -> acb:
    """
    Principal asinh:
        asinh z = log(z + sqrt(1+z^2)).
    """
    s = (ONE + z*z).sqrt(analytic=analytic)
    return (z + s).log(analytic=analytic)


def sinsq_map(u: acb, a: acb, b: acb):
    s = (PI*u/TWO).sin() ** 2
    ds = (PI/TWO) * (PI*u).sin()
    return a + (b-a)*s, (b-a)*ds


# ---------------------------------------------------------------------------
# I1 exact branches
# ---------------------------------------------------------------------------

def c_from_z(z: acb, analytic: bool) -> acb:
    return z / (SQRT15 * sqrt_nonneg(ONE-z*z, analytic))


def A4_lens(z: acb, analytic: bool) -> acb:
    c = c_from_z(z, analytic)
    den = ONE - c*c
    arg1 = (c*c + ONE/THREE) / den
    arg2 = SQRT2*c / sqrt_nonneg(den, analytic)
    L = TWO*acos01(arg1, analytic) - acb(4)*c*acos01(arg2, analytic)
    return acb(8)*PI*c - acb(4)*PI + SIX*L


def A4_linear(z: acb, analytic: bool) -> acb:
    c = c_from_z(z, analytic)
    return acb(8)*PI*c - acb(4)*PI


def I1_density(z: acb, branch: str, analytic: bool) -> acb:
    if branch == "lens":
        A4 = A4_lens(z, analytic)
    elif branch == "linear":
        A4 = A4_linear(z, analytic)
    elif branch == "full":
        A4 = acb(4)*PI
    else:
        raise ValueError(branch)

    return (
        (FIVE/SIXTY_FOUR)
        * acb(8)
        * (TWO - SQRT2*TWO*z/SQRT5)
        * sqrt_nonneg(ONE-z*z, analytic)
        * A4
    )


def i1_piece(w: acb, za: acb, zb: acb, branch: str, analytic: bool) -> acb:
    z, jac = sinsq_map(w, za, zb)
    return I1_density(z, branch, analytic) * jac


# ---------------------------------------------------------------------------
# Exact theta-antiderivative for full-A3 I2 pieces
# ---------------------------------------------------------------------------

def F_antiderivative(u: acb, z: acb, analytic: bool) -> acb:
    a = THREE*z*z/FIVE
    m2 = SIX*z*z/FIVE
    sa = sqrt_nonneg(a, analytic)
    r = sqrt_nonneg(a + u*u, analytic)
    q = u/sa

    return (
        acb(26)/FIVE*u
        + TWO*m2/(FIVE*sa) * acb_atan_pos(q, analytic)
        - SQRT2 * (u*r + a*acb_asinh_pos(q, analytic))
        - SIX*SQRT2/FIVE * acb_asinh_pos(q, analytic)
        - TWO*SQRT2*m2/FIVE * u/(a*r)
    )


def full_i2_density(z: acb, which: str, analytic: bool) -> acb:
    if which == "I2_02":
        ua = z * sqrt_nonneg(THREE/FIVE, analytic)
        ub = sqrt_nonneg(ONE - SIX*z*z/FIVE, analytic)
    elif which == "I2_03":
        ua = sqrt_nonneg(ONE-z*z, analytic)
        ub = sqrt_nonneg(ONE - SIX*z*z/FIVE, analytic)
    elif which == "I2_05":
        ua = sqrt_nonneg(ONE-z*z, analytic)
        ub = ZERO
    else:
        raise ValueError(which)

    return (FIVE*PI/TWO) * (
        F_antiderivative(ua, z, analytic)
        - F_antiderivative(ub, z, analytic)
    )


def i2_full_piece(w: acb, za: acb, zb: acb, which: str, analytic: bool) -> acb:
    z, jac = sinsq_map(w, za, zb)
    return full_i2_density(z, which, analytic) * jac


# ---------------------------------------------------------------------------
# Validated 1D integration
# ---------------------------------------------------------------------------

def integrate_1d(fun, tol: float, args) -> acb:
    def wrapped(x, analytic):
        return fun(x, analytic)
    return acb.integral(
        wrapped, 0, 1,
        rel_tol=tol,
        abs_tol=tol,
        deg_limit=args.deg_limit,
        eval_limit=args.eval_limit,
        depth_limit=args.depth_limit,
        use_heap=True,
        verbose=0,
    )


def verify_real_integral(name: str, z: acb) -> arb:
    if not finite_real(z.real) or not finite_real(z.imag):
        raise RuntimeError(f"{name}: nonfinite acb enclosure {z}")
    if not (z.imag.lower() <= 0 <= z.imag.upper()):
        raise RuntimeError(
            f"{name}: imaginary enclosure does not contain zero: {z.imag}"
        )
    return z.real


# ---------------------------------------------------------------------------
# Decimal certification
# ---------------------------------------------------------------------------

def rounded_integer(seed: Decimal, digits: int) -> int:
    scale = Decimal(10) ** digits
    return int((seed*scale).to_integral_value(rounding=ROUND_HALF_UP))


def decimal_from_integer(k: int, digits: int) -> str:
    sign = "-" if k < 0 else ""
    k = abs(k)
    scale = 10**digits
    a, b = divmod(k, scale)
    return f"{sign}{a}.{b:0{digits}d}"


def certify_rounding(total: arb, digits: int) -> dict:
    k = rounded_integer(NUMERICAL_SEED, digits)
    den = arb(2*(10**digits))
    left = arb(2*k-1)/den
    right = arb(2*k+1)/den
    ok = total.lower() > left.upper() and total.upper() < right.lower()
    return {
        "digits": digits,
        "candidate": decimal_from_integer(k, digits),
        "left": left.str(digits+12),
        "right": right.str(digits+12),
        "certified": bool(ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dps", type=int, default=40)
    ap.add_argument("--tol-digits", type=int, default=13)
    ap.add_argument("--eval-limit", type=int, default=500000)
    ap.add_argument("--depth-limit", type=int, default=200)
    ap.add_argument("--deg-limit", type=int, default=120)
    ap.add_argument("--output", default="u4_5dp_hybrid_certificate.json")
    args = ap.parse_args()

    if args.dps < 30:
        raise SystemExit("Use --dps >= 30.")
    if args.tol_digits < 10:
        raise SystemExit("Use --tol-digits >= 10.")

    ctx.dps = args.dps
    tol = 10.0 ** (-args.tol_digits)

    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)

    print("=== U4 hybrid rigorous 5-decimal certificate ===")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"python-flint : {getattr(flint, '__version__', 'unknown')}")
    print(f"platform     : {platform.platform()}")
    print(f"Arb dps      : {args.dps}")
    print(f"tol target   : ~1e-{args.tol_digits}")
    print(f"script SHA256: {script_hash}")
    print()
    print("Imported 2D proof provenance:")
    print(f"  latest manuscript SHA256: {LATEST_MANUSCRIPT_SHA256}")
    for name, h in ESTABLISHED_CERTIFICATE_HASHES.items():
        print(f"  {name}: {h}")
    print()
    print("Trust boundary:")
    print("  exact      : ten-piece split + full-A3 theta antiderivative")
    print("  fresh Arb  : 3 I1 pieces + 3 full-A3 I2 pieces")
    print("  prior Arb  : 4 middle-A3 2D intervals from established 4096^2 certificate")
    print()

    fresh = {}
    elapsed = {}
    t0 = time.time()

    jobs = [
        ("I1_01", lambda x, a: i1_piece(x, zD, zC, "lens", a)),
        ("I1_02", lambda x, a: i1_piece(x, zC, zE, "linear", a)),
        ("I1_03", lambda x, a: i1_piece(x, zE, ONE, "full", a)),
        ("I2_02", lambda x, a: i2_full_piece(x, zB, zD, "I2_02", a)),
        ("I2_03", lambda x, a: i2_full_piece(x, zD, zC, "I2_03", a)),
        ("I2_05", lambda x, a: i2_full_piece(x, zC, ONE, "I2_05", a)),
    ]

    with prevent_windows_sleep():
        for idx, (name, fun) in enumerate(jobs, 1):
            ts = time.time()
            val = integrate_1d(fun, tol, args)
            real = verify_real_integral(name, val)
            elapsed[name] = time.time() - ts
            fresh[name] = real
            print(f"[{idx}/6] {name}")
            print(f"    real  : {real.str(55)}")
            print(f"    imag  : {val.imag.str(30)}")
            print(f"    width : {width(real).str(30, radius=False)}")
            print(f"    time  : {elapsed[name]:.3f}s")

    imported = {
        name: imported_interval(*bounds)
        for name, bounds in IMPORTED_MIDDLE_2D.items()
    }

    print("\nImported rigorous middle-A3 intervals:")
    for name in ("I2_01", "I2_04", "I3_01", "I3_02"):
        print(f"  {name}: {imported[name].str(45)}")

    # Exact ten-piece accounting.
    all_pieces = {
        **fresh,
        **imported,
    }
    required = {
        "I1_01", "I1_02", "I1_03",
        "I2_01", "I2_02", "I2_03", "I2_04", "I2_05",
        "I3_01", "I3_02",
    }
    if set(all_pieces) != required:
        raise AssertionError("ten-piece accounting failed")

    I1 = fresh["I1_01"] + fresh["I1_02"] + fresh["I1_03"]
    I2 = (
        imported["I2_01"] + fresh["I2_02"] + fresh["I2_03"]
        + imported["I2_04"] + fresh["I2_05"]
    )
    I3 = imported["I3_01"] + imported["I3_02"]
    total = I1 + I2 + I3

    print("\n=== rigorous hybrid sum ===")
    print(f"I1 in {I1.str(55)}")
    print(f"I2 in {I2.str(55)}")
    print(f"I3 in {I3.str(55)}")
    print(f"TOTAL in {total.str(60)}")
    print(f"TOTAL width = {width(total).str(40, radius=False)}")

    # Sanity: the hybrid interval must be contained in the previous global theorem.
    old_lo = arb("0.2218338853720453")
    old_hi = arb("0.2218382994641744")
    contained = (
        total.lower() >= old_lo.lower()
        and total.upper() <= old_hi.upper()
    )
    print(f"contained in previous global rigorous interval: {contained}")
    if not contained:
        raise RuntimeError(
            "Hybrid interval is not contained in the previous theorem interval; "
            "stop and audit formulas/provenance."
        )

    cert5 = certify_rounding(total, 5)
    cert6 = certify_rounding(total, 6)
    print()
    print(
        f"5 decimals: {cert5['candidate']}  "
        f"cell=({cert5['left']}, {cert5['right']})  "
        f"certified={'YES' if cert5['certified'] else 'NO'}"
    )
    print(
        f"6 decimals: {cert6['candidate']}  "
        f"cell=({cert6['left']}, {cert6['right']})  "
        f"certified={'YES' if cert6['certified'] else 'NO'}"
    )

    result = {
        "schema_version": 1,
        "statement": "hybrid rigorous decimal certificate for Vol_4(U_4^(1))",
        "script": script_path.name,
        "script_sha256": script_hash,
        "python": sys.version,
        "python_flint": getattr(flint, "__version__", "unknown"),
        "platform": platform.platform(),
        "dps": args.dps,
        "tol_digits": args.tol_digits,
        "latest_manuscript_sha256": LATEST_MANUSCRIPT_SHA256,
        "established_certificate_hashes": ESTABLISHED_CERTIFICATE_HASHES,
        "imported_middle_2d_decimal_endpoints": IMPORTED_MIDDLE_2D,
        "fresh_piece_balls": {
            k: {
                "ball": v.str(80),
                "lower": v.lower().str(80, radius=False),
                "upper": v.upper().str(80, radius=False),
                "width": width(v).str(80, radius=False),
                "elapsed_seconds": elapsed[k],
            }
            for k, v in fresh.items()
        },
        "components": {
            "I1": I1.str(80),
            "I2": I2.str(80),
            "I3": I3.str(80),
        },
        "total": {
            "ball": total.str(90),
            "lower": total.lower().str(90, radius=False),
            "upper": total.upper().str(90, radius=False),
            "width": width(total).str(90, radius=False),
        },
        "contained_in_previous_global_rigorous_interval": bool(contained),
        "rounding_5dp": cert5,
        "rounding_6dp": cert6,
        "numerical_seed_used_only_to_choose_rounding_cell": str(NUMERICAL_SEED),
        "elapsed_seconds": time.time() - t0,
        "trust_boundary": {
            "exact": "ten-piece reduction and theta-antiderivative",
            "fresh_validated": "six 1D pieces with FLINT/Arb",
            "imported_validated": "four middle-A3 2D bounds from established 4096^2 certificate",
        },
    }

    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSaved: {out.resolve()}")
    print(f"Output SHA256: {sha256_file(out)}")

    if cert5["certified"]:
        print("\n============================================================")
        print("RIGOROUS 5-DECIMAL CERTIFICATE: SUCCESS")
        print("Vol_4(U4^(1)) = 0.22184  (correctly rounded to 5 decimals)")
        print("============================================================")
        return 0

    print("\n5-decimal target not yet certified.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

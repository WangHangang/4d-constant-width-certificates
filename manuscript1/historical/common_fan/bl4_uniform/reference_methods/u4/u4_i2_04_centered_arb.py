#!/usr/bin/env python3
"""
Rigorous centered-form Arb verifier for the remaining difficult U4 piece I2_04.

The mathematical piece is

    I2_04 = integral_{z_B}^{z_C} integral_{nu_2(z)}^{r_2(z)} G_2(z,theta) dtheta dz,

in the middle A3 branch, after the same sine-square maps used by
u4_validated_interval_arb.py.

Instead of enclosing the whole integrand by its natural interval extension on
an n x n box, this script uses a centered first-derivative form on interior
boxes.  If B is a rectangle centered at c and D_i encloses partial_i F on B,
write D_i = m_i + [-r_i,r_i].  The line-segment identity gives

    F(x)-F(c) = sum_i (x_i-c_i) * average_i(x),   average_i(x) in D_i.

The constant m_i terms integrate to zero on a symmetric box.  Therefore

  | integral_B F - |B| F(c) |
      <= sum_i r_i * integral_B |x_i-c_i|.

For a square of side Delta this becomes

  error_B <= Delta^3/4 * (r_w + r_t).

This is rigorous and uses only first-derivative interval enclosures.  Cells
that touch a transformed-domain endpoint, or for which interval automatic
differentiation encounters a derivative-domain singularity, automatically
fall back to the original natural Arb box enclosure.  Thus every reported
endpoint remains theorem-level rigorous.

Place this file beside u4_validated_interval_arb.py and run with the same
Python environment containing python-flint, e.g.

  & "D:\\桌面搬运\\Canada001\\Presentation\\.venv311\\Scripts\\python.exe" `
    "D:\\桌面搬运\\Canada001\\Presentation\\u4_i2_04_centered_arb.py" `
    --n 256 --dps 60 --outdir "D:\\桌面搬运\\Canada001\\Presentation\\u4_i2_04_centered_n256"

If the n=256 result is promising, rerun with --n 512 or --n 1024.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import flint
    from flint import arb, ctx
except Exception as exc:
    raise SystemExit(
        "python-flint is required in THIS Python environment.\n"
        f"Original import error: {exc}"
    )

try:
    import u4_validated_interval_arb as K
except Exception as exc:
    raise SystemExit(
        "Could not import u4_validated_interval_arb.py from the same directory.\n"
        f"Original import error: {exc}"
    )


ZERO = arb(0)
ONE = arb(1)
TWO = arb(2)
THREE = arb(3)
FOUR = arb(4)
FIVE = arb(5)
SIX = arb(6)
TEN = arb(10)
SIXTY_FOUR = arb(64)
PI = arb.pi()
SQRT2 = TWO.sqrt()
SQRT_3_OVER_2 = (THREE / TWO).sqrt()
SQRT_6_OVER_5 = (SIX / FIVE).sqrt()


class DerivativeDomainError(ArithmeticError):
    pass


def _coerce(x: Any) -> "D2":
    if isinstance(x, D2):
        return x
    if isinstance(x, arb):
        return D2(x, ZERO, ZERO)
    return D2(arb(x), ZERO, ZERO)


@dataclass(frozen=True)
class D2:
    """First-order two-variable interval automatic-differentiation object."""

    v: arb
    dw: arb
    dt: arb

    @staticmethod
    def w(v: arb) -> "D2":
        return D2(v, ONE, ZERO)

    @staticmethod
    def t(v: arb) -> "D2":
        return D2(v, ZERO, ONE)

    def __add__(self, other: Any) -> "D2":
        o = _coerce(other)
        return D2(self.v + o.v, self.dw + o.dw, self.dt + o.dt)

    __radd__ = __add__

    def __neg__(self) -> "D2":
        return D2(-self.v, -self.dw, -self.dt)

    def __sub__(self, other: Any) -> "D2":
        return self + (-_coerce(other))

    def __rsub__(self, other: Any) -> "D2":
        return _coerce(other) - self

    def __mul__(self, other: Any) -> "D2":
        o = _coerce(other)
        return D2(
            self.v * o.v,
            self.dw * o.v + self.v * o.dw,
            self.dt * o.v + self.v * o.dt,
        )

    __rmul__ = __mul__

    def inv(self) -> "D2":
        if self.v.lower() <= ZERO and self.v.upper() >= ZERO:
            raise DerivativeDomainError("division interval contains zero")
        iv = ONE / self.v
        fac = -(iv * iv)
        return D2(iv, fac * self.dw, fac * self.dt)

    def __truediv__(self, other: Any) -> "D2":
        return self * _coerce(other).inv()

    def __rtruediv__(self, other: Any) -> "D2":
        return _coerce(other) * self.inv()

    def sin(self) -> "D2":
        sv = self.v.sin()
        cv = self.v.cos()
        return D2(sv, cv * self.dw, cv * self.dt)

    def cos(self) -> "D2":
        cv = self.v.cos()
        sv = self.v.sin()
        return D2(cv, -sv * self.dw, -sv * self.dt)

    def sqrt_pos(self) -> "D2":
        # Strict positivity is required only for the derivative formula.
        if self.v.lower() <= ZERO:
            raise DerivativeDomainError("sqrt derivative interval reaches zero")
        sv = self.v.sqrt()
        fac = ONE / (TWO * sv)
        return D2(sv, fac * self.dw, fac * self.dt)

    def asin_strict(self) -> "D2":
        if self.v.lower() <= -ONE or self.v.upper() >= ONE:
            raise DerivativeDomainError("asin derivative interval reaches +/-1")
        av = self.v.asin()
        den = (ONE - self.v * self.v).sqrt()
        fac = ONE / den
        return D2(av, fac * self.dw, fac * self.dt)

    def acos_strict(self) -> "D2":
        if self.v.lower() <= -ONE or self.v.upper() >= ONE:
            raise DerivativeDomainError("acos derivative interval reaches +/-1")
        av = self.v.acos()
        den = (ONE - self.v * self.v).sqrt()
        fac = -ONE / den
        return D2(av, fac * self.dw, fac * self.dt)


def sinsq01_d(u: D2) -> tuple[D2, D2]:
    a = (PI * u / TWO).sin()
    s = a * a
    ds = (PI / TWO) * (PI * u).sin()
    return s, ds


def map_interval_d(u: D2, a: arb, b: arb) -> tuple[D2, D2]:
    s, ds = sinsq01_d(u)
    z = a + (b - a) * s
    jac = (b - a) * ds
    return z, jac


def qfun_d(z: D2) -> D2:
    return z * z / (FIVE * (ONE - z * z))


def nu2_d(z: D2) -> D2:
    q = qfun_d(z)
    return q.sqrt_pos().asin_strict()


def r2_bound_d(z: D2) -> D2:
    x = (FIVE - SIX * z * z) / (TEN * (ONE - z * z))
    return x.sqrt_pos().acos_strict()


def theta_map_d(t: D2, z: D2) -> tuple[D2, D2]:
    L = nu2_d(z)
    H = r2_bound_d(z)
    s, ds = sinsq01_d(t)
    theta = L + (H - L) * s
    jac = (H - L) * ds
    return theta, jac


def A3_middle_d(x: D2) -> D2:
    arg = SQRT_3_OVER_2 * x
    return TWO * PI - SIX * arg.acos_strict()


def G2_middle_d(z: D2, theta: D2) -> D2:
    one_minus_z2 = ONE - z * z
    root = one_minus_z2.sqrt_pos()
    ct = theta.cos()
    st = theta.sin()
    y = root * st
    m = SQRT_6_OVER_5 * z
    r2 = (THREE / FIVE) * z * z + one_minus_z2 * ct * ct
    r = r2.sqrt_pos()

    a3 = A3_middle_d(m / (THREE * y))
    delta = FOUR * (
        TWO - SQRT2 * (THREE * r2 + m * m) / (FIVE * r * r * r)
    )
    return (
        (TEN / SIXTY_FOUR)
        * y
        * TWO
        * a3
        * (TWO - SQRT2 * r)
        * delta
    )


def transformed_i2_04_d(w: D2, t: D2) -> D2:
    z, jz = map_interval_d(w, K.zB, K.zC)
    theta, jt = theta_map_d(t, z)
    return G2_middle_d(z, theta) * jz * jt


def natural_i2_04(w: arb, t: arb) -> arb:
    return K.I2_transformed(w, t, K.zB, K.zC, K.nu2, K.r2_bound, "middle")


def point_cell_center(i: int, n: int) -> arb:
    return arb(f"{2*i+1}/{2*n}")


def box_cell(i: int, n: int) -> arb:
    # Reuse the kernel's validated closed-cell representation.
    return K.dyadic_cell(i, n)


def interval_radius(x: arb) -> arb:
    return (x.upper() - x.lower()) / TWO


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def lower_str(x: arb, digits: int) -> str:
    return x.lower().str(digits, radius=False)


def upper_str(x: arb, digits: int) -> str:
    return x.upper().str(digits, radius=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rigorous centered first-derivative Arb verifier for U4 I2_04."
    )
    ap.add_argument("--n", type=int, default=256, help="n x n grid (default 256)")
    ap.add_argument("--dps", type=int, default=60)
    ap.add_argument(
        "--boundary-layers",
        type=int,
        default=1,
        help="number of outer cell layers forced to natural interval fallback",
    )
    ap.add_argument("--report-every", type=int, default=32, help="progress every rows")
    ap.add_argument("--print-digits", type=int, default=40)
    ap.add_argument("--outdir", default="u4_i2_04_centered_output")
    args = ap.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if args.dps < 30:
        raise SystemExit("use at least --dps 30")
    if args.boundary_layers < 0 or 2 * args.boundary_layers >= args.n:
        raise SystemExit("bad --boundary-layers")

    ctx.dps = args.dps
    n = args.n
    delta = ONE / arb(n)
    half = delta / TWO
    area = delta * delta

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    kernel_path = Path(K.__file__).resolve()
    script_hash = sha256_file(script_path)
    kernel_hash = sha256_file(kernel_path)

    print("=== U4 I2_04 centered-form rigorous Arb verifier ===")
    print(f"python        : {sys.version.split()[0]}")
    print(f"python-flint  : {getattr(flint, '__version__', 'unknown')}")
    print(f"platform      : {platform.platform()}")
    print(f"Arb dps       : {args.dps}")
    print(f"grid          : {n} x {n} = {n*n} boxes")
    print(f"boundary layer: {args.boundary_layers}")
    print(f"driver SHA256 : {script_hash}")
    print(f"kernel SHA256 : {kernel_hash}")
    print()

    total_lo = ZERO
    total_hi = ZERO
    natural_boxes = 0
    centered_boxes = 0
    derivative_fallbacks = 0
    t0 = time.time()

    for i in range(n):
        wb = box_cell(i, n)
        wc = point_cell_center(i, n)
        for j in range(n):
            tb = box_cell(j, n)
            tc = point_cell_center(j, n)

            force_natural = (
                i < args.boundary_layers
                or i >= n - args.boundary_layers
                or j < args.boundary_layers
                or j >= n - args.boundary_layers
            )

            if not force_natural:
                try:
                    dval = transformed_i2_04_d(D2.w(wb), D2.t(tb))
                    rw = interval_radius(dval.dw)
                    rt = interval_radius(dval.dt)

                    # Rigorous point evaluation at the exact box center.
                    fc = natural_i2_04(wc, tc)
                    center_contrib = area * fc

                    # For a square of side Delta,
                    # integral |w-wc| = integral |t-tc| = Delta^3/4.
                    err = area * half * (rw + rt) / TWO
                    e = err.upper()
                    lo = center_contrib.lower() - e
                    hi = center_contrib.upper() + e
                    centered_boxes += 1
                except (DerivativeDomainError, ZeroDivisionError, ValueError):
                    force_natural = True
                    derivative_fallbacks += 1

            if force_natural:
                nv = natural_i2_04(wb, tb) * area
                lo = nv.lower()
                hi = nv.upper()
                natural_boxes += 1

            total_lo += lo
            total_hi += hi

        if args.report_every and ((i + 1) % args.report_every == 0 or i + 1 == n):
            elapsed = time.time() - t0
            print(
                f"rows {i+1:5d}/{n}; centered={centered_boxes}; natural={natural_boxes}; "
                f"fallbacks={derivative_fallbacks}; elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    width = total_hi - total_lo
    midpoint = (total_lo + total_hi) / TWO

    print("\n=== rigorous I2_04 enclosure ===")
    print(f"lower    : {lower_str(total_lo, args.print_digits)}")
    print(f"upper    : {upper_str(total_hi, args.print_digits)}")
    print(f"width    : {upper_str(width, args.print_digits)}")
    print(f"midpoint : {midpoint.str(args.print_digits)}")
    print(f"centered boxes : {centered_boxes}")
    print(f"natural boxes  : {natural_boxes}")
    print(f"AD fallbacks   : {derivative_fallbacks}")
    print(f"elapsed        : {elapsed:.3f} s")

    record = {
        "certificate": "U4 I2_04 rigorous centered first-derivative enclosure",
        "piece": "I2_04_mid_zB_zC_nu_r2",
        "proof_kernel": (
            "symmetric-box centered form using interval first derivatives; "
            "natural Arb fallback on boundary/singular derivative boxes"
        ),
        "n": n,
        "dps": args.dps,
        "boundary_layers": args.boundary_layers,
        "lower": lower_str(total_lo, args.print_digits),
        "upper": upper_str(total_hi, args.print_digits),
        "width": upper_str(width, args.print_digits),
        "midpoint_ball": midpoint.str(args.print_digits),
        "centered_boxes": centered_boxes,
        "natural_boxes": natural_boxes,
        "derivative_fallbacks": derivative_fallbacks,
        "elapsed_seconds": elapsed,
        "script": script_path.name,
        "script_sha256": script_hash,
        "kernel": kernel_path.name,
        "kernel_sha256": kernel_hash,
        "python": sys.version,
        "python_flint": getattr(flint, "__version__", "unknown"),
        "platform": platform.platform(),
    }
    manifest = outdir / "u4_i2_04_centered_manifest.json"
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"manifest       : {manifest}")
    print(f"manifest SHA256: {sha256_file(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

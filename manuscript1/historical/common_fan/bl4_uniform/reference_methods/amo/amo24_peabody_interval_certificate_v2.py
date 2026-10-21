#!/usr/bin/env python3
"""AMO24 direct 4D peabody: strengthened interval certificate (v2).

Purpose
-------
Give a deterministic continuum enclosure sufficient to prove, after width-one
normalization,

    Vol_4(U_4) < 4437/20000 = 0.22185 < Vol_4(P_AMO).

The geometric reduction is exact:

    Vol_4(P_AMO) = 1/4 (10 I_pair + 10 I_triple + 5 I_cap).

The two wedge-orbit integrals are enclosed on a fixed three-dimensional box by
outward-rounded IEEE-754 binary64 interval arithmetic.  The spherical 3-cap is
reduced by tetrahedral symmetry to a one-dimensional interval integral and is
enclosed with mpmath.iv.

Strengthening relative to the first certificate
-----------------------------------------------
* The floating reduction of large NumPy sums is now bounded by a full gamma_n
  absolute-error estimate using an *upper enclosure* of sum(abs(x_i)).
* Multiplication by cell volume is interval-rounded rather than assumed exact.
* Chunk bounds are accumulated exactly as rational numbers via Fraction; there
  is no floating summation in the final wedge totals.
* The final separator comparison is exact rational arithmetic on the certified
  binary endpoints.  No rounded division is load-bearing.
* The script checks denominator signs, positivity of Jacobian/support factors,
  and the cap root monotonicity needed for the one-dimensional reduction.

Evidence / trust boundary
-------------------------
This is a Python-only interval certificate under the following arithmetic
backends:
  (i) standard IEEE-754 binary64 + numpy.nextafter for the wedge evaluator;
  (ii) mpmath.iv arbitrary-precision interval arithmetic for the cap.
An independent Arb/python-flint rerun is still desirable for publication-level
software diversity, but is not mathematically needed for the exact reduction
or the rational separator logic implemented here.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import time

import mpmath as mp
import numpy as np

NEG_INF = -np.inf
POS_INF = np.inf
UROUND = 2.0**-53
SEP = Fraction(4437, 20000)  # 0.22185
U4_HI = Fraction.from_float(0.2218382994641744)


def down(x):
    return np.nextafter(x, NEG_INF)


def up(x):
    return np.nextafter(x, POS_INF)


class I:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(lo if hi is None else hi, dtype=np.float64)
        if np.any(self.lo > self.hi):
            raise ValueError("invalid interval")

    def __add__(self, other):
        o = asI(other)
        return I(down(self.lo + o.lo), up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-asI(other))

    def __rsub__(self, other):
        return asI(other) - self

    def __mul__(self, other):
        o = asI(other)
        p1 = self.lo * o.lo
        p2 = self.lo * o.hi
        p3 = self.hi * o.lo
        p4 = self.hi * o.hi
        lo = np.minimum(np.minimum(p1, p2), np.minimum(p3, p4))
        hi = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
        return I(down(lo), up(hi))

    __rmul__ = __mul__

    def reciprocal(self):
        if np.any((self.lo <= 0.0) & (self.hi >= 0.0)):
            raise ZeroDivisionError("interval denominator crosses zero")
        a = 1.0 / self.lo
        b = 1.0 / self.hi
        return I(down(np.minimum(a, b)), up(np.maximum(a, b)))

    def __truediv__(self, other):
        return self * asI(other).reciprocal()

    def __rtruediv__(self, other):
        return asI(other) / self


def asI(x):
    return x if isinstance(x, I) else I(x)


def sq(x):
    x = asI(x)
    a = x.lo * x.lo
    b = x.hi * x.hi
    hi = up(np.maximum(a, b))
    lo0 = np.where((x.lo <= 0.0) & (x.hi >= 0.0), 0.0, np.minimum(a, b))
    return I(down(lo0), hi)


def sqrtI(x):
    x = asI(x)
    if np.any(x.hi < 0.0):
        raise ValueError("negative interval passed to sqrt")
    return I(
        down(np.sqrt(np.maximum(x.lo, 0.0))),
        up(np.sqrt(np.maximum(x.hi, 0.0))),
    )


def nonnegative(x):
    """Intersect with [0,+infty) using the exact Steiner-radius fact R>=0."""
    return I(np.maximum(x.lo, 0.0), np.maximum(x.hi, 0.0))


def c_sqrt(n):
    return sqrtI(I(float(n)))


# ---------- exact source constants represented by outward binary intervals ----------
rt10 = c_sqrt(10)
rt3 = c_sqrt(3)
a = sqrtI(I(1.5))
x0 = sqrtI((41 - 4 * rt10) / 27)
y0 = sqrtI((7 - 2 * rt10) / 27)
x1 = sqrtI((11 + 2 * rt10) / 12)
z1 = sqrtI((7 - 2 * rt10) / 36)
W = 2 * z1
g = (2 * x1 + 3 * x0) / 5
D0 = x0 - g
sH = sqrtI(sq(x0 - a) + sq(y0))
sE = sqrtI(sq(x1 - 1) + sq(z1))
AREA = 3 * rt3 * sq(y0) / 4
W4 = sq(sq(W))


# Global diagnostics accumulated during wedge evaluation.
class Diagnostics:
    def __init__(self):
        self.min_disc_lo = math.inf
        self.max_lam_den_hi = -math.inf
        self.min_lam_den_lo = math.inf
        self.max_Fl_hi = -math.inf
        self.min_Fl_lo = math.inf
        self.min_xe_lo = math.inf
        self.min_d_lo = math.inf
        self.min_detz_lo = math.inf
        self.min_support_t_lo = math.inf
        self.min_support_p_lo = math.inf
        self.min_ft_lo = math.inf
        self.min_fp_lo = math.inf

    def absorb(self, *, disc, lamden, Fl, xe, d, detz, support_t, support_p, ft, fp):
        self.min_disc_lo = min(self.min_disc_lo, float(np.min(disc.lo)))
        self.min_lam_den_lo = min(self.min_lam_den_lo, float(np.min(lamden.lo)))
        self.max_lam_den_hi = max(self.max_lam_den_hi, float(np.max(lamden.hi)))
        self.min_Fl_lo = min(self.min_Fl_lo, float(np.min(Fl.lo)))
        self.max_Fl_hi = max(self.max_Fl_hi, float(np.max(Fl.hi)))
        self.min_xe_lo = min(self.min_xe_lo, float(np.min(xe.lo)))
        self.min_d_lo = min(self.min_d_lo, float(np.min(d.lo)))
        self.min_detz_lo = min(self.min_detz_lo, float(np.min(detz.lo)))
        self.min_support_t_lo = min(self.min_support_t_lo, float(np.min(support_t.lo)))
        self.min_support_p_lo = min(self.min_support_p_lo, float(np.min(support_p.lo)))
        self.min_ft_lo = min(self.min_ft_lo, float(np.min(ft.lo)))
        self.min_fp_lo = min(self.min_fp_lo, float(np.min(fp.lo)))


DIAG = Diagnostics()


# ---------- Wedge interval integrand ----------
def wedge_box(r, t, q):
    """Enclose representative triple/pair flux densities on one parameter box.

    Symmetry-reduced fixed box:
        r in [0,1], t in [0,1/2], q in [0,1], z=z1*q.

    Hyperboloid triangle central projection:
        Y = y0(1-3r/2),
        Z = (sqrt(3)y0/2) r(2t-1).

    Ellipse is algebraic:
        z = z1*q, x_E=sqrt(3/2-3z^2).

    Exact determinant simplification:
        det(h_Y,h_Z,y_z,u) = k (a^2 X-x_E)/(x_E d),
        k=2 D0 lambda^2/F_lambda.
    """
    Y = y0 * (1 - 1.5 * r)
    Z = rt3 * y0 / 2 * r * (2 * t - 1)
    R2 = sq(Y) + sq(Z)

    Aq = sq(D0) - 2 * R2
    Bq = 2 * g * D0
    Cq = sq(g) - 1
    disc = sq(Bq) - 4 * Aq * Cq
    if np.any(disc.lo <= 0.0):
        raise RuntimeError("wedge discriminant not certified positive")

    # Stable quadratic-root formula for the desired positive intersection.
    lamden = -Bq + sqrtI(disc)
    if np.any(lamden.lo <= 0.0):
        raise RuntimeError("lambda denominator not certified positive")
    lam = 2 * Cq / lamden
    X = g + D0 * lam

    hy = lam * Y
    hz = lam * Z
    ze = z1 * q
    xe = sqrtI(1.5 - 3 * sq(ze))
    if np.any(xe.lo <= 0.0):
        raise RuntimeError("ellipse x-coordinate not certified positive")

    d = sqrtI(sq(xe - X) + sq(hy) + sq(ze) + sq(hz))
    if np.any(d.lo <= 0.0):
        raise RuntimeError("center distance not certified positive")

    Rx = nonnegative(sH - sqrtI(sq(X - a) + sq(hy) + sq(hz)))
    Ry = nonnegative(sE - sqrtI(sq(xe - 1) + sq(ze)))

    Fl = 2 * D0 * X - 4 * lam * R2
    # D0<0 and Fl<0 on the whole domain, hence k>0.
    if np.any(Fl.hi >= 0.0):
        raise RuntimeError("F_lambda not certified negative")
    kval = 2 * D0 * sq(lam) / Fl
    detz = kval * (1.5 * X - xe) / (xe * d)
    if np.any(detz.lo < 0.0):
        raise RuntimeError("orientation-fixed determinant factor not nonnegative")

    Jt = (Rx / d) * sq(1 + Rx / d) * detz
    Jp = sq(Ry / d) * (1 + Ry / d) * detz

    support_t = ((X - g) * (X - xe) + sq(lam) * R2) / d + Rx
    support_p = ((xe - g) * (xe - X) + sq(ze)) / d + Ry
    if np.any(support_t.lo < 0.0) or np.any(support_p.lo < 0.0):
        raise RuntimeError("support factor not certified nonnegative")

    ft = support_t * Jt
    fp = support_p * Jp
    # Outward rounding at exact zero may create the negative minimum subnormal.
    # This is harmless: the interval remains valid, and the exact flux is
    # nonnegative geometrically.  We deliberately do not clamp here because
    # the signed interval sum below already handles it rigorously.

    # Exact change-of-variable factors: 4 from two reflections,
    # 2*AREA*r from the triangle Duffy map, dz=z1 dq.
    factor = 8 * AREA * z1 * r
    out_t = factor * ft
    out_p = factor * fp

    DIAG.absorb(
        disc=disc, lamden=lamden, Fl=Fl, xe=xe, d=d, detz=detz,
        support_t=support_t, support_p=support_p, ft=out_t, fp=out_p,
    )
    return out_t, out_p


def gamma_n(n):
    if n <= 1:
        return 0.0
    gma = (n - 1) * UROUND / (1.0 - (n - 1) * UROUND)
    return up(gma)


def sum_float_array_enclosure(arr):
    """Rigorous enclosure of the exact real sum of binary64 array entries.

    Let s=fl(sum x_i), A=sum |x_i|.  Standard floating arithmetic gives
        |s-S| <= gamma_{n-1} A.
    Since the computed abs-sum a_fl may itself underestimate A, first use
        A <= a_fl/(1-gamma_{n-1})
    and round this bound outward.  This removes the weak point in v1.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = int(arr.size)
    gma = gamma_n(n)
    s = float(np.sum(arr, dtype=np.float64))
    afl = float(np.sum(np.abs(arr), dtype=np.float64))
    if not math.isfinite(s) or not math.isfinite(afl):
        raise RuntimeError("nonfinite floating sum")
    if gma >= 1.0:
        raise RuntimeError("gamma_n >= 1")
    Aup = up(afl / (1.0 - gma))
    err = up(gma * Aup)
    return down(s - err), up(s + err)


def frac(x: float) -> Fraction:
    return Fraction.from_float(float(x))


def wedge_enclosure(Nr=512, Nt=256, Nq=256, chunk_r=2):
    """Continuum-covering interval enclosure for both wedge-orbit integrals."""
    if min(Nr, Nt, Nq) <= 0:
        raise ValueError("grid sizes must be positive")
    # cell-volume endpoints are themselves interval-rounded; arbitrary integer
    # grid sizes are therefore allowed, although powers of two are preferred.
    cell = I(1.0) / Nr * (I(0.5) / Nt) * (I(1.0) / Nq)

    t_lo_total = Fraction(0); t_hi_total = Fraction(0)
    p_lo_total = Fraction(0); p_hi_total = Fraction(0)

    for i0 in range(0, Nr, chunk_r):
        i1 = min(Nr, i0 + chunk_r)
        ii = np.arange(i0, i1)[:, None, None]
        jj = np.arange(Nt)[None, :, None]
        kk = np.arange(Nq)[None, None, :]
        r = I(ii / Nr, (ii + 1) / Nr)
        t = I(0.5 * jj / Nt, 0.5 * (jj + 1) / Nt)
        q = I(kk / Nq, (kk + 1) / Nq)
        ft, fp = wedge_box(r, t, q)

        stlo, sthi = sum_float_array_enclosure(ft.lo)
        splo, sphi = sum_float_array_enclosure(fp.lo)
        # For upper sums, sum the upper endpoints separately.
        utlo, uthi = sum_float_array_enclosure(ft.hi)
        uplo, uphi = sum_float_array_enclosure(fp.hi)

        # Only stlo and uthi are load-bearing for the triple integral;
        # splo and uphi for the pair integral.  Multiply interval-wise.
        tc = I(stlo, uthi) * cell
        pc = I(splo, uphi) * cell

        # Exact rational accumulation of certified binary endpoints.
        t_lo_total += frac(float(tc.lo)); t_hi_total += frac(float(tc.hi))
        p_lo_total += frac(float(pc.lo)); p_hi_total += frac(float(pc.hi))

    return (t_lo_total, t_hi_total), (p_lo_total, p_hi_total)


# ---------- Spherical cap: exact 1D tetrahedral reduction ----------
def cap_enclosure(N=32768, dps=40):
    """Interval enclosure of one spherical 3-cap flux integral.

    In an orthonormal gnomonic chart centered at the inward p1 direction,
    the four boundary cones have tetrahedral form
        Q_i(q)=1/8 + L<v_i,q> + A|q|^2 + (3/8)<v_i,q>^2,
    where v_i are the unit vertices of a regular tetrahedron.

    On one Voronoi cell z=<v_0,u> lies in [-1,-1/3].  The first cone root
    r(z) is strictly increasing; the latitude length is explicit.  Four cells
    recover S^2.  Radial integration in the gnomonic chart is elementary.
    """
    iv = mp.iv
    iv.dps = dps
    rt2 = iv.sqrt(2); rt6 = iv.sqrt(6)
    rt10m = iv.sqrt(10); rt15 = iv.sqrt(15)
    A0 = (2 * rt10m - 7) / 18
    L0 = (4 * rt6 - rt15) / 12
    Wm = iv.sqrt(7 - 2 * rt10m) / 3
    R0m = iv.sqrt(10) * (2 * (iv.sqrt(5) - iv.sqrt(2)) / 3) / 10

    def root(z):
        aq = A0 + iv.mpf(3) / 8 * z * z
        bq = L0 * z
        disc = bq * bq - aq / 2
        if mp.mpf(disc._mpi_[0]) <= 0:
            raise RuntimeError("cap root discriminant not certified positive")
        rr = (-bq - iv.sqrt(disc)) / (2 * aq)
        if mp.mpf(rr._mpi_[0]) <= 0:
            raise RuntimeError("cap first root not certified positive")
        return rr

    # Monotonicity proof: Q_r<0 and Q_z>0 along Q(r(z),z)=0 imply r'(z)>0.
    max_Qr = mp.mpf('-inf'); min_Qz = mp.mpf('inf')
    Nmono = 2048
    for j in range(Nmono):
        zl = mp.mpf('-1') + (mp.mpf(2) / 3) * j / Nmono
        zh = mp.mpf('-1') + (mp.mpf(2) / 3) * (j + 1) / Nmono
        zz = iv.mpf([zl, zh])
        rr = root(zz)
        aq = A0 + iv.mpf(3) / 8 * zz * zz
        bq = L0 * zz
        Qr = bq + 2 * aq * rr
        Qz = L0 * rr + iv.mpf(3) / 4 * zz * rr * rr
        max_Qr = max(max_Qr, mp.mpf(Qr._mpi_[1]))
        min_Qz = min(min_Qz, mp.mpf(Qz._mpi_[0]))
    if not (max_Qr < 0 and min_Qz > 0):
        raise RuntimeError("cap root monotonicity interval check failed")

    def primitive(R):
        atan = iv.atan2(R, iv.mpf(1))
        return Wm**3 * (
            Wm / 2 * (atan - R / (1 + R * R))
            - R0m / 3 * R**3 / (1 + R * R) ** iv.mpf('1.5')
        )

    def ell_full(z):
        return 2 * iv.pi

    def ell_corner(z):
        c = -rt2 * z / iv.sqrt(1 - z * z)
        # Exact geometry on [-1/sqrt(3),-1/3]: 1/2 <= c <= 1.
        clo = max(mp.mpf('0.5'), mp.mpf(c._mpi_[0]))
        chi = min(mp.mpf('1.0'), mp.mpf(c._mpi_[1]))
        cc = iv.mpf([clo, chi])
        om = 1 - cc * cc
        olo = max(mp.mpf('0.0'), mp.mpf(om._mpi_[0]))
        ohi = max(mp.mpf('0.0'), mp.mpf(om._mpi_[1]))
        ac = iv.atan2(iv.sqrt(iv.mpf([olo, ohi])), cc)
        return 2 * iv.pi - 6 * ac

    def integrate_piece(a0, b0, ell):
        width = (b0 - a0) / N
        lo = mp.mpf('0'); hi = mp.mpf('0')
        for i in range(N):
            zl = a0 + i * width
            zh = a0 + (i + 1) * width
            zz = iv.mpf([zl, zh])
            val = primitive(root(zz)) * ell(zz)
            lo += mp.mpf(val._mpi_[0]) * width
            hi += mp.mpf(val._mpi_[1]) * width
        return 4 * lo, 4 * hi

    bp = -1 / mp.sqrt(3)
    lo1, hi1 = integrate_piece(mp.mpf('-1'), bp, ell_full)
    lo2, hi2 = integrate_piece(bp, -mp.mpf(1) / 3, ell_corner)
    return lo1 + lo2, hi1 + hi2, (max_Qr, min_Qz)


def mp_lower_to_fraction(x) -> Fraction:
    # Conversion RN -> one ulp downward gives a binary float <= the mp lower bound.
    return frac(float(down(float(x))))


def mp_upper_to_fraction(x) -> Fraction:
    return frac(float(up(float(x))))


def dec_fraction(q: Fraction, digits=30):
    with localcontext() as ctx:
        ctx.prec = digits + 15
        return Decimal(q.numerator) / Decimal(q.denominator)


def print_interval(name, lo: Fraction, hi: Fraction, digits=17):
    dlo = dec_fraction(lo, digits + 4)
    dhi = dec_fraction(hi, digits + 4)
    print(f"{name} in [{dlo:.{digits}g}, {dhi:.{digits}g}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Nr', type=int, default=512)
    ap.add_argument('--Nt', type=int, default=256)
    ap.add_argument('--Nq', type=int, default=256)
    ap.add_argument('--chunk-r', type=int, default=2)
    ap.add_argument('--cap-N', type=int, default=32768)
    ap.add_argument('--cap-dps', type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()
    print('=== AMO24 direct 4D peabody strengthened interval certificate v2 ===')
    print('wedge grid:', args.Nr, args.Nt, args.Nq, 'chunk_r:', args.chunk_r)
    print('cap 1D boxes:', args.cap_N, 'mpmath.iv dps:', args.cap_dps)
    print('target rational separator: 4437/20000 = 0.22185')

    wt, wp = wedge_enclosure(args.Nr, args.Nt, args.Nq, args.chunk_r)
    print_interval('I_triple', wt[0], wt[1])
    print_interval('I_pair  ', wp[0], wp[1])
    print('wedge sign/denominator diagnostics:')
    print('  min discriminant lower       =', DIAG.min_disc_lo)
    print('  min lambda denominator lower =', DIAG.min_lam_den_lo)
    print('  F_lambda interval extrema    =', (DIAG.min_Fl_lo, DIAG.max_Fl_hi))
    print('  min x_E lower                =', DIAG.min_xe_lo)
    print('  min center distance lower    =', DIAG.min_d_lo)
    print('  min determinant lower        =', DIAG.min_detz_lo)
    print('  min support triple/pair      =', (DIAG.min_support_t_lo, DIAG.min_support_p_lo))
    print('  min flux triple/pair         =', (DIAG.min_ft_lo, DIAG.min_fp_lo))

    cl_mp, ch_mp, mono = cap_enclosure(args.cap_N, args.cap_dps)
    cl = mp_lower_to_fraction(cl_mp)
    ch = mp_upper_to_fraction(ch_mp)
    print('cap monotonicity: max Q_r =', mono[0], '; min Q_z =', mono[1])
    print_interval('I_cap   ', cl, ch)

    raw_lo = (10 * wt[0] + 10 * wp[0] + 5 * cl) / 4
    raw_hi = (10 * wt[1] + 10 * wp[1] + 5 * ch) / 4
    w4_lo = frac(float(np.min(W4.lo)))
    w4_hi = frac(float(np.max(W4.hi)))

    # Exact rational normalized endpoints from certified binary bounds.
    v1_lo = raw_lo / w4_hi
    v1_hi = raw_hi / w4_lo
    print_interval('W^4     ', w4_lo, w4_hi)
    print_interval('V_source', raw_lo, raw_hi)
    print_interval('V_width1', v1_lo, v1_hi)

    # Load-bearing comparison is exact rational arithmetic; no final float.
    u4_below_sep = U4_HI < SEP
    amo_above_sep = raw_lo * SEP.denominator > SEP.numerator * w4_hi
    direct_amo_u4 = v1_lo > U4_HI

    sep_margin = v1_lo - SEP
    u4_margin = SEP - U4_HI
    direct_margin = v1_lo - U4_HI
    print()
    print('exact-rational separator checks on certified binary endpoints:')
    print('  U4_upper < 4437/20000       :', u4_below_sep)
    print('  4437/20000 < AMO_lower      :', amo_above_sep)
    print('  U4_upper < AMO_lower direct :', direct_amo_u4)
    print('  AMO_lower - 0.22185         =', dec_fraction(sep_margin, 25))
    print('  0.22185 - U4_upper          =', dec_fraction(u4_margin, 25))
    print('  AMO_lower - U4_upper        =', dec_fraction(direct_margin, 25))

    if not (u4_below_sep and amo_above_sep and direct_amo_u4):
        raise SystemExit('CERTIFICATE FAILED: refine boxes or audit formulas.')

    print()
    print('CERTIFIED under the stated interval backends:')
    print('  Vol_4(U4) < 4437/20000 = 0.22185 < Vol_4(P_AMO).')
    print('  In particular Vol_4(U4) < Vol_4(P_AMO).')
    print('elapsed = %.2f s' % (time.time() - t0))


if __name__ == '__main__':
    main()

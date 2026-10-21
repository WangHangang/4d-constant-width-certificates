#!/usr/bin/env python3
r"""End-to-end computer-assisted upper bound for the canonical raised U3.

Prerequisite certificate
------------------------
The companion program ``raised_u3_certify_centres.py`` proves, by outward-
rounded interval arithmetic, that the finite negative-half centre set is a
subset of the true positive half K_+.  This program checks the SHA-256 digest
of that certified centre file before using it.

Cubature
--------
One tetrahedral chamber of S^2 is parameterized exactly by

    h(a,s)=(1,a,a s)/sqrt(D),  D=1+a^2(1+s^2),
    0<=a<=1, -1<=s<=1,

with area element a D^{-3/2} da ds.  The vertical angle is represented by
a rational half-angle coordinate q in [0,1], using two charts:

    u_+(a,s,q)=( 2q/(1+q^2) h(a,s), +(1-q^2)/(1+q^2)),
    u_-(a,s,q)=( 2q/(1+q^2) h(a,s), -(1-q^2)/(1+q^2)).

The S^3 area factor in either chart is 8q^2/(1+q^2)^3.  Since the chamber
has 24 images and Vol_4=(1/4) int_{S^3} rho^4, the integrand is

    48 rho(u)^4 a q^2 / [D^(3/2)(1+q^2)^3].

On each parameter box a single admissible unit-ball centre is fixed.  Its
radial root is a pointwise upper bound for the true radial function.  Tensor
2x2x2 Gauss quadrature is supplemented by outward-rounded bounds for each
pure fourth derivative, giving a rigorous tensor-product remainder bound.
No trigonometric library call enters the validated integrand or its jets.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parent
DEFAULT_CENTERS = HERE / "raised_u3_s4_certified_centers.npz"
DEFAULT_CERTIFICATE = HERE / "raised_u3_s4_centres_certificate.json"
BODY5_LOWER = 0.222881612903959
ARC_GRID_N = 16384
GL = 1.0 / math.sqrt(3.0)

VERTEX_SIGNS = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.int8
)
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@njit(inline="always")
def dn(x):
    return np.nextafter(x, -np.inf)


@njit(inline="always")
def up(x):
    return np.nextafter(x, np.inf)


@njit(inline="always")
def iadd(al, ah, bl, bh):
    return dn(al + bl), up(ah + bh)


@njit(inline="always")
def isub(al, ah, bl, bh):
    return dn(al - bh), up(ah - bl)


@njit(inline="always")
def imul(al, ah, bl, bh):
    p1, p2, p3, p4 = al * bl, al * bh, ah * bl, ah * bh
    return dn(min(p1, p2, p3, p4)), up(max(p1, p2, p3, p4))


@njit(inline="always")
def isqr(al, ah):
    if al <= 0.0 <= ah:
        return 0.0, up(max(al * al, ah * ah))
    return max(0.0, dn(min(al * al, ah * ah))), up(max(al * al, ah * ah))


@njit(inline="always")
def irecip(al, ah):
    return dn(1.0 / ah), up(1.0 / al)


@njit(inline="always")
def idiv(al, ah, bl, bh):
    rl, rh = irecip(bl, bh)
    return imul(al, ah, rl, rh)


@njit(inline="always")
def isqrt(al, ah):
    if al < 0.0:
        al = 0.0
    return max(0.0, dn(math.sqrt(al))), up(math.sqrt(max(0.0, ah)))


@njit
def jconst(lo, hi):
    L = np.zeros(5)
    H = np.zeros(5)
    L[0], H[0] = lo, hi
    return L, H


@njit
def jvar(lo, hi):
    L = np.zeros(5)
    H = np.zeros(5)
    L[0], H[0] = lo, hi
    L[1] = H[1] = 1.0
    return L, H


@njit
def jadd(AL, AH, BL, BH):
    L, H = np.empty(5), np.empty(5)
    for n in range(5):
        L[n], H[n] = iadd(AL[n], AH[n], BL[n], BH[n])
    return L, H


@njit
def jsub(AL, AH, BL, BH):
    L, H = np.empty(5), np.empty(5)
    for n in range(5):
        L[n], H[n] = isub(AL[n], AH[n], BL[n], BH[n])
    return L, H


@njit
def jmul(AL, AH, BL, BH):
    L, H = np.empty(5), np.empty(5)
    for n in range(5):
        sl = sh = 0.0
        for k in range(n + 1):
            pl, ph = imul(AL[k], AH[k], BL[n - k], BH[n - k])
            sl, sh = iadd(sl, sh, pl, ph)
        L[n], H[n] = sl, sh
    return L, H


@njit
def jsqr(AL, AH):
    L, H = jmul(AL, AH, AL, AH)
    L[0], H[0] = isqr(AL[0], AH[0])
    return L, H


@njit
def jrecip(AL, AH):
    L, H = np.empty(5), np.empty(5)
    L[0], H[0] = irecip(AL[0], AH[0])
    for n in range(1, 5):
        sl = sh = 0.0
        for k in range(1, n + 1):
            pl, ph = imul(AL[k], AH[k], L[n - k], H[n - k])
            sl, sh = iadd(sl, sh, pl, ph)
        pl, ph = imul(L[0], H[0], sl, sh)
        L[n], H[n] = -ph, -pl
    return L, H


@njit
def jdiv(AL, AH, BL, BH):
    RL, RH = jrecip(BL, BH)
    return jmul(AL, AH, RL, RH)


@njit
def jsqrt(AL, AH):
    L, H = np.empty(5), np.empty(5)
    L[0], H[0] = isqrt(AL[0], AH[0])
    dyl, dyh = imul(2.0, 2.0, L[0], H[0])
    for n in range(1, 5):
        sl = sh = 0.0
        for k in range(1, n):
            pl, ph = imul(L[k], H[k], L[n - k], H[n - k])
            sl, sh = iadd(sl, sh, pl, ph)
        nl, nh = isub(AL[n], AH[n], sl, sh)
        L[n], H[n] = idiv(nl, nh, dyl, dyh)
    return L, H


@njit
def integrand_jet(bounds, clo, chi, vertical_sign, diff_axis):
    jetsL, jetsH = np.empty((3, 5)), np.empty((3, 5))
    for r in range(3):
        lo = np.nextafter(bounds[r, 0], -np.inf)
        hi = np.nextafter(bounds[r, 1], np.inf)
        if r == diff_axis:
            L, H = jvar(lo, hi)
        else:
            L, H = jconst(lo, hi)
        jetsL[r], jetsH[r] = L, H
    AL, AH = jetsL[0], jetsH[0]
    SL, SH = jetsL[1], jetsH[1]
    QL, QH = jetsL[2], jetsH[2]

    A2L, A2H = jsqr(AL, AH)
    S2L, S2H = jsqr(SL, SH)
    oneS2L, oneS2H = jadd(*jconst(1.0, 1.0), S2L, S2H)
    tmpL, tmpH = jmul(A2L, A2H, oneS2L, oneS2H)
    DL, DH = jadd(*jconst(1.0, 1.0), tmpL, tmpH)
    sqrtDL, sqrtDH = jsqrt(DL, DH)

    Q2L, Q2H = jsqr(QL, QH)
    denL, denH = jadd(*jconst(1.0, 1.0), Q2L, Q2H)
    twoQL, twoQH = jmul(*jconst(2.0, 2.0), QL, QH)
    horizL, horizH = jdiv(twoQL, twoQH, denL, denH)
    oneMinusQ2L, oneMinusQ2H = jsub(*jconst(1.0, 1.0), Q2L, Q2H)
    vertL, vertH = jdiv(oneMinusQ2L, oneMinusQ2H, denL, denH)
    if vertical_sign < 0:
        vertL, vertH = -vertH, -vertL

    facL, facH = jdiv(horizL, horizH, sqrtDL, sqrtDH)
    U0L, U0H = facL, facH
    U1L, U1H = jmul(facL, facH, AL, AH)
    ASL, ASH = jmul(AL, AH, SL, SH)
    U2L, U2H = jmul(facL, facH, ASL, ASH)
    U3L, U3H = vertL, vertH

    LL, LH = jconst(0.0, 0.0)
    N2L, N2H = jconst(0.0, 0.0)
    for j, (UL, UH) in enumerate(((U0L, U0H), (U1L, U1H), (U2L, U2H), (U3L, U3H))):
        CL, CH = jconst(clo[j], chi[j])
        xL, xH = jmul(CL, CH, UL, UH)
        LL, LH = jadd(LL, LH, xL, xH)
        xL, xH = jsqr(CL, CH)
        N2L, N2H = jadd(N2L, N2H, xL, xH)

    deltaL, deltaH = jsub(*jconst(1.0, 1.0), N2L, N2H)
    L2L, L2H = jsqr(LL, LH)
    radL, radH = jadd(L2L, L2H, deltaL, deltaH)
    radL, radH = jsqrt(radL, radH)
    rootL, rootH = jadd(LL, LH, radL, radH)
    root2L, root2H = jsqr(rootL, rootH)
    root4L, root4H = jsqr(root2L, root2H)

    DsqrtL, DsqrtH = jsqrt(DL, DH)
    D32L, D32H = jmul(DL, DH, DsqrtL, DsqrtH)
    den2L, den2H = jsqr(denL, denH)
    den3L, den3H = jmul(den2L, den2H, denL, denH)
    aq2L, aq2H = jmul(AL, AH, Q2L, Q2H)
    jacDenL, jacDenH = jmul(D32L, D32H, den3L, den3H)
    jacL, jacH = jdiv(aq2L, aq2H, jacDenL, jacDenH)

    FL, FH = jmul(root4L, root4H, jacL, jacH)
    FL, FH = jmul(*jconst(48.0, 48.0), FL, FH)
    return FL, FH


@njit
def integrand_point(a, s, q, clo, chi, vertical_sign):
    D = 1.0 + a * a * (1.0 + s * s)
    sqrtD = math.sqrt(D)
    den = 1.0 + q * q
    horizontal = 2.0 * q / den
    vertical = (1.0 - q * q) / den
    if vertical_sign < 0:
        vertical = -vertical
    u0 = horizontal / sqrtD
    u1 = a * u0
    u2 = a * s * u0
    u3 = vertical

    Llo = Lhi = N2lo = N2hi = 0.0
    for j, u in enumerate((u0, u1, u2, u3)):
        pl, ph = imul(clo[j], chi[j], u, u)
        Llo, Lhi = iadd(Llo, Lhi, pl, ph)
        pl, ph = isqr(clo[j], chi[j])
        N2lo, N2hi = iadd(N2lo, N2hi, pl, ph)
    dl, dh = isub(1.0, 1.0, N2lo, N2hi)
    l2, h2 = isqr(Llo, Lhi)
    radl, radh = iadd(l2, h2, dl, dh)
    radl, radh = isqrt(radl, radh)
    rl, rh = iadd(Llo, Lhi, radl, radh)
    r2l, r2h = isqr(rl, rh)
    r4l, r4h = isqr(r2l, r2h)
    J = 48.0 * a * q * q / (D * sqrtD * den * den * den)
    return imul(J, J, r4l, r4h)


@njit(parallel=True)
def eval_boxes(bounds, signs, clo, chi):
    n = len(bounds)
    ubs, proxies = np.empty(n), np.empty(n)
    for b in prange(n):
        mid, half, lengths = np.empty(3), np.empty(3), np.empty(3)
        for r in range(3):
            mid[r] = 0.5 * (bounds[b, r, 0] + bounds[b, r, 1])
            lengths[r] = bounds[b, r, 1] - bounds[b, r, 0]
            half[r] = 0.5 * lengths[r]
        qsum = 0.0
        for mask in range(8):
            x = np.empty(3)
            for r in range(3):
                x[r] = mid[r] + half[r] * GL * (1.0 if ((mask >> r) & 1) else -1.0)
            fl, fh = integrand_point(x[0], x[1], x[2], clo[b], chi[b], signs[b])
            qsum = up(qsum + fh)
        weight = half[0] * half[1] * half[2]
        quadrature = up(qsum * up(weight))

        M = np.empty(3)
        for r in range(3):
            FL, FH = integrand_jet(bounds[b], clo[b], chi[b], signs[b], r)
            M[r] = up(24.0 * max(abs(FL[4]), abs(FH[4])))
        la, ls, lq = lengths[0], lengths[1], lengths[2]
        remainder = (
            ls * lq * la**5 * M[0]
            + la * lq * ls**5 * M[1]
            + la * ls * lq**5 * M[2]
        ) / 4320.0
        ubs[b] = up(quadrature + up(remainder))
        proxies[b] = quadrature
    return ubs, proxies


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tetrahedral_signed_permutations():
    group = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            if signs[0] * signs[1] * signs[2] == 1.0:
                group.append((np.asarray(perm, dtype=np.int64), np.asarray(signs, dtype=np.float64)))
    assert len(group) == 24
    return group


def exact_symmetry_closure(centres: np.ndarray) -> bool:
    rows = {tuple(row) for row in centres.tolist()}
    for perm, signs in tetrahedral_signed_permutations():
        image = centres.copy()
        image[:, :3] = centres[:, perm] * signs
        for row in image.tolist():
            if tuple(row) not in rows:
                return False
    return True


def interval_to_float_bounds(x):
    lo = float(x.a)
    hi = float(x.b)
    return np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)


def build_arc_grid(N: int):
    """Rigorous enclosures of exact arc points at fixed grid parameters."""
    mp.iv.dps = 70
    lo = np.empty((6, N + 1, 4))
    hi = np.empty_like(lo)
    mid = np.empty_like(lo)
    s0 = mp.iv.mpf(1) / (2 * mp.iv.sqrt(2))
    for j in range(N + 1):
        t = mp.iv.pi * j / (2 * N)
        cosine = mp.iv.cos(t)
        sine = mp.iv.sin(t)
        for arc, (ia, ib) in enumerate(PAIRS):
            for k in range(3):
                coordinate = s0 * (
                    int(VERTEX_SIGNS[ia, k]) * cosine
                    + int(VERTEX_SIGNS[ib, k]) * sine
                )
                lo[arc, j, k], hi[arc, j, k] = interval_to_float_bounds(coordinate)
            lo[arc, j, 3] = hi[arc, j, 3] = 0.0
            mid[arc, j] = 0.5 * (lo[arc, j] + hi[arc, j])
    return lo, hi, mid


def direction(a, s, q, vertical_sign):
    D = 1.0 + a * a * (1.0 + s * s)
    den = 1.0 + q * q
    horizontal = 2.0 * q / den
    scale = horizontal / math.sqrt(D)
    vertical = vertical_sign * (1.0 - q * q) / den
    return np.array([scale, a * scale, a * s * scale, vertical])


def assign(bounds, signs, centres, qlo, qhi, qmid):
    n = len(bounds)
    clo, chi = np.empty((n, 4)), np.empty((n, 4))
    centre_norm2 = np.sum(centres * centres, axis=1)
    step = max(1, ARC_GRID_N // 128)
    coarse = np.arange(0, ARC_GRID_N + 1, step, dtype=int)
    for b in range(n):
        m = 0.5 * (bounds[b, :, 0] + bounds[b, :, 1])
        u = direction(m[0], m[1], m[2], int(signs[b]))
        if signs[b] < 0:
            dots = centres @ u
            roots = dots + np.sqrt(np.maximum(0.0, dots * dots + 1.0 - centre_norm2))
            j = int(np.argmin(roots))
            clo[b] = chi[b] = centres[j]
        else:
            best, best_arc, best_index = 1.0e99, 0, 0
            for arc in range(6):
                qq = qmid[arc, coarse]
                dots = qq @ u
                delta = 1.0 - np.sum(qq * qq, axis=1)
                roots = dots + np.sqrt(np.maximum(0.0, dots * dots + delta))
                local = int(np.argmin(roots))
                if roots[local] < best:
                    best = float(roots[local])
                    best_arc = arc
                    best_index = int(coarse[local])
            lower = max(0, best_index - step)
            upper = min(ARC_GRID_N, best_index + step)
            ids = np.arange(lower, upper + 1)
            qq = qmid[best_arc, ids]
            dots = qq @ u
            delta = 1.0 - np.sum(qq * qq, axis=1)
            roots = dots + np.sqrt(np.maximum(0.0, dots * dots + delta))
            best_index = int(ids[int(np.argmin(roots))])
            clo[b], chi[b] = qlo[best_arc, best_index], qhi[best_arc, best_index]
    return clo, chi


def make_boxes(na, ns, nq):
    ga = np.linspace(0.0, 1.0, na + 1)
    gs = np.linspace(-1.0, 1.0, ns + 1)
    gq = np.linspace(0.0, 1.0, nq + 1)
    boxes, signs = [], []
    for vertical_sign in (1, -1):
        for i in range(na):
            for j in range(ns):
                for k in range(nq):
                    boxes.append(np.array(
                        [[ga[i], ga[i + 1]], [gs[j], gs[j + 1]], [gq[k], gq[k + 1]]],
                        dtype=float,
                    ))
                    signs.append(vertical_sign)
    return boxes, np.asarray(signs, dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Certified S4-chamber cubature for the true canonical raised U3.")
    parser.add_argument("--centers-file", type=Path, default=DEFAULT_CENTERS)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--na", type=int, default=4)
    parser.add_argument("--ns", type=int, default=8)
    parser.add_argument("--nq", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--max-boxes", type=int, default=180000)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--safety-margin", type=float, default=1.0e-5)
    args = parser.parse_args()

    centers_path = args.centers_file.resolve()
    certificate_path = args.certificate.resolve()
    if not centers_path.exists() or not certificate_path.exists():
        raise FileNotFoundError("Place this script, the certified-centre NPZ, and its JSON certificate together.")
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(centers_path)
    if actual_hash != certificate.get("output_centers_sha256"):
        raise RuntimeError("Certified-centre SHA-256 does not match the membership certificate.")
    data = np.load(centers_path)
    centres = data["centers"]

    print("script SHA-256:", sha256_file(Path(__file__).resolve()))
    print("centre certificate SHA-256:", sha256_file(certificate_path))
    print("certified centres SHA-256:", actual_hash)
    print("centres:", len(centres))
    print("membership certificate worst slack lower:", certificate["worst_certified_slack_lower"])
    closure = exact_symmetry_closure(centres)
    print("exact S4 closure:", "PASS" if closure else "FAIL")
    if not closure:
        raise RuntimeError("The certified centre set is not exactly S4-closed.")

    target = BODY5_LOWER - args.safety_margin
    print(f"target upper: {target:.15f} (Body5 lower minus {args.safety_margin:.3e})")
    print("precomputing interval-certified positive arc centres...", flush=True)
    qlo, qhi, qmid = build_arc_grid(ARC_GRID_N)

    records, record_signs = make_boxes(args.na, args.ns, args.nq)
    bounds = np.asarray(records)
    clo, chi = assign(bounds, record_signs, centres, qlo, qhi, qmid)
    print("evaluating initial", len(records), "boxes...", flush=True)
    ub, proxy_values = eval_boxes(bounds, record_signs, clo, chi)
    total = float(np.sum(ub))
    proxy = float(np.sum(proxy_values))

    heap = []
    uid = 0
    for box, sign, upper, prox in zip(records, record_signs, ub, proxy_values):
        heapq.heappush(heap, (-(float(upper) - float(prox)), uid, box, int(sign), float(upper), float(prox)))
        uid += 1

    iteration = 0
    while len(heap) < args.max_boxes and total >= target:
        picked = [heapq.heappop(heap) for _ in range(min(args.batch, len(heap)))]
        children, child_signs = [], []
        previous_upper = previous_proxy = 0.0
        for _, _, box, sign, upper, prox in picked:
            previous_upper += upper
            previous_proxy += prox
            lengths = box[:, 1] - box[:, 0]
            axis = int(np.argmax(lengths))
            middle = 0.5 * (box[axis, 0] + box[axis, 1])
            first, second = box.copy(), box.copy()
            first[axis, 1], second[axis, 0] = middle, middle
            children.extend((first, second))
            child_signs.extend((sign, sign))

        child_bounds = np.asarray(children)
        child_sign_array = np.asarray(child_signs, dtype=np.int8)
        child_clo, child_chi = assign(child_bounds, child_sign_array, centres, qlo, qhi, qmid)
        child_upper, child_proxy = eval_boxes(child_bounds, child_sign_array, child_clo, child_chi)
        new_upper, new_proxy = float(np.sum(child_upper)), float(np.sum(child_proxy))
        total = np.nextafter(total - previous_upper + new_upper, np.inf)
        proxy = proxy - previous_proxy + new_proxy

        for box, sign, upper, prox in zip(children, child_sign_array, child_upper, child_proxy):
            heapq.heappush(heap, (-(float(upper) - float(prox)), uid, box, int(sign), float(upper), float(prox)))
            uid += 1
        iteration += 1
        if iteration % args.report_every == 0 or total < target:
            print(
                f"iter={iteration:5d} boxes={len(heap):7d} upper={total:.12f} "
                f"proxy={proxy:.12f} validation_slack={total-proxy:.3e} "
                f"Body5_gap={BODY5_LOWER-total:.3e}",
                flush=True,
            )

    print("\n=== fully certified raised-U3 upper-bound report ===")
    print("boxes:", len(heap))
    print(f"validated upper: {total:.15f}")
    print(f"Gauss proxy for the same boxwise constraints: {proxy:.15f}")
    print(f"total validation slack: {total-proxy:.3e}")
    print(f"Body5 certified lower: {BODY5_LOWER:.15f}")
    print(f"strict comparison gap: {BODY5_LOWER-total:.3e}")
    if total < BODY5_LOWER:
        print("FULL COMPUTER-ASSISTED INEQUALITY PROVED:")
        print("Vol_4(canonical raised U3) < Vol_4(Body 5).")
    else:
        print("NOT YET SEPARATED; increase --max-boxes or improve the finite centre set.")


if __name__ == "__main__":
    main()

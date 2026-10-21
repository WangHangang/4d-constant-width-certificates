#!/usr/bin/env python3
"""Raised-body certificates for the five-candidate volume ordering.

No local imports, floating Qhull, stored per-box bounds, or centre proposal
program is trusted. Required Python packages: numpy, numba, mpmath.
Only the previously certified U4 and Body5 endpoints are external inputs.
Both lower and upper certificates for each raised body are rechecked here.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,itertools,json,math,os,sys,time
from pathlib import Path
from fractions import Fraction
from typing import Optional, Sequence
import numpy as np
import mpmath as mp
from numba import njit,prange,set_num_threads
VERTEX_SIGNS=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=np.int8)
SIGNS=VERTEX_SIGNS
PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
RM_EXPECTED_SHA256='43997a1ed71f4c3240146112a74b886d1355113c18c14963ee9b9798d3954412'
VERT={'A':(1,0,0),'B':(0,1,0),'C':(0,0,1),'D':(-1,-1,-1)}
FACE_ARCS=[('A','B','C','D'),('A','C','B','D'),('A','D','B','C')]


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
    if al <= 0.0:
        return np.nan,np.nan
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
    if AL[0] <= 0.0:
        return np.full(5,np.nan),np.full(5,np.nan)
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
def integrand_jet(bounds, clo, chi, vertical_sign, diff_axis, rep=-1, factor=48.0):
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
    if rep >= 0:
        U0L,U0H = jmul(U0L,U0H,*jconst(float(VERTEX_SIGNS[rep,0]),float(VERTEX_SIGNS[rep,0])))
        U1L,U1H = jmul(U1L,U1H,*jconst(float(VERTEX_SIGNS[rep,1]),float(VERTEX_SIGNS[rep,1])))
        U2L,U2H = jmul(U2L,U2H,*jconst(float(VERTEX_SIGNS[rep,2]),float(VERTEX_SIGNS[rep,2])))

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
    FL, FH = jmul(*jconst(factor, factor), FL, FH)
    return FL, FH


@njit
def integrand_interval(aL,aH,sL,sH,qL,qH,clo,chi,vertical_sign,rep,factor):
    a2L,a2H=isqr(aL,aH)
    s2L,s2H=isqr(sL,sH)
    osL,osH=iadd(1.0,1.0,s2L,s2H)
    tL,tH=imul(a2L,a2H,osL,osH)
    DL,DH=iadd(1.0,1.0,tL,tH)
    sdL,sdH=isqrt(DL,DH)
    q2L,q2H=isqr(qL,qH)
    denL,denH=iadd(1.0,1.0,q2L,q2H)
    tqL,tqH=imul(2.0,2.0,qL,qH)
    hL,hH=idiv(tqL,tqH,denL,denH)
    omL,omH=isub(1.0,1.0,q2L,q2H)
    vL,vH=idiv(omL,omH,denL,denH)
    if vertical_sign < 0:
        vL,vH=-vH,-vL
    u0L,u0H=idiv(hL,hH,sdL,sdH)
    u1L,u1H=imul(aL,aH,u0L,u0H)
    asL,asH=imul(aL,aH,sL,sH)
    u2L,u2H=imul(asL,asH,u0L,u0H)
    if rep >= 0:
        u0L,u0H=imul(u0L,u0H,float(VERTEX_SIGNS[rep,0]),float(VERTEX_SIGNS[rep,0]))
        u1L,u1H=imul(u1L,u1H,float(VERTEX_SIGNS[rep,1]),float(VERTEX_SIGNS[rep,1]))
        u2L,u2H=imul(u2L,u2H,float(VERTEX_SIGNS[rep,2]),float(VERTEX_SIGNS[rep,2]))
    Llo = Lhi = N2lo = N2hi = 0.0
    for j, (ul,uh) in enumerate(((u0L,u0H),(u1L,u1H),(u2L,u2H),(vL,vH))):
        pl, ph = imul(clo[j], chi[j], ul, uh)
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
    d32L,d32H=imul(DL,DH,sdL,sdH)
    den2L,den2H=isqr(denL,denH)
    den3L,den3H=imul(den2L,den2H,denL,denH)
    aqL,aqH=imul(aL,aH,q2L,q2H)
    jdL,jdH=imul(d32L,d32H,den3L,den3H)
    jL,jH=idiv(aqL,aqH,jdL,jdH)
    fL,fH=imul(jL,jH,r4l,r4h)
    return imul(factor,factor,fL,fH)


@njit(parallel=True)
def eval_boxes(bounds, signs, clo, chi, reps, factor):
    n = len(bounds)
    ubs, proxies = np.empty(n), np.empty(n)
    for b in prange(n):
        midL,midH,halfL,halfH,lengthsH=(np.empty(3),np.empty(3),np.empty(3),np.empty(3),np.empty(3))
        # sqrt(3) and its reciprocal are themselves interval operations.
        sqrt3L,sqrt3H=isqrt(3.0,3.0)
        gl,gh=idiv(1.0,1.0,sqrt3L,sqrt3H)
        for r in range(3):
            bl,bh=bounds[b,r,0],bounds[b,r,1]
            sl,sh=iadd(bl,bl,bh,bh)
            midL[r],midH[r]=imul(0.5,0.5,sl,sh)
            ll,lh=isub(bh,bh,bl,bl)
            lengthsH[r]=lh
            halfL[r],halfH[r]=imul(0.5,0.5,ll,lh)
        qsum = 0.0
        for mask in range(8):
            xL,xH = np.empty(3),np.empty(3)
            for r in range(3):
                sl,sh=imul(halfL[r],halfH[r],gl,gh)
                if ((mask>>r)&1)==0:
                    sl,sh=-sh,-sl
                xL[r],xH[r]=iadd(midL[r],midH[r],sl,sh)
            fl,fh=integrand_interval(xL[0],xH[0],xL[1],xH[1],xL[2],xH[2],clo[b],chi[b],signs[b],reps[b],factor)
            qsum = up(qsum + fh)
        weight=up(up(halfH[0]*halfH[1])*halfH[2])
        quadrature = up(qsum * weight)

        M = np.empty(3)
        for r in range(3):
            FL,FH=integrand_jet(bounds[b],clo[b],chi[b],signs[b],r,reps[b],factor)
            M[r] = up(24.0 * max(abs(FL[4]), abs(FH[4])))
        terms=np.empty(3)
        for r in range(3):
            t=up(lengthsH[(r+1)%3]*lengthsH[(r+2)%3])
            for j in range(5):
                t=up(t*lengthsH[r])
            terms[r]=up(t*M[r])
        remainder=up(up(up(terms[0]+terms[1])+terms[2])/4320.0)
        ubs[b] = up(quadrature + remainder)
        proxies[b] = quadrature
    return ubs, proxies


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
        if p_box.a > 0:
            candidate_values.append(distance2(mp.iv.mpf([right, right])))
        elif p_box.b < 0:
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
    if not (h.a >= 0):
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
    passed = bool(maximum.b < threshold.a)
    slack = threshold - maximum
    return passed, slack, visited, terminal


def fiv(x):
 return np.nextafter(float(x.a),-np.inf),np.nextafter(float(x.b),np.inf)


def build_arcs(N):
 mp.iv.dps=70;one=mp.iv.mpf(1);s0=one/(2*mp.iv.sqrt(2));verts=[(s0,s0,s0),(s0,-s0,-s0),(-s0,s0,-s0),(-s0,-s0,s0)];data=((0,1,2,3),(0,2,1,3),(0,3,1,2));lo=np.empty((3,N+1,4));hi=np.empty_like(lo);mid=np.empty_like(lo)
 ca=mp.iv.mpf(1)/3;sa=2*mp.iv.sqrt(2)/3;rho=mp.iv.sqrt(3)/2
 for arc,(ix,iy,iz,iw) in enumerate(data):
  m=[(verts[iz][k]+verts[iw][k])/2 for k in range(3)];e1=[(verts[ix][k]-m[k])/rho for k in range(3)];e2=[((verts[iy][k]-m[k])/rho-ca*e1[k])/sa for k in range(3)]
  for j in range(N+1):
   t=(2*mp.iv.atan2(mp.iv.mpf(1),mp.iv.sqrt(2)))*j/N;c=mp.iv.cos(t);ss=mp.iv.sin(t)
   for k in range(3):lo[arc,j,k],hi[arc,j,k]=fiv(m[k]+rho*(c*e1[k]+ss*e2[k]))
   lo[arc,j,3]=hi[arc,j,3]=0.;mid[arc,j]=.5*(lo[arc,j]+hi[arc,j])
 return lo,hi,mid


@njit(inline='always')
def idot3(zl,zh,vl,vh):
 lo=0.0;hi=0.0
 for d in range(3):
  a,b=imul(zl[d],zh[d],vl[d],vh[d]);lo,hi=iadd(lo,hi,a,b)
 return lo,hi


@njit(inline='always')
def center_from_y(y0,y1,y2,midlo,midhi,rholo,rhohi,e1lo,e1hi,e2lo,e2hi,calo,cahi,salo,sahi):
 # Exact binary64 y; rigorous upper enclosure of max distance^2 to 3 arcs.
 yl=np.array([y0,y1,y2]); yh=yl.copy(); maxF=-1e300
 for ar in range(3):
  zl=np.empty(3);zh=np.empty(3)
  for d in range(3):zl[d],zh[d]=isub(yl[d],yh[d],midlo[ar,d],midhi[ar,d])
  Alo,Ahi=idot3(zl,zh,e1lo[ar],e1hi[ar]);Blo,Bhi=idot3(zl,zh,e2lo[ar],e2hi[ar])
  E0lo,E0hi=Alo,Ahi
  xlo,xhi=imul(Alo,Ahi,calo,cahi);wlo,whi=imul(Blo,Bhi,salo,sahi);E1lo,E1hi=iadd(xlo,xhi,wlo,whi)
  mlo=min(E0lo,E1lo)
  # Interior minimum may occur when A<=0, B<=0, B-2sqrt(2)A>=0.
  # If its validity is uncertain, including it remains a safe lower bound.
  tlo,thi=imul(Alo,Ahi,dn(2*math.sqrt(2.0)),up(2*math.sqrt(2.0)))
  Clo,Chi=isub(Blo,Bhi,tlo,thi)
  possible=not (Alo>0.0 or Blo>0.0 or Chi<0.0)
  if possible:
   a2l,a2h=isqr(Alo,Ahi);b2l,b2h=isqr(Blo,Bhi);r2l,r2h=iadd(a2l,a2h,b2l,b2h);rl,rh=isqrt(r2l,r2h);mlo=min(mlo,-rh)
  z2lo=z2hi=0.0
  for d in range(3):
   a,b=isqr(zl[d],zh[d]);z2lo,z2hi=iadd(z2lo,z2hi,a,b)
  r2lo,r2hi=isqr(rholo[ar],rhohi[ar]);baseL,baseH=iadd(z2lo,z2hi,r2lo,r2hi)
  pmL,pmH=imul(rholo[ar],rhohi[ar],mlo,mlo);twL,twH=imul(2.0,2.0,pmL,pmH);FL,FH=isub(baseL,baseH,twL,twH)
  if FH>maxF:maxF=FH
 if maxF>1.0:return -1.0,maxF
 gap=dn(1.0-maxF)
 if gap<0.0:gap=0.0
 h=dn(math.sqrt(gap))
 return h,maxF


def need(ok,msg):
    if not ok: raise RuntimeError(msg)


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


@njit
def dyadic_leaf_keys(boxes,reps):
    # A 3D leaf is encoded by a root ID and a binary path. Sentinel bit 1
    # makes paths of different depths distinct. All arithmetic below is exact.
    n=len(boxes);keys=np.empty(n,np.int64);depths=np.empty(n,np.int64)
    for i in range(n):
        b=boxes[i]
        ra=int(math.floor(b[0,0]*4));rs=int(math.floor((b[1,0]+1)*4));rq=int(math.floor(b[2,0]*8))
        if not(0<=ra<4 and 0<=rs<8 and 0<=rq<8):raise ValueError('Root index')
        lo=np.array([ra/4.0,-1+rs/4.0,rq/8.0]);hi=lo+np.array([.25,.25,.125])
        if np.any(b[:,0]<lo) or np.any(b[:,1]>hi):raise ValueError('Leaf crosses root boundary')
        path=1;dep=0
        while not (np.all(b[:,0]==lo) and np.all(b[:,1]==hi)):
            axis=int(np.argmax(hi-lo));m=(lo[axis]+hi[axis])*.5
            if b[axis,1]<=m:
                hi[axis]=m;path=2*path
            elif b[axis,0]>=m:
                lo[axis]=m;path=2*path+1
            else:raise ValueError('Noncanonical leaf spanning bisection')
            dep+=1
            if dep>40:raise ValueError('Leaf depth limit')
        root=((int(reps[i])*4+ra)*8+rs)*8+rq
        keys[i]=(root<<42)+path;depths[i]=dep
    return keys,depths


def coverage(boxes,reps):
    keys,depths=dyadic_leaf_keys(boxes,reps)
    need(len(np.unique(keys))==len(keys),'Duplicate leaves')
    # Collapse sibling pairs exactly. Every root must collapse to its sentinel.
    leaves=set(map(int,keys));maximum=int(depths.max())
    for d in range(maximum,0,-1):
        at=[k for k in leaves if (k&((1<<42)-1)).bit_length()==d+1]
        for k in at:
            if k not in leaves:continue
            sibling=k^1
            need(sibling in leaves,'Missing sibling/domain gap or overlapping ancestor')
            root=k>>42;parent=(root<<42)+((k&((1<<42)-1))//2)
            need(parent not in leaves,'Overlapping ancestor leaf')
            leaves.remove(k);leaves.remove(sibling);leaves.add(parent)
    need(leaves=={(r<<42)+1 for r in range(1024)},'Missing parameter-domain root')
    print(f'[exact coverage] all 4 representatives and 1024 roots covered once; max depth={maximum}: PASS',flush=True)


def meissner_constants():
    import mpmath as mp
    mp.iv.dps=80
    iv=mp.iv;one=iv.mpf(1);s=one/(2*iv.sqrt(2));verts=[(s,s,s),(s,-s,-s),(-s,s,-s),(-s,-s,s)]
    ca=one/3;sa=2*iv.sqrt(2)/3;rho=iv.sqrt(3)/2
    def enc(x):return float(np.nextafter(float(x.a),-np.inf)),float(np.nextafter(float(x.b),np.inf))
    midlo=np.empty((3,3));midhi=np.empty_like(midlo);e1lo=np.empty_like(midlo);e1hi=np.empty_like(midlo);e2lo=np.empty_like(midlo);e2hi=np.empty_like(midlo)
    for ar,(i,j,k,l) in enumerate(((0,1,2,3),(0,2,1,3),(0,3,1,2))):
        m=[(verts[k][d]+verts[l][d])/2 for d in range(3)]
        u=[(verts[i][d]-m[d])/rho for d in range(3)]
        v=[((verts[j][d]-m[d])/rho-ca*u[d])/sa for d in range(3)]
        for d in range(3):
            midlo[ar,d],midhi[ar,d]=enc(m[d]);e1lo[ar,d],e1hi[ar,d]=enc(u[d]);e2lo[ar,d],e2hi[ar,d]=enc(v[d])
    rL,rH=enc(rho);cL,cH=enc(ca);sL,sH=enc(sa)
    return midlo,midhi,np.full(3,rL),np.full(3,rH),e1lo,e1hi,e2lo,e2hi,cL,cH,sL,sH


def exact_sum(values):
    need(np.all(np.isfinite(values)) and np.all(values>=0),'Nonfinite/negative box upper')
    # Every binary64 is an exact multiple of 2^-1074. No rounded reduction.
    total=0
    for x in values:
        num,den=float(x).as_integer_ratio();total+=num<<(1074-(den.bit_length()-1))
    return Fraction(total,1<<1074)


def rounded_up(x):
    f=float(x)
    if Fraction.from_float(f)<x:f=float(np.nextafter(f,np.inf))
    return f


def fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


def raw_chart_rays(n, sign):
    grid=np.arange(n+1,dtype=np.int64)*2-n
    cap=np.empty(((n+1)**3,4),dtype=np.int64)
    cap[:,:3]=np.stack(np.meshgrid(grid,grid,grid,indexing="ij"),axis=-1).reshape(-1,3)
    cap[:,3]=sign*n
    nodes=[]; index={}
    for i in range(n+1):
        for j in range(n+1-i):
            index[i,j]=len(nodes);nodes.append((i+j,-i+j))
    triangles=[]
    for i in range(n):
        for j in range(n-i):
            triangles.append((index[i,j],index[i+1,j],index[i,j+1]))
            if i+j<=n-2:
                triangles.append((index[i+1,j],index[i+1,j+1],index[i,j+1]))
    ac=np.asarray(nodes,dtype=np.int64)
    equi=np.empty(((n+1)*len(ac),4),dtype=np.int64)
    equi[:,0]=n
    equi[:,1:3]=np.tile(ac,(n+1,1))
    equi[:,3]=sign*np.repeat(np.arange(n+1,dtype=np.int64),len(ac))
    return np.vstack((cap,equi)), np.asarray(triangles,dtype=np.int64),len(cap),len(ac)


def arc_grid(N):
    mp.iv.dps=70
    lo=np.empty((6,N+1,3));hi=np.empty_like(lo)
    s0=mp.iv.mpf(1)/(2*mp.iv.sqrt(2))
    for j in range(N+1):
        angle=mp.iv.pi*j/(2*N)
        co,si=mp.iv.cos(angle),mp.iv.sin(angle)
        for a,(ia,ib) in enumerate(PAIRS):
            for k in range(3):
                value=s0*(int(SIGNS[ia,k])*co+int(SIGNS[ib,k])*si)
                lo[a,j,k]=np.nextafter(float(value.a),-np.inf)
                hi[a,j,k]=np.nextafter(float(value.b),np.inf)
    # If |y|<=1, f(t)=|y-gamma(t)|^2 has |f''(t)|<3.
    # The maximum between grid nodes is at most max(endpoint f)+3*delta^2/8.
    err=3*(mp.iv.pi/(2*N))**2/8
    return lo,hi,np.nextafter(float(err.b),np.inf)


@njit(parallel=True)
def positive_membership(points,lo,hi,error):
    passed=np.ones(len(points),dtype=np.uint8)
    slack=np.empty(len(points))
    for i in prange(len(points)):
        p=points[i];norm2=0.0
        for k in range(3):
            norm2=up(norm2+up(p[k]*p[k]))
        best=0.0
        for a in range(len(lo)):
            for j in range(lo.shape[1]):
                value=0.0
                for k in range(3):
                    dl=dn(p[k]-hi[a,j,k]);dh=up(p[k]-lo[a,j,k])
                    value=up(value+up(max(dl*dl,dh*dh)))
                best=max(best,value)
        total=up(up(best+error)+up(p[3]*p[3]))
        slack[i]=dn(1.0-total)
        if p[3]<0 or norm2>1.0 or slack[i]<=0.0:
            passed[i]=0
    return passed,slack


def negative_membership(raw,scales,witnesses,D,report=True):
    DD=D*D
    start=time.perf_counter()
    for i,(v,t,m) in enumerate(zip(raw,scales,witnesses),1):
        v=tuple(map(int,v));t=int(t);m=tuple(map(int,m))
        need(v[3]<=0,"Negative U3 point has positive height")
        # In Cartesian coordinates T is the intersection of
        # sigma_i.m >= -sqrt(2)/4.  Negative dot products can be safely squared.
        for signs in SIGNS:
            dot=sum(int(signs[k])*m[k] for k in range(3))
            need(dot>=0 or 8*dot*dot<=DD,f"U3 witness outside T at {i-1}")
        A=sum((v[k]*t-m[k])**2 for k in range(3))+(v[3]*t)**2
        B=5*DD//8+sum(x*x for x in m)
        C=DD+B-A
        need(B<DD and C>=0 and C*C>4*B*DD,
             f"U3 negative ball membership failed at {i-1}")
        if report and (i%50000==0 or i==len(raw)):
            elapsed=time.perf_counter()-start
            print(f"[U3 exact negative membership] {i}/{len(raw)}; "
                  f"elapsed={fmt_time(elapsed)}; ETA={fmt_time(elapsed*(len(raw)-i)/i)}",
                  flush=True)


def integer_fan_sum(scales,n,triangles,cap_count,node_count):
    """No floating reduction: every object-array product and sum is a Python int."""
    cap=scales[:cap_count].astype(object).reshape(n+1,n+1,n+1)
    equi=scales[cap_count:].astype(object).reshape(n+1,node_count)
    cap_sum=0;eq_sum=0
    for perm in itertools.permutations(range(3)):
        off=np.zeros(3,dtype=int);product=np.ones((n,n,n),dtype=object)
        for step in range(4):
            product*=cap[off[0]:off[0]+n,off[1]:off[1]+n,off[2]:off[2]+n]
            if step<3:
                off[perm[step]]+=1
        cap_sum+=int(product.sum())
    ia,ib,ic=triangles.T
    for k in range(n):
        A,B,C=equi[k,ia],equi[k,ib],equi[k,ic]
        Ap,Bp,Cp=equi[k+1,ia],equi[k+1,ib],equi[k+1,ic]
        eq_sum+=int((A*B*C*Cp+A*B*Bp*Cp+A*Ap*Bp*Cp).sum())
    # Exact raw-ray determinants are 8*n (cap) and 2*n (chamber).
    # The chamber has 24 disjoint tetrahedral images.
    return 8*n*cap_sum+48*n*eq_sum


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def det3(a,b,c,d,e,f,g,h,i) -> int:
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)


def det4_rows(r0, r1, r2, r3) -> int:
    a,b,c,d = (int(x) for x in r0)
    e,f,g,h = (int(x) for x in r1)
    i,j,k,l = (int(x) for x in r2)
    m,n,o,p = (int(x) for x in r3)
    return (a*det3(f,g,h,j,k,l,n,o,p)
            - b*det3(e,g,h,i,k,l,m,o,p)
            + c*det3(e,f,h,i,j,l,m,n,p)
            - d*det3(e,f,g,i,j,k,m,n,o))


def sign_int(x: int) -> int:
    return (x > 0) - (x < 0)


def sub4(x, y):
    return (int(x[0])-int(y[0]), int(x[1])-int(y[1]),
            int(x[2])-int(y[2]), int(x[3])-int(y[3]))


def dot4(x, y) -> int:
    return sum(int(a)*int(b) for a,b in zip(x,y))


def affine_tet_rank3(points: Sequence[Sequence[int]], tet: Sequence[int]) -> bool:
    """Exact affine nondegeneracy of four points in R^4 (rank of 3 differences = 3)."""
    p0 = points[int(tet[0])]
    r = [sub4(points[int(tet[j])], p0) for j in (1,2,3)]
    # Some 3x3 minor of the 3x4 difference matrix must be nonzero.
    cols = ((0,1,2),(0,1,3),(0,2,3),(1,2,3))
    for c0,c1,c2 in cols:
        if det3(r[0][c0],r[0][c1],r[0][c2],
                r[1][c0],r[1][c1],r[1][c2],
                r[2][c0],r[2][c1],r[2][c2]) != 0:
            return True
    return False


def solve_exact_barycentric(nums5: np.ndarray):
    A = [[Fraction(int(nums5[col,row])) for col in range(5)] + [Fraction(0)] for row in range(4)]
    A.append([Fraction(1) for _ in range(5)] + [Fraction(1)])
    r = 0
    pivcols = []
    for c in range(5):
        piv = next((rr for rr in range(r,5) if A[rr][c] != 0), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][c]
        A[r] = [x/pv for x in A[r]]
        for rr in range(5):
            if rr == r:
                continue
            f = A[rr][c]
            if f:
                A[rr] = [A[rr][j]-f*A[r][j] for j in range(6)]
        pivcols.append(c)
        r += 1
        if r == 5:
            break
    if r < 5:
        return None
    lam = [Fraction(0) for _ in range(5)]
    for rr,c in enumerate(pivcols):
        lam[c] = A[rr][5]
    if sum(lam) != 1:
        return None
    for j in range(4):
        if sum(lam[i]*int(nums5[i,j]) for i in range(5)) != 0:
            return None
    return lam


def unique_facets_check(facets: np.ndarray):
    sf = np.sort(facets.astype(np.int32,copy=False),axis=1)
    order = np.lexsort((sf[:,3],sf[:,2],sf[:,1],sf[:,0]))
    ss = sf[order]
    dup = np.all(ss[1:]==ss[:-1],axis=1)
    if np.any(dup):
        raise AssertionError(f"duplicate tetrahedra: {int(np.count_nonzero(dup))}")
    print("[topology] tetrahedra are unique")


def build_triangle_pairs(facets: np.ndarray):
    F = len(facets)
    rec = np.empty((4*F,4),dtype=np.int32)
    choices = ((1,2,3),(0,2,3),(0,1,3),(0,1,2))
    for k,ch in enumerate(choices):
        block = np.sort(facets[:,ch],axis=1).astype(np.int32,copy=False)
        rec[k*F:(k+1)*F,:3] = block
        rec[k*F:(k+1)*F,3] = np.arange(F,dtype=np.int32)
    order = np.lexsort((rec[:,2],rec[:,1],rec[:,0]))
    s = rec[order]
    same_prev = np.all(s[1:,:3]==s[:-1,:3],axis=1)
    starts = np.r_[0, np.flatnonzero(~same_prev)+1]
    ends = np.r_[starts[1:],len(s)]
    counts = ends-starts
    bad = np.flatnonzero(counts!=2)
    if len(bad):
        i=int(bad[0]); tri=tuple(map(int,s[starts[i],:3]))
        raise AssertionError(f"triangle incidence !=2 for {tri}: count={int(counts[i])}; bad groups={len(bad)}")
    T=len(starts)
    tris = s[starts,:3].copy()
    pairs = np.column_stack((s[starts,3],s[starts+1,3])).astype(np.int32,copy=False)
    print(f"[topology] unique triangles={T:,}; every triangle occurs exactly twice")
    del rec, order, s, same_prev, starts, ends, counts
    return tris, pairs


def dual_connected(F: int, pairs: np.ndarray):
    parent=np.arange(F,dtype=np.int32)
    rank=np.zeros(F,dtype=np.int8)
    def find(x):
        x=int(x)
        while int(parent[x])!=x:
            parent[x]=parent[int(parent[x])]
            x=int(parent[x])
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra==rb:return
        if rank[ra]<rank[rb]: ra,rb=rb,ra
        parent[rb]=ra
        if rank[ra]==rank[rb]:rank[ra]+=1
    t0=time.perf_counter()
    for i,(a,b) in enumerate(pairs,1):
        union(int(a),int(b))
        if i%500000==0:
            print(f"[topology] dual unions {i:,}/{len(pairs):,}; elapsed={fmt_time(time.perf_counter()-t0)}")
    root=find(0)
    roots=sum(find(i)==i for i in range(F))
    if roots!=1:
        raise AssertionError(f"facet-dual graph has {roots} connected components")
    print("[topology] facet-dual graph connected")


def vertex_links_check(N: int, facets: np.ndarray, tris: np.ndarray):
    """Check each vertex link is a connected closed triangulated surface with chi=2."""
    facet_count=np.bincount(facets.ravel(),minlength=N)
    T=len(tris)
    rec=np.empty((3*T,3),dtype=np.int32)
    # center vertex, the link edge endpoints
    rec[0*T:1*T]=np.column_stack((tris[:,0],tris[:,1],tris[:,2]))
    rec[1*T:2*T]=np.column_stack((tris[:,1],tris[:,0],tris[:,2]))
    rec[2*T:3*T]=np.column_stack((tris[:,2],tris[:,0],tris[:,1]))
    order=np.argsort(rec[:,0],kind="stable")
    s=rec[order]
    centers=s[:,0]
    starts=np.r_[0,np.flatnonzero(centers[1:]!=centers[:-1])+1]
    ends=np.r_[starts[1:],len(s)]
    seen_centers=set(map(int,s[starts,0]))
    if len(seen_centers)!=N:
        raise AssertionError(f"only {len(seen_centers)}/{N} vertices appear in triangle links")
    t0=time.perf_counter()
    for gi,(st,en) in enumerate(zip(starts,ends),1):
        v=int(s[st,0]); edges=s[st:en,1:3]
        adj={}
        for aa,bb in edges:
            a,b=int(aa),int(bb)
            adj.setdefault(a,[]).append(b)
            adj.setdefault(b,[]).append(a)
        # 1-skeleton connectivity of the link.
        start=next(iter(adj))
        q=[start]; vis={start}
        for x in q:
            for y in adj[x]:
                if y not in vis:
                    vis.add(y); q.append(y)
        if len(vis)!=len(adj):
            raise AssertionError(f"vertex {v} link disconnected: {len(vis)}/{len(adj)} link vertices reached")
        chi=len(adj)-len(edges)+int(facet_count[v])
        if chi!=2:
            raise AssertionError(f"vertex {v} link Euler characteristic {chi}, expected 2")
        if gi%25000==0 or gi==len(starts):
            dt=time.perf_counter()-t0
            rate=gi/dt if dt else 0
            eta=(len(starts)-gi)/rate if rate else 0
            print(f"[topology] vertex links {gi:,}/{len(starts):,}; elapsed={fmt_time(dt)}; ETA={fmt_time(eta)}")
    print("[topology] every vertex link is connected with Euler characteristic 2")
    del rec,order,s,centers,starts,ends


def edge_records(facets: np.ndarray):
    F=len(facets)
    rec=np.empty((6*F,4),dtype=np.int32)
    specs=((0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1))
    for k,(i,j,r,s) in enumerate(specs):
        a=facets[:,i]; b=facets[:,j]
        lo=np.minimum(a,b); hi=np.maximum(a,b)
        block=rec[k*F:(k+1)*F]
        block[:,0]=lo; block[:,1]=hi
        block[:,2]=facets[:,r]; block[:,3]=facets[:,s]
    order=np.lexsort((rec[:,1],rec[:,0]))
    s=rec[order]
    keychange=(s[1:,0]!=s[:-1,0]) | (s[1:,1]!=s[:-1,1])
    starts=np.r_[0,np.flatnonzero(keychange)+1]
    ends=np.r_[starts[1:],len(s)]
    return s,starts,ends


def link_cycle_from_pairs(pairs: np.ndarray):
    adj={}
    for cc,dd in pairs:
        c,d=int(cc),int(dd)
        if c==d:
            return None,"link edge has repeated vertex"
        adj.setdefault(c,[]).append(d)
        adj.setdefault(d,[]).append(c)
    if len(adj)<3:
        return None,f"link has only {len(adj)} vertices"
    bad=[(v,len(nb)) for v,nb in adj.items() if len(nb)!=2]
    if bad:
        return None,f"link degrees not 2, first={bad[0]}"
    start=min(adj)
    cycle=[start]
    prev=None; cur=start
    for _ in range(len(adj)+1):
        n0,n1=adj[cur]
        nxt=n0 if n0!=prev else n1
        if nxt==start:
            if len(cycle)==len(adj):
                return cycle,None
            return None,"link closes before visiting all vertices"
        if nxt in cycle:
            return None,"link repeats a vertex before closure"
        cycle.append(nxt); prev,cur=cur,nxt
    return None,"link failed to close"


def edge_local_convexity_check(nums: np.ndarray, facets: np.ndarray, checkpoint: Optional[Path], state_key: str, checkpoint_every:int=50000):
    P=[tuple(int(x) for x in row) for row in nums]
    s,starts,ends=edge_records(facets)
    E=len(starts)
    print(f"[topology/local] unique edges={E:,}; edge-tetrahedron incidences={len(s):,}")

    resume=0; planar=0; generic=0
    if checkpoint is not None and checkpoint.exists():
        try:
            state=json.loads(checkpoint.read_text(encoding="utf-8"))
            if (state.get("format_version")==2 and state.get("state_key")==state_key
                    and state.get("facet_count")==len(facets)
                    and state.get("edge_count")==E):
                resume=int(state.get("processed_edges",0))
                planar=int(state.get("planar",0)); generic=int(state.get("generic",0))
                if resume:
                    print(f"[resume] local-convexity audit resumes at edge {resume:,}/{E:,}")
            elif state:
                print("[resume] existing checkpoint belongs to a different certificate; ignoring it")
        except Exception:
            print("[resume] unreadable checkpoint; starting local-convexity stage from zero")
            resume=0; planar=0; generic=0

    t0=time.perf_counter()
    # A resume only skips already successful deterministic checks; no partial sums are involved.
    for gi in range(resume,E):
        st,en=int(starts[gi]),int(ends[gi])
        a,b=int(s[st,0]),int(s[st,1])
        cycle,err=link_cycle_from_pairs(s[st:en,2:4])
        if err:
            raise AssertionError(f"edge ({a},{b}) link is not one cycle: {err}")
        m=len(cycle)
        d=sub4(P[b],P[a])
        vs=[sub4(P[c],P[a]) for c in cycle]
        # Find a nonzero consecutive triple. If none exists, all transverse rays are coplanar.
        k0=None; sig0=0
        for k in range(m):
            sg=sign_int(det4_rows(d,vs[k],vs[(k+1)%m],vs[(k+2)%m]))
            if sg:
                k0=k; sig0=sg; break
        if k0 is None:
            planar += 1
            raise RuntimeError(
                f"edge ({a},{b}) has a fully planar transverse link. "
                "The current strict proof kernel deliberately stops here rather than "
                "silently using an unproved degenerate-case shortcut; report this edge "
                "so that the planar local-convexity case can be certified separately."
            )
        else:
            generic+=1
            if k0:
                cycle=cycle[k0:]+cycle[:k0]
                vs=vs[k0:]+vs[:k0]
            sgn=sign_int(det4_rows(d,vs[0],vs[1],vs[2]))
            if not sgn:
                raise AssertionError("internal rotation error: initial edge-star sign vanished")
            # Rybnikov Lemma 10 / Algorithm 3: never turn to the opposite orientation,
            # and never cross the reference fan from ray 0.
            for i in range(m):
                t1=sign_int(det4_rows(d,vs[i],vs[(i+1)%m],vs[(i+2)%m]))
                if t1 == -sgn:
                    raise AssertionError(f"edge ({a},{b}) fails local turn condition at link index {i}")
                t2=sign_int(det4_rows(d,vs[0],vs[(i+1)%m],vs[(i+2)%m]))
                if t2 == -sgn:
                    raise AssertionError(f"edge ({a},{b}) fails local non-self-intersection condition at link index {i}")

        done=gi+1
        if done%checkpoint_every==0 or done==E:
            dt=time.perf_counter()-t0
            rate=max(1e-12,(done-resume)/dt)
            eta=(E-done)/rate
            print(f"[local] edges {done:,}/{E:,}; generic={generic:,}; planar={planar:,}; elapsed={fmt_time(dt)}; ETA={fmt_time(eta)}")
            if checkpoint is not None:
                tmp=checkpoint.with_suffix(checkpoint.suffix+".tmp")
                tmp.write_text(json.dumps({"format_version":2,"state_key":state_key,"facet_count":len(facets),"edge_count":E,"processed_edges":done,"generic":generic,"planar":planar},indent=2,sort_keys=True),encoding="utf-8")
                os.replace(tmp,checkpoint)
    print(f"[local] every edge star passed exact local-convexity test; planar edge stars={planar:,}")
    return E,planar


def exact_det_sum(nums: np.ndarray, facets: np.ndarray, report_every:int=100000):
    P=[tuple(int(x) for x in row) for row in nums]
    S=0; zero=0; t0=time.perf_counter()
    for i,tet in enumerate(facets,1):
        d=det4_rows(P[int(tet[0])],P[int(tet[1])],P[int(tet[2])],P[int(tet[3])])
        if d==0: zero+=1
        S+=abs(d)
        if i%report_every==0 or i==len(facets):
            dt=time.perf_counter()-t0; rate=i/dt if dt else 0; eta=(len(facets)-i)/rate if rate else 0
            print(f"[volume] {i:,}/{len(facets):,}; zero-cones={zero}; elapsed={fmt_time(dt)}; ETA={fmt_time(eta)}")
    return S,zero


def above_rational(S:int,D:int,p:int,q:int)->bool:
    # S*sqrt(2)/(192 D^4) > p/q
    return 2*(q*S)**2 > (192*p*(D**4))**2


def volume_float(S:int,D:int)->float:
    return float(S)*math.sqrt(2.0)/(192.0*float(D)**4)


def qform(a):
    x,y,z=map(int,a)
    return 3*(x*x+y*y+z*z)-2*(x*y+x*z+y*z)


def bil(a,b):
    a0,a1,a2=map(int,a); b0,b1,b2=map(int,b)
    return 3*(a0*b0+a1*b1+a2*b2) - (a0*b1+a1*b0+a0*b2+a2*b0+a1*b2+a2*b1)


def vecsub(a,b): return tuple(int(x)-int(y) for x,y in zip(a,b))


def vecscale(c,a): return tuple(int(c)*int(x) for x in a)


def certify_negative(n,D,mw,W):
    # Lift the point and witness to the common denominator L=lcm(D,W).
    g=math.gcd(D,W); L=D//g*W
    xs=L//D; ms=L//W
    xn=tuple(xs*int(n[i]) for i in range(4))
    mn=tuple(ms*int(v) for v in mw)
    diff=tuple(xn[i]-mn[i] for i in range(3))
    A=qform(diff)+8*xn[3]**2            # a=A/(8L^2)
    B=5*L*L+qform(mn)                   # b=B/(8L^2)
    C=8*L*L+B-A                         # 1+b-a=C/(8L^2)
    ok=(B < 8*L*L and C>=0 and C*C>=32*B*L*L)
    slack1=C
    slack2=C*C-32*B*L*L
    return ok,slack1,slack2


def arc_scaled_data(D,arc):
    X,Y,Z,Wn=[VERT[s] for s in arc]
    XD=vecscale(D,X); YD=vecscale(D,Y); ZD=vecscale(D,Z); WD=vecscale(D,Wn)
    mid=tuple((ZD[i]+WD[i])//2 for i in range(3))
    if any(2*mid[i]!=ZD[i]+WD[i] for i in range(3)):
        raise ValueError('D must make arc midpoints integral')
    U=vecsub(XD,mid)
    Yminus=vecsub(YD,mid)
    Vvec=vecsub(vecscale(3,Yminus),U)  # V uses Vvec/(2D)
    return XD,YD,mid,U,Vvec


def certify_positive(n,D,arcs):
    nh=tuple(map(int,n[:3])); t=int(n[3])
    if t<0: return False,None
    worst=None
    for arc in arcs:
        XD,YD,mid,Uvec,Vvec=arc_scaled_data(D,arc)
        # Endpoint checks: ||y-X||^2+t^2 <= 1
        for ED in (XD,YD):
            de=vecsub(nh,ED)
            slack=8*D*D-qform(de)-8*t*t
            if worst is None or slack<worst: worst=slack
            if slack<0: return False,('endpoint',arc,slack)
        z=vecsub(nh,mid)
        uN=bil(z,Uvec)                  # U=uN/(8D^2)
        vN=bil(z,Vvec)                  # V=vN/(16D^2)
        # Interior minimum lies on [0,alpha] iff V<=0 and V-4U>=0.
        if vN<=0 and vN-8*uN>=0:
            HN=2*D*D-8*t*t-qform(z)    # H=HN/(8D^2)
            s1=HN
            s2=2*HN*HN-(8*uN*uN+vN*vN)
            local=min(s1,s2)
            if worst is None or local<worst: worst=local
            if s1<0 or s2<0: return False,('interior',arc,s1,s2,uN,vN)
    return True,worst


def load_rm(path):
    need(sha256_file(path) == RM_EXPECTED_SHA256, "RM certificate SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as z:
        need(set(z.files) == {"format_version", "points_num", "denominator",
            "sides", "witness_num", "witness_denominator", "facets",
            "origin_indices", "exact_det_sum_S"}, "Unexpected RM schema")
        data = {k: z[k].copy() for k in z.files}
    need(data["format_version"].shape == () and int(data["format_version"]) == 3,
         "Unsupported RM format")
    nums, facets = data["points_num"], data["facets"]
    sides, witnesses = data["sides"], data["witness_num"]
    need(nums.shape == (105654,4) and nums.dtype == np.int64, "RM point schema")
    need(facets.shape == (708899,4) and facets.dtype == np.int32, "RM facet schema")
    need(sides.shape == (len(nums),) and sides.dtype == np.int8, "RM side schema")
    need(witnesses.shape == (len(nums),3) and witnesses.dtype == np.int64,
         "RM witness schema")
    for k in ("denominator", "witness_denominator"):
        need(data[k].shape == () and data[k].dtype == np.int64 and int(data[k]) > 0,
             "RM denominator must be a positive scalar int64")
    need(int(data["denominator"]) == 1048576 and int(data["witness_denominator"]) == 64,
         "Unexpected RM denominators")
    need(np.array_equal(np.unique(sides), [-1,1]), "RM sides must be -1/+1")
    need(np.all(witnesses[sides>0] == 0), "Positive RM witness sentinel must be zero")
    need(np.all(nums[sides>0,3] >= 0) and np.all(nums[sides<0,3] <= 0),
         "RM point in wrong half")
    need(facets.min() >= 0 and facets.max() < len(nums), "RM facets are zero-based")
    need(np.all(np.diff(np.sort(facets,axis=1),axis=1) != 0), "Repeated facet vertex")
    need(np.array_equal(np.unique(facets), np.arange(len(nums))), "Unused RM vertex")
    oi=data["origin_indices"]
    need(oi.shape == (5,) and oi.dtype == np.int64 and len(np.unique(oi)) == 5
         and oi.min() >= 0 and oi.max() < len(nums), "Invalid origin witness indices")
    need(data["exact_det_sum_S"].shape == () and data["exact_det_sum_S"].dtype.kind == "U",
         "RM determinant sum must be a decimal string scalar")
    print("[RM preflight] hash, shapes, integer types, zero-based indices and sentinels PASS",
          flush=True)
    return data


def verify_rm(path, target):
    data=load_rm(path)
    nums, facets = data["points_num"], data["facets"]
    D, W = int(data["denominator"]), int(data["witness_denominator"])
    start=time.perf_counter()
    for i,(n,s,mw) in enumerate(zip(nums,data["sides"],data["witness_num"]),1):
        if s>0:
            ok,_=certify_positive(n,D,FACE_ARCS)
        else:
            mw=tuple(map(int,mw))
            # m=(lambda_A-lambda_D,lambda_B-lambda_D,lambda_C-lambda_D)/W.
            rem=W-sum(mw)
            need(rem % 4 == 0, "RM witness barycentric denominator")
            ld=rem//4
            need(ld >= 0 and all(x+ld >= 0 for x in mw), "RM witness outside T")
            ok,_,_=certify_negative(n,D,mw,W)
        need(ok, f"RM membership failed at point {i-1}")
        if i%25000 == 0 or i==len(nums):
            elapsed=time.perf_counter()-start
            print(f"[RM membership] {i}/{len(nums)}; elapsed={fmt_time(elapsed)}; "
                  f"ETA={fmt_time(elapsed*(len(nums)-i)/i)}",flush=True)
    lam=solve_exact_barycentric(nums[data["origin_indices"]])
    need(lam is not None and all(x>0 for x in lam), "Origin is not strictly interior")
    unique_facets_check(facets)
    P=[tuple(map(int,row)) for row in nums]
    need(all(affine_tet_rank3(P,tet) for tet in facets), "Degenerate boundary tetrahedron")
    del P
    tris,pairs=build_triangle_pairs(facets)
    dual_connected(len(facets),pairs)
    vertex_links_check(len(nums),facets,tris)
    # Always replay every edge: no cached successful predicate is accepted.
    E,planar=edge_local_convexity_check(nums,facets,None,"fresh")
    S,zero=exact_det_sum(nums,facets)
    need(zero == 0 and S == int(data["exact_det_sum_S"].item()), "RM determinant mismatch")
    above=above_rational(S,D,target.numerator,target.denominator)
    result={"body":"RaisedMeissnerFace","arithmetic":"exact",
            "volume_expression":f"{S}*sqrt(2)/(192*{D}^4)",
            "exact_det_sum":str(S),"above_threshold":above,
            "threshold":str(target),"points":len(nums),"facets":len(facets),
            "edges":E,"planar_edge_stars":planar,
            "elapsed_seconds":time.perf_counter()-start}
    need(above,"RM lower certificate is valid but below the requested threshold")
    print("PASS RaisedMeissnerFace: exact lower > "+str(target),flush=True)
    return result


"""Driver appended to the mechanically assembled single-file checker."""

@njit
def decode_keys(keys):
    boxes=np.empty((len(keys),3,2));reps=np.empty(len(keys),np.int8)
    for i in range(len(keys)):
        root=int(keys[i]>>42);path=int(keys[i]&((1<<42)-1));needpath=path>0
        if not needpath:raise ValueError('Invalid path sentinel')
        reps[i]=root//256;root%=256;ra=root//64;rs=(root%64)//8;rq=root%8
        lo=np.array([ra*.25,-1+rs*.25,rq*.125]);hi=lo+np.array([.25,.25,.125])
        depth=0;t=path
        while t>1:depth+=1;t//=2
        if depth>40:raise ValueError('Path depth limit')
        for d in range(depth-1,-1,-1):
            axis=int(np.argmax(hi-lo));middle=(lo[axis]+hi[axis])*.5
            if (path>>d)&1:lo[axis]=middle
            else:hi[axis]=middle
        boxes[i,:,0]=lo;boxes[i,:,1]=hi
    return boxes,reps

def restore_keys(x,max_root):
    need(x.ndim==1 and x.dtype==np.int64 and 1<=len(x)<=2500000 and np.all(x>0) and np.all(x<2**53),'Key delta schema')
    k=np.cumsum(x,dtype=np.int64)
    need(np.all(k>0) and np.all(k<(max_root<<42)) and np.all(k[1:]>k[:-1]),'Key overflow/range/order')
    return k

def load_comparison_data(path):
    expected={'format_version','witness_bits','rm_pos_key_deltas','rm_neg_key_deltas','rm_pos_tags','rm_neg_witness_deltas','rm_pos_sum_num','rm_pos_sum_den','rm_neg_sum_num','rm_neg_sum_den','u3_seeds','u3_upper_key_deltas','u3_upper_tags','u3_lower_mesh_n','u3_lower_bits','u3_lower_arc_grid','u3_lower_positive_scales_deltas','u3_lower_negative_scales_deltas','u3_lower_negative_witness_num_deltas'}
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==expected,'Unexpected data schema keys');d={k:z[k].copy() for k in z.files}
    for k,value in [('format_version',1),('witness_bits',20),('u3_lower_mesh_n',80),('u3_lower_bits',42),('u3_lower_arc_grid',2048)]:
        need(d[k].shape==() and d[k].dtype==np.int64 and int(d[k])==value,k+' scalar schema')
    d['rm_pos_keys']=restore_keys(d['rm_pos_key_deltas'],1024);d['rm_neg_keys']=restore_keys(d['rm_neg_key_deltas'],1024);d['u3_upper_keys']=restore_keys(d['u3_upper_key_deltas'],512)
    need(len(d['rm_pos_keys'])==716800 and len(d['rm_neg_keys'])==1650688 and len(d['u3_upper_keys'])==86016,'Unexpected leaf counts')
    t=d['rm_pos_tags'];need(t.shape==(716800,2) and t.dtype==np.int16 and np.all((t[:,0]>=0)&(t[:,0]<3)) and np.all((t[:,1]>=0)&(t[:,1]<=16384)),'Meissner arc tags')
    y=d['rm_neg_witness_deltas'];need(y.shape==(1650688,3) and y.dtype==np.int32,'Meissner witness delta schema')
    y=np.cumsum(y,dtype=np.int64,axis=0);need(np.max(np.abs(y))<2**30,'Meissner witness range');d['rm_neg_witness_num']=y
    seeds=d['u3_seeds'];need(seeds.shape==(435,4) and seeds.dtype==np.float64 and np.isfinite(seeds).all() and np.all(seeds[:,3]>=0) and np.max(abs(seeds))<1,'U3 seed schema')
    d['u3_centres']=build_exact_orbit_centres(seeds);need(d['u3_centres'].shape==(10141,4),'U3 orbit count')
    t=d['u3_upper_tags'];need(t.shape==(86016,2) and t.dtype==np.int16,'U3 upper tag schema')
    roots=d['u3_upper_keys']>>42;pos=roots<256
    need(np.all((t[pos,0]>=0)&(t[pos,0]<6)) and np.all((t[pos,1]>=0)&(t[pos,1]<=16384)),'U3 positive tag ranges')
    need(np.all(t[~pos,0]==-1) and np.all((t[~pos,1]>=0)&(t[~pos,1]<10141)),'U3 negative tag ranges/sentinel')
    for k in ('positive_scales','negative_scales','negative_witness_num'):
        x=d['u3_lower_'+k+'_deltas'];shape=(800442,3) if k=='negative_witness_num' else (800442,)
        need(x.shape==shape and x.dtype==np.int64 and np.max(abs(x))<2**42,'U3 lower delta schema')
        x=np.cumsum(x,dtype=np.int64,axis=0)
        if k=='negative_witness_num':need(np.max(abs(x))<2**52,'U3 witness numerator range')
        else:need(np.all(x>=0) and int(x.max())*80<2**52,'U3 endpoint range')
        d['u3_lower_'+k]=x
    print('[preflight] all schemas, dyadic formats, 0-based arc/centre indices, sentinels and ranges PASS',flush=True)
    return d

@njit(parallel=True)
def checked_meissner_centres(ys,midlo,midhi,rholo,rhohi,e1lo,e1hi,e2lo,e2hi,calo,cahi,salo,sahi):
    cl=np.empty((len(ys),4));bad=np.zeros(len(ys),np.uint8)
    for i in prange(len(ys)):
        h,F=center_from_y(ys[i,0],ys[i,1],ys[i,2],midlo,midhi,rholo,rhohi,e1lo,e1hi,e2lo,e2hi,calo,cahi,salo,sahi)
        if h<0:bad[i]=1
        cl[i,:3]=ys[i];cl[i,3]=h
    return cl,bad

def replay_meissner(d,kind,batch):
    boxes,reps=decode_keys(d['rm_'+kind+'_keys']);coverage(boxes,reps)
    if kind=='pos':lo,hi,_=build_arcs(16384)
    else:const=meissner_constants()
    values=np.empty(len(boxes));start=time.perf_counter()
    for a in range(0,len(boxes),batch):
        b=min(a+batch,len(boxes));bb=boxes[a:b];rr=reps[a:b]
        if kind=='pos':
            tags=d['rm_pos_tags'][a:b];cl=lo[tags[:,0],tags[:,1]];ch=hi[tags[:,0],tags[:,1]];sg=np.ones(b-a,np.int8)
        else:
            ys=d['rm_neg_witness_num'][a:b].astype(float)/2**20;cl,bad=checked_meissner_centres(ys,*const)
            need(not np.any(bad),'Negative centre continuous-arc membership failed');ch=cl;sg=-np.ones(b-a,np.int8)
        values[a:b],_=eval_boxes(bb,sg,cl,ch,rr,12.)
        elapsed=time.perf_counter()-start;print(f'[RM {kind}] {b}/{len(boxes)}; elapsed={elapsed:.1f}s; ETA={elapsed*(len(boxes)-b)/b:.1f}s',flush=True)
    return exact_sum(values)

def replay_u3_upper(d,batch):
    mp.mp.dps=60;mp.iv.dps=60
    for i,p in enumerate(d['u3_seeds']):
        passed,slack,_,_=certify_seed(p,48);need(passed,f'U3 seed {i} membership failed')
        if (i+1)%50==0:print(f'[U3 centres] {i+1}/435 PASS',flush=True)
    lo,hi,_=build_arc_grid(16384);centres=d['u3_centres']
    boxes,roots=decode_keys(d['u3_upper_keys']);signs=np.where(roots==0,1,-1).astype(np.int8)
    # Coverage of the two original charts is equivalent to coverage of two
    # copies of each chart labeled as four independent representative domains.
    b0,b1=boxes[roots==0],boxes[roots==1]
    coverage(np.concatenate([b0,b1,b0,b1]),np.concatenate([np.full(len(b0),0,np.int8),np.full(len(b1),1,np.int8),np.full(len(b0),2,np.int8),np.full(len(b1),3,np.int8)]))
    values=np.empty(len(boxes))
    for a in range(0,len(boxes),batch):
        b=min(a+batch,len(boxes));t=d['u3_upper_tags'][a:b];pos=t[:,0]>=0;cl=np.empty((b-a,4));ch=np.empty_like(cl)
        cl[pos]=lo[t[pos,0],t[pos,1]];ch[pos]=hi[t[pos,0],t[pos,1]];cl[~pos]=ch[~pos]=centres[t[~pos,1]]
        values[a:b],_=eval_boxes(boxes[a:b],signs[a:b],cl,ch,-np.ones(b-a,np.int8),48.)
        print(f'[U3 upper replay] {b}/{len(boxes)}',flush=True)
    return exact_sum(values)

def replay_u3_lower(d):
    n=80;D=2**42;raw,tri,cap,nodes=raw_chart_rays(n,1)
    nums=raw*d['u3_lower_positive_scales'][:,None];lo,hi,err=arc_grid(2048)
    ok,slack=positive_membership(nums.astype(float)/D,lo,hi,err);need(np.all(ok),'U3 positive endpoint membership failed')
    print(f'[U3 lower positive] all {len(raw)} endpoints PASS; min slack={slack.min():.9g}',flush=True)
    Splus=integer_fan_sum(d['u3_lower_positive_scales'],n,tri,cap,nodes)
    raw,_,_,_=raw_chart_rays(n,-1)
    negative_membership(raw,d['u3_lower_negative_scales'],d['u3_lower_negative_witness_num'],D)
    Sminus=integer_fan_sum(d['u3_lower_negative_scales'],n,tri,cap,nodes)
    return Fraction(Splus+Sminus,24*D**4)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data',type=Path,default=Path(__file__).with_name('raised_comparison_compact.npz'));parser.add_argument('--rm-lower',type=Path,default=Path(__file__).with_name('rm_lower.npz'));parser.add_argument('--workers',type=int,default=8);parser.add_argument('--batch',type=int,default=65536);parser.add_argument('--out',type=Path,default=Path(__file__).with_name('verification_result.json'));args=parser.parse_args()
    set_num_threads(args.workers);start=time.perf_counter()
    need(sys.float_info.radix==2 and sys.float_info.mant_dig==53 and sys.float_info.rounds==1,'Binary64 arithmetic contract')
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        d=load_comparison_data(args.data)
        rm_lower_result=verify_rm(args.rm_lower,Fraction('0.221869166132387'))
        rmpos=replay_meissner(d,'pos',args.batch);rmneg=replay_meissner(d,'neg',args.batch);rmupper=rmpos+rmneg
        u3upper=replay_u3_upper(d,args.batch);u3lower=replay_u3_lower(d)
        # Previously certified U4/Body5 endpoints, rounded conservatively.
        # This run independently checks all the raised-body endpoints,
        # but does not pretend to recheck U4 or the separate Body5 enclosure.
        u4upper=Fraction('0.221838300');rmlower=Fraction('0.221869166132387');body5lower=Fraction('0.22287764647773');body5upper=Fraction('0.22299117642699')
        mp.iv.dps=80;iv=mp.iv;m4=iv.pi/32*((24-11*iv.sqrt(2))*iv.pi-24)
        need(m4.a>iv.mpf('0.248'),'M4 analytic lower bound')
        comparisons={'U4_lt_RM':u4upper<rmlower,'RM_lt_U3':rmupper<u3lower,'U3_lt_Body5':u3upper<body5lower,'Body5_lt_M4':body5upper<Fraction('.248')}
        need(all(comparisons.values()),'Full comparison did not separate')
        result={'status':'PASS','scope':'Independent raised-body lower/upper replay, combined with previously certified U4 upper and Body5 enclosure; M4 exact formula','comparisons':comparisons,'rm_lower_certificate':rm_lower_result,'rm_upper_outward':rounded_up(rmupper),'u3_upper_outward':rounded_up(u3upper),'u3_lower_display':float(u3lower),'rm_u3_gap_display':float(u3lower-rmupper),'u3_body5_gap_display':float(body5lower-u3upper),'exact_raised_bounds':{name:{'numerator':str(v.numerator),'denominator':str(v.denominator)} for name,v in [('rm_positive_upper',rmpos),('rm_negative_upper',rmneg),('rm_upper',rmupper),('u3_upper',u3upper),('u3_lower',u3lower)]},'checker_sha256':digest(Path(__file__)),'data_sha256':digest(args.data),'elapsed_seconds':time.perf_counter()-start}
        args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()

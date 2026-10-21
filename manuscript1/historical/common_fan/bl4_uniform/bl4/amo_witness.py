"""Standard-library AMO endpoint-witness verifier.

This module implements the *proof* side of the direct AMO peabody fixed-fan
adapter.  It does not search numerically for a boundary point.  A generator may
use floating point / SciPy to propose a small parameter box, but a stored
witness is accepted here only after all interval and inward-shrink inequalities
are rechecked with ``fractions.Fraction`` and integer square-root enclosures.

Coordinates
-----------
We use the coefficient chart of the first four vertices v1,...,v4 of the
width-one regular 4-simplex, centered at its barycenter.  The fifth vertex is
v5=-(v1+...+v4).  In this chart

    |a|^2 = (5*sum(a_i^2) - (sum a_i)^2)/10,

and the Euclidean coordinate Jacobian is sqrt(5)/20.

Boundary witness
----------------
A witness identifies either
  * one of the two representative wedge patches (``triple`` or ``pair``), or
  * the representative spherical 3-cap (``cap``),
together with a rational parameter box and a simplex permutation.  Exact
rational interval arithmetic encloses the whole image of that box by a
coordinate box C.

Let c be the rational midpoint of C and let rho be a rational upper bound for
the Euclidean radius of C about c.  Since every parameter box is nonempty and
lies in a source-proved boundary patch, there exists x in K with |x-c|<=rho.

For width one, the AMO body has the full regular-4-simplex symmetry.  Its
Chebyshev center is therefore the simplex barycenter.  Jung's theorem gives
circumradius <=sqrt(2/5), and constant width gives an inball of radius
1-sqrt(2/5)>1/3.  We deliberately use the exact rational r0=1/3.

If z=lambda*c and
    |p-z| + lambda*rho <= (1-lambda)/3,
then p lies in K by convexity.  The verifier squares only after proving the
right-hand slack is nonnegative.

Trust boundary
--------------
The algebraic boundary parametrizations are the exact formulas already used by
the manuscript's AMO boundary-orbit reduction.  This module certifies their
evaluation and the shrink step.  It does not turn floating Qhull data into a
proof and it does not assume a global membership oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from math import isqrt
from typing import Iterable

SQRT_BITS = 140
R0 = Fraction(1, 3)


def F(x=0, y=1):
    return x if isinstance(x, Fraction) and y == 1 else Fraction(x, y)


@dataclass(frozen=True)
class RI:
    lo: Fraction
    hi: Fraction

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError("invalid rational interval")

    @staticmethod
    def point(x):
        q = F(x)
        return RI(q, q)

    def __add__(self, other):
        o = as_ri(other)
        return RI(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return RI(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + (-as_ri(other))

    def __rsub__(self, other):
        return as_ri(other) - self

    def __mul__(self, other):
        o = as_ri(other)
        p = (self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi)
        return RI(min(p), max(p))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError("interval denominator crosses zero")
        vals = (1/self.lo, 1/self.hi)
        return RI(min(vals), max(vals))

    def __truediv__(self, other):
        return self * as_ri(other).reciprocal()

    def __rtruediv__(self, other):
        return as_ri(other) / self

    def square(self):
        if self.lo <= 0 <= self.hi:
            return RI(F(0), max(self.lo*self.lo, self.hi*self.hi))
        return RI(min(self.lo*self.lo, self.hi*self.hi),
                  max(self.lo*self.lo, self.hi*self.hi))


def as_ri(x):
    return x if isinstance(x, RI) else RI.point(x)


def _sqrt_floor_scaled(x: Fraction, bits: int = SQRT_BITS):
    if x < 0:
        raise ValueError("sqrt of negative rational")
    scale = 1 << bits
    # floor(sqrt(x)*scale) = isqrt(floor(num*scale^2/den)).
    n = (x.numerator * scale * scale) // x.denominator
    k = isqrt(n)
    while (k + 1) * (k + 1) * x.denominator <= x.numerator * scale * scale:
        k += 1
    while k * k * x.denominator > x.numerator * scale * scale:
        k -= 1
    return k, scale


def sqrt_fraction_interval(x: Fraction, bits: int = SQRT_BITS):
    x = F(x)
    k, scale = _sqrt_floor_scaled(x, bits)
    lo = F(k, scale)
    if k*k*x.denominator == x.numerator*scale*scale:
        return RI(lo, lo)
    return RI(lo, F(k+1, scale))


def sqrt_ri(x: RI, bits: int = SQRT_BITS):
    x = as_ri(x)
    if x.lo < 0:
        raise ValueError("sqrt interval has negative lower endpoint")
    return RI(sqrt_fraction_interval(x.lo, bits).lo,
              sqrt_fraction_interval(x.hi, bits).hi)


def nonnegative(x: RI):
    """Intersect with [0,+inf), using the source theorem R_x,R_y>=0."""
    x = as_ri(x)
    if x.hi < 0:
        raise ValueError("source radius interval is strictly negative")
    return RI(max(F(0), x.lo), max(F(0), x.hi))


# ---------- simplex metric / symmetries ----------

def qform4(v):
    v = tuple(v)
    return 5*sum(x*x for x in v) - sum(v)*sum(v)


def bil4(a, b):
    a, b = tuple(a), tuple(b)
    return 5*sum(a[i]*b[i] for i in range(4)) - sum(a)*sum(b)


def norm2(v):
    return qform4(v) / 10


def qform4_interval(v):
    s2 = RI.point(0)
    s = RI.point(0)
    for x in v:
        x = as_ri(x)
        s2 = s2 + x.square()
        s = s + x
    return 5*s2 - s.square()


VERTEX_COEFF = (
    (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (-1,-1,-1,-1)
)


def permutation_matrix(perm):
    """Integer matrix for v_i -> v_{perm[i]} in the first-four-vertex chart."""
    perm = tuple(map(int, perm))
    if sorted(perm) != list(range(5)):
        raise ValueError("not a permutation of five simplex vertices")
    cols = [VERTEX_COEFF[perm[i]] for i in range(4)]
    return tuple(tuple(cols[j][i] for j in range(4)) for i in range(4))


def matvec_int(M, v):
    return tuple(sum(M[i][j]*v[j] for j in range(4)) for i in range(4))


def matvec_interval(M, v):
    return tuple(sum((as_ri(v[j])*M[i][j] for j in range(4)), RI.point(0))
                 for i in range(4))


# ---------- exact source constants as rational intervals ----------

RT10 = sqrt_fraction_interval(F(10))
RT3 = sqrt_fraction_interval(F(3))
AFOC = sqrt_fraction_interval(F(3,2))
X0 = sqrt_ri((41 - 4*RT10)/27)
Y0 = sqrt_ri((7 - 2*RT10)/27)
X1 = sqrt_ri((11 + 2*RT10)/12)
Z1 = sqrt_ri((7 - 2*RT10)/36)
WIDTH = 2*Z1
G = (2*X1 + 3*X0)/5
D0 = X0 - G
SH = sqrt_ri((X0-AFOC).square() + Y0.square())
SE = sqrt_ri((X1-1).square() + Z1.square())


def source_basis_columns():
    """Centered source-coordinate columns p1-G,...,p4-G as intervals."""
    return (
        (X1-G, RI.point(0), Z1, RI.point(0)),
        (X1-G, RI.point(0), -Z1, RI.point(0)),
        (X0-G, Y0, RI.point(0), RI.point(0)),
        (X0-G, -Y0/2, RI.point(0), -RT3*Y0/2),
    )


SOURCE_BASIS = source_basis_columns()


def audit_source_simplex_similarity():
    """Audit the exact normalization from AMO source coordinates to width one.

    The source paper's regular simplex has edge length ``WIDTH=2*Z1``.  The
    four centered source columns therefore have Gram matrix ``WIDTH**2``
    times the width-one regular-simplex Gram matrix.  The coefficient chart
    divides out this common similarity factor implicitly.  This finite
    interval check guards the normalization used by the AMO adapter.
    """
    w2 = WIDTH.square()
    for i in range(4):
        for j in range(4):
            gij = RI.point(0)
            for k in range(4):
                gij = gij + SOURCE_BASIS[i][k] * SOURCE_BASIS[j][k]
            target = w2 * (F(2,5) if i == j else F(-1,10))
            if gij.hi < target.lo or target.hi < gij.lo:
                raise AssertionError("source/simplex Gram normalization mismatch")
    # In the normalized coefficient metric every simplex edge has length one.
    if norm2((1,-1,0,0)) != 1:
        raise AssertionError("normalized simplex edge does not have length one")
    return {"status":"PASS", "source_width_interval":[str(WIDTH.lo),str(WIDTH.hi)]}


def coeff_to_source_vector(v):
    """Centered source vector corresponding to coefficient vector v."""
    out = []
    for k in range(4):
        out.append(sum((as_ri(v[j])*SOURCE_BASIS[j][k] for j in range(4)), RI.point(0)))
    return tuple(out)


def source_point_to_coeff(x):
    """Source point x -> coefficient chart of x-G, via exact analytic inverse."""
    X,Y,Z,W = map(as_ri, x)
    a4 = -2*W/(RT3*Y0)
    a3 = Y/Y0 + a4/2
    A = X1-G
    B = X0-G
    S = ((X-G) - B*(a3+a4))/A
    D = Z/Z1
    return ((S+D)/2, (S-D)/2, a3, a4)


# ---------- wedge patches ----------

def _parse_box(box, expected):
    if not isinstance(box, list) or len(box) != expected:
        raise ValueError("invalid parameter box")
    out=[]
    for pair in box:
        if not isinstance(pair, list) or len(pair)!=2:
            raise ValueError("invalid interval pair")
        lo=Fraction(pair[0]); hi=Fraction(pair[1])
        out.append(RI(lo,hi))
    return out


def wedge_boundary_box(kind, box):
    """Enclose the representative AMO wedge boundary in coefficient coordinates.

    Full source domains are r,t in [0,1], q in [-1,1].
    ``triple`` is phi_1 on H_345 x E_12; ``pair`` is phi_2.
    """
    if kind not in ("triple","pair"):
        raise ValueError("unknown wedge kind")
    r,t,q = _parse_box(box,3)
    if r.lo < 0 or r.hi > 1 or t.lo < 0 or t.hi > 1 or q.lo < -1 or q.hi > 1:
        raise ValueError("wedge parameter box outside source domain")

    Y = Y0*(1 - F(3,2)*r)
    Z = RT3*Y0/2 * r*(2*t - 1)
    R2 = Y.square()+Z.square()
    Aq = D0.square()-2*R2
    Bq = 2*G*D0
    Cq = G.square()-1
    disc = Bq.square()-4*Aq*Cq
    if disc.lo <= 0:
        raise ValueError("wedge discriminant not certified positive")
    lamden = -Bq + sqrt_ri(disc)
    if lamden.lo <= 0:
        raise ValueError("wedge lambda denominator not certified positive")
    lam = 2*Cq/lamden
    X = G + D0*lam
    hy = lam*Y
    hw = lam*Z
    ze = Z1*q
    xe_sq = RI.point(F(3,2))-3*ze.square()
    if xe_sq.lo <= 0:
        raise ValueError("ellipse x^2 not certified positive")
    xe = sqrt_ri(xe_sq)
    d2 = (xe-X).square()+hy.square()+ze.square()+hw.square()
    if d2.lo <= 0:
        raise ValueError("center distance not certified positive")
    d = sqrt_ri(d2)
    Rx = nonnegative(SH - sqrt_ri((X-AFOC).square()+hy.square()+hw.square()))
    Ry = nonnegative(SE - sqrt_ri((xe-1).square()+ze.square()))
    ux=(xe-X)/d
    uy=(-hy)/d
    uz=ze/d
    uw=(-hw)/d
    if kind=="triple":
        p=(X-Rx*ux, hy-Rx*uy, -Rx*uz, hw-Rx*uw)
    else:
        p=(xe+Ry*ux, Ry*uy, ze+Ry*uz, Ry*uw)
    return source_point_to_coeff(p)


# ---------- spherical cap ----------

# Rational tangent basis at -v1.  Each b satisfies <b,v1>=0 in the simplex metric.
CAP_TANGENT = ((1,4,0,0),(1,0,4,0),(1,0,0,4))
E1=(1,0,0,0)


def _cap_direction(box):
    q1,q2,q3=_parse_box(box,3)
    for q in (q1,q2,q3):
        # Finite but intentionally generous guard; actual accepted domain is much smaller.
        if q.lo < -2 or q.hi > 2:
            raise ValueError("cap chart parameter outside guard range")
    d=[RI.point(-E1[i]) for i in range(4)]
    for q,b in zip((q1,q2,q3),CAP_TANGENT):
        for i in range(4):
            d[i]=d[i]+q*b[i]
    return tuple(d)


def _qcanon_source(dsrc):
    dx,dy,dz,dw=dsrc
    return (X1*dz-Z1*dx).square()-2*Z1.square()*(dy.square()+dw.square())-dz.square()


def _swap_perm_2_with(j):
    # source notation p1..p5; base cone is one boundary, and the four
    # transpositions p2<->pj generate the four cap inequalities used in the
    # exact gnomonic reduction.
    p=list(range(5))
    p[1],p[j]=p[j],p[1]
    return tuple(p)


CAP_CONE_PERMS=tuple(_swap_perm_2_with(j) for j in range(1,5))



RT6 = sqrt_fraction_interval(F(6))
RT15 = sqrt_fraction_interval(F(15))
CAP_A0 = (2*RT10-7)/18
CAP_L0 = (4*RT6-RT15)/12
_CAP_COMPONENT_AUDITED = None


def _cap_model_Q(r, z):
    """Exact interval form Q(r,z)=1/8+L*z*r+(A+3 z^2/8)r^2."""
    r,z=as_ri(r),as_ri(z)
    aa=CAP_A0+F(3,8)*z.square()
    return RI.point(F(1,8))+CAP_L0*z*r+aa*r.square()


def _cap_model_dQ(r, z):
    r,z=as_ri(r),as_ri(z)
    aa=CAP_A0+F(3,8)*z.square()
    return CAP_L0*z+2*aa*r


def audit_cap_first_component(max_depth=20):
    """Prove that every positive root with 0<=r<=1 is a down-crossing.

    Hence, because Q(0,z)=1/8>0, the set {0<=r<=1: Q(r,z)>=0}
    is exactly the component containing r=0.  The proof is a finite rational
    interval branch-and-bound on (r,z) in [0,1]x[-1,1].
    """
    stack=[(RI(F(0),F(1)),RI(F(-1),F(1)),0)]
    leaves=0
    while stack:
        r,z,depth=stack.pop()
        q=_cap_model_Q(r,z)
        dq=_cap_model_dQ(r,z)
        if q.lo>0 or q.hi<0 or dq.hi<0:
            leaves+=1
            continue
        if depth>=max_depth:
            raise AssertionError("cap first-component audit did not close")
        # Split the geometrically wider box (z starts with twice r's width).
        if (z.hi-z.lo) >= 2*(r.hi-r.lo):
            mid=(z.lo+z.hi)/2
            stack.append((r,RI(z.lo,mid),depth+1))
            stack.append((r,RI(mid,z.hi),depth+1))
        else:
            mid=(r.lo+r.hi)/2
            stack.append((RI(r.lo,mid),z,depth+1))
            stack.append((RI(mid,r.hi),z,depth+1))
    return {"status":"PASS","leaf_boxes":leaves,"max_depth":max_depth}


def _ensure_cap_component_audit():
    global _CAP_COMPONENT_AUDITED
    if _CAP_COMPONENT_AUDITED is None:
        _CAP_COMPONENT_AUDITED=audit_cap_first_component()
    return _CAP_COMPONENT_AUDITED


def cap_boundary_box(box):
    """Enclose the representative cap (2345) centered on the p1 sphere."""
    d=_cap_direction(box)
    _ensure_cap_component_audit()
    # Our rational tangent chart has d=-v1+w.  Relative to the orthonormal
    # gnomonic chart used in the manuscript, |q|^2=(5/2)|w|^2_simplex.
    # Restricting this to <=1, together with the global down-crossing audit,
    # rules out the remote second positive component of a cone quadratic.
    w=tuple(d[i]+RI.point(E1[i]) for i in range(4))
    r2=F(5,2)*(qform4_interval(w)/10)
    if r2.hi > 1:
        raise ValueError("cap parameter box leaves the certified first-component guard")
    # Exact project-side cap characterization: all four homogeneous cone
    # quadratics are nonnegative.  Strict lower bounds are used so that a
    # parameter box cannot straddle a wedge seam unnoticed.
    for perm in CAP_CONE_PERMS:
        M=permutation_matrix(perm)
        dp=matvec_interval(M,d)
        Q=_qcanon_source(coeff_to_source_vector(dp))
        if Q.lo < 0:
            raise ValueError("cap parameter box is not certified inside all cone domains")
    metric=qform4_interval(d)/10
    if metric.lo <= 0:
        raise ValueError("cap direction norm not positive")
    s=sqrt_ri(metric)
    return tuple(RI.point(E1[i])+d[i]/s for i in range(4))


# ---------- permutation / enclosure / shrink ----------

def permute_box(box4, perm):
    return matvec_interval(permutation_matrix(perm), box4)


def box_midpoint_radius(box4):
    """Return rational midpoint c and rational Euclidean radius rho."""
    c=tuple((x.lo+x.hi)/2 for x in box4)
    diffs=tuple(RI(x.lo-c[i], x.hi-c[i]) for i,x in enumerate(box4))
    q=qform4_interval(diffs)/10
    if q.hi < 0:
        raise AssertionError("metric interval upper bound negative")
    rho=sqrt_fraction_interval(max(F(0),q.hi)).hi
    return c,rho


def parse_permutation(obj):
    if not isinstance(obj,list) or len(obj)!=5:
        raise ValueError("invalid simplex permutation")
    p=tuple(map(int,obj))
    if sorted(p)!=list(range(5)):
        raise ValueError("invalid simplex permutation")
    return p


def boundary_enclosure(witness):
    if not isinstance(witness,dict):
        raise ValueError("witness is not an object")
    kind=witness.get("patch")
    perm=parse_permutation(witness.get("permutation"))
    box=witness.get("parameter_box")
    if kind in ("triple","pair"):
        b=wedge_boundary_box(kind,box)
    elif kind=="cap":
        b=cap_boundary_box(box)
    else:
        raise ValueError("unknown AMO patch type")
    b=permute_box(b,perm)
    return b


def _metric_dist2_fraction(a,b):
    d=tuple(F(a[i])-F(b[i]) for i in range(4))
    return F(qform4(d),10)


def verify_endpoint_witness(point_num, point_den, witness):
    """Verify one rational endpoint by boundary enclosure + inward shrink."""
    if point_den<=0 or len(point_num)!=4:
        return False
    try:
        box=boundary_enclosure(witness)
        c,rho=box_midpoint_radius(box)
        lam=Fraction(witness["lambda"])
    except (KeyError,ValueError,ZeroDivisionError,TypeError):
        return False
    if not (0 <= lam <= 1):
        return False

    # The stored c/rho are redundant audit fields, not trusted inputs.
    if "center" in witness:
        try:
            stored=tuple(Fraction(x) for x in witness["center"])
        except Exception:
            return False
        if stored!=c:
            return False
    if "rho" in witness:
        try:
            if Fraction(witness["rho"])!=rho:
                return False
        except Exception:
            return False

    slack=(1-lam)*R0-lam*rho
    if slack < 0:
        return False
    z=tuple(lam*x for x in c)
    p=tuple(Fraction(int(n),int(point_den)) for n in point_num)
    return _metric_dist2_fraction(p,z) <= slack*slack


def complete_witness(point_num, point_den, base_witness, lam):
    """Fill redundant center/rho fields and return witness if it verifies."""
    w=dict(base_witness)
    w["lambda"]=str(Fraction(lam))
    box=boundary_enclosure(w)
    c,rho=box_midpoint_radius(box)
    w["center"]=[str(x) for x in c]
    w["rho"]=str(rho)
    if not verify_endpoint_witness(point_num,point_den,w):
        return None
    return w


CHART_FACTOR={"num":1,"den":20,"sqrt_radicand":5}

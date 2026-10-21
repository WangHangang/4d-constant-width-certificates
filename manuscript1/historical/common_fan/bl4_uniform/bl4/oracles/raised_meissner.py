"""Exact conservative oracle for the face-type canonical raised Meissner body.

The positive half is tested against every preserved arc.  The negative half
uses a finite rational witness-ball family.  Consequently every accepted point
is in the true raised body; rejection may be conservative.
"""
from math import gcd, sqrt
from .base import Oracle, qform3, bil3

VERT = {"A": (1,0,0), "B": (0,1,0), "C": (0,0,1), "D": (-1,-1,-1)}
ARCS = (("A","B","C","D"), ("A","C","B","D"), ("A","D","B","C"))
WITNESS_DEN = 18


def _witnesses(den=WITNESS_DEN):
    out=[]
    for ia in range(den+1):
        for ib in range(den+1-ia):
            for ic in range(den+1-ia-ib):
                id_=den-ia-ib-ic
                m=(ia-id_,ib-id_,ic-id_)
                if qform3(m) < 3*den*den:
                    out.append(m)
    return tuple(out)


WITNESSES = _witnesses()


def _negative_one(n, D, m, W=WITNESS_DEN):
    g=gcd(D,W); L=D//g*W
    xs, ms=L//D, L//W
    xn=tuple(xs*int(v) for v in n)
    mn=tuple(ms*int(v) for v in m)
    diff=tuple(xn[i]-mn[i] for i in range(3))
    A=qform3(diff)+8*xn[3]*xn[3]
    B=5*L*L+qform3(mn)
    C=8*L*L+B-A
    return B < 8*L*L and C >= 0 and C*C >= 32*B*L*L


def _positive_even(n, D):
    nh=tuple(map(int,n[:3])); t=int(n[3])
    if t < 0: return False
    for arc in ARCS:
        X,Y,Z,W=[VERT[s] for s in arc]
        XD=tuple(D*v for v in X); YD=tuple(D*v for v in Y)
        mid=tuple(D*(Z[i]+W[i])//2 for i in range(3))
        U=tuple(XD[i]-mid[i] for i in range(3))
        Vvec=tuple(3*(YD[i]-mid[i])-U[i] for i in range(3))
        for endpoint in (XD,YD):
            de=tuple(nh[i]-endpoint[i] for i in range(3))
            if 8*D*D-qform3(de)-8*t*t < 0:
                return False
        z=tuple(nh[i]-mid[i] for i in range(3))
        uN=bil3(z,U); vN=bil3(z,Vvec)
        if vN <= 0 and vN-8*uN >= 0:
            HN=2*D*D-8*t*t-qform3(z)
            if HN < 0 or 2*HN*HN < 8*uN*uN+vN*vN:
                return False
    return True


def contains(point_num, point_den):
    if point_den <= 0 or len(point_num) != 4:
        raise ValueError("invalid rational point")
    n=tuple(map(int,point_num)); D=int(point_den)
    if n[3] >= 0:
        # Lift to an even denominator so every arc midpoint is integral.
        return _positive_even(tuple(2*v for v in n), 2*D)
    return any(_negative_one(n,D,m) for m in WITNESSES)


def _qf(a):
    x,y,z=a
    return 3*(x*x+y*y+z*z)-2*(x*y+x*z+y*z)


def _bf(a,b):
    x,y,z=a; X,Y,Z=b
    return 3*(x*X+y*Y+z*Z)-(x*Y+y*X+x*Z+z*X+y*Z+z*Y)


WF=tuple(tuple(v/WITNESS_DEN for v in m) for m in WITNESSES)


def contains_float(x):
    h=x[:3]; t=x[3]
    if t < 0:
        for m in WF:
            mn2=_qf(m)/8.0
            eta=1.0-sqrt(5.0/8.0+mn2)
            diff=tuple(h[i]-m[i] for i in range(3))
            if _qf(diff)/8.0+t*t <= eta*eta:
                return True
        return False
    for arc in ARCS:
        X,Y,Z,W=[VERT[s] for s in arc]
        mid=tuple((Z[i]+W[i])/2 for i in range(3))
        U=tuple(X[i]-mid[i] for i in range(3))
        Vv=tuple(3*(Y[i]-mid[i])-U[i] for i in range(3))
        for endpoint in (X,Y):
            de=tuple(h[i]-endpoint[i] for i in range(3))
            if _qf(de)/8.0+t*t > 1.0:
                return False
        z=tuple(h[i]-mid[i] for i in range(3))
        u=_bf(z,U)/8.0; v=_bf(z,Vv)/16.0
        if v <= 0 and v-4*u >= 0:
            H=(2.0-8*t*t-_qf(z))/8.0
            if H < 0 or 2*H*H < 8*u*u+(2*v)*(2*v):
                # Float path is advisory only; exact final check controls proof.
                return False
    return True


def radial_float(direction, max_radius):
    if direction[3] >= 0:
        lo,hi=0.0,float(max_radius)
        if contains_float(tuple(hi*v for v in direction)):
            raise ValueError("max_radius does not bracket the body")
        for _ in range(55):
            mid=(lo+hi)/2
            if contains_float(tuple(mid*v for v in direction)): lo=mid
            else: hi=mid
        return lo
    qh=direction[:3]; qt=direction[3]
    qnorm=_qf(qh)/8.0+qt*qt
    best=0.0
    for m in WF:
        mn2=_qf(m)/8.0
        eta=1.0-sqrt(5.0/8.0+mn2)
        md=_bf(m,qh)/8.0
        disc=md*md+qnorm*(eta*eta-mn2)
        if disc>=0:
            root=(md+sqrt(disc))/qnorm
            if root>best: best=root
    return min(best,float(max_radius))


ORACLE = Oracle(
    name="RaisedMeissnerFace",
    contains=contains,
    contains_float=contains_float,
    description=("Face-type canonical raise of a width-one Meissner body in "
                 "tetrahedral coefficient coordinates. Positive arc maximum "
                 "and negative rational witness-ball tests are exact integers."),
    # det(A,B,C in R^3)=sqrt(2)/8; the fourth coordinate is Cartesian.
    chart_factor={"num": 1, "den": 8, "sqrt_radicand": 2},
    radial_float_fn=radial_float,
)

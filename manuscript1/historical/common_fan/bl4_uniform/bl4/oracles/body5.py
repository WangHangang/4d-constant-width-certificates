"""Exact Body-5 membership via the manuscript's seven-candidate theorem."""
from fractions import Fraction
from .base import Oracle
from ..quadratic5 import Q5, inverse_matrix, matvec, nonnegative_a_plus_b_sqrt_q, a_plus_sqrt_b_leq


def q(v): return tuple(Q5.coerce(x) for x in v)
BASE=(q((1,0,0,0)),q((0,1,0,0)),q((0,0,1,0)),q((0,0,0,1)))
PHI=Q5(Fraction(1,2),Fraction(1,2))
E=tuple(PHI/2 for _ in range(4))
G=tuple(Q5(Fraction(1,4),Fraction(1,20)) for _ in range(4))


def add(a,b): return tuple(a[i]+b[i] for i in range(4))
def sub(a,b): return tuple(a[i]-b[i] for i in range(4))
def scale(c,a): return tuple(c*x for x in a)
def dot(a,b): return sum((a[i]*b[i] for i in range(4)),Q5())/2


def solve_coeff(gs,x):
    gram=[[dot(a,b) for b in gs] for a in gs]
    return matvec(inverse_matrix(gram),[dot(x,g) for g in gs])


PATCHES=[]
for i in range(4):
    for j in range(i+1,4):
        rem=[k for k in range(4) if k not in (i,j)]
        c=scale(Fraction(1,2),add(BASE[rem[0]],BASE[rem[1]]))
        gs=(BASE[i],BASE[j],E)
        gram=[[dot(a,b) for b in gs] for a in gs]
        inv=inverse_matrix(gram)
        ccoeff=solve_coeff(gs,c)
        edges=[]
        for omit in range(3):
            es=tuple(g for k,g in enumerate(gs) if k!=omit)
            egram=[[dot(a,b) for b in es] for a in es]
            einv=inverse_matrix(egram)
            cproj_coeff=matvec(einv,[dot(c,g) for g in es])
            cproj=tuple(sum((cproj_coeff[k]*es[k][d] for k in range(2)),Q5()) for d in range(4))
            dist=sub(c,cproj)
            r2=Q5(Fraction(3,4))-dot(dist,dist)
            edges.append((es,einv,cproj,cproj_coeff,r2))
        PATCHES.append((c,gs,inv,ccoeff,tuple(edges)))


def contains(point_num,point_den):
    if point_den<=0 or len(point_num)!=4: raise ValueError("invalid rational point")
    p=tuple(Q5(Fraction(int(n),int(point_den))) for n in point_num)
    x=add(G,p)
    xx=dot(x,x)
    threshold=Q5(Fraction(1,4))-xx/2
    for c,gs,inv,ccoeff,edges in PATCHES:
        v=sub(c,x)
        if any(dot(v,g)>threshold for g in gs): return False
        b=[dot(v,g) for g in gs]
        alpha=matvec(inv,b)
        p2=sum((b[i]*alpha[i] for i in range(3)),Q5())
        if p2:
            factor2=Q5(Fraction(3,4))/p2
            feasible=all(nonnegative_a_plus_b_sqrt_q(ccoeff[i],alpha[i],factor2) for i in range(3))
            if feasible and not a_plus_sqrt_b_leq(dot(v,c),Q5(Fraction(3,4))*p2,threshold):
                return False
        for es,einv,cproj,cproj_coeff,r2 in edges:
            eb=[dot(v,g) for g in es]
            ealpha=matvec(einv,eb)
            ep2=sum((eb[i]*ealpha[i] for i in range(2)),Q5())
            if ep2:
                factor2=r2/ep2
                feasible=all(nonnegative_a_plus_b_sqrt_q(cproj_coeff[i],ealpha[i],factor2) for i in range(2))
                if feasible and not a_plus_sqrt_b_leq(dot(v,cproj),r2*ep2,threshold):
                    return False
    return True


def _inverse_float(a):
    n=len(a); m=[list(a[i])+[float(i==j) for j in range(n)] for i in range(n)]
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(m[i][k])); m[k],m[p]=m[p],m[k]
        z=m[k][k]; m[k]=[x/z for x in m[k]]
        for i in range(n):
            if i!=k:
                z=m[i][k]; m[i]=[x-z*y for x,y in zip(m[i],m[k])]
    return [r[n:] for r in m]


def _mv(a,x): return [sum(a[i][j]*x[j] for j in range(len(x))) for i in range(len(a))]
def _fvec(v): return tuple(x.to_float() for x in v)
def _fdot(a,b): return sum(a[i]*b[i] for i in range(4))/2.0


FPATCHES=[]
for c,gs,_,_,_ in PATCHES:
    c=_fvec(c); gs=tuple(_fvec(g) for g in gs)
    inv=_inverse_float([[_fdot(a,b) for b in gs] for a in gs])
    ccoeff=_mv(inv,[_fdot(c,g) for g in gs])
    edges=[]
    for omit in range(3):
        es=tuple(g for k,g in enumerate(gs) if k!=omit)
        einv=_inverse_float([[_fdot(a,b) for b in es] for a in es])
        cpcoef=_mv(einv,[_fdot(c,g) for g in es])
        cp=tuple(sum(cpcoef[k]*es[k][d] for k in range(2)) for d in range(4))
        dist=tuple(c[d]-cp[d] for d in range(4))
        edges.append((es,einv,cp,cpcoef,.75-_fdot(dist,dist)))
    FPATCHES.append((c,gs,inv,ccoeff,tuple(edges)))


def contains_float(p):
    x=tuple(G[i].to_float()+p[i] for i in range(4)); xx=_fdot(x,x); threshold=.25-xx/2
    for c,gs,inv,ccoeff,edges in FPATCHES:
        v=tuple(c[i]-x[i] for i in range(4))
        if any(_fdot(v,g)>threshold for g in gs): return False
        b=[_fdot(v,g) for g in gs]; alpha=_mv(inv,b); p2=sum(b[i]*alpha[i] for i in range(3))
        if p2>0:
            factor=(.75/p2)**.5
            if all(ccoeff[i]+factor*alpha[i]>=-1e-13 for i in range(3)):
                if _fdot(v,c)+(.75*p2)**.5>threshold: return False
        for es,einv,cp,cpcoef,r2 in edges:
            eb=[_fdot(v,g) for g in es]; ea=_mv(einv,eb); ep2=sum(eb[i]*ea[i] for i in range(2))
            if ep2>0:
                factor=(r2/ep2)**.5
                if all(cpcoef[i]+factor*ea[i]>=-1e-13 for i in range(2)):
                    if _fdot(v,cp)+(r2*ep2)**.5>threshold: return False
    return True


ORACLE=Oracle(
    name="Body5",contains=contains,contains_float=contains_float,
    description=("Body 5 in the orthogonal A,B,C,D coefficient chart centered at G. "
                 "Exact Q(sqrt(5)) implementation of the six-patch, seven-candidate support theorem."),
    chart_factor={"num":1,"den":4,"sqrt_radicand":1},
)


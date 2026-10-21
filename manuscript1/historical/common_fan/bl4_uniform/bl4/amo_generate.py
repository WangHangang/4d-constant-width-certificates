"""Numerical proposal layer for the direct AMO witness adapter.

This file is intentionally *not* part of the proof kernel.  It may use NumPy
and SciPy to find a boundary point close to a prescribed rational fan ray.
Every proposed endpoint is subsequently certified by ``bl4.amo_witness`` using
standard-library exact rational interval arithmetic and the inward-shrink
lemma.

The generator uses exact *rational parameter points* (degenerate parameter
boxes) after the floating search.  Thus the proof-side enclosure radius comes
only from rigorous enclosures of the algebraic constants and radicals, not
from a floating parameter uncertainty box.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .amo_witness import (
    R0, boundary_enclosure, box_midpoint_radius, bil4, qform4,
    permutation_matrix, sqrt_fraction_interval, verify_endpoint_witness,
)

# ---------------- floating source geometry (proposal only) ----------------
rt10=math.sqrt(10.0); afoc=math.sqrt(1.5)
x0=math.sqrt((41-4*rt10)/27); y0=math.sqrt((7-2*rt10)/27)
x1=math.sqrt((11+2*rt10)/12); z1=math.sqrt((7-2*rt10)/36)
g=(2*x1+3*x0)/5
sH=math.hypot(x0-afoc,y0); sE=math.hypot(x1-1,z1)
D0=x0-g

P=np.array([
    [x1,0,z1,0], [x1,0,-z1,0], [x0,y0,0,0],
    [x0,-y0/2,0,-math.sqrt(3)*y0/2],
    [x0,-y0/2,0, math.sqrt(3)*y0/2],
],dtype=float)
Gsrc=P.mean(axis=0)
Bsrc=(P[:4]-Gsrc).T
Bsrc_inv=np.linalg.inv(Bsrc)

GM=(5*np.eye(4)-np.ones((4,4)))/10.0
ORTH=np.linalg.cholesky(GM).T


def coeff_to_orth(v):
    return ORTH @ np.asarray(v,dtype=float)


def unit_direction(v):
    y=coeff_to_orth(v); n=np.linalg.norm(y)
    if not n>0: raise ValueError("zero direction")
    return y/n


def source_to_coeff(p):
    return Bsrc_inv @ (np.asarray(p,dtype=float)-Gsrc)


def perm_apply_float(perm,v):
    M=np.asarray(permutation_matrix(tuple(perm)),dtype=float)
    return M @ np.asarray(v,dtype=float)


def wedge_point_float(kind,par):
    r,t,q=map(float,par)
    if not (0<=r<=1 and 0<=t<=1 and -1<=q<=1):
        raise ValueError("wedge parameter outside domain")
    Y=y0*(1-1.5*r)
    W=math.sqrt(3)*y0/2*r*(2*t-1)
    R2=Y*Y+W*W
    A=D0*D0-2*R2; Bq=2*g*D0; Cq=g*g-1
    disc=Bq*Bq-4*A*Cq
    if disc<=0: raise ValueError("negative wedge discriminant")
    den=-Bq+math.sqrt(disc)
    if den<=0: raise ValueError("bad wedge root")
    lam=2*Cq/den
    X=g+D0*lam; hy=lam*Y; hw=lam*W
    ze=z1*q; xe2=1.5-3*ze*ze
    if xe2<=0: raise ValueError("ellipse endpoint outside")
    xe=math.sqrt(xe2)
    dvec=np.array([xe-X,-hy,ze,-hw],float)
    d=np.linalg.norm(dvec)
    if d<=0: raise ValueError("zero center distance")
    u=dvec/d
    Rx=sH-math.sqrt((X-afoc)**2+hy*hy+hw*hw)
    Ry=sE-math.sqrt((xe-1)**2+ze*ze)
    if Rx < -1e-12 or Ry < -1e-12:
        raise ValueError("negative source radius")
    Rx=max(0.0,Rx); Ry=max(0.0,Ry)
    if kind=="triple":
        p=np.array([X,hy,0,hw])-Rx*u
    elif kind=="pair":
        p=np.array([xe,0,ze,0])+Ry*u
    else:
        raise ValueError("bad wedge kind")
    return source_to_coeff(p)


CAP_TANGENT=np.array(((1,4,0,0),(1,0,4,0),(1,0,0,4)),dtype=float)
E1=np.array((1,0,0,0),dtype=float)


def cap_point_float(par):
    q=np.asarray(par,dtype=float)
    d=-E1 + q @ CAP_TANGENT
    n=math.sqrt(float(d@GM@d))
    if n<=0: raise ValueError("zero cap direction")
    return E1+d/n


def qcanon_source_float(dsrc):
    dx,dy,dz,dw=np.asarray(dsrc,dtype=float)
    return (x1*dz-z1*dx)**2 - 2*z1*z1*(dy*dy+dw*dw) - dz*dz


def cap_valid_float(par, tol=2e-12):
    q=np.asarray(par,dtype=float)
    d=-E1 + q @ CAP_TANGENT
    w=d+E1
    if 2.5*float(w@GM@w)>1+tol:
        return False
    for j in range(1,5):
        perm=list(range(5)); perm[1],perm[j]=perm[j],perm[1]
        dp=perm_apply_float(perm,d)
        if qcanon_source_float(Bsrc@dp)<-tol:
            return False
    return True


def wedge_permutations():
    out=[]
    for a in range(5):
        for b in range(a+1,5):
            rest=[j for j in range(5) if j not in (a,b)]
            out.append((a,b,*rest))
    return tuple(out)


def cap_permutations():
    out=[]
    for a in range(5):
        rest=[j for j in range(5) if j!=a]
        out.append((a,*rest))
    return tuple(out)

WEDGE_PERMS=wedge_permutations()
CAP_PERMS=cap_permutations()


@dataclass(frozen=True)
class Seed:
    patch: str
    perm: tuple[int,...]
    params: tuple[float,float,float]


class ProposalLibrary:
    """Small deterministic nearest-neighbour library for local solves."""
    def __init__(self,wedge_grid=5,cap_grid=7,cap_bound=0.13):
        seeds=[]; dirs=[]
        wg=np.linspace(0,1,wedge_grid); qg=np.linspace(-1,1,wedge_grid)
        for kind in ("triple","pair"):
            for par in itertools.product(wg,wg,qg):
                try: base=wedge_point_float(kind,par)
                except ValueError: continue
                for perm in WEDGE_PERMS:
                    p=perm_apply_float(perm,base)
                    if np.linalg.norm(coeff_to_orth(p))<1e-14: continue
                    seeds.append(Seed(kind,perm,tuple(map(float,par))))
                    dirs.append(unit_direction(p))
        cg=np.linspace(-cap_bound,cap_bound,cap_grid)
        for par in itertools.product(cg,cg,cg):
            if not cap_valid_float(par): continue
            base=cap_point_float(par)
            for perm in CAP_PERMS:
                p=perm_apply_float(perm,base)
                if np.linalg.norm(coeff_to_orth(p))<1e-14: continue
                seeds.append(Seed("cap",perm,tuple(map(float,par))))
                dirs.append(unit_direction(p))
        self.seeds=seeds
        self.dirs=np.asarray(dirs,dtype=float)
        self.tree=cKDTree(self.dirs)

    def nearest(self,target,k=12):
        k=min(int(k),len(self.seeds))
        _,idx=self.tree.query(unit_direction(target),k=k)
        idx=np.atleast_1d(idx)
        return [self.seeds[int(i)] for i in idx]


def _eval_seed(seed,par):
    if seed.patch in ("triple","pair"):
        p=wedge_point_float(seed.patch,par)
    else:
        p=cap_point_float(par)
    return perm_apply_float(seed.perm,p)


def _residual(seed,par,target_unit):
    try:
        p=_eval_seed(seed,par)
        return unit_direction(p)-target_unit
    except Exception:
        return np.full(4,10.0)


def optimize_seed(seed,target,max_nfev=160):
    target_unit=unit_direction(target)
    if seed.patch in ("triple","pair"):
        lb=np.array((0.,0.,-1.)); ub=np.array((1.,1.,1.))
    else:
        # The certified cap component lies comfortably inside this generous guard.
        lb=np.array((-0.18,-0.18,-0.18)); ub=-lb
    x0=np.clip(np.array(seed.params,dtype=float),lb,ub)
    res=least_squares(lambda x:_residual(seed,x,target_unit),x0,bounds=(lb,ub),
                      xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=max_nfev)
    p=_eval_seed(seed,res.x)
    err=float(np.linalg.norm(unit_direction(p)-target_unit))
    return tuple(map(float,res.x)),err


def dyadic_point_box(params,patch,exp):
    D=1<<int(exp); out=[]
    for i,x in enumerate(params):
        k=int(round(float(x)*D))
        if patch in ("triple","pair"):
            lo,hi=((0,D) if i<2 else (-D,D))
        else:
            lo,hi=(-2*D,2*D)
        k=min(hi,max(lo,k))
        s=str(Fraction(k,D)); out.append([s,s])
    return out


def _ray_projection_data(u,c):
    u=tuple(map(int,u))
    qu=Fraction(qform4(u),10)
    if qu<=0: raise ValueError("zero ray")
    ip=Fraction(bil4(u,c),10)
    alpha=ip/qu
    c2=Fraction(qform4(c),10)
    perp2=c2-ip*ip/qu
    if perp2<0 and abs(float(perp2))<1e-30: perp2=Fraction(0)
    if perp2<0: raise AssertionError("negative perpendicular square")
    perp=sqrt_fraction_interval(perp2).hi
    unorm=sqrt_fraction_interval(qu).hi
    return alpha,perp,unorm


def build_endpoint_for_witness(u,scale,radius_den,base_witness,lambda_exp=52):
    """Choose a dyadic inward factor and nearest radial grid point, then verify."""
    box=boundary_enclosure(base_witness)
    c,rho=box_midpoint_radius(box)
    alpha,perp,unorm=_ray_projection_data(u,c)
    if alpha<=0: return None
    point_den=int(scale)*int(radius_den)
    # Sufficient continuous bound including half a radial-grid step.
    grid=unorm/Fraction(2*point_den)
    B=R0-grid
    if B<=0: return None
    lam_bound=B/(R0+perp+rho)
    if lam_bound<=0: return None
    D=1<<int(lambda_exp)
    k=min(D-1,(lam_bound.numerator*D)//lam_bound.denominator - 1)
    if k<=0: return None
    # Try a short descending sequence to absorb all rational-grid effects.
    for drop in range(0,64):
        kk=k-drop
        if kk<=0: break
        lam=Fraction(kk,D)
        real_r=Fraction(point_den)*lam*alpha
        base=(real_r.numerator*2+real_r.denominator)//(2*real_r.denominator)
        for rr in (base,base-1,base+1,base-2,base+2):
            if rr<0: continue
            w=dict(base_witness); w["lambda"]=str(lam)
            point=tuple(int(rr)*int(x) for x in u)
            if verify_endpoint_witness(point,point_den,w):
                return int(rr),w,{"lambda":float(lam),"perp":float(perp),"rho":float(rho)}
    return None


def certify_ray(u,scale,radius_den,library,nearest=16,param_exps=(36,44,52),
                max_nfev=180, preferred_residual=1e-6):
    """Find and rigorously certify one AMO endpoint on a fan ray.

    Patch seams can make the nearest coarse-library seeds all belong to a cap
    chart whose rationalized optimizer lies infinitesimally on the wrong side
    of a cone inequality.  We therefore expand the nearest-neighbour pool
    before accepting a visibly off-ray shrink witness.  This affects efficiency
    only; every returned endpoint is still checked by the exact verifier.
    """
    tried=set(); best=[]; failures=[]; fallback=None
    schedule=[]
    for k in (nearest,max(48,3*nearest),max(128,8*nearest)):
        k=min(k,len(library.seeds))
        if not schedule or k>schedule[-1]: schedule.append(k)
    for k in schedule:
        for seed in library.nearest(u,k=k):
            key=(seed.patch,seed.perm,seed.params)
            if key in tried: continue
            tried.add(key)
            try:
                par,err=optimize_seed(seed,u,max_nfev=max_nfev)
                best.append((err,seed,par))
            except Exception:
                continue
        best.sort(key=lambda z:z[0])
        for err,seed,par in best:
            # Do not repeatedly retry candidates already exact-tested.
            testkey=(seed.patch,seed.perm,tuple(round(x,15) for x in par))
            if any(f and f[0]==testkey for f in failures):
                continue
            candidate_failed=[]
            for exp in param_exps:
                bw={"patch":seed.patch,"permutation":list(seed.perm),
                    "parameter_box":dyadic_point_box(par,seed.patch,exp)}
                try:
                    out=build_endpoint_for_witness(u,scale,radius_den,bw)
                except Exception as e:
                    candidate_failed.append((exp,type(e).__name__,str(e)[:120])); continue
                if out is not None:
                    r,w,diag=out
                    diag.update({"direction_residual":err,"parameter_exp":exp,"patch":seed.patch,
                                 "nearest_pool":k})
                    if err<=preferred_residual:
                        return r,w,diag
                    if fallback is None or err<fallback[2]["direction_residual"]:
                        fallback=(r,w,diag)
                    break
            failures.append((testkey,candidate_failed))
        if best and best[0][0]<=preferred_residual and fallback is not None:
            # An accurate optimizer existed but all accurate exact charts failed;
            # expand once more to look for its adjacent seam patch.
            continue
    if fallback is not None:
        return fallback
    raise RuntimeError("AMO ray certification failed; best diagnostics="+repr(failures[:8]))

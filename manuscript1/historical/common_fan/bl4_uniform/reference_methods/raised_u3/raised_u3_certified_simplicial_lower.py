#!/usr/bin/env python3
r"""Computer-assisted strict lower bound for the canonical raised U3.

The proof uses an explicit simplicial fan, not statistical sampling.
Each open hemisphere is partitioned into two bounded gnomonic charts:

(1) polar cap: directions proportional to (x1,x2,x3,sign), x_i in [-1,1];
(2) equatorial chamber: directions proportional to (1,a,c,sign*b),
    0<=a<=1, |c|<=a, 0<=b<=1, followed by the 24 tetrahedral images.

The parameter domains are triangulated explicitly.  If p_i=r_i u_i are
certified points of a convex target body on the four rays of one parameter
tetrahedron, conv(0,p_1,p_2,p_3,p_4) is contained in the target body.  The
gnomonic affine relation shows that interiors of these cones are disjoint.
Therefore the sum of their 4-simplex volumes is a rigorous lower bound.

Positive vertices are certified in the true K_{U,+} by the six-arc interval
certificate from raised_u3_certify_centres.py.  Negative vertices carry an
explicit ball B(m,eta(m)) contained in the rigorous inner model L_-.
All stored binary64 coordinates are treated as exact dyadic rationals, and
4x4 determinants are evaluated by interval arithmetic.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, time
from pathlib import Path
import mpmath as mp
import numpy as np

HERE=Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(HERE))
import body12_radial_exact_meissner_u3_elliptic as backend
import raised_u3_certify_centres as arc_cert
import raised_u3_negative_inner_validated_lower as neg_backend
import raised_u3_validated_cubature_s4_certified as upper_backend
from numba import njit, prange

PERMS4=list(itertools.permutations(range(4)))

def sha256_file(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def exact_iv(x:float):
    n,d=float(x).as_integer_ratio();return mp.iv.mpf(n)/mp.iv.mpf(d)

def det4_interval(cols:np.ndarray):
    A=[[exact_iv(cols[r,c]) for c in range(4)] for r in range(4)]
    total=mp.iv.mpf(0)
    for p in PERMS4:
        inv=sum(1 for i in range(4) for j in range(i+1,4) if p[i]>p[j])
        term=mp.iv.mpf(-1 if inv&1 else 1)
        for i in range(4):term*=A[i][p[i]]
        total+=term
    return total

def abs_lower(iv):
    if iv>0:return mp.mpf(iv.a)
    if iv<0:return -mp.mpf(iv.b)
    return mp.mpf(0)

def interval_leq(a,b):return bool(a<=b)

def certify_negative_point(u:np.ndarray):
    m,eta,_=neg_backend.candidate_ball(np.asarray(u,dtype=np.float64))
    m=np.asarray(m,dtype=np.float64);eta=float(eta)
    s0=mp.iv.mpf(1)/(2*mp.iv.sqrt(2));mi=[exact_iv(x) for x in m]
    for signs in arc_cert.VERTEX_SIGNS:
        dotiv=s0*sum((int(signs[k])*mi[k] for k in range(3)),mp.iv.mpf(0))
        if not (dotiv>=-mp.iv.mpf(1)/8):raise RuntimeError('negative witness m not certified in T')
    dot=float(np.dot(m,u[:3]));n2=float(np.dot(m,m));disc=dot*dot+eta*eta-n2
    if disc < -1.0e-14:raise RuntimeError('selected inner ball misses its assigned ray')
    disc=max(0.0,disc)
    root=dot+math.sqrt(disc)
    ci=[exact_iv(m[0]),exact_iv(m[1]),exact_iv(m[2]),mp.iv.mpf(0)];e2=exact_iv(eta)*exact_iv(eta)
    r=np.nextafter(root,-np.inf)
    for _ in range(160):
        x=np.asarray(r*u,dtype=np.float64);xi=[exact_iv(v) for v in x]
        dist2=sum(((xi[k]-ci[k])**2 for k in range(4)),mp.iv.mpf(0))
        if interval_leq(dist2,e2):return x,m,eta
        r=np.nextafter(r,-np.inf)
    # For a nearly degenerate ball, its centre itself is always a certified point of the ball union.
    x=np.zeros(4,dtype=np.float64);x[:3]=m
    xi=[exact_iv(v) for v in x];dist2=sum(((xi[k]-ci[k])**2 for k in range(4)),mp.iv.mpf(0))
    if interval_leq(dist2,e2):return x,m,eta
    raise RuntimeError('could not certify negative ball point')


@njit(inline="always")
def _sqr_interval_exact(x):
    y=x*x
    return np.nextafter(y,-np.inf),np.nextafter(y,np.inf)
@njit(inline="always")
def _sum3_norm2_interval(x0,x1,x2):
    l0,h0=_sqr_interval_exact(x0);l1,h1=_sqr_interval_exact(x1);l2,h2=_sqr_interval_exact(x2)
    lo=np.nextafter(np.nextafter(l0+l1,-np.inf)+l2,-np.inf)
    hi=np.nextafter(np.nextafter(h0+h1,np.inf)+h2,np.inf)
    return lo,hi
@njit(inline="always")
def _point_ball_dist2_upper(r,u,m):
    total=0.0
    for k in range(3):
        x=np.nextafter(r*u[k],np.inf) if r*u[k]>=0 else np.nextafter(r*u[k],-np.inf)
        d=x-m[k];sq=np.nextafter(d*d,np.inf);total=np.nextafter(total+sq,np.inf)
    x=np.nextafter(r*u[3],np.inf) if r*u[3]>=0 else np.nextafter(r*u[3],-np.inf)
    total=np.nextafter(total+np.nextafter(x*x,np.inf),np.inf)
    return total

@njit(inline="always")
def _ball_root_value(u,m):
    n2=m[0]*m[0]+m[1]*m[1]+m[2]*m[2]
    eta=1.0-math.sqrt(0.625+n2)
    d=m[0]*u[0]+m[1]*u[1]+m[2]*u[2]
    disc=d*d+eta*eta-n2
    if disc<=0.0:return -1.0e100
    return d+math.sqrt(disc)
@njit(inline="always")
def _fast_witness_m(u,iterations=24):
    m=np.zeros(3);value=_ball_root_value(u,m)
    for it in range(iterations):
        n2=m[0]*m[0]+m[1]*m[1]+m[2]*m[2];w=math.sqrt(0.625+n2);eta=1.0-w
        d=m[0]*u[0]+m[1]*u[1]+m[2]*u[2];disc=d*d+eta*eta-n2
        if disc<=1.0e-20:break
        S=math.sqrt(disc);g=np.empty(3)
        for k in range(3):g[k]=u[k]+(d*u[k]-m[k]/w)/S
        gn=math.sqrt(g[0]*g[0]+g[1]*g[1]+g[2]*g[2])
        if gn<1.0e-13:break
        step=0.15/(1.0+0.05*it);improved=False
        for _ in range(14):
            z=m+step*g;cand=neg_backend.project_tetra(z);nv=_ball_root_value(u,cand)
            if nv>value:
                m=cand;value=nv;improved=True;break
            step*=0.5
        if not improved:break
    m*=1.0-1.0e-11
    return m

@njit
def negative_points_batch(dirs):
    n=len(dirs);pts=np.empty((n,4));valid=np.ones(n,dtype=np.uint8);etas=np.empty(n);margins=np.empty(n)
    s0lo=np.nextafter(1.0/(2.0*math.sqrt(2.0)),-np.inf);s0hi=np.nextafter(1.0/(2.0*math.sqrt(2.0)),np.inf)
    for i in range(n):
        u=dirs[i];m=_fast_witness_m(u)
        # certify m in T using interval products
        minmargin=1e300
        for j in range(4):
            dlo=dhi=0.0
            for k in range(3):
                coeff_lo=s0lo*arc_cert.VERTEX_SIGNS[j,k];coeff_hi=s0hi*arc_cert.VERTEX_SIGNS[j,k]
                if coeff_lo>coeff_hi:coeff_lo,coeff_hi=coeff_hi,coeff_lo
                pl,ph=neg_backend.imul(coeff_lo,coeff_hi,m[k],m[k]);dlo,dhi=neg_backend.iadd(dlo,dhi,pl,ph)
            margin=np.nextafter(dlo+0.125,-np.inf)
            if margin<minmargin:minmargin=margin
            if margin<0.0:valid[i]=0
        margins[i]=minmargin
        n2lo,n2hi=_sum3_norm2_interval(m[0],m[1],m[2])
        tmp=np.nextafter(0.625+n2hi,np.inf);sq=np.nextafter(math.sqrt(tmp),np.inf);eta=np.nextafter(1.0-sq,-np.inf)
        if eta<0.0:eta=0.0
        etas[i]=eta
        dot=m[0]*u[0]+m[1]*u[1]+m[2]*u[2]
        disc=dot*dot+eta*eta-(m[0]*m[0]+m[1]*m[1]+m[2]*m[2])
        if disc<-1e-12:valid[i]=0;disc=0.0
        if disc<0.0:disc=0.0
        root=dot+math.sqrt(disc)
        found=False
        # pull inward until the exact-float point is certified in the conservative ball
        for attempt in range(40):
            factor=1.0-(2.0**attempt)*1e-13
            if factor<0.0:factor=0.0
            r=root*factor
            upper=_point_ball_dist2_upper(r,u,m)
            e2lo=np.nextafter(eta*eta,-np.inf)
            if upper<=e2lo:
                for k in range(4):pts[i,k]=r*u[k]
                found=True;break
        if not found:
            # Degenerate vertex case: if m is collinear with u, use m itself.
            mn=math.sqrt(m[0]*m[0]+m[1]*m[1]+m[2]*m[2])
            err=abs(m[0]-mn*u[0])+abs(m[1]-mn*u[1])+abs(m[2]-mn*u[2])+abs(mn*u[3])
            if err<1e-9:
                pts[i,0]=m[0];pts[i,1]=m[1];pts[i,2]=m[2];pts[i,3]=0.0
            else:valid[i]=0
    return pts,valid,etas,margins

def direction_cap(x,sign):
    v=np.asarray([x[0],x[1],x[2],float(sign)],dtype=np.float64);return v/np.linalg.norm(v)
def direction_equi(a,c,b,sign):
    v=np.asarray([1.0,a,c,float(sign)*b],dtype=np.float64);return v/np.linalg.norm(v)


@njit(inline="always")
def _up(x): return np.nextafter(x,np.inf)
@njit(inline="always")
def _dn(x): return np.nextafter(x,-np.inf)
@njit(inline="always")
def _isq_upper(lo,hi):
    if lo<=0.0<=hi:return _up(max(lo*lo,hi*hi))
    return _up(max(lo*lo,hi*hi))
@njit(parallel=True)
def certify_positive_grid(points,qlo,qhi,interp_error):
    n=len(points);slack=np.empty(n);passed=np.ones(n,dtype=np.uint8)
    narcs=qlo.shape[0];ngrid=qlo.shape[1]
    for i in prange(n):
        y0,y1,y2,h=points[i,0],points[i,1],points[i,2],points[i,3]
        best=0.0
        for a in range(narcs):
            for j in range(ngrid):
                d0lo=_dn(y0-qhi[a,j,0]);d0hi=_up(y0-qlo[a,j,0])
                d1lo=_dn(y1-qhi[a,j,1]);d1hi=_up(y1-qlo[a,j,1])
                d2lo=_dn(y2-qhi[a,j,2]);d2hi=_up(y2-qlo[a,j,2])
                val=_up(_up(_isq_upper(d0lo,d0hi)+_isq_upper(d1lo,d1hi))+_isq_upper(d2lo,d2hi))
                if val>best:best=val
        yn2=_up(_up(_up(y0*y0)+_up(y1*y1))+_up(y2*y2))
        h2=_up(h*h);total=_up(_up(best+interp_error)+h2);slack[i]=_dn(1.0-total)
        if not (h>=0.0 and yn2<=1.0 and slack[i]>0.0):passed[i]=0
    return passed,slack

def positive_points_batch(directions,shrink,qlo,qhi,interp_error,boundary_iterations,root_iterations):
    arc_a,arc_b=backend.u3_elliptical_arc_arrays()
    rho=backend.positive_radial_u3_kernel(directions,arc_a,arc_b,boundary_iterations,root_iterations);points=np.asarray(np.float64(shrink)*rho[:,None]*directions,dtype=np.float64)
    passed,slack=certify_positive_grid(points,qlo,qhi,interp_error)
    if not np.all(passed):
        bad=int(np.flatnonzero(passed==0)[0]);raise RuntimeError(f'positive grid certificate failed at {bad}, slack={slack[bad]}')
    return points,slack

def positive_point(u,shrink,terminal_depth):
    rho=float(backend.positive_radial_u3(u[None,:])[0]);p=np.asarray(np.float64(shrink)*rho*u,dtype=np.float64)
    passed,slack,visited,terminal=arc_cert.certify_seed(p,terminal_depth)
    if not passed:raise RuntimeError(f'positive point failed: {p}, slack={slack}')
    return p,slack,visited,terminal

def triangle_grid(n):
    nodes=[];idx={}
    for i in range(n+1):
        for j in range(n+1-i):
            idx[(i,j)]=len(nodes);nodes.append(((i+j)/n,(-i+j)/n))
    tris=[]
    for i in range(n):
        for j in range(n-i):
            tris.append((idx[(i,j)],idx[(i+1,j)],idx[(i,j+1)]))
            if i+j<=n-2:tris.append((idx[(i+1,j)],idx[(i+1,j+1)],idx[(i,j+1)]))
    return np.asarray(nodes),np.asarray(tris,dtype=np.int64)

def cube_tets():
    out=[]
    for perm in itertools.permutations(range(3)):
        cur=np.zeros(3,dtype=np.int64);verts=[cur.copy()]
        for ax in perm:cur=cur.copy();cur[ax]+=1;verts.append(cur.copy())
        out.append(np.asarray(verts))
    return out
CUBE_TETS=cube_tets()

def build_chart_points(n,sign,shrink,terminal_depth,report_every,qlo,qhi,interp_error,candidate_boundary_iterations,candidate_root_iterations):
    grid=np.linspace(-1.0,1.0,n+1)
    cap_dirs=np.empty(((n+1)**3,4));idx=0
    for x in grid:
      for y in grid:
       for z in grid:
        cap_dirs[idx]=direction_cap((x,y,z),sign);idx+=1
    ac,tris=triangle_grid(n);equi_dirs=np.empty(((n+1)*len(ac),4));idx=0
    for b in np.linspace(0.0,1.0,n+1):
      for a,c in ac:
        equi_dirs[idx]=direction_equi(a,c,b,sign);idx+=1
    dirs=np.vstack([cap_dirs,equi_dirs])
    visited=terminal=0;worst=None;witnesses=[]
    if sign>0:
        pts,slack=positive_points_batch(dirs,shrink,qlo,qhi,interp_error,candidate_boundary_iterations,candidate_root_iterations);worst=float(np.min(slack))
    else:
        pts,ok,etas,margins=negative_points_batch(dirs)
        if not np.all(ok):
            bad=int(np.flatnonzero(ok==0)[0]);raise RuntimeError(f'negative batch certificate failed at {bad}, eta={etas[bad]}, margin={margins[bad]}')
        if report_every:print(f'  certified negative vertices: {len(dirs)}/{len(dirs)} (certified batch)',flush=True)
    cap=pts[:len(cap_dirs)].reshape(n+1,n+1,n+1,4)
    equi=pts[len(cap_dirs):].reshape(n+1,len(ac),4)
    return cap,equi,tris,visited,terminal,worst,witnesses


PERM_ARRAY=np.asarray(PERMS4,dtype=np.int64)
PERM_SIGN=np.asarray([(-1.0 if sum(1 for i in range(4) for j in range(i+1,4) if p[i]>p[j])&1 else 1.0) for p in PERMS4],dtype=np.float64)
@njit(inline="always")
def det_abs_lower_float(M):
    lo=hi=0.0
    for k in range(24):
        tl=th=PERM_SIGN[k]
        for i in range(4):
            x=M[i,PERM_ARRAY[k,i]]
            tl,th=neg_backend.imul(tl,th,x,x)
        lo,hi=neg_backend.iadd(lo,hi,tl,th)
    if lo>0.0:return lo
    if hi<0.0:return -hi
    return 0.0
@njit(parallel=True)
def det_batch_lower(mats):
    out=np.empty(len(mats))
    for i in prange(len(mats)):out[i]=det_abs_lower_float(mats[i])/24.0
    return out

def _sum_mats(mats):
    if len(mats)==0:return 0.0,0.0,0
    mats=np.asarray(mats,dtype=np.float64)
    lows=det_batch_lower(mats)
    flt=np.abs(np.linalg.det(mats))/24.0
    return float(np.sum(lows)),float(np.sum(flt)),int(np.sum(lows==0.0))

def sum_chart_volumes(cap,equi,tris,n):
    cap_lo=cap_fl=eq_lo=eq_fl=0.0;zero=0
    # Cap: process each Freudenthal family as one batch.
    base=np.indices((n,n,n)).reshape(3,-1).T
    for offs in CUBE_TETS:
        cols=[]
        for off in offs:
            ind=base+off;cols.append(cap[ind[:,0],ind[:,1],ind[:,2]])
        mats=np.stack(cols,axis=2)
        lo,fl,z=_sum_mats(mats);cap_lo+=lo;cap_fl+=fl;zero+=z
    # Equatorial triangular prisms, three fixed tetrahedra per prism.
    for kb in range(n):
        bot=equi[kb];top=equi[kb+1]
        A=bot[tris[:,0]];B=bot[tris[:,1]];C=bot[tris[:,2]]
        Ap=top[tris[:,0]];Bp=top[tris[:,1]];Cp=top[tris[:,2]]
        for mats in (np.stack((A,B,C,Cp),axis=2),np.stack((A,B,Bp,Cp),axis=2),np.stack((A,Ap,Bp,Cp),axis=2)):
            lo,fl,z=_sum_mats(mats);eq_lo+=lo;eq_fl+=fl;zero+=z
    return mp.mpf(cap_lo)+24*mp.mpf(eq_lo),cap_fl+24*eq_fl,mp.mpf(cap_lo),24*mp.mpf(eq_lo),zero

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=4);ap.add_argument('--shrink',type=float,default=1-1e-7);ap.add_argument('--interval-dps',type=int,default=50);ap.add_argument('--terminal-depth',type=int,default=40);ap.add_argument('--arc-grid',type=int,default=2048);ap.add_argument('--candidate-boundary-iterations',type=int,default=24);ap.add_argument('--candidate-root-iterations',type=int,default=24);ap.add_argument('--report-every',type=int,default=250);ap.add_argument('--json',type=Path,default=HERE/'raised_u3_certified_simplicial_lower.json');args=ap.parse_args()
    mp.mp.dps=args.interval_dps;mp.iv.dps=args.interval_dps
    print('script SHA-256:',sha256_file(Path(__file__).resolve()));print('backend SHA-256:',sha256_file(Path(backend.__file__).resolve()));print('arc certifier SHA-256:',sha256_file(Path(arc_cert.__file__).resolve()));print('negative witness backend SHA-256:',sha256_file(Path(neg_backend.__file__).resolve()));print('mesh n:',args.n);print('building rigorous arc grid:',args.arc_grid,flush=True)
    qlo,qhi,qmid=upper_backend.build_arc_grid(args.arc_grid)
    delta=math.pi/(2*args.arc_grid);interp_error=np.nextafter(3.0*delta*delta/8.0,np.inf)
    print('continuous-arc interpolation error in squared distance:',format(interp_error,'.3e'))
    total=mp.mpf(0);report={}
    for sign,name in ((1,'positive'),(-1,'negative_inner')):
        print(f'building and certifying {name} chart vertices...',flush=True);start=time.time();cap,equi,tris,visited,terminal,worst,wit=build_chart_points(args.n,sign,args.shrink,args.terminal_depth,args.report_every,qlo,qhi,interp_error,args.candidate_boundary_iterations,args.candidate_root_iterations)
        print(f'summing {name} interval simplex volumes...',flush=True);lo,flt,caplo,eqlo,zero=sum_chart_volumes(cap,equi,tris,args.n);total+=lo
        report[name]={'certified_lower':mp.nstr(lo,30),'floating_same_mesh':format(flt,'.17g'),'cap_lower':mp.nstr(caplo,30),'equatorial_lower':mp.nstr(eqlo,30),'vertex_count':int(cap.size//4+equi.size//4),'simplex_count':int(6*args.n**3+3*len(tris)*args.n),'zero_sign_determinants':int(zero),'interval_arc_boxes':int(visited),'terminal_arc_boxes':int(terminal),'worst_positive_slack_lower':None if worst is None else format(float(worst),'.17g'),'elapsed_seconds':time.time()-start}
        print(f'{name} certified lower = {mp.nstr(lo,20)}; floating same mesh = {flt:.15f}')
    report.update({'statement':'The reported total is a strict computer-assisted lower bound for Vol_4(canonical raised U3).','proof_type':'explicit simplicial fan + interval point membership + interval determinants','mesh_n':args.n,'positive_shrink_binary64':format(np.float64(args.shrink),'.17g'),'interval_dps':args.interval_dps,'terminal_depth':args.terminal_depth,'candidate_boundary_iterations':args.candidate_boundary_iterations,'candidate_root_iterations':args.candidate_root_iterations,'total_certified_lower':mp.nstr(total,40),'script_sha256':sha256_file(Path(__file__).resolve())})
    args.json.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n=== certified simplicial lower-bound report ===');print('total strict lower:',mp.nstr(total,40));print('JSON:',args.json)
if __name__=='__main__':main()

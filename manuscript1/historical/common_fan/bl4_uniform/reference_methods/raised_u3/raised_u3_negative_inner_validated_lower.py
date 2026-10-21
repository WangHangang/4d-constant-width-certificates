#!/usr/bin/env python3
r"""Validated lower cubature for the negative inner model of raised U3.

For T=conv{A,B,C,D}, the rigorous inner model is

    L_- = {x_4<=0} intersect union_{m in T} B_4(m, eta(m)),
    eta(m)=1-sqrt(5/8+||m||^2).

On each S4 chamber parameter box a floating-point search chooses one m in T.
The proof uses only the explicitly checked inclusion m in T and a downward-
rounded radius eta.  The selected ball is contained in L_-, hence its radial
function is a pointwise lower bound.  Tensor 2x2x2 Gauss quadrature is combined
with outward-rounded pure fourth-derivative bounds; the certified contribution
is Gauss_lower - remainder_upper.
"""
from __future__ import annotations
import argparse, hashlib, heapq, math
from pathlib import Path
import numpy as np
from numba import njit, prange

HERE=Path(__file__).resolve().parent
GL=1.0/math.sqrt(3.0)
VERTEX_SIGNS=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=np.float64)
S0=1.0/(2.0*math.sqrt(2.0))
TETRA=S0*VERTEX_SIGNS

# face data for metric projection to T
_FACE_P0=np.zeros((15,3)); _FACE_Q=np.zeros((15,3,3)); _FACE_A=np.zeros((15,4,3)); _FACE_B=np.zeros((15,4)); _FACE_K=np.zeros(15,dtype=np.int64)
fi=0
for mask in range(1,16):
    ids=np.asarray([i for i in range(4) if (mask>>i)&1],dtype=np.int64)
    p0=TETRA[ids[0]]; k=len(ids)
    if k==1:
        projector=np.zeros((3,3)); bary_A=np.zeros((1,3)); bary_B=np.ones(1)
    else:
        E=(TETRA[ids[1:]]-p0).T; G=np.linalg.inv(E.T@E); projector=E@G@E.T; other=G@E.T
        bary_A=np.empty((k,3)); bary_B=np.empty(k)
        bary_A[1:]=other; bary_B[1:]=-other@p0
        bary_A[0]=-np.sum(other,axis=0); bary_B[0]=1.0+np.sum(other@p0)
    _FACE_P0[fi]=p0; _FACE_Q[fi]=projector; _FACE_A[fi,:k]=bary_A; _FACE_B[fi,:k]=bary_B; _FACE_K[fi]=k; fi+=1

def sha256_file(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()

@njit(inline='always')
def dn(x): return np.nextafter(x,-np.inf)
@njit(inline='always')
def up(x): return np.nextafter(x,np.inf)
@njit(inline='always')
def iadd(al,ah,bl,bh): return dn(al+bl),up(ah+bh)
@njit(inline='always')
def isub(al,ah,bl,bh): return dn(al-bh),up(ah-bl)
@njit(inline='always')
def imul(al,ah,bl,bh):
    p1,p2,p3,p4=al*bl,al*bh,ah*bl,ah*bh
    return dn(min(p1,p2,p3,p4)),up(max(p1,p2,p3,p4))
@njit(inline='always')
def isqr(al,ah):
    if al<=0<=ah: return 0.0,up(max(al*al,ah*ah))
    return max(0.0,dn(min(al*al,ah*ah))),up(max(al*al,ah*ah))
@njit(inline='always')
def irecip(al,ah): return dn(1.0/ah),up(1.0/al)
@njit(inline='always')
def idiv(al,ah,bl,bh):
    rl,rh=irecip(bl,bh); return imul(al,ah,rl,rh)
@njit(inline='always')
def isqrt(al,ah):
    if al<0: al=0.0
    return max(0.0,dn(math.sqrt(al))),up(math.sqrt(max(0.0,ah)))

@njit
def jconst(lo,hi):
    L=np.zeros(5);H=np.zeros(5);L[0]=lo;H[0]=hi;return L,H
@njit
def jvar(lo,hi):
    L=np.zeros(5);H=np.zeros(5);L[0]=lo;H[0]=hi;L[1]=H[1]=1.0;return L,H
@njit
def jadd(AL,AH,BL,BH):
    L=np.empty(5);H=np.empty(5)
    for n in range(5): L[n],H[n]=iadd(AL[n],AH[n],BL[n],BH[n])
    return L,H
@njit
def jsub(AL,AH,BL,BH):
    L=np.empty(5);H=np.empty(5)
    for n in range(5): L[n],H[n]=isub(AL[n],AH[n],BL[n],BH[n])
    return L,H
@njit
def jmul(AL,AH,BL,BH):
    L=np.empty(5);H=np.empty(5)
    for n in range(5):
        sl=sh=0.0
        for k in range(n+1):
            pl,ph=imul(AL[k],AH[k],BL[n-k],BH[n-k]);sl,sh=iadd(sl,sh,pl,ph)
        L[n]=sl;H[n]=sh
    return L,H
@njit
def jsqr(AL,AH):
    L,H=jmul(AL,AH,AL,AH);L[0],H[0]=isqr(AL[0],AH[0]);return L,H
@njit
def jrecip(AL,AH):
    L=np.empty(5);H=np.empty(5);L[0],H[0]=irecip(AL[0],AH[0])
    for n in range(1,5):
        sl=sh=0.0
        for k in range(1,n+1):
            pl,ph=imul(AL[k],AH[k],L[n-k],H[n-k]);sl,sh=iadd(sl,sh,pl,ph)
        pl,ph=imul(L[0],H[0],sl,sh);L[n],H[n]=-ph,-pl
    return L,H
@njit
def jdiv(AL,AH,BL,BH):
    RL,RH=jrecip(BL,BH);return jmul(AL,AH,RL,RH)
@njit
def jsqrt(AL,AH):
    L=np.empty(5);H=np.empty(5);L[0],H[0]=isqrt(AL[0],AH[0])
    if L[0]<=0.0: # caller will mark invalid
        for n in range(1,5): L[n],H[n]=-np.inf,np.inf
        return L,H
    dyl,dyh=imul(2.0,2.0,L[0],H[0])
    for n in range(1,5):
        sl=sh=0.0
        for k in range(1,n):
            pl,ph=imul(L[k],H[k],L[n-k],H[n-k]);sl,sh=iadd(sl,sh,pl,ph)
        nl,nh=isub(AL[n],AH[n],sl,sh);L[n],H[n]=idiv(nl,nh,dyl,dyh)
    return L,H

@njit(inline='always')
def direction(a,s,q):
    D=1+a*a*(1+s*s); den=1+q*q; horizontal=2*q/den; scale=horizontal/math.sqrt(D)
    u=np.empty(4);u[0]=scale;u[1]=a*scale;u[2]=a*s*scale;u[3]=-(1-q*q)/den;return u

@njit(inline='always')
def distance2_to_tetra(z):
    best=1e300
    for f in range(15):
        k=_FACE_K[f]; feasible=True
        for j in range(k):
            lam=_FACE_B[f,j]+_FACE_A[f,j,0]*z[0]+_FACE_A[f,j,1]*z[1]+_FACE_A[f,j,2]*z[2]
            if lam < -2e-13: feasible=False; break
        if not feasible: continue
        d0=z[0]-_FACE_P0[f,0];d1=z[1]-_FACE_P0[f,1];d2=z[2]-_FACE_P0[f,2]
        q0=_FACE_Q[f,0,0]*d0+_FACE_Q[f,0,1]*d1+_FACE_Q[f,0,2]*d2
        q1=_FACE_Q[f,1,0]*d0+_FACE_Q[f,1,1]*d1+_FACE_Q[f,1,2]*d2
        q2=_FACE_Q[f,2,0]*d0+_FACE_Q[f,2,1]*d1+_FACE_Q[f,2,2]*d2
        val=(d0-q0)**2+(d1-q1)**2+(d2-q2)**2
        if val<best: best=val
    return best
@njit(inline='always')
def project_tetra(z):
    best=1e300;pbest=np.zeros(3)
    for f in range(15):
        k=_FACE_K[f]; feasible=True
        for j in range(k):
            lam=_FACE_B[f,j]+_FACE_A[f,j,0]*z[0]+_FACE_A[f,j,1]*z[1]+_FACE_A[f,j,2]*z[2]
            if lam < -2e-13: feasible=False;break
        if not feasible: continue
        d0=z[0]-_FACE_P0[f,0];d1=z[1]-_FACE_P0[f,1];d2=z[2]-_FACE_P0[f,2]
        q0=_FACE_Q[f,0,0]*d0+_FACE_Q[f,0,1]*d1+_FACE_Q[f,0,2]*d2
        q1=_FACE_Q[f,1,0]*d0+_FACE_Q[f,1,1]*d1+_FACE_Q[f,1,2]*d2
        q2=_FACE_Q[f,2,0]*d0+_FACE_Q[f,2,1]*d1+_FACE_Q[f,2,2]*d2
        val=(d0-q0)**2+(d1-q1)**2+(d2-q2)**2
        if val<best:
            best=val;pbest[0]=_FACE_P0[f,0]+q0;pbest[1]=_FACE_P0[f,1]+q1;pbest[2]=_FACE_P0[f,2]+q2
    return pbest
@njit(inline='always')
def dual_value2(x,s):
    z=np.empty(3);z[0]=x[0]/s;z[1]=x[1]/s;z[2]=x[2]/s
    d2=distance2_to_tetra(z);n2=x[0]**2+x[1]**2+x[2]**2+x[3]**2
    return n2+.625*s+(s*s*d2+x[3]*x[3])/(s-1)
@njit(inline='always')
def farthest2(x):
    lo=1+1e-12;hi=8.0;g=(math.sqrt(5)-1)/2;c=hi-g*(hi-lo);d=lo+g*(hi-lo);fc=dual_value2(x,c);fd=dual_value2(x,d)
    for _ in range(64):
        if fc<fd: hi,d,fd=d,c,fc;c=hi-g*(hi-lo);fc=dual_value2(x,c)
        else: lo,c,fc=c,d,fd;d=lo+g*(hi-lo);fd=dual_value2(x,d)
    return min(fc,fd)
@njit(inline='always')
def candidate_ball(u):
    lo=0.0;hi=1.0;x=np.empty(4)
    for _ in range(52):
        r=.5*(lo+hi)
        for j in range(4): x[j]=r*u[j]
        if farthest2(x)<=1: lo=r
        else: hi=r
    rho=lo
    for j in range(4): x[j]=rho*u[j]
    slo=1+1e-12;shi=8.0;g=(math.sqrt(5)-1)/2;c=shi-g*(shi-slo);d=slo+g*(shi-slo);fc=dual_value2(x,c);fd=dual_value2(x,d)
    for _ in range(64):
        if fc<fd: shi,d,fd=d,c,fc;c=shi-g*(shi-slo);fc=dual_value2(x,c)
        else: slo,c,fc=c,d,fd;d=slo+g*(shi-slo);fd=dual_value2(x,d)
    ss=.5*(slo+shi);z=np.empty(3);z[0]=x[0]/ss;z[1]=x[1]/ss;z[2]=x[2]/ss
    m=project_tetra(z);m*=1.0-1e-11
    n2=m[0]**2+m[1]**2+m[2]**2
    eta=dn(1.0-up(math.sqrt(up(.625+up(n2)))))
    if eta<0: eta=0.0
    return m,eta,rho

@njit(parallel=True)
def assign_balls(bounds):
    n=len(bounds);centers=np.empty((n,4));radii=np.empty(n);ideal=np.empty(n);valid=np.ones(n,dtype=np.uint8)
    for b in prange(n):
        a=.5*(bounds[b,0,0]+bounds[b,0,1]);s=.5*(bounds[b,1,0]+bounds[b,1,1]);q=.5*(bounds[b,2,0]+bounds[b,2,1]);u=direction(a,s,q)
        m,e,r=candidate_ball(u);centers[b,0]=m[0];centers[b,1]=m[1];centers[b,2]=m[2];centers[b,3]=0.0;radii[b]=e;ideal[b]=r
        # exact tetra halfspace test with outward error
        for i in range(4):
            dot=VERTEX_SIGNS[i,0]*S0*m[0]+VERTEX_SIGNS[i,1]*S0*m[1]+VERTEX_SIGNS[i,2]*S0*m[2]
            if dn(dot)<-0.125: valid[b]=0
    return centers,radii,ideal,valid

@njit
def integrand_jet(bounds,c,radius,diff_axis):
    jetsL=np.empty((3,5));jetsH=np.empty((3,5))
    for j in range(3):
        lo=np.nextafter(bounds[j,0],-np.inf);hi=np.nextafter(bounds[j,1],np.inf)
        if j==diff_axis: L,H=jvar(lo,hi)
        else: L,H=jconst(lo,hi)
        jetsL[j]=L;jetsH[j]=H
    AL,AH=jetsL[0],jetsH[0];SL,SH=jetsL[1],jetsH[1];QL,QH=jetsL[2],jetsH[2]
    A2L,A2H=jsqr(AL,AH);S2L,S2H=jsqr(SL,SH);oneS2L,oneS2H=jadd(*jconst(1,1),S2L,S2H);tmpL,tmpH=jmul(A2L,A2H,oneS2L,oneS2H);DL,DH=jadd(*jconst(1,1),tmpL,tmpH);sqrtDL,sqrtDH=jsqrt(DL,DH)
    Q2L,Q2H=jsqr(QL,QH);denL,denH=jadd(*jconst(1,1),Q2L,Q2H);twoQL,twoQH=jmul(*jconst(2,2),QL,QH);hL,hH=jdiv(twoQL,twoQH,denL,denH);omL,omH=jsub(*jconst(1,1),Q2L,Q2H);vL,vH=jdiv(omL,omH,denL,denH);vL,vH=-vH,-vL
    facL,facH=jdiv(hL,hH,sqrtDL,sqrtDH);U0L,U0H=facL,facH;U1L,U1H=jmul(facL,facH,AL,AH);ASL,ASH=jmul(AL,AH,SL,SH);U2L,U2H=jmul(facL,facH,ASL,ASH)
    LL,LH=jconst(0,0);N2L,N2H=jconst(0,0)
    for j,(UL,UH) in enumerate(((U0L,U0H),(U1L,U1H),(U2L,U2H),(vL,vH))):
        CL,CH=jconst(c[j],c[j]);xL,xH=jmul(CL,CH,UL,UH);LL,LH=jadd(LL,LH,xL,xH);xL,xH=jsqr(CL,CH);N2L,N2H=jadd(N2L,N2H,xL,xH)
    r2lo,r2hi=isqr(radius,radius);deltaL,deltaH=jsub(*jconst(r2lo,r2hi),N2L,N2H);L2L,L2H=jsqr(LL,LH);radL,radH=jadd(L2L,L2H,deltaL,deltaH)
    if radL[0]<=0: return np.zeros(5),np.zeros(5),False
    sqL,sqH=jsqrt(radL,radH);rootL,rootH=jadd(LL,LH,sqL,sqH)
    if rootL[0]<=0: return np.zeros(5),np.zeros(5),False
    r2L,r2H=jsqr(rootL,rootH);r4L,r4H=jsqr(r2L,r2H)
    D32L,D32H=jmul(DL,DH,*jsqrt(DL,DH));den2L,den2H=jsqr(denL,denH);den3L,den3H=jmul(den2L,den2H,denL,denH);aq2L,aq2H=jmul(AL,AH,Q2L,Q2H);jdL,jdH=jmul(D32L,D32H,den3L,den3H);jacL,jacH=jdiv(aq2L,aq2H,jdL,jdH);FL,FH=jmul(r4L,r4H,jacL,jacH);FL,FH=jmul(*jconst(48,48),FL,FH);return FL,FH,True

@njit
def integrand_point(a,s,q,c,radius):
    u=direction(a,s,q);dot=0.0;n2=0.0
    for j in range(4): dot+=c[j]*u[j];n2+=c[j]*c[j]
    dl,dh=isub(radius*radius,radius*radius,n2,n2);l2,h2=isqr(dot,dot);rl,rh=iadd(l2,h2,dl,dh)
    if rh<=0: return 0.0,0.0
    rl,rh=isqrt(rl,rh);rootl,rooth=iadd(dot,dot,rl,rh)
    if rooth<=0:return 0.0,0.0
    rootl=max(0.0,rootl);r2l,r2h=isqr(rootl,rooth);r4l,r4h=isqr(r2l,r2h);D=1+a*a*(1+s*s);den=1+q*q;J=48*a*q*q/(D*math.sqrt(D)*den**3);return imul(J,J,r4l,r4h)

@njit(parallel=True)
def eval_boxes(bounds,centers,radii):
    n=len(bounds);lbs=np.empty(n);prox=np.empty(n);valid=np.ones(n,dtype=np.uint8)
    for b in prange(n):
        mid=np.empty(3);half=np.empty(3);length=np.empty(3)
        for j in range(3): mid[j]=.5*(bounds[b,j,0]+bounds[b,j,1]);length[j]=bounds[b,j,1]-bounds[b,j,0];half[j]=.5*length[j]
        qlo=qhi=0.0
        for mask in range(8):
            x=np.empty(3)
            for j in range(3): x[j]=mid[j]+half[j]*GL*(1 if ((mask>>j)&1) else -1)
            fl,fh=integrand_point(x[0],x[1],x[2],centers[b],radii[b]);qlo=dn(qlo+fl);qhi=up(qhi+fh)
        w=half[0]*half[1]*half[2];quadlo=dn(qlo*dn(w));quadhi=up(qhi*up(w));M=np.empty(3);ok=True
        for j in range(3):
            FL,FH,good=integrand_jet(bounds[b],centers[b],radii[b],j)
            if not good: ok=False;M[j]=np.inf
            else:M[j]=up(24*max(abs(FL[4]),abs(FH[4])))
        if ok:
            la,ls,lq=length[0],length[1],length[2];rem=up((ls*lq*la**5*M[0]+la*lq*ls**5*M[1]+la*ls*lq**5*M[2])/4320.0);lbs[b]=max(0.0,dn(quadlo-rem))
        else: lbs[b]=0.0;valid[b]=0
        prox[b]=quadhi
    return lbs,prox,valid

def make_boxes(na,ns,nq):
    ga=np.linspace(0,1,na+1);gs=np.linspace(-1,1,ns+1);gq=np.linspace(0,1,nq+1);return [np.array([[ga[i],ga[i+1]],[gs[j],gs[j+1]],[gq[k],gq[k+1]]],float) for i in range(na) for j in range(ns) for k in range(nq)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--na',type=int,default=4);ap.add_argument('--ns',type=int,default=8);ap.add_argument('--nq',type=int,default=8);ap.add_argument('--batch',type=int,default=2048);ap.add_argument('--max-boxes',type=int,default=160000);ap.add_argument('--target',type=float,default=0.05298);ap.add_argument('--report-every',type=int,default=5);args=ap.parse_args()
    print('script SHA-256:',sha256_file(Path(__file__).resolve()));records=make_boxes(args.na,args.ns,args.nq);bounds=np.asarray(records);centers,radii,ideal,cv=assign_balls(bounds)
    if not np.all(cv):raise RuntimeError('candidate outside T')
    lb,pr,v=eval_boxes(bounds,centers,radii);total=float(np.sum(lb));proxy=float(np.sum(pr));heap=[];uid=0
    for box,l,p,ok in zip(records,lb,pr,v):heapq.heappush(heap,(-(float(p)-float(l)+(0 if ok else 1.0)),uid,box,float(l),float(p)));uid+=1
    it=0
    print(f'initial boxes={len(heap)} lower={total:.15f} selected-ball-proxy={proxy:.15f}')
    while len(heap)<args.max_boxes and total<args.target:
        picked=[heapq.heappop(heap) for _ in range(min(args.batch,len(heap)))];children=[];oldl=oldp=0.0
        for _,_,box,l,p in picked:
            oldl+=l;oldp+=p;length=box[:,1]-box[:,0];axis=int(np.argmax(length));mid=.5*(box[axis,0]+box[axis,1]);b1=box.copy();b2=box.copy();b1[axis,1]=mid;b2[axis,0]=mid;children.extend((b1,b2))
        cb=np.asarray(children);cc,rr,ii,cv=assign_balls(cb)
        if not np.all(cv):raise RuntimeError('candidate outside T')
        nl,npv,nvalid=eval_boxes(cb,cc,rr);total+=float(np.sum(nl))-oldl;proxy+=float(np.sum(npv))-oldp
        for box,l,p,ok in zip(children,nl,npv,nvalid):heapq.heappush(heap,(-(float(p)-float(l)+(0 if ok else 1.0)),uid,box,float(l),float(p)));uid+=1
        it+=1
        if it%args.report_every==0 or total>=args.target:print(f'iter={it:4d} boxes={len(heap):7d} lower={total:.15f} selected-ball-proxy={proxy:.15f} slack={proxy-total:.3e}',flush=True)
    print('\n=== validated negative-inner lower report ===');print('boxes:',len(heap));print(f'validated lower: {total:.15f}');print(f'selected-ball Gauss upper proxy: {proxy:.15f}');print('target reached:',total>=args.target)
if __name__=='__main__':main()

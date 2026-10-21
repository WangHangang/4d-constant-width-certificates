#!/usr/bin/env python3
"""Geometry-derived numerical volume for the direct AMO 4D peabody.

Arelio--Montejano--Oliveros, Mathematika 71 (2025), e70014.
Evidence: floating deterministic quadrature + randomized Sobol QMC; NOT certified.
"""
import argparse, math, time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import null_space
from scipy.stats import qmc

rt10=math.sqrt(10.0); a=math.sqrt(1.5); b=1/math.sqrt(2)
x0=math.sqrt((41-4*rt10)/27); y0=math.sqrt((7-2*rt10)/27)
x1=math.sqrt((11+2*rt10)/12); z1=math.sqrt((7-2*rt10)/36); WIDTH=2*z1
g=(2*x1+3*x0)/5
sH=math.hypot(x0-a,y0); sE=math.hypot(x1-1,z1); s1=math.acos(x1/a)
P=np.array([[x1,0,z1,0],[x1,0,-z1,0],[x0,y0,0,0],
 [x0,-y0/2,0,-math.sqrt(3)*y0/2],[x0,-y0/2,0,math.sqrt(3)*y0/2]],float)
G=P.mean(0); PC=P-G; p1=P[0]
fh=np.array([a,0,0,0.]); fe=np.array([1.,0,0,0.])
R0=np.linalg.norm(p1-G)
T3=np.array([y0,0.]); T4=np.array([-y0/2,-math.sqrt(3)*y0/2]); T5=np.array([-y0/2,math.sqrt(3)*y0/2])
TRI_AREA=abs(np.linalg.det(np.stack([T4-T3,T5-T3],axis=1)))/2; D0=x0-g
N0=(G-p1)/R0; BCHART=null_space(N0.reshape(1,4))

def perm_matrix(perm):
    return PC[np.asarray(perm[:4])].T @ np.linalg.inv(PC[:4].T)
CAP_MAPS=[]
for j in (1,2,3,4):
    p=list(range(5)); p[1],p[j]=p[j],p[1]; CAP_MAPS.append(perm_matrix(p))

def lam_h(Y,W):
    A=D0*D0-2*(Y*Y+W*W); Bq=2*g*D0; Cq=g*g-1
    disc=Bq*Bq-4*A*Cq; sd=np.sqrt(np.maximum(disc,0))
    r1=(-Bq+sd)/(2*A); r2=(-Bq-sd)/(2*A)
    return np.where((r1>0)&((r1<r2)|(r2<=0)),r1,r2)

def hdata(Y,W):
    L=lam_h(Y,W); X=g+D0*L; r2=Y*Y+W*W
    h=np.column_stack([X,L*Y,np.zeros_like(L),L*W])
    Fl=2*D0*X-4*L*r2; LY=4*L*L*Y/Fl; LW=4*L*L*W/Fl
    hY=np.column_stack([D0*LY,L+LY*Y,np.zeros_like(L),LY*W])
    hW=np.column_stack([D0*LW,LW*Y,np.zeros_like(L),L+LW*W])
    return h,hY,hW,L

def edata(s):
    cs=np.cos(s); ss=np.sin(s)
    y=np.column_stack([a*cs,np.zeros_like(s),b*ss,np.zeros_like(s)])
    ys=np.column_stack([-a*ss,np.zeros_like(s),b*cs,np.zeros_like(s)])
    return y,ys

def jac(t1,t2,t3):
    T=(t1,t2,t3); gram=np.empty((len(t1),3,3))
    for i in range(3):
        for j in range(3): gram[:,i,j]=np.einsum('ij,ij->i',T[i],T[j])
    return np.sqrt(np.maximum(np.linalg.det(gram),0))

def wedge_integrands(Y,W,s,checks=False):
    h,hY,hW,L=hdata(Y,W); y,ys=edata(s)
    hv=h-fh; hd=np.linalg.norm(hv,axis=1); Rx=sH-hd
    RxY=-np.einsum('ij,ij->i',hv,hY)/hd; RxW=-np.einsum('ij,ij->i',hv,hW)/hd
    ev=y-fe; ed=np.linalg.norm(ev,axis=1); Ry=sE-ed; Rys=-np.einsum('ij,ij->i',ev,ys)/ed
    diff=y-h; d=np.linalg.norm(diff,axis=1); u=diff/d[:,None]
    def du(delta):
        ud=np.einsum('ij,ij->i',u,delta); return (delta-u*ud[:,None])/d[:,None]
    uY,uW,us=du(-hY),du(-hW),du(ys)
    pt=h-Rx[:,None]*u; pp=y+Ry[:,None]*u
    tY=hY-RxY[:,None]*u-Rx[:,None]*uY; tW=hW-RxW[:,None]*u-Rx[:,None]*uW; ts=-Rx[:,None]*us
    pY=Ry[:,None]*uY; pW=Ry[:,None]*uW; ps=ys+Rys[:,None]*u+Ry[:,None]*us
    Jt,Jp=jac(tY,tW,ts),jac(pY,pW,ps)
    ft=np.einsum('ij,ij->i',pt-G,-u)*Jt; fp=np.einsum('ij,ij->i',pp-G,u)*Jp
    if not checks: return ft,fp
    ck=dict(width_err=float(np.max(np.abs(d+Rx+Ry-WIDTH))),
      H_err=float(np.max(np.abs(h[:,0]**2-2*(h[:,1]**2+h[:,3]**2)-1))),
      E_err=float(np.max(np.abs(y[:,0]**2/1.5+y[:,2]**2/0.5-1))),
      min_Rx=float(Rx.min()),min_Ry=float(Ry.min()),max_Rx=float(Rx.max()),max_Ry=float(Ry.max()),
      tan_t=float(max(np.max(abs(np.einsum('ij,ij->i',tY,u))),np.max(abs(np.einsum('ij,ij->i',tW,u))),np.max(abs(np.einsum('ij,ij->i',ts,u))))),
      tan_p=float(max(np.max(abs(np.einsum('ij,ij->i',pY,u))),np.max(abs(np.einsum('ij,ij->i',pW,u))),np.max(abs(np.einsum('ij,ij->i',ps,u))))),
      lam_min=float(L.min()),lam_max=float(L.max()))
    return ft,fp,ck

def wedge_gauss(n=20):
    xr,wr=leggauss(n); xt,wt=leggauss(n); xs,ws=leggauss(n)
    r=(xr+1)/2; wr/=2; t=(xt+1)/2; wt/=2; uu=(xs+1)/2; ws/=2
    R,T,U=np.meshgrid(r,t,uu,indexing='ij')
    YW=(1-R.ravel())[:,None]*T3 + R.ravel()[:,None]*((1-T.ravel())[:,None]*T4+T.ravel()[:,None]*T5)
    s=-s1+2*s1*U.ravel(); ft,fp,ck=wedge_integrands(YW[:,0],YW[:,1],s,True)
    WR,WT,WU=np.meshgrid(wr,wt,ws,indexing='ij')
    ww=(WR*WT*WU*R).ravel()*(2*TRI_AREA)*(2*s1)
    return float(ww@ft),float(ww@fp),ck

def Qcanon(d):
    dx,dy,dz,dw=d[:,0],d[:,1],d[:,2],d[:,3]
    return (x1*dz-z1*dx)**2-2*z1*z1*(dy*dy+dw*dw)-dz*dz

def capQ(q):
    d=N0+q@BCHART.T
    return np.stack([Qcanon(d@A) for A in CAP_MAPS],axis=1)
Q0=capQ(np.zeros((1,3)))[0]

def posroot(A,B,c):
    D=B*B-4*A*c; sd=np.sqrt(np.maximum(D,0)); out=np.full_like(A,np.inf)
    m=(D>=0)&(np.abs(A)>1e-14)
    if np.any(m):
        r1=(-B[m]-sd[m])/(2*A[m]); r2=(-B[m]+sd[m])/(2*A[m]); rr=np.stack([r1,r2],1); rr=np.where(rr>1e-13,rr,np.inf); out[m]=rr.min(1)
    ml=(D>=0)&(np.abs(A)<=1e-14)&(np.abs(B)>1e-14)
    if np.any(ml):
        rr=-c/B[ml]; out[ml]=np.where(rr>1e-13,rr,np.inf)
    return out

def rmax(u):
    qp,qm=capQ(u),capQ(-u); aa=(qp+qm)/2-Q0; bb=(qp-qm)/2
    roots=np.stack([posroot(aa[:,i],bb[:,i],Q0[i]) for i in range(4)],axis=1)
    return roots.min(1),roots.argmin(1)

def Frad(R):
    return WIDTH**3*(.5*WIDTH*(np.arctan(R)-R/(1+R*R))-(R0/3)*R**3/(1+R*R)**1.5)

def cap_qmc(power=22,reps=8,seed=20260823,chunk_power=18):
    total=1<<power; chunk=1<<min(power,chunk_power); ans=[]
    for k in range(reps):
        sob=qmc.Sobol(2,scramble=True,seed=seed+k); acc=0.; done=0
        while done<total:
            n=min(chunk,total-done); U=sob.random(n); z=1-2*U[:,0]; ph=2*math.pi*U[:,1]; rr=np.sqrt(np.maximum(0,1-z*z))
            u=np.column_stack([rr*np.cos(ph),rr*np.sin(ph),z]); rm,_=rmax(u); acc+=Frad(rm).sum(); done+=n
        ans.append(4*math.pi*acc/total)
    return np.array(ans)

def cap_product(nz=256,nphi=1024):
    z,w=leggauss(nz); ph=(np.arange(nphi)+.5)*2*math.pi/nphi; acc=0.
    for zz,ww in zip(z,w):
        rr=math.sqrt(max(0,1-zz*zz)); u=np.column_stack([rr*np.cos(ph),rr*np.sin(ph),np.full(nphi,zz)]); rm,_=rmax(u); acc+=ww*Frad(rm).sum()
    return acc*2*math.pi/nphi

def geometry_checks():
    D=np.linalg.norm(P[:,None,:]-P[None,:,:],axis=2); mask=~np.eye(5,dtype=bool)
    edge=float(np.max(np.abs(D[mask]-WIDTH))); ortho=0.; verr=0.
    for j,A in zip((1,2,3,4),CAP_MAPS):
        ortho=max(ortho,float(np.max(abs(A.T@A-np.eye(4))))); perm=list(range(5)); perm[1],perm[j]=perm[j],perm[1]; verr=max(verr,float(np.max(abs(PC@A.T-PC[np.array(perm)]))))
    return edge,ortho,verr

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gauss',type=int,default=20); ap.add_argument('--cap-power',type=int,default=22); ap.add_argument('--cap-replicates',type=int,default=8); ap.add_argument('--chunk-power',type=int,default=18); ap.add_argument('--crosscheck',action='store_true'); args=ap.parse_args(); t0=time.time()
    print('=== AMO direct 4D peabody volume ==='); print('Evidence: numerical, NOT certified')
    print(f'width W = {WIDTH:.16g}; W^4 = {WIDTH**4:.16g}; barycenter G = {G}')
    edge,orth,ve=geometry_checks(); print(f'checks: edge_err={edge:.3e}, orth_err={orth:.3e}, perm_err={ve:.3e}')
    It,Ip,ck=wedge_gauss(args.gauss); print(f'I_triple={It:.15g}  I_pair={Ip:.15g}'); print('wedge checks:',ck)
    reps=cap_qmc(args.cap_power,args.cap_replicates,chunk_power=args.chunk_power); Ic=float(reps.mean()); sd=float(reps.std(ddof=1)) if len(reps)>1 else float('nan')
    print('I_cap reps:',np.array2string(reps,precision=15)); print(f'I_cap mean={Ic:.15g}, replicate_SD={sd:.3e}')
    if args.crosscheck:
        cc=cap_product(); print(f'I_cap product crosscheck={cc:.15g}, cross-QMC={cc-Ic:+.3e}')
    raw=.25*(10*Ip+10*It+5*Ic); vn=raw/WIDTH**4
    lo=0.2218338853720453; hi=0.2218382994641744
    print(f'V_source={raw:.15g}'); print(f'V_width_one={vn:.15g}')
    print(f'U4=[{lo:.16g},{hi:.16g}]'); print(f'AMO-U4_upper={vn-hi:+.15g}'); print(f'AMO-U4_lower={vn-lo:+.15g}')
    print('Numerical ordering only until a deterministic/interval enclosure is supplied.')
    print(f'elapsed={time.time()-t0:.2f}s')
if __name__=='__main__': main()

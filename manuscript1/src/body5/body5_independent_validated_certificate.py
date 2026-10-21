import math, time, sys, hashlib, inspect, os
from dataclasses import dataclass

# Validated interval arithmetic:
#   - binary64 +,-,*,/,sqrt rounded outward immediately with nextafter;
#   - atan and pi enclosed by an explicit alternating Taylor series after
#     the identity atan(x)=pi/4+atan((x-1)/(x+1)) for x>0.
# Grid endpoints are dyadic binary64 numbers.
NEG_INF=-math.inf; POS_INF=math.inf

def down(x): return math.nextafter(float(x), NEG_INF)
def up(x): return math.nextafter(float(x), POS_INF)

@dataclass(frozen=True, slots=True)
class I:
    lo: float
    hi: float
    @staticmethod
    def point(x):
        x=float(x); return I(x,x)
    def __add__(self,o):
        o=o if isinstance(o,I) else I.point(o)
        return I(down(self.lo+o.lo), up(self.hi+o.hi))
    __radd__=__add__
    def __neg__(self): return I(-self.hi,-self.lo)
    def __sub__(self,o): return self+(-(o if isinstance(o,I) else I.point(o)))
    def __rsub__(self,o): return I.point(o)-self
    def __mul__(self,o):
        o=o if isinstance(o,I) else I.point(o)
        vals=(self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi)
        return I(down(min(vals)),up(max(vals)))
    __rmul__=__mul__
    def inv(self):
        if self.lo <= 0.0 <= self.hi: raise ZeroDivisionError((self.lo,self.hi))
        vals=(1.0/self.lo,1.0/self.hi)
        return I(down(min(vals)),up(max(vals)))
    def __truediv__(self,o): return self*(o if isinstance(o,I) else I.point(o)).inv()
    def __rtruediv__(self,o): return I.point(o)*self.inv()
    def sqrt(self):
        if self.lo < 0.0:
            raise ValueError(("sqrt negative", self))
        # math.sqrt is only an initial guess.  The returned endpoints are
        # verified by exact rational squaring, so correctness does not depend
        # on a correct-rounding claim for libm sqrt.
        from fractions import Fraction
        lo = down(math.sqrt(self.lo))
        hi = up(math.sqrt(self.hi))
        xlo = Fraction(*self.lo.as_integer_ratio())
        xhi = Fraction(*self.hi.as_integer_ratio())
        while Fraction(*lo.as_integer_ratio())**2 > xlo:
            lo = down(lo)
        while Fraction(*hi.as_integer_ratio())**2 < xhi:
            hi = up(hi)
        return I(lo, hi)
    def atan(self):
        loI = atan_point_signed(self.lo)
        hiI = atan_point_signed(self.hi)
        return I(loI.lo, hiI.hi)
    def abs_upper(self): return max(abs(self.lo),abs(self.hi))
    def width(self): return self.hi-self.lo

ONE=I.point(1); TWO=I.point(2); THREE=I.point(3)

def atan_series_small_positive(T, n=55):
    # 0 <= T < 1.  Direct power-series enclosure plus a symmetric rigorous
    # remainder bound |R_n| <= T_max^(2n+3)/(2n+3).
    if T.lo < 0.0 or T.hi >= 1.0:
        raise ValueError(("atan series domain", T))
    t2=T*T; term=T; S=I.point(0)
    sign=1
    for k in range(n+1):
        add=term/(2*k+1)
        S = S+add if sign>0 else S-add
        term=term*t2; sign=-sign
    R=I.point(1)
    for _ in range(2*n+3):
        R=R*T
    R=R/(2*n+3)
    r=R.hi
    return S+I(-r,r)

def fraction_to_interval_early(fr):
    from fractions import Fraction
    if not isinstance(fr,Fraction): fr=Fraction(fr)
    f=float(fr); lo=down(f); hi=up(f)
    while Fraction(*lo.as_integer_ratio()) > fr: lo=down(lo)
    while Fraction(*hi.as_integer_ratio()) < fr: hi=up(hi)
    return I(lo,hi)

def atan_rational_exact_enclosure(num,den,K):
    from fractions import Fraction
    t=Fraction(num,den); S=Fraction(0)
    for k in range(K+1):
        S += ((-1)**k)*t**(2*k+1)/Fraction(2*k+1)
    nxt=t**(2*K+3)/Fraction(2*K+3)
    lo=min(S,S-nxt); hi=max(S,S-nxt)
    return I(fraction_to_interval_early(lo).lo,fraction_to_interval_early(hi).hi)

# Machin formula with exact-rational alternating-series enclosures.
PI=16*atan_rational_exact_enclosure(1,5,24)-4*atan_rational_exact_enclosure(1,239,6)

def atan_positive_point(x):
    # x is an exact positive binary64 endpoint.  Transform to |z|<0.762 on
    # the entire range used by the Body-5 certificate, then use the rigorous
    # alternating series.
    X=I.point(x)
    Z=(X-1)/(X+1)
    if Z.hi <= 0.0:
        T=I(-Z.hi,-Z.lo)
        A=atan_series_small_positive(T)
        return PI/4-A
    if Z.lo >= 0.0:
        A=atan_series_small_positive(Z)
        return PI/4+A
    # If outward rounding makes Z straddle zero, use |atan z| <= |z|.
    # Hence PI/4 + [Z.lo,Z.hi] is a valid (slightly conservative) enclosure.
    return PI/4 + I(Z.lo,Z.hi)

def atan_point_signed(x):
    if x == 0.0:
        return I.point(0.0)
    if x > 0.0:
        return atan_positive_point(x)
    a=atan_positive_point(-x)
    return I(-a.hi,-a.lo)

SQRT2=TWO.sqrt(); SQRT3=THREE.sqrt(); SQRT5=I.point(5).sqrt()

@dataclass(slots=True)
class J1:
    v:I; d:I; dd:I
    @staticmethod
    def const(c): return J1(c if isinstance(c,I) else I.point(c),I.point(0),I.point(0))
    @staticmethod
    def var(c): return J1(c if isinstance(c,I) else I.point(c),I.point(1),I.point(0))
    def __add__(self,o):
        o=o if isinstance(o,J1) else J1.const(o); return J1(self.v+o.v,self.d+o.d,self.dd+o.dd)
    __radd__=__add__
    def __neg__(self): return J1(-self.v,-self.d,-self.dd)
    def __sub__(self,o): return self+(-(o if isinstance(o,J1) else J1.const(o)))
    def __rsub__(self,o): return J1.const(o)-self
    def __mul__(self,o):
        o=o if isinstance(o,J1) else J1.const(o)
        return J1(self.v*o.v, self.d*o.v+self.v*o.d,
                  self.dd*o.v+2*self.d*o.d+self.v*o.dd)
    __rmul__=__mul__
    def unary(self,fv,fp,fpp): return J1(fv,fp*self.d,fpp*self.d*self.d+fp*self.dd)
    def inv(self):
        v=self.v; return self.unary(ONE/v,-ONE/(v*v),TWO/(v*v*v))
    def __truediv__(self,o): return self*(o if isinstance(o,J1) else J1.const(o)).inv()
    def __rtruediv__(self,o): return J1.const(o)*self.inv()
    def sqrt(self):
        s=self.v.sqrt(); return self.unary(s,ONE/(2*s),-ONE/(4*self.v*s))
    def atan(self):
        den=ONE+self.v*self.v; return self.unary(self.v.atan(),ONE/den,-TWO*self.v/(den*den))

@dataclass(slots=True)
class J2:
    v:I; x:I; y:I; xx:I; xy:I; yy:I
    @staticmethod
    def const(c):
        c=c if isinstance(c,I) else I.point(c); z=I.point(0); return J2(c,z,z,z,z,z)
    @staticmethod
    def varx(c):
        c=c if isinstance(c,I) else I.point(c); z=I.point(0); return J2(c,I.point(1),z,z,z,z)
    @staticmethod
    def vary(c):
        c=c if isinstance(c,I) else I.point(c); z=I.point(0); return J2(c,z,I.point(1),z,z,z)
    def __add__(self,o):
        o=o if isinstance(o,J2) else J2.const(o)
        return J2(self.v+o.v,self.x+o.x,self.y+o.y,self.xx+o.xx,self.xy+o.xy,self.yy+o.yy)
    __radd__=__add__
    def __neg__(self): return J2(-self.v,-self.x,-self.y,-self.xx,-self.xy,-self.yy)
    def __sub__(self,o): return self+(-(o if isinstance(o,J2) else J2.const(o)))
    def __rsub__(self,o): return J2.const(o)-self
    def __mul__(self,o):
        o=o if isinstance(o,J2) else J2.const(o)
        return J2(self.v*o.v,
                  self.x*o.v+self.v*o.x,
                  self.y*o.v+self.v*o.y,
                  self.xx*o.v+2*self.x*o.x+self.v*o.xx,
                  self.xy*o.v+self.x*o.y+self.y*o.x+self.v*o.xy,
                  self.yy*o.v+2*self.y*o.y+self.v*o.yy)
    __rmul__=__mul__
    def unary(self,fv,fp,fpp):
        return J2(fv,fp*self.x,fp*self.y,
                  fpp*self.x*self.x+fp*self.xx,
                  fpp*self.x*self.y+fp*self.xy,
                  fpp*self.y*self.y+fp*self.yy)
    def inv(self):
        v=self.v; return self.unary(ONE/v,-ONE/(v*v),TWO/(v*v*v))
    def __truediv__(self,o): return self*(o if isinstance(o,J2) else J2.const(o)).inv()
    def __rtruediv__(self,o): return J2.const(o)*self.inv()
    def sqrt(self):
        s=self.v.sqrt(); return self.unary(s,ONE/(2*s),-ONE/(4*self.v*s))
    def atan(self):
        den=ONE+self.v*self.v; return self.unary(self.v.atan(),ONE/den,-TWO*self.v/(den*den))

# transformed integrands
def f_face(x):
    sqrt5=J1.const(SQRT5); sqrt3=J1.const(SQRT3); sqrt6=J1.const(6).sqrt()
    phi=(1+sqrt5)/2; z0=-1/sqrt3; z1=(phi-1)/sqrt3; z=z0+(z1-z0)*x
    a=(1+sqrt3*z)/(sqrt6*(1-z*z).sqrt())
    asin_a=(a/(1-a*a).sqrt()).atan()
    omega=J1.const(PI/2)-2*asin_a
    r=sqrt3/2; J0=5*J1.const(PI)/12-3*sqrt3/4; Jc=J1.const(11)/12-sqrt3*J1.const(PI)/6
    return (z1-z0)*omega*(J0-(r+z/2)*Jc)

def f_apex_arc(x):
    sqrt3=J1.const(SQRT3); sqrt6=J1.const(6).sqrt(); sqrt30=J1.const(30).sqrt()
    q=sqrt3*x; R=(1+q*q).sqrt()/(1+sqrt3*q); C=1/(1+R*R/2).sqrt(); ra=(J1.const(2)/3).sqrt()
    A=(1-C*C)/2-ra*(1-C); B=(1-C*C*C)/3-ra*(1-C*C)/2
    beta=J1.const(15).sqrt().atan(); Ma=(J1.const(3)/5).sqrt()*(-5*sqrt6/48+sqrt30/16)
    return 6*(beta*A-(ra*beta+Ma)*B)*sqrt3/(1+q*q)

def f_base_arc(x):
    q=x; a=J1.const(SQRT2).inv(); c=(1+q)/(2*(1+q+q*q)).sqrt()
    def P(u): return u*u/2-a*u-a*u*u*u/3+a*a*u*u/2
    return J1.const(PI)*(P(c)-P(a))/(1+q*q)

def F_rho(r):
    s=1+r*r
    return (r.atan()-r/s)/2-r*r*r/(J2.const(3)*J2.const(SQRT2)*s*s.sqrt())

def f_base_vertex(x,y):
    sqrt3=J2.const(SQRT3); t=x/sqrt3; q0=t/(1-2*t*t).sqrt(); q=q0+(1-q0)*y
    rho_hi=1/((1+2*t*t).sqrt()-2*t)
    rho_lo=1/(t+(1-t*t).sqrt()*(1+q)/(1+q*q).sqrt())
    jac=(1/sqrt3)*(1-q0)/(1+q*q)
    return 6*jac*(F_rho(rho_hi)-F_rho(rho_lo))

# interval Taylor integration on dyadic grids
def d(k,N): return k/N  # exact binary for power-of-two N and integer k within range

def integrate1(fun,N):
    total=I.point(0)
    for i in range(N):
        a=d(i,N); b=d(i+1,N); m=(a+b)/2; h=(b-a)/2; w=b-a
        fc=fun(J1.var(I.point(m))).v
        F=fun(J1.var(I(a,b))); M=I.point(F.dd.abs_upper())
        err=I.point(w)*M*I.point(h)*I.point(h)/6
        total=total+I.point(w)*fc+I(-err.hi,err.hi)
    return total

def cell2(fun,ax,bx,ay,by):
    mx=(ax+bx)/2; my=(ay+by)/2; hx=(bx-ax)/2; hy=(by-ay)/2; area=(bx-ax)*(by-ay)
    fc=fun(J2.varx(I.point(mx)),J2.vary(I.point(my))).v
    F=fun(J2.varx(I(ax,bx)),J2.vary(I(ay,by)))
    Mxx=I.point(F.xx.abs_upper()); Mxy=I.point(F.xy.abs_upper()); Myy=I.point(F.yy.abs_upper())
    err=I.point(area)*(Mxx*I.point(hx*hx)/6+Myy*I.point(hy*hy)/6+Mxy*I.point(hx*hy)/4)
    val=I.point(area)*fc+I(-err.hi,err.hi)
    return val,err.hi

def integrate2_uniform(fun,N,progress=False):
    total=I.point(0)
    for i in range(N):
        if progress and i%max(1,N//8)==0: print(' row',i,'/',N,flush=True)
        ax=d(i,N); bx=d(i+1,N)
        for j in range(N):
            ay=d(j,N); by=d(j+1,N); v,_=cell2(fun,ax,bx,ay,by); total=total+v
    return total

# Apex E: plain inclusion on Duffy cubes; no atan
def apex_val(X,Y,Z):
    l1=X; l2=(1-X)*Y; l3=(1-X)*(1-Y)*Z; l4=(1-X)*(1-Y)*(1-Z)
    s=l1*l1+l2*l2+l3*l3+l4*l4; R=((1+s)/2).sqrt()
    integ=1/(R*R*R*R)-((5+SQRT5)/8)/(R*R*R*R*R); jac=(1-X)*(1-X)*(1-Y)
    return SQRT5/4*integ*jac

def integrate3_inclusion(N):
    total=I.point(0); w=1/N; vol=I.point(w*w*w)
    for i in range(N):
        X=I(d(i,N),d(i+1,N))
        for j in range(N):
            Y=I(d(j,N),d(j+1,N))
            for k in range(N):
                Z=I(d(k,N),d(k+1,N)); total=total+vol*apex_val(X,Y,Z)
    return total



# --- Exact rational moment certificate for the apex term I_E ---
def fraction_interval(fr):
    from fractions import Fraction
    if not isinstance(fr,Fraction): fr=Fraction(fr)
    f=float(fr)
    lo=down(f); hi=up(f)
    # Exact verification that the binary64 bracket contains the rational.
    while Fraction(*lo.as_integer_ratio()) > fr: lo=down(lo)
    while Fraction(*hi.as_integer_ratio()) < fr: hi=up(hi)
    return I(lo,hi)

def apex_exact_moment_certificate(K=20):
    from fractions import Fraction
    from math import factorial, comb
    # M_j exact via the manuscript's Dirichlet monomial formula.
    Ms=[]
    for j in range(K+2):
        sm=0
        for m1 in range(j+1):
            for m2 in range(j-m1+1):
                for m3 in range(j-m1-m2+1):
                    m4=j-m1-m2-m3
                    prod=1
                    for m in (m1,m2,m3,m4): prod *= factorial(2*m)//factorial(m)
                    sm += prod
        Ms.append(Fraction(factorial(j)*sm, factorial(2*j+3)))
    mus=[]
    for k in range(K+2):
        mu=Fraction(0)
        for j in range(k+1):
            mu += Fraction(comb(k,j)) * Fraction(-5,8)**(k-j) * Ms[j]
        mus.append(mu)
    a0=Fraction(13,8)
    S1=Fraction(0); S2=Fraction(0)
    poch=Fraction(1)
    for k in range(K+1):
        xk=mus[k]/(a0**k)
        S1 += ((-1)**k)*Fraction(k+1)*xk
        if k>0: poch *= Fraction(2*k+3,2)  # multiply previous by (k+3/2)? check below
        # Easier compute (5/2)_k/k! recurrence directly separately below.
    S2=Fraction(0); coeff=Fraction(1)
    for k in range(K+1):
        if k>0: coeff *= Fraction(2*k+3,2*k)  # ((5/2)+(k-1))/k=(2k+3)/(2k)
        S2 += ((-1)**k)*coeff*(mus[k]/(a0**k))
    q=Fraction(3,13); vol=Fraction(1,6)
    # Tail1 sum_{k=K+1}inf (k+1)q^k = q^n*((n+1)-n*q)/(1-q)^2, n=K+1
    n=K+1
    tail1=vol*(q**n)*Fraction((n+1),1)-vol*(q**n)*Fraction(n,1)*q
    tail1 /= (1-q)**2
    # Tail2: first omitted b_n and geometric ratio upper r=q*(n+5/2)/(n+1)
    coeff_n=Fraction(1)
    for k in range(1,n+1): coeff_n *= Fraction(2*k+3,2*k)
    b_n=coeff_n*(q**n)
    ratio=q*Fraction(2*n+5,2*n+2)
    tail2=vol*b_n/(1-ratio)
    S1I=fraction_interval(S1)+I(-fraction_interval(tail1).hi,fraction_interval(tail1).hi)
    S2I=fraction_interval(S2)+I(-fraction_interval(tail2).hi,fraction_interval(tail2).hi)
    a0I=fraction_interval(a0)
    pref1=4/(a0I*a0I)
    pref2=((5+SQRT5)/8)*(4*SQRT2)/(a0I*a0I*a0I.sqrt())
    IE=SQRT5/4*(pref1*S1I-pref2*S2I)
    return IE, tail1, tail2, S1, S2

def f_face2(x,y):
    sqrt5=J2.const(SQRT5); sqrt3=J2.const(SQRT3); sqrt6=J2.const(6).sqrt()
    phi=(1+sqrt5)/2; z0=-1/sqrt3; z1=(phi-1)/sqrt3; z=z0+(z1-z0)*x
    a=(1+sqrt3*z)/(sqrt6*(1-z*z).sqrt())
    q0=a/(1-a*a).sqrt()
    q=q0+(1-q0)*y
    r=sqrt3/2; J0=5*J2.const(PI)/12-3*sqrt3/4; Jc=J2.const(11)/12-sqrt3*J2.const(PI)/6
    jac=2*(z1-z0)*(1-q0)/(1+q*q)
    return jac*(J0-(r+z/2)*Jc)


# --- Production certificate driver ---
KU_VALIDATED_UPPER_DECIMAL = "0.222868116845342"


def base_vertex_chunk(args):
    N, i0, i1 = args
    total = I.point(0)
    for i in range(i0, i1):
        ax=d(i,N); bx=d(i+1,N)
        for j in range(N):
            ay=d(j,N); by=d(j+1,N)
            v,_=cell2(f_base_vertex,ax,bx,ay,by)
            total=total+v
    return (i0,i1,total.lo,total.hi)


def integrate_base_vertex_parallel(N=128, workers=8):
    import multiprocessing as multiprocessing
    step=(N+workers-1)//workers
    args=[(N,i,min(N,i+step)) for i in range(0,N,step)]
    with multiprocessing.Pool(len(args)) as pool:
        out=pool.map(base_vertex_chunk,args)
    total=I.point(0)
    for i0,i1,lo,hi in sorted(out):
        total=total+I(lo,hi)
    return total,out


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def exact_decimal_fraction(s):
    from fractions import Fraction
    if '.' not in s: return Fraction(int(s),1)
    a,b=s.split('.')
    sign=-1 if a.startswith('-') else 1
    if a.startswith('-'): a=a[1:]
    return sign*Fraction(int(a+b),10**len(b))


def preflight():
    assert sys.float_info.radix == 2 and sys.float_info.mant_dig == 53
    # Freeze the arithmetic mode used by the one-ulp outward-expansion proof.
    # FLT_ROUNDS == 1 means round-to-nearest at interpreter startup.
    assert sys.float_info.rounds == 1
    assert hasattr(math,'nextafter')
    for N in (128,1024): assert N & (N-1) == 0
    # Exact dyadic grid sanity checks.
    assert (1/128).as_integer_ratio() == (1,128)
    assert (1/1024).as_integer_ratio() == (1,1024)
    # The Machin interval must be narrow and positive; its construction itself
    # is exact-rational series + outward binary64 arithmetic.
    assert PI.lo > 3.14 and PI.hi < 3.15 and PI.width() < 5e-15


def production_certificate(workers=8):
    from fractions import Fraction
    preflight()
    t0=time.time()
    face=integrate2_uniform(f_face2,64,progress=False)
    apex_arc=integrate1(f_apex_arc,1024)
    base_arc=integrate1(f_base_arc,1024)
    base_vertex,chunks=integrate_base_vertex_parallel(128,workers)
    apex_E,tail1,tail2,_,_=apex_exact_moment_certificate(20)
    total=(6*face+4*apex_arc+6*base_arc+4*base_vertex+apex_E)/4
    target=exact_decimal_fraction(KU_VALIDATED_UPPER_DECIMAL)
    lower_exact=Fraction(*total.lo.as_integer_ratio())
    strict=lower_exact > target
    gap=lower_exact-target
    # Non-proof diagnostic: verify the historical deterministic point values fall inside.
    diagnostics={
        'face': (face, 0.002160697924731),
        'apex_arc': (apex_arc, 0.016197695368860),
        'base_arc': (base_arc, 0.003845437283688),
        'base_vertex': (base_vertex, 0.201996862101295),
        'apex_E': (apex_E, -0.017104135014280),
    }
    for name,(J,x) in diagnostics.items():
        assert J.lo <= x <= J.hi, (name,J,x)
    assert strict
    elapsed=time.time()-t0
    return dict(face=face,apex_arc=apex_arc,base_arc=base_arc,
                base_vertex=base_vertex,apex_E=apex_E,total=total,
                chunks=chunks,tail1=tail1,tail2=tail2,gap=gap,elapsed=elapsed)


def main():
    r=production_certificate()
    print('=== BODY 5 INDEPENDENT VALIDATED CERTIFICATE ===')
    print('python:',sys.version.split()[0])
    print('binary64: radix=2 mant_dig=53 rounds=nearest')
    print('script_sha256:',sha256_file(__file__))
    print('arithmetic: immediate nextafter outward rounding for +,-,*,/;')
    print('sqrt: libm candidate followed by exact-rational endpoint verification;')
    print('atan/pi: explicit alternating Taylor enclosures; pi via Machin formula;')
    print('no mpmath interval arithmetic; no probabilistic input')
    for key in ('face','apex_arc','base_arc','base_vertex','apex_E'):
        J=r[key]; print(f'{key}: [{J.lo:.17g}, {J.hi:.17g}] width={J.width():.3e}')
    print('volume: [{:.17g}, {:.17g}]'.format(r['total'].lo,r['total'].hi))
    print('raised_U3_validated_upper:',KU_VALIDATED_UPPER_DECIMAL)
    print('strict_lower_gap_exact_positive: YES')
    print('strict_lower_gap_float:',float(r['gap']))
    print('apex_tail1_bound:',float(r['tail1']))
    print('apex_tail2_bound:',float(r['tail2']))
    print('base_vertex_chunks:',[(a,b) for a,b,_,_ in r['chunks']])
    print('elapsed_sec:',round(r['elapsed'],3))
    print('BODY5_VALIDATED_LOWER_CERTIFICATE_SUCCESS')

if __name__=='__main__':
    main()

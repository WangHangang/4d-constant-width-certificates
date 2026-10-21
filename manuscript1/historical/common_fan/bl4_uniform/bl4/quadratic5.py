"""Exact arithmetic in Q(sqrt(5)), implemented with Fraction pairs."""
from fractions import Fraction


class Q5:
    __slots__=("a","b")
    def __init__(self,a=0,b=0): self.a=Fraction(a); self.b=Fraction(b)
    @staticmethod
    def coerce(x): return x if isinstance(x,Q5) else Q5(x)
    def __add__(self,o): o=Q5.coerce(o); return Q5(self.a+o.a,self.b+o.b)
    __radd__=__add__
    def __neg__(self): return Q5(-self.a,-self.b)
    def __sub__(self,o): return self+-Q5.coerce(o)
    def __rsub__(self,o): return Q5.coerce(o)+-self
    def __mul__(self,o):
        o=Q5.coerce(o); return Q5(self.a*o.a+5*self.b*o.b,self.a*o.b+self.b*o.a)
    __rmul__=__mul__
    def __truediv__(self,o):
        o=Q5.coerce(o); den=o.a*o.a-5*o.b*o.b
        if not den: raise ZeroDivisionError
        return Q5((self.a*o.a-5*self.b*o.b)/den,(self.b*o.a-self.a*o.b)/den)
    def __rtruediv__(self,o): return Q5.coerce(o)/self
    def __eq__(self,o): o=Q5.coerce(o); return self.a==o.a and self.b==o.b
    def __bool__(self): return bool(self.a or self.b)
    def sign(self):
        if not self.b: return (self.a>0)-(self.a<0)
        if not self.a: return (self.b>0)-(self.b<0)
        if self.a>0 and self.b>0: return 1
        if self.a<0 and self.b<0: return -1
        cmp=self.a*self.a-5*self.b*self.b
        if not cmp: return 0
        if self.a>0: return 1 if cmp>0 else -1
        return 1 if cmp<0 else -1
    def __lt__(self,o): return (self-Q5.coerce(o)).sign()<0
    def __le__(self,o): return (self-Q5.coerce(o)).sign()<=0
    def __gt__(self,o): return (self-Q5.coerce(o)).sign()>0
    def __ge__(self,o): return (self-Q5.coerce(o)).sign()>=0
    def to_float(self): return float(self.a)+float(self.b)*(5.0**0.5)
    def __repr__(self): return f"Q5({self.a!r},{self.b!r})"


def inverse_matrix(matrix):
    n=len(matrix)
    m=[[Q5.coerce(matrix[i][j]) for j in range(n)]+[Q5(i==j) for j in range(n)] for i in range(n)]
    for k in range(n):
        p=next(i for i in range(k,n) if m[i][k])
        m[k],m[p]=m[p],m[k]
        pivot=m[k][k]; m[k]=[x/pivot for x in m[k]]
        for i in range(n):
            if i!=k and m[i][k]:
                z=m[i][k]; m[i]=[x-z*y for x,y in zip(m[i],m[k])]
    return [row[n:] for row in m]


def matvec(matrix,vector):
    return [sum((matrix[i][j]*vector[j] for j in range(len(vector))),Q5()) for i in range(len(matrix))]


def nonnegative_a_plus_b_sqrt_q(a,b,q):
    """Exact test a+b*sqrt(q)>=0 for a,b,q in Q(sqrt(5)), q>=0."""
    a,b,q=Q5.coerce(a),Q5.coerce(b),Q5.coerce(q)
    if q<0: raise ValueError("negative radicand")
    sa,sb=a.sign(),b.sign()
    if sb==0 or not q: return sa>=0
    if sa>=0 and sb>0: return True
    if sa<0 and sb<0: return False
    cmp=a*a-b*b*q
    return cmp>=0 if sa>=0 else cmp<=0


def a_plus_sqrt_b_leq(a,b,s):
    """Exact test a+sqrt(b)<=s for nonnegative b in Q(sqrt(5))."""
    a,b,s=Q5.coerce(a),Q5.coerce(b),Q5.coerce(s)
    if b<0: raise ValueError("negative radicand")
    t=s-a
    return t>=0 and t*t>=b


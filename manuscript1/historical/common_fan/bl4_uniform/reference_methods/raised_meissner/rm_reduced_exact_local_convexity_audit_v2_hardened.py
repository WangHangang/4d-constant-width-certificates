#!/usr/bin/env python3
"""
Exact local-to-global convexity audit for the reduced raised-Meissner certificate.

Mathematical basis
------------------
For a finite cellwise-flat PL realization of a connected closed 3-manifold in
R^4, local convexity at one point of every edge implies that the realization is
an embedding onto the boundary of a convex body (the bounded n=4 case of
Rybnikov, Theorem 8, "An Efficient Local Approach to Convexity Testing of
Piecewise-Linear Hypersurfaces", arXiv:cs/0703030).

For a simplicial 3-manifold, local convexity at an edge e=[a,b] is checked on a
3-dimensional transverse section.  The link of e is a cycle c_1,...,c_m.  After
orthogonal projection to (b-a)^perp, its star becomes the cone with cyclic rays
q_i.  Rybnikov's Lemma 10 reduces convexity of this cone to signs of 3x3
oriented volumes.  In R^4 those signs can be computed without division or
projection:

  sign det[b-a, q_i, q_j, q_k]
    = sign det[b-a, c_i-a, c_j-a, c_k-a],

because the projection only adds multiples of b-a and multiplies by a positive
factor.

Trust boundary
--------------
* Qhull is NOT trusted as a convexity oracle.  Its tetrahedron list is treated
  only as a proposed finite simplicial complex.
* Combinatorial manifold checks are recomputed from that list.
* All local-convexity predicates are exact Python-integer determinant signs.
* The origin-interior witness is found numerically but verified exactly with
  fractions before use.
* The present strict kernel handles the nonplanar edge-link case covered directly
  by the 3-dimensional cone sign criterion.  If a fully planar transverse edge
  link is encountered, it stops explicitly for a separate exact treatment.
* The volume determinant sum and rational threshold comparison are exact Python
  integers.
* If the program prints EXACT_LOCAL_CONVEXITY_CERTIFICATE_SUCCESS, the only
  external mathematical ingredient is the stated local-to-global PL convexity
  theorem; no floating-point geometric predicate is used in the certificate.

The script is intentionally strict.  Unexpected schema/indexing conventions or
unsupported local degeneracies cause an explicit failure rather than a silent
assumption.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linprog
except Exception as exc:
    raise SystemExit(f"SciPy is required only to FIND an origin simplex; exact verification is rational. Import error: {exc}")


# ------------------------------ utilities ---------------------------------

class KeepAwake:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled and os.name == "nt")
        self.active = False

    def __enter__(self):
        if self.enabled:
            flags = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED
            try:
                ret = ctypes.windll.kernel32.SetThreadExecutionState(flags | self.ES_AWAYMODE_REQUIRED)
            except Exception:
                ret = 0
            if not ret:
                ret = ctypes.windll.kernel32.SetThreadExecutionState(flags)
            self.active = bool(ret)
            print("[keep-awake] automatic Windows system sleep inhibited." if self.active else
                  "[keep-awake] WARNING: SetThreadExecutionState failed.")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.active:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
            except Exception:
                pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def certificate_state_key(
    script_sha256: str,
    reduced_sha256: str,
    parent_sha256: str,
    target_num: int,
    target_den: int,
) -> str:
    """Bind resumable geometric work to source, both data files, and target."""
    payload = {
        "format_version": 2,
        "script_sha256": script_sha256,
        "reduced_sha256": reduced_sha256,
        "parent_sha256": parent_sha256,
        "target_num": int(target_num),
        "target_den": int(target_den),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fmt_time(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m {s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.0f}s"


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


# ------------------------------ loading -----------------------------------

def load_reduced(path: Path, parent: Optional[Path]):
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        print(f"[preflight] reduced keys={keys}")
        required = {"points_num", "denominator", "source_indices", "qhull_simplices"}
        missing = sorted(required - set(keys))
        if missing:
            raise ValueError(f"reduced checkpoint missing keys: {missing}")
        nums = np.asarray(z["points_num"])
        Darr = np.asarray(z["denominator"])
        src = np.asarray(z["source_indices"])
        facets = np.asarray(z["qhull_simplices"])
        saved_S = None
        if "exact_det_sum_S" in keys:
            saved_S = int(str(np.asarray(z["exact_det_sum_S"]).item()))

    if nums.ndim != 2 or nums.shape[1] != 4 or not np.issubdtype(nums.dtype, np.integer):
        raise ValueError(f"points_num must be integer (N,4), got {nums.shape}, {nums.dtype}")
    if Darr.ndim != 0:
        raise ValueError("denominator must be scalar")
    D = int(Darr.item())
    if D <= 0:
        raise ValueError("denominator must be positive")
    if src.ndim != 1 or not np.issubdtype(src.dtype, np.integer):
        raise ValueError("source_indices must be a 1D integer array")
    if len(src) != len(nums):
        raise ValueError("source_indices length != points_num length")
    if len(np.unique(src)) != len(src):
        raise ValueError("source_indices contains duplicates")
    if facets.ndim != 2 or facets.shape[1] != 4 or not np.issubdtype(facets.dtype, np.integer):
        raise ValueError(f"qhull_simplices must be integer (F,4), got {facets.shape}, {facets.dtype}")

    nums = np.asarray(nums, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    facets = np.asarray(facets, dtype=np.int64)
    N = len(nums)
    if facets.size == 0 or int(facets.min()) < 0 or int(facets.max()) >= N:
        raise ValueError(f"qhull_simplices must use zero-based local indices in [0,{N-1}]; range={facets.min()}..{facets.max()}")
    if np.any(np.diff(np.sort(facets,axis=1),axis=1)==0):
        raise ValueError("a tetrahedron contains a repeated vertex")

    used = np.unique(facets.ravel())
    if len(used) != N:
        raise ValueError(f"strict reduced certificate expects every selected point to be a boundary vertex; used {len(used):,}/{N:,}")
    if not np.array_equal(used, np.arange(N, dtype=used.dtype)):
        raise ValueError("boundary vertex labels are not exactly 0,...,N-1")
    reduced_hash = sha256_file(path)
    print(f"[preflight] points={N:,}, facets={len(facets):,}, D={D}, point_dtype={nums.dtype}, facet_range=[{facets.min()},{facets.max()}]")
    print(f"[preflight] every selected point occurs in the proposed boundary complex")
    print(f"[preflight] reduced sha256={reduced_hash}")

    parent_hash = None
    if parent is not None:
        parent_hash = sha256_file(parent)
        with np.load(parent, allow_pickle=False) as z:
            pkeys = list(z.files)
            print(f"[preflight] parent keys={pkeys}")
            if "numerators" not in pkeys:
                raise ValueError("parent checkpoint has no numerators")
            pnums = np.asarray(z["numerators"])
            if pnums.ndim != 2 or pnums.shape[1] != 4 or not np.issubdtype(pnums.dtype, np.integer):
                raise ValueError("parent numerators schema mismatch")
            if int(src.min()) < 0 or int(src.max()) >= len(pnums):
                raise ValueError("source_indices out of parent range")
            if not np.array_equal(nums, np.asarray(pnums[src], dtype=np.int64)):
                raise AssertionError("points_num != parent[numerators][source_indices]")
            if "denominator" in pkeys:
                pD = int(np.asarray(z["denominator"]).item())
                if pD != D:
                    raise AssertionError(f"parent denominator {pD} != reduced denominator {D}")
        print(f"[preflight] parent sha256={parent_hash}")
        print("[preflight] reduced points exactly match the recorded parent source indices")

    return nums, D, src, facets, saved_S, parent_hash, reduced_hash


# ---------------------- exact origin-interior witness ----------------------

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


def find_exact_origin_simplex(nums: np.ndarray, tries: int = 8, seed: int = 20260811):
    # Find a sparse feasible convex combination numerically; certify the resulting 5 points exactly.
    Y = nums.astype(np.float64)
    scale = max(1.0, float(np.max(np.abs(Y))))
    Aeq = np.vstack([Y.T/scale, np.ones(len(Y))])
    beq = np.array([0.,0.,0.,0.,1.])
    rng = np.random.default_rng(seed)
    for attempt in range(1, tries+1):
        c = rng.random(len(Y))*1e-12
        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=(0,None), method="highs")
        if not res.success:
            continue
        cand = np.flatnonzero(res.x > 1e-10)
        # HiGHS basic solution normally has <=5 nonzeros.  Try the largest weights first.
        if len(cand) > 5:
            cand = cand[np.argsort(res.x[cand])[-5:]]
        if len(cand) != 5:
            continue
        lam = solve_exact_barycentric(nums[cand])
        if lam is not None and all(x > 0 for x in lam):
            print(f"[origin] exact interior simplex found on attempt {attempt}: local indices {' '.join(map(str,cand))}")
            print("[origin] barycentric weights: " + " ; ".join(str(x) for x in lam))
            return cand, lam
    raise RuntimeError("could not find an exactly certified 5-point origin simplex")


# --------------------------- combinatorics ---------------------------------

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


def projected_integer_ray(P, a: int, b: int, c: int):
    """q=(d.d)(c-a)-((c-a).d)d, an integer representative in d^perp."""
    d=sub4(P[b],P[a]); v=sub4(P[c],P[a]); L=dot4(d,d); vd=dot4(v,d)
    return tuple(L*v[j]-vd*d[j] for j in range(4))


def planar_cycle_is_once_around(P, a:int,b:int,cycle:Sequence[int]) -> bool:
    qs=[projected_integer_ray(P,a,b,c) for c in cycle]
    q0,q1=qs[0],qs[1]
    # Find an ambient 2x2 coordinate minor giving exact coordinates in span(q0,q1).
    ij=None; den=0
    for i in range(4):
        for j in range(i+1,4):
            dd=q0[i]*q1[j]-q0[j]*q1[i]
            if dd:
                ij=(i,j); den=dd; break
        if ij: break
    if ij is None:
        return False
    i,j=ij
    rays=[]
    for q in qs:
        x=q[i]*q1[j]-q[j]*q1[i]   # alpha numerator
        y=q0[i]*q[j]-q0[j]*q[i]   # beta numerator
        if den<0: x,y=-x,-y
        if x==0 and y==0:
            return False
        rays.append((x,y))
    # Distinct rays are required; consecutive independence already follows from simplex nondegeneracy,
    # but nonconsecutive duplicate rays would create multiple wrapping/self-overlap.
    def same_ray(u,v):
        return u[0]*v[1]-u[1]*v[0]==0 and u[0]*v[0]+u[1]*v[1]>0
    m=len(rays)
    for i0 in range(m):
        for j0 in range(i0+1,m):
            if same_ray(rays[i0],rays[j0]):
                return False
    def half(u):
        x,y=u
        return 0 if (y>0 or (y==0 and x>0)) else 1
    def cmp(u,v):
        hu,hv=half(u),half(v)
        if hu!=hv:return -1 if hu<hv else 1
        cr=u[0]*v[1]-u[1]*v[0]
        if cr>0:return -1
        if cr<0:return 1
        return 0
    order=sorted(range(m), key=functools.cmp_to_key(lambda ii,jj: cmp(rays[ii],rays[jj])))
    given=list(range(m))
    # Compare circularly, allowing either orientation.
    def cyclic_equal(x,y):
        n=len(x)
        pos={val:i for i,val in enumerate(y)}
        if x[0] not in pos:return False
        k=pos[x[0]]
        return all(x[t]==y[(k+t)%n] for t in range(n))
    return cyclic_equal(given,order) or cyclic_equal(given,list(reversed(order)))


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


# ------------------------ volume / threshold -------------------------------

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


# ------------------------------- main -------------------------------------

def main()->int:
    ap=argparse.ArgumentParser(description="Exact Python-only local-convexity audit for the reduced raised-Meissner fan")
    ap.add_argument("reduced",type=Path)
    ap.add_argument(
        "--parent",type=Path,required=True,
        help="audited parent NPZ; required to certify source_indices and containment inheritance",
    )
    ap.add_argument("--target-num",type=int,default=4437)
    ap.add_argument("--target-den",type=int,default=20000)
    ap.add_argument("--checkpoint",type=Path,default=Path("rm_reduced_local_convexity_checkpoint_v2.json"))
    ap.add_argument("--report",type=Path,default=Path("rm_reduced_local_convexity_report_v2.json"))
    ap.add_argument("--expect-reduced-sha256",default="")
    ap.add_argument("--expect-parent-sha256",default="")
    ap.add_argument("--keep-awake",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--preflight-only",action="store_true",help="verify schema/indexing/hashes, then stop before geometric work")
    args=ap.parse_args()

    if args.target_num <= 0 or args.target_den <= 0:
        raise SystemExit("target numerator and denominator must be positive")

    t_all=time.perf_counter()
    script_path=Path(__file__).resolve()
    script_hash=sha256_file(script_path)
    reduced_path=args.reduced.resolve()
    parent_path=args.parent.resolve()

    print("=== Exact reduced raised-Meissner local-convexity audit (Python only; hardened v2) ===")
    print(f"started={time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"python={sys.version.split()[0]} numpy={np.__version__} platform={platform.platform()}")
    print(f"script={script_path}")
    print(f"script_sha256={script_hash}")
    print(f"reduced={reduced_path}")
    print(f"parent={parent_path}")
    print(f"target={args.target_num}/{args.target_den}={args.target_num/args.target_den:.12f}")

    with KeepAwake(args.keep_awake):
        nums,D,src,facets,saved_S,parent_hash,reduced_hash=load_reduced(reduced_path,parent_path)

        if args.expect_reduced_sha256 and reduced_hash.lower()!=args.expect_reduced_sha256.lower():
            raise AssertionError(
                f"reduced SHA-256 mismatch: expected {args.expect_reduced_sha256}, got {reduced_hash}"
            )
        if args.expect_parent_sha256 and parent_hash.lower()!=args.expect_parent_sha256.lower():
            raise AssertionError(
                f"parent SHA-256 mismatch: expected {args.expect_parent_sha256}, got {parent_hash}"
            )

        state_key=certificate_state_key(
            script_hash,reduced_hash,parent_hash,args.target_num,args.target_den
        )
        print(f"[preflight] checkpoint_state_key={state_key}")
        unique_facets_check(facets)
        if args.preflight_only:
            print("PREFLIGHT_SUCCESS")
            return 0

        # Exact affine simplex check.
        P=[tuple(int(x) for x in row) for row in nums]
        t0=time.perf_counter(); bad=0
        for i,tet in enumerate(facets,1):
            if not affine_tet_rank3(P,tet):
                bad+=1
                raise AssertionError(f"affinely degenerate boundary tetrahedron at row {i-1}: {tuple(map(int,tet))}")
            if i%200000==0 or i==len(facets):
                print(f"[geometry] affine tetrahedra {i:,}/{len(facets):,}; elapsed={fmt_time(time.perf_counter()-t0)}")
        print("[geometry] all boundary tetrahedra are affinely nondegenerate")
        del P

        # Origin is strictly interior to the convex hull of the vertex set, independently of Qhull.
        origin_idx,origin_lam=find_exact_origin_simplex(nums)

        tris,pairs=build_triangle_pairs(facets)
        dual_connected(len(facets),pairs)
        vertex_links_check(len(nums),facets,tris)
        E,planar=edge_local_convexity_check(nums,facets,args.checkpoint,state_key)

        # Rybnikov Theorem 8 now promotes the finite bounded locally convex PL 3-manifold
        # to an embedded convex boundary.  Only after this point do we interpret the cone sum
        # as the polytope volume.
        print("[global] hypotheses of the bounded n=4 local-to-global PL convexity criterion are certified")

        S,zero=exact_det_sum(nums,facets)
        if zero:
            raise AssertionError(f"{zero} boundary cone determinants vanish although origin was certified interior")
        if saved_S is not None and S!=saved_S:
            raise AssertionError(f"exact determinant sum mismatch: recomputed {S}, saved {saved_S}")
        ok=above_rational(S,D,args.target_num,args.target_den)
        vf=volume_float(S,D)
        print(f"[volume] S={S}")
        print(f"[volume] algebraic value≈{vf:.18f}")
        print(f"[volume] exact threshold > {args.target_num}/{args.target_den}: {ok}")

        report={
            "certificate":"Reduced exact raised-Meissner local-to-global lower certificate",
            "format_version":2,
            "script":str(script_path),
            "script_sha256":script_hash,
            "checkpoint_state_key":state_key,
            "reduced":str(reduced_path),
            "reduced_sha256":reduced_hash,
            "parent":str(parent_path),
            "parent_sha256":parent_hash,
            "points":int(len(nums)),"facets":int(len(facets)),"edges":int(E),
            "planar_edge_stars":int(planar),"denominator":int(D),
            "indexing":{
                "reduced_qhull_simplices":"zero-based local indices",
                "source_indices":"zero-based parent numerator indices",
            },
            "origin_simplex_local_indices":[int(x) for x in origin_idx],
            "origin_barycentric_weights":[str(x) for x in origin_lam],
            "exact_det_sum_S":str(S),
            "algebraic_volume_float":vf,
            "target_num":args.target_num,"target_den":args.target_den,
            "exact_threshold_pass":bool(ok),
            "theorem_basis":"Rybnikov 2007/2009, Theorem 8 (bounded n=4) and Lemma 10 edge-link cone test",
            "proof_scope":(
                "Qhull tetrahedra are treated only as a proposed finite complex; "
                "manifold/local-convexity predicates and volume comparison are exact."
            ),
            "elapsed_seconds":time.perf_counter()-t_all,
            "python":sys.version,
            "numpy":np.__version__,
            "platform":platform.platform(),
        }
        args.report.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
        print(f"report={args.report}")
        print(f"report_sha256={sha256_file(args.report)}")
        print(f"checkpoint={args.checkpoint}")
        print(f"checkpoint_sha256={sha256_file(args.checkpoint) if args.checkpoint.exists() else 'MISSING'}")
        print(f"total time={fmt_time(time.perf_counter()-t_all)}")
        if ok:
            print("EXACT_LOCAL_CONVEXITY_CERTIFICATE_SUCCESS")
            return 0
        print("EXACT_LOCAL_CONVEXITY_CERTIFICATE_BELOW_TARGET")
        return 2


if __name__=="__main__":
    raise SystemExit(main())

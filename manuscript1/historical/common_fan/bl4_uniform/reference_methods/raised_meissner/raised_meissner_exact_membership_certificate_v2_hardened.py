#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def preflight_parent(path: Path, expected_sha256: str = ""):
    """Verify the exact checkpoint schema and indexing conventions before use."""
    actual_hash = sha256_file(path)
    if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
        raise AssertionError(
            f"input SHA-256 mismatch: expected {expected_sha256}, got {actual_hash}"
        )

    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        required = {
            "numerators", "denominator", "sides",
            "witness_indices", "witness_denominator", "hull_simplices",
        }
        missing = sorted(required - set(keys))
        if missing:
            raise ValueError(f"checkpoint missing required keys: {missing}")

        nums0 = np.asarray(z["numerators"])
        Darr = np.asarray(z["denominator"])
        sides0 = np.asarray(z["sides"])
        wi0 = np.asarray(z["witness_indices"])
        Warr = np.asarray(z["witness_denominator"])
        hull0 = np.asarray(z["hull_simplices"])

    if nums0.ndim != 2 or nums0.shape[1] != 4 or not np.issubdtype(nums0.dtype, np.integer):
        raise ValueError(f"numerators must be integer (N,4), got {nums0.shape}, {nums0.dtype}")
    if Darr.ndim != 0 or int(Darr.item()) <= 0:
        raise ValueError("denominator must be a positive scalar integer")
    if sides0.ndim != 1 or len(sides0) != len(nums0) or not np.issubdtype(sides0.dtype, np.integer):
        raise ValueError("sides must be a 1D integer array of length N")
    if set(map(int, np.unique(sides0))) != {-1, 1}:
        raise ValueError(f"sides must contain exactly -1/+1, got {np.unique(sides0)}")
    if wi0.ndim != 1 or len(wi0) != len(nums0) or not np.issubdtype(wi0.dtype, np.integer):
        raise ValueError("witness_indices must be a 1D integer array of length N")
    if Warr.ndim != 0 or int(Warr.item()) <= 0:
        raise ValueError("witness_denominator must be a positive scalar integer")
    if not np.all(wi0[sides0 > 0] == -1):
        raise ValueError("positive-half records must use witness_indices == -1 sentinel")
    if np.any(wi0[sides0 < 0] < 0):
        raise ValueError("negative-half records must have nonnegative witness indices")
    if hull0.ndim != 2 or hull0.shape[1] != 4 or not np.issubdtype(hull0.dtype, np.integer):
        raise ValueError(f"hull_simplices must be integer (F,4), got {hull0.shape}, {hull0.dtype}")
    if hull0.size == 0:
        raise ValueError("hull_simplices is empty")

    # Historical parent convention is explicitly one-based because row 0 is
    # reserved for the origin when cone determinants are accumulated.
    N = len(nums0)
    if int(hull0.min()) < 1 or int(hull0.max()) > N:
        raise ValueError(
            f"hull_simplices must be one-based indices in [1,{N}], "
            f"got range [{hull0.min()},{hull0.max()}]"
        )
    if np.any(np.diff(np.sort(hull0, axis=1), axis=1) == 0):
        raise ValueError("hull_simplices contains a row with a repeated vertex")

    nums = np.asarray(nums0, dtype=np.int64)
    sides = np.asarray(sides0, dtype=np.int8)
    wi = np.asarray(wi0, dtype=np.int64)
    hull = np.asarray(hull0, dtype=np.int64)
    D = int(Darr.item())
    W = int(Warr.item())

    return {
        "keys": keys,
        "input_sha256": actual_hash,
        "N": N,
        "D": D,
        "W": W,
        "positive": int(np.count_nonzero(sides > 0)),
        "negative": int(np.count_nonzero(sides < 0)),
        "hull_facets": int(len(hull)),
        "hull_indexing": "one-based parent indices in [1,N]; row 0 reserved for origin after prepend",
        "nums": nums,
        "sides": sides,
        "wi": wi,
        "hull": hull,
    }


M = ((3,-1,-1),(-1,3,-1),(-1,-1,3))
VERT = {
    'A': (1,0,0), 'B': (0,1,0), 'C': (0,0,1), 'D': (-1,-1,-1)
}
FACE_ARCS = [('A','B','C','D'),('A','C','B','D'),('A','D','B','C')]
VERTEX_ARCS = [('B','C','A','D'),('B','D','A','C'),('C','D','A','B')]

def qform(a):
    x,y,z=map(int,a)
    return 3*(x*x+y*y+z*z)-2*(x*y+x*z+y*z)

def bil(a,b):
    a0,a1,a2=map(int,a); b0,b1,b2=map(int,b)
    return 3*(a0*b0+a1*b1+a2*b2) - (a0*b1+a1*b0+a0*b2+a2*b0+a1*b2+a2*b1)

def vecadd(a,b): return tuple(int(x)+int(y) for x,y in zip(a,b))
def vecsub(a,b): return tuple(int(x)-int(y) for x,y in zip(a,b))
def vecscale(c,a): return tuple(int(c)*int(x) for x in a)

def witness_integer_coeffs(den:int):
    out=[]; bary=[]
    for ia in range(den+1):
      for ib in range(den+1-ia):
       for ic in range(den+1-ia-ib):
        id_=den-ia-ib-ic
        m=(ia-id_,ib-id_,ic-id_)
        # eta>0 iff 5/8 + ||m/den||^2 < 1
        # qform(m)/(8 den^2) < 3/8
        if qform(m) < 3*den*den:
            out.append(m); bary.append((ia,ib,ic,id_))
    return out,bary

def certify_negative(n,D,mw,W):
    # Lift the point and witness to the common denominator L=lcm(D,W).
    g=math.gcd(D,W); L=D//g*W
    xs=L//D; ms=L//W
    xn=tuple(xs*int(n[i]) for i in range(4))
    mn=tuple(ms*int(v) for v in mw)
    diff=tuple(xn[i]-mn[i] for i in range(3))
    A=qform(diff)+8*xn[3]**2            # a=A/(8L^2)
    B=5*L*L+qform(mn)                   # b=B/(8L^2)
    C=8*L*L+B-A                         # 1+b-a=C/(8L^2)
    ok=(B < 8*L*L and C>=0 and C*C>=32*B*L*L)
    slack1=C
    slack2=C*C-32*B*L*L
    return ok,slack1,slack2

def arc_scaled_data(D,arc):
    X,Y,Z,Wn=[VERT[s] for s in arc]
    XD=vecscale(D,X); YD=vecscale(D,Y); ZD=vecscale(D,Z); WD=vecscale(D,Wn)
    mid=tuple((ZD[i]+WD[i])//2 for i in range(3))
    if any(2*mid[i]!=ZD[i]+WD[i] for i in range(3)):
        raise ValueError('D must make arc midpoints integral')
    U=vecsub(XD,mid)
    Yminus=vecsub(YD,mid)
    Vvec=vecsub(vecscale(3,Yminus),U)  # V uses Vvec/(2D)
    return XD,YD,mid,U,Vvec

def certify_positive(n,D,arcs):
    nh=tuple(map(int,n[:3])); t=int(n[3])
    if t<0: return False,None
    worst=None
    for arc in arcs:
        XD,YD,mid,Uvec,Vvec=arc_scaled_data(D,arc)
        # Endpoint checks: ||y-X||^2+t^2 <= 1
        for ED in (XD,YD):
            de=vecsub(nh,ED)
            slack=8*D*D-qform(de)-8*t*t
            if worst is None or slack<worst: worst=slack
            if slack<0: return False,('endpoint',arc,slack)
        z=vecsub(nh,mid)
        uN=bil(z,Uvec)                  # U=uN/(8D^2)
        vN=bil(z,Vvec)                  # V=vN/(16D^2)
        # Interior minimum lies on [0,alpha] iff V<=0 and V-4U>=0.
        if vN<=0 and vN-8*uN>=0:
            HN=2*D*D-8*t*t-qform(z)    # H=HN/(8D^2)
            s1=HN
            s2=2*HN*HN-(8*uN*uN+vN*vN)
            local=min(s1,s2)
            if worst is None or local<worst: worst=local
            if s1<0 or s2<0: return False,('interior',arc,s1,s2,uN,vN)
    return True,worst

def det4_bareiss(rows):
    a=[[int(x) for x in row] for row in rows]
    sign=1; prev=1
    for k in range(3):
        if a[k][k]==0:
            sw=next((i for i in range(k+1,4) if a[i][k]),None)
            if sw is None:return 0
            a[k],a[sw]=a[sw],a[k]; sign=-sign
        piv=a[k][k]
        for i in range(k+1,4):
            for j in range(k+1,4):
                a[i][j]=(a[i][j]*piv-a[i][k]*a[k][j])//prev
        prev=piv
    return sign*a[3][3]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        'input',
        nargs='?',
        default='rm_rational_v4_stage4_m16_exact.npz',
        help=(
            'm16 checkpoint; default: '
            'rm_rational_v4_stage4_m16_exact.npz in the current directory'
        ),
    )
    ap.add_argument('--kind',choices=['face','vertex'],default='face')
    ap.add_argument('--skip-volume',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--expect-input-sha256',default='')
    ap.add_argument('--output',default='')
    args=ap.parse_args()
    start=time.time()
    input_path = Path(args.input)
    if not input_path.exists():
        script_candidate = Path(__file__).resolve().parent / input_path.name
        if script_candidate.exists():
            input_path = script_candidate
        else:
            raise SystemExit(
                'Input checkpoint not found. Pass its path explicitly, e.g.\n'
                '  python raised_meissner_exact_membership_certificate_v2.py '
                'rm_rational_v4_stage4_m16_exact.npz'
            )

    input_path = input_path.resolve()
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    pf = preflight_parent(input_path, args.expect_input_sha256)

    nums = pf["nums"]
    D = pf["D"]
    sides = pf["sides"]
    wi = pf["wi"]
    W = pf["W"]
    simplices = pf["hull"]

    print('=== Raised-Meissner exact parent membership certificate ===')
    print(f'python={sys.version.split()[0]} numpy={np.__version__} platform={platform.platform()}')
    print(f'input={input_path}')
    print(f'script_sha256={script_hash}')
    print(f'input_sha256={pf["input_sha256"]}')
    print(f'[preflight] keys={pf["keys"]}')
    print(
        f'[preflight] points={pf["N"]} D={D} sides=({pf["positive"]} positive, '
        f'{pf["negative"]} negative) witness_denominator={W}'
    )
    print(
        f'[preflight] hull_simplices={pf["hull_facets"]} '
        f'indexing={pf["hull_indexing"]}'
    )

    witnesses,bary=witness_integer_coeffs(W)
    neg_wi = wi[sides < 0]
    if len(neg_wi) and int(neg_wi.max()) >= len(witnesses):
        raise RuntimeError(
            f'negative witness index max {int(neg_wi.max())} exceeds '
            f'reconstructed witness count {len(witnesses)}'
        )
    print(f'reconstructed_witnesses={len(witnesses)}')
    if args.preflight_only:
        print('PREFLIGHT_SUCCESS')
        return

    arcs=FACE_ARCS if args.kind=='face' else VERTEX_ARCS
    pos_count=neg_count=0; min_pos=None; min_neg1=None; min_neg2=None
    failures=[]
    for idx,(n,s,widx) in enumerate(zip(nums,sides,wi)):
        if int(s)>0:
            pos_count+=1; ok,sl=certify_positive(n,D,arcs)
            if ok:
                if min_pos is None or sl<min_pos:min_pos=sl
            else: failures.append((idx,'positive',sl))
        else:
            neg_count+=1; ok,s1,s2=certify_negative(n,D,witnesses[int(widx)],W)
            if min_neg1 is None or s1<min_neg1:min_neg1=s1
            if min_neg2 is None or s2<min_neg2:min_neg2=s2
            if not ok: failures.append((idx,'negative',s1,s2,int(widx)))
        if (idx + 1) % 10000 == 0 or idx + 1 == len(nums):
            print(
                f'  audited {idx + 1}/{len(nums)} points; '
                f'failures={len(failures)}',
                flush=True,
            )
        if failures and len(failures)>=20:
            break
    print(f'positive={pos_count} negative={neg_count} failures={len(failures)}')
    print(f'min exact positive integer slack={min_pos}')
    print(f'min exact negative linear slack={min_neg1}')
    print(f'min exact negative squared slack={min_neg2}')
    if failures:
        print('first failures:',failures[:20])
        raise SystemExit(2)

    S=None
    if not args.skip_volume:
        # The stored parent fan uses one-based point indices.  Prepending the
        # origin therefore makes the stored rows directly usable as cone vertices.
        allnums=np.vstack([np.zeros((1,4),dtype=np.int64),nums])
        total=0
        for j,simp in enumerate(simplices):
            total+=abs(det4_bareiss(allnums[simp]))
            if (j+1)%100000==0: print(f'volume determinants {j+1}/{len(simplices)}')
        S=total
        print(f'S={S}')
        print(f'Vol={S}*sqrt(2)/(192*{D}^4)')
        # Historical exact comparison of the listed parent simplicial chain.
        lhs=2*(S*10000)**2; rhs=(2219*192*D**4)**2
        print(f'Vol > 2219/10000 exact integer test: {lhs>rhs}; difference={lhs-rhs}')

    payload={
        'certificate':'Raised-Meissner exact parent membership certificate',
        'proof_scope':(
            'All stored parent points are exactly certified members. '
            'The optional determinant sum is exact for the stored one-based '
            'simplicial list but is not by itself a combinatorial hull certificate.'
        ),
        'script_sha256':script_hash,
        'input':str(input_path),
        'input_sha256':pf['input_sha256'],
        'schema':{
            'keys':pf['keys'],
            'numerators_shape':[int(x) for x in nums.shape],
            'numerators_dtype':str(nums.dtype),
            'denominator':D,
            'sides_shape':[int(x) for x in sides.shape],
            'positive_count':pos_count,
            'negative_count':neg_count,
            'positive_witness_sentinel':-1,
            'witness_denominator':W,
            'hull_simplices_shape':[int(x) for x in simplices.shape],
            'hull_indexing':pf['hull_indexing'],
        },
        'membership_success':True,
        'min_positive_integer_slack':str(min_pos),
        'min_negative_linear_slack':str(min_neg1),
        'min_negative_squared_slack':str(min_neg2),
        'S':None if S is None else str(S),
        'elapsed_seconds':time.time()-start,
        'python':sys.version,
        'numpy':np.__version__,
        'platform':platform.platform(),
    }
    out=Path(args.output or (input_path.stem+'_exact_membership_certificate_v2.json'))
    out.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print('saved',out,'elapsed',payload['elapsed_seconds'])
    print('EXACT_PARENT_MEMBERSHIP_CERTIFICATE_SUCCESS')


if __name__=='__main__':
    main()

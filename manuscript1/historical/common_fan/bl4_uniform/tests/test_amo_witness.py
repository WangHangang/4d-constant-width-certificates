from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bl4.amo_witness import audit_cap_first_component, audit_source_simplex_similarity, permutation_matrix, qform4, verify_endpoint_witness

# Exact audit of the source-coordinate similarity normalization.
assert audit_source_simplex_similarity()["status"]=="PASS"

# Finite proof that the cap cone inequalities select the first radial component
# on the certified gnomonic guard.
a=audit_cap_first_component()
assert a["status"]=="PASS" and a["leaf_boxes"]==55

# Every regular-simplex permutation is an isometry for qform4.
perms=((0,1,2,3,4),(4,1,2,3,0),(2,4,0,3,1))
vs=((1,2,-3,4),(2,-1,0,3),(-4,1,2,0))
for p in perms:
    M=permutation_matrix(p)
    for v in vs:
        mv=tuple(sum(M[i][j]*v[j] for j in range(4)) for i in range(4))
        assert qform4(mv)==qform4(v)

# A nonzero proof-side cap witness; no NumPy/SciPy is involved.
w={"patch":"cap","permutation":[0,1,2,3,4],
   "parameter_box":[["0","0"],["0","0"],["0","0"]],"lambda":"1/2"}
assert verify_endpoint_witness((-1,0,0,0),4,w)
print("PASS AMO proof-side witness tests")

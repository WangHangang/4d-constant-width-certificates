from pathlib import Path
import json, tempfile, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bl4.amo_ray_cache import (ray_key, parse_ray_key, new_cache, put_record,
                               atomic_write_cache, load_cache)

# Inherited directions have the same canonical key under a common scaling.
assert ray_key((1,2,-1,0),4)==ray_key((2,4,-2,0),8)
assert ray_key((-1,-2,1,0),4)!=ray_key((1,2,-1,0),4)
parse_ray_key(ray_key((3,0,-1,2),8))

w={"patch":"cap","permutation":[0,1,2,3,4],
   "parameter_box":[["0","0"],["0","0"],["0","0"]],"lambda":"1/2"}
c=new_cache(1<<22)
put_record(c,(1,0,0,0),4,123,w,{"source":"unit-test"})
with tempfile.TemporaryDirectory() as td:
    p=Path(td)/"cache.json"
    atomic_write_cache(p,c)
    d=load_cache(p,1<<22)
    assert d["records"][ray_key((2,0,0,0),8)]["radius_num"]==123
print("PASS AMO shared ray-cache schema/canonical-key tests")

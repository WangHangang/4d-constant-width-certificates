from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bl4.fan import build_fan, audit_fan

for level, rays, cells in ((0,8,16),(1,32,128),(2,192,1024),(3,1408,8192)):
    v,c,s=build_fan(level); meta=audit_fan(level,v,c,s)
    assert meta["ray_count"]==rays and meta["cell_count"]==cells
print("PASS fan levels 0..3")

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bl4.fan import build_adaptive_fan,audit_adaptive_fan

schedule=[[0,3],[1]]
v,c,s=build_adaptive_fan(0,schedule)
meta=audit_adaptive_fan(0,schedule,v,c,s)
assert meta["cell_count"]==16+2*7+7
print("PASS adaptive fan coverage/nonoverlap invariant")


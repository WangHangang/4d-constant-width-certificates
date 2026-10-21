from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bl4.oracles import REGISTRY

for name,oracle in REGISTRY.items():
    assert oracle.contains((0,0,0,0),1024), name
print("PASS exact-origin checks:", ", ".join(sorted(REGISTRY)))

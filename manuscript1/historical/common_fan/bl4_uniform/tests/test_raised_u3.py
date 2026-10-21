from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bl4.oracles.raised_u3 import (
    _quartic_nonnegative_unit_interval,
    _quartic_for_arc,
    PAIRS, VERT,
)

assert _quartic_nonnegative_unit_interval((1,0,0,0,0))
assert _quartic_nonnegative_unit_interval((0,0,1,0,0))
assert _quartic_nonnegative_unit_interval((1,-4,4,0,0))
assert not _quartic_nonnegative_unit_interval((-1,0,0,0,0))
assert not _quartic_nonnegative_unit_interval((1,-4,3,0,0))

# Origin is strictly inside every positive-half arc constraint.
for X,Y in PAIRS:
    c=_quartic_for_arc((0,0,0),0,1024,VERT[X],VERT[Y])
    assert _quartic_nonnegative_unit_interval(c)

print("PASS RaisedU3 quartic/Bernstein smoke checks")

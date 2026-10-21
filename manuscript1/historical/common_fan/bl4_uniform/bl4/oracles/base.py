"""Oracle protocol and common exact integer forms."""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Oracle:
    name: str
    contains: Callable[[tuple[int, int, int, int], int], bool]
    contains_float: Callable[[tuple[float, float, float, float]], bool]
    description: str
    chart_factor: dict
    proof_status: str = "ready"
    radial_float_fn: Optional[Callable[[tuple[float, float, float, float], float], float]] = None
    verify_witness: Optional[Callable[[tuple[int, int, int, int], int, dict], bool]] = None

    def radial_float(self, direction, max_radius):
        if self.radial_float_fn is not None:
            return self.radial_float_fn(direction, max_radius)
        lo, hi = 0.0, float(max_radius)
        if self.contains_float(tuple(hi * x for x in direction)):
            raise ValueError("max_radius does not bracket the body")
        for _ in range(55):
            mid = (lo + hi) / 2
            if self.contains_float(tuple(mid * x for x in direction)):
                lo = mid
            else:
                hi = mid
        return lo


def qform3(a):
    x, y, z = map(int, a)
    return 3 * (x*x + y*y + z*z) - 2 * (x*y + x*z + y*z)


def bil3(a, b):
    x, y, z = map(int, a); X, Y, Z = map(int, b)
    return 3*(x*X+y*Y+z*Z) - (x*Y+y*X+x*Z+z*X+y*Z+z*Y)

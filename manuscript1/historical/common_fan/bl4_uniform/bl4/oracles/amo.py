"""Direct AMO peabody witness-based proof adapter.

There is intentionally no claimed global membership predicate.  Nonzero
endpoints are certified by a stored boundary-parameter enclosure plus the
inward-shrink lemma, independently rechecked in ``bl4.amo_witness``.
"""
from .base import Oracle
from ..amo_witness import verify_endpoint_witness, CHART_FACTOR


def contains(point_num, point_den):
    if point_den <= 0 or len(point_num) != 4:
        raise ValueError("invalid rational point")
    # The only witness-free point used by the common framework is the
    # barycenter/origin.  All nonzero AMO endpoints require explicit witnesses.
    return all(int(x) == 0 for x in point_num)


def contains_float(x):
    # Not a proof oracle.  A body-specific generator proposes boundary boxes.
    return sum(float(v)*float(v) for v in x) < 1e-30


ORACLE = Oracle(
    name="AMO",
    contains=contains,
    contains_float=contains_float,
    description=("Direct width-one AMO peabody in the regular-4-simplex "
                 "coefficient chart. Nonzero endpoints use explicit boundary "
                 "parameter-box enclosures and the exact inward-shrink witness; "
                 "no global membership predicate is assumed."),
    chart_factor=CHART_FACTOR,
    proof_status="witness",
    verify_witness=verify_endpoint_witness,
)

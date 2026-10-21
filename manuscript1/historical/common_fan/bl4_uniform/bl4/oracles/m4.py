"""Exact Cartesian oracle for the width-one M_4 body."""
from .base import Oracle


def contains(point_num, point_den):
    if point_den <= 0 or len(point_num) != 4:
        raise ValueError("invalid rational point")
    apos = sum(max(n, 0) ** 2 for n in point_num)
    bneg = sum(max(-n, 0) ** 2 for n in point_num)
    c = point_den * point_den - 2 * (apos + bneg)
    return c >= 0 and c*c >= 8 * bneg * point_den * point_den


def contains_float(x):
    apos = sum(max(v, 0.0) ** 2 for v in x)
    bneg = sum(max(-v, 0.0) ** 2 for v in x)
    c = 1.0 - 2.0 * (apos + bneg)
    return c >= 0.0 and c*c >= 8.0*bneg


ORACLE = Oracle(
    name="M4",
    contains=contains,
    contains_float=contains_float,
    description=("Width-one M_4 in Cartesian coordinates. Exact integer form of "
                 "the positive/negative-part norm inequality."),
    chart_factor={"num": 1, "den": 1, "sqrt_radicand": 1},
)


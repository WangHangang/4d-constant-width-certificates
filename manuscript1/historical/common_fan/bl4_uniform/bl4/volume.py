"""Exact determinant sum and algebraic threshold comparison."""
from fractions import Fraction
from math import sqrt
from .fan import determinant4


def radial_volume(vertices, cells, radii, point_den):
    total = 0
    for cell in cells:
        product = 1
        for i in cell:
            product *= radii[i]
        total += abs(determinant4([vertices[i] for i in cell])) * product
    return total, 24 * point_den**4


def apply_chart_factor(total, denominator, chart_factor):
    num = total * int(chart_factor["num"])
    den = denominator * int(chart_factor["den"])
    radicand = int(chart_factor.get("sqrt_radicand", 1))
    if num < 0 or den <= 0 or radicand <= 0:
        raise ValueError("invalid positive volume representation")
    return num, den, radicand


def above_target(num, den, radicand, target_num, target_den):
    if radicand == 1:
        return target_den * num > target_num * den
    return radicand * (target_den * num) ** 2 > (target_num * den) ** 2


def decimal_value(num, den, radicand):
    return (num / den) * sqrt(radicand)


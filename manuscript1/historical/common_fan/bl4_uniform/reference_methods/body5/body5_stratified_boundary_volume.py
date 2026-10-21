#!/usr/bin/env python3
"""
Deterministic stratified-boundary volume formula for Body 5.

This script evaluates the four-dimensional volume of the body M from
Mercado-Flores--Raggi--Roldan-Pensado, "A Four-Dimensional Body of
Constant Width" (2026), without radial bisection and without a finite
center cloud.

The smooth boundary is decomposed according to the stratum of M_0 that
contains the unique diametrically opposite point:

    6 two-dimensional spherical patches,
    4 apex arcs F_{X,E},
    6 base arcs S_{X,Y},
    4 base vertices,
    1 apex vertex E.

The divergence theorem gives

    Vol_4(M) = (6 I_face + 4 I_apex_arc + 6 I_base_arc
                + 4 I_base_vertex + I_apex_vertex) / 4.

All integrals below are one-, two-, or three-dimensional deterministic
integrals.  No Monte-Carlo sampling is used.
"""

from __future__ import annotations

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad


SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
SQRT5 = math.sqrt(5.0)
PHI = (1.0 + SQRT5) / 2.0


def face_patch_geometry() -> tuple[float, float]:
    """
    Return:
        A_T = area of the unit-sphere parameter domain T,
        M_T = integral_T c.s dA, where |c|=1/2.

    For the representative patch F_AB, write q=c+(sqrt(3)/2)s.
    With z=s_3, the angular width of T is explicit.
    """
    r = SQRT3 / 2.0
    z_min = -1.0 / SQRT3
    z_max = (PHI - 1.0) / SQRT3

    def angular_width(z: float) -> float:
        value = (1.0 + SQRT3 * z) / (
            math.sqrt(6.0) * math.sqrt(1.0 - z * z)
        )
        value = min(1.0, max(-1.0, value))
        return math.pi / 2.0 - 2.0 * math.asin(value)

    area = quad(
        angular_width, z_min, z_max,
        epsabs=2e-13, epsrel=2e-13, limit=200
    )[0]
    first_moment = 0.5 * quad(
        lambda z: z * angular_width(z),
        z_min, z_max,
        epsabs=2e-13, epsrel=2e-13, limit=200
    )[0]
    return area, first_moment


def face_contribution() -> float:
    r = SQRT3 / 2.0
    area, first_moment = face_patch_geometry()

    j0 = 5.0 * math.pi / 12.0 - 3.0 * SQRT3 / 4.0
    j1 = 11.0 / 12.0 - SQRT3 * math.pi / 6.0

    return area * j0 - (r * area + first_moment) * j1


def apex_arc_normal_moments() -> tuple[float, float]:
    """
    Normal cell over an interior point of F_{A,E} is a regular spherical
    triangle.  K0 and K1 are the two moments required by the Jacobian.
    """
    r = math.sqrt(2.0 / 3.0)

    def radial_boundary(phi: float) -> float:
        return 1.0 / (math.cos(phi) + SQRT3 * math.sin(phi))

    def cos_theta_max(phi: float) -> float:
        radius = radial_boundary(phi)
        return 1.0 / math.sqrt(1.0 + 0.5 * radius * radius)

    def inner_k0(phi: float) -> float:
        c = cos_theta_max(phi)
        return 0.5 * (1.0 - c * c) - r * (1.0 - c)

    def inner_k1(phi: float) -> float:
        c = cos_theta_max(phi)
        return (1.0 - c**3) / 3.0 - 0.5 * r * (1.0 - c * c)

    k0 = 6.0 * quad(
        inner_k0, 0.0, math.pi / 3.0,
        epsabs=2e-13, epsrel=2e-13, limit=200
    )[0]
    k1 = 6.0 * quad(
        inner_k1, 0.0, math.pi / 3.0,
        epsabs=2e-13, epsrel=2e-13, limit=200
    )[0]
    return k0, k1


def apex_arc_contribution() -> float:
    r = math.sqrt(2.0 / 3.0)
    beta = math.acos(0.25)
    k0, k1 = apex_arc_normal_moments()

    # Integral of c.s along the unit-circle parameter of the arc.
    c_sum = -5.0 * math.sqrt(6.0) / 48.0 + math.sqrt(30.0) / 16.0
    curve_moment = math.tan(beta / 2.0) * c_sum

    return beta * k0 - (r * beta + curve_moment) * k1


def base_arc_contribution() -> float:
    """
    Representative new arc S_AB.  The normal cell is described by
        phi in [0,pi/2],
        theta_min(phi) <= theta <= pi/4,
    tan(theta_min)=1/(cos(phi)+sin(phi)).
    """
    a = 1.0 / SQRT2

    def primitive(u: float) -> float:
        # Integral of (1-a u)(u-a) du.
        return (
            0.5 * u * u
            - a * u
            - (a / 3.0) * u**3
            + 0.5 * a * a * u * u
        )

    base = primitive(a)

    def outer(phi: float) -> float:
        s = math.cos(phi) + math.sin(phi)
        c0 = s / math.sqrt(1.0 + s * s)
        return primitive(c0) - base

    return (math.pi / 2.0) * quad(
        outer, 0.0, math.pi / 2.0,
        epsabs=2e-13, epsrel=2e-13, limit=200
    )[0]


def base_vertex_contribution() -> float:
    """
    Representative base vertex A.

    In gnomonic coordinates
        m=(1,-y1,-y2,-y3)/sqrt(1+|y|^2),
    the normal cell is
        yi >= 0,
        y1+y2+y3 >= 1,
        yj^2+yk^2-yi^2-4 yi-1 <= 0  (cyclic).

    In positive spherical coordinates y=r u, this becomes
        1/(u1+u2+u3) <= r <= R(min ui),
    where
        R(t)=1/(sqrt(1+2t^2)-2t).

    S_3 symmetry reduces the angular integral to the chamber
        0 <= u1 <= u2 <= u3.
    """
    a = 1.0 / SQRT2

    def radial_primitive(r: float) -> float:
        return (
            0.5 * (math.atan(r) - r / (1.0 + r * r))
            - a * r**3 / (3.0 * (1.0 + r * r)**1.5)
        )

    def upper_radius(t: float) -> float:
        return 1.0 / (math.sqrt(1.0 + 2.0 * t * t) - 2.0 * t)

    def inner(t: float) -> float:
        root = math.sqrt(1.0 - t * t)
        psi_min = 0.0 if t == 0.0 else math.asin(t / root)
        upper_value = radial_primitive(upper_radius(t))

        def in_psi(psi: float) -> float:
            coordinate_sum = t + root * (math.sin(psi) + math.cos(psi))
            lower_radius = 1.0 / coordinate_sum
            return upper_value - radial_primitive(lower_radius)

        return quad(
            in_psi, psi_min, math.pi / 4.0,
            epsabs=2e-13, epsrel=2e-13, limit=200
        )[0]

    return 6.0 * quad(
        inner, 0.0, 1.0 / SQRT3,
        epsabs=2e-12, epsrel=2e-12, limit=200
    )[0]


def apex_vertex_contribution(order: int = 32) -> float:
    """
    Apex E.  Its normal cone is the simplicial cone generated by E-A_i.

    With lambda_i >=0 and sum lambda_i=1,
        |V lambda|^2 = (1 + sum lambda_i^2)/2,
        |det V| = sqrt(5)/4,
        E.(V lambda) = (5+sqrt(5))/8.

    A tensor-product Gauss--Legendre rule is used after a Duffy transform
    from the unit cube to the 3-simplex.
    """
    nodes, weights = leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    determinant = SQRT5 / 4.0
    dot_value = (5.0 + SQRT5) / 8.0
    total = 0.0

    for i, x in enumerate(nodes):
        one_minus_x = 1.0 - x
        for j, y in enumerate(nodes):
            one_minus_y = 1.0 - y

            l1 = x
            l2 = one_minus_x * y
            l3 = one_minus_x * one_minus_y * nodes
            l4 = one_minus_x * one_minus_y * (1.0 - nodes)

            norm_sq = 0.5 * (1.0 + l1*l1 + l2*l2 + l3*l3 + l4*l4)
            norm = np.sqrt(norm_sq)

            values = determinant * (
                norm**-4 - dot_value * norm**-5
            )
            jacobian = one_minus_x**2 * one_minus_y
            total += (
                weights[i] * weights[j] * jacobian
                * float(np.dot(weights, values))
            )

    return total


def compute_volume() -> dict[str, float]:
    i_face = face_contribution()
    i_apex_arc = apex_arc_contribution()
    i_base_arc = base_arc_contribution()
    i_base_vertex = base_vertex_contribution()
    i_apex_vertex = apex_vertex_contribution()

    boundary_integral = (
        6.0 * i_face
        + 4.0 * i_apex_arc
        + 6.0 * i_base_arc
        + 4.0 * i_base_vertex
        + i_apex_vertex
    )
    volume = boundary_integral / 4.0

    return {
        "I_face": i_face,
        "I_apex_arc": i_apex_arc,
        "I_base_arc": i_base_arc,
        "I_base_vertex": i_base_vertex,
        "I_apex_vertex": i_apex_vertex,
        "boundary_integral": boundary_integral,
        "volume": volume,
        "effective_radius": (2.0 * volume / math.pi**2)**0.25,
        "ball_volume_ratio": volume / (math.pi**2 / 32.0),
    }


def main() -> None:
    values = compute_volume()

    print("Body 5 deterministic stratified-boundary computation")
    print("----------------------------------------------------")
    for key in (
        "I_face",
        "I_apex_arc",
        "I_base_arc",
        "I_base_vertex",
        "I_apex_vertex",
    ):
        print(f"{key:>20s} = {values[key]:.15f}")

    print(f"{'boundary integral':>20s} = {values['boundary_integral']:.15f}")
    print(f"{'Vol_4(M)':>20s} = {values['volume']:.15f}")
    print(f"{'effective radius':>20s} = {values['effective_radius']:.15f}")
    print(f"{'ratio to width-1 ball':>20s} = {values['ball_volume_ratio']:.15f}")


if __name__ == "__main__":
    main()

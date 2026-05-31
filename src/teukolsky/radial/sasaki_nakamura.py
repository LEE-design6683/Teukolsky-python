"""Sasaki-Nakamura radial solver for the s = -2 Teukolsky equation.

The Sasaki-Nakamura (SN) transformation converts the Teukolsky equation into
a form with a short-range potential, which is numerically easier to integrate.
The transformation is specific to spin-weight s = -2.

Reference: S. E. Gralla, A. P. Porfyriadis, N. Warburton,
Phys. Rev. D 92, 064029 (2015), arXiv:1506.08496.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from teukolsky.core import RadialSolution


def _rp(a: float, mass: float = 1.0) -> float:
    return mass + math.sqrt(mass * mass - a * a)


def _rm(a: float, mass: float = 1.0) -> float:
    return mass - math.sqrt(mass * mass - a * a)


def _tortoise(r: float, a: float, mass: float = 1.0) -> complex:
    """Tortoise coordinate r*."""
    rp = _rp(a, mass)
    rm = _rm(a, mass)
    if rp == rm:
        return complex(r + 2.0 * mass * math.log((r - rp) / (2.0 * mass)) - 2.0 * mass * mass / (r - rp))
    return complex(
        r + (2.0 * mass * rp) / (rp - rm) * math.log((r - rp) / (2.0 * mass))
        - (2.0 * mass * rm) / (rp - rm) * math.log((r - rm) / (2.0 * mass))
    )


def _sn_outer_boundary_coefficients(
    lam: complex, m: int, a: float, omega: complex, n_coeff: int = 20
) -> list[complex]:
    r"""Compute asymptotic expansion coefficients c[k] for the SN function.

    The SN function X(r) at large r behaves as:

        X(r) = exp(i ω r*) \sum_{j=0}^{∞} c[j] (ω r)^{-j}

    The coefficients satisfy a 13-term linear recurrence derived from the
    SN equation.  c[0] = 1 and c[j] = 0 for j < 0.

    The recurrence formula was obtained from the Mathematica
    BlackHolePerturbationToolkit (Kernel/SasakiNakamura.m), which implements
    Eq. (8)-(10) of Gralla-Porfyriadis-Warburton (2015).
    """
    M = 1.0
    q = a
    lam_sq = lam * lam
    lam_cu = lam * lam_sq
    m2 = m * m
    denom = 2.0 * lam + lam_sq - 12.0 * omega * (-a * m + 1.0j * M + a * a * omega)

    c = [0.0j] * (n_coeff + 14)  # padding for negative indices
    offset = 13
    c[offset] = 1.0 + 0.0j  # c[0] = 1

    for k_val in range(1, n_coeff + 1):
        idx = k_val + offset
        val = 0.0j

        # --- c[k-13] term ---
        if k_val >= 13:
            val += (
                -6.0j
                * a**12
                * (210.0 - 29.0 * k_val + k_val * k_val)
                * omega**12
                * c[idx - 13]
                / (k_val * denom)
            )

        # --- c[k-12] term ---
        if k_val >= 12:
            val += (
                12.0
                * a**10
                * (-14.0 + k_val)
                * omega**11
                * (
                    -a * (-12.0 + k_val) * m
                    + 3.0j * (-13.0 + k_val) * M
                    + a * a * (-13.0 + k_val) * omega
                )
                * c[idx - 12]
                / (k_val * denom)
            )

        # --- c[k-11] term ---
        if k_val >= 11:
            kk = k_val
            val += (
                6.0j
                * a**8
                * omega**10
                * (
                    -2.0j * a * (729.0 - 121.0 * kk + 5.0 * kk * kk) * m * M
                    - 12.0 * (156.0 - 25.0 * kk + kk * kk) * M * M
                    - 4.0 * a**3 * (143.0 - 24.0 * kk + kk * kk) * m * omega
                    + 2.0 * a**4 * (156.0 - 25.0 * kk + kk * kk) * omega * omega
                    + a * a
                    * (
                        kk * kk * (-5.0 + 2.0 * m2 + 10.0j * M * omega)
                        + kk * (123.0 - 46.0 * m2 - 250.0j * M * omega)
                        + 3.0 * (-251.0 + 87.0 * m2 + 520.0j * M * omega)
                    )
                )
                * c[idx - 11]
                / (kk * denom)
            )

        # --- c[k-10] term ---
        if k_val >= 10:
            kk = k_val
            val += (
                4.0
                * a**6
                * omega**9
                * (
                    -6.0 * a * (493.0 - 89.0 * kk + 4.0 * kk * kk) * m * M * M
                    + 12.0j * (132.0 - 23.0 * kk + kk * kk) * M**3
                    - 12.0 * a**5 * (-12.0 + kk) * m * omega * omega
                    + 6.0 * a**6 * (-12.0 + kk) * omega**3
                    - 3.0j
                    * a * a
                    * M
                    * (
                        4.0 * kk * kk * (-3.0 + m2 + 2.0j * M * omega)
                        - 2.0 * kk * (-135.0 + 43.0 * m2 + 91.0j * M * omega)
                        + 3.0 * (-503.0 + 153.0 * m2 + 344.0j * M * omega)
                    )
                    + a**4
                    * omega
                    * (
                        -kk * kk * (-15.0 + lam + 12.0j * M * omega)
                        + kk * (-339.0 + 6.0 * m2 + 20.0 * lam + 276.0j * M * omega)
                        - 3.0 * (-635.0 + 25.0 * m2 + 32.0 * lam + 528.0j * M * omega)
                    )
                    + a**3
                    * m
                    * (
                        kk * kk * (-12.0 + lam + 24.0j * M * omega)
                        + kk * (264.0 - 20.0 * lam - 534.0j * M * omega)
                        + 3.0 * (-481.0 + m2 + 32.0 * lam + 986.0j * M * omega)
                    )
                )
                * c[idx - 10]
                / (kk * denom)
            )

        # --- c[k-9] term ---
        if k_val >= 9:
            kk = k_val
            val += (
                -1.0j
                * a**5
                * omega**8
                * (
                    96.0j * (-11.0 + kk) * (-9.0 + kk) * m * M**3
                    + 48.0 * a**6 * m * omega**3
                    + 48.0
                    * a
                    * M * M
                    * (
                        3.0
                        - 2.0 * m2
                        + (-9.0 + kk) ** 2 * (9.0 - 2.0 * m2 - 2.0j * M * omega)
                        + (-9.0 + kk) * (-21.0 + 4.0 * m2 + 4.0j * M * omega)
                    )
                    + 4.0
                    * a**5
                    * omega * omega
                    * (
                        -2970.0
                        - 27.0 * kk * kk
                        - 30.0 * m2
                        + 46.0 * lam
                        + 264.0j * M * omega
                        + kk * (567.0 - 4.0 * lam - 24.0j * M * omega)
                    )
                    - 8.0j
                    * a * a
                    * m
                    * M
                    * (
                        9.0 * m2
                        - 11.0 * lam
                        + 36.0j * M * omega
                        + 4.0 * (-9.0 + kk) ** 2 * (-12.0 + lam + 6.0j * M * omega)
                        - 2.0 * (-9.0 + kk) * (-45.0 + lam + 30.0j * M * omega)
                    )
                    + 4.0
                    * a**4
                    * m
                    * omega
                    * (
                        51.0 * kk * kk
                        + kk * (-1011.0 + 4.0 * lam + 48.0j * M * omega)
                        + 6.0 * (4.0 * m2 - 9.0 * (-92.0 + lam + 10.0j * M * omega))
                    )
                    + a**3
                    * (
                        (-9.0 + kk) ** 2
                        * (
                            120.0
                            - 96.0 * m2
                            + 2.0 * lam
                            + lam_sq
                            - 492.0j * M * omega
                            + 32.0j * M * lam * omega
                            - 96.0 * M * M * omega * omega
                        )
                        - 4.0
                        * (
                            -9.0
                            + 6.0 * m2 * m2
                            + 2.0 * lam_sq
                            + 54.0j * M * omega
                            + 48.0 * M * M * omega * omega
                            + lam * (4.0 + 22.0j * M * omega)
                            - m2 * (33.0 + 8.0 * lam + 78.0j * M * omega)
                        )
                        + (-9.0 + kk)
                        * (
                            lam_sq
                            + 96.0 * m2 * (1.0 - 1.0j * M * omega)
                            + 2.0 * lam * (1.0 - 8.0j * M * omega)
                            + 12.0 * (-22.0 + 95.0j * M * omega + 24.0 * M * M * omega * omega)
                        )
                    )
                )
                * c[idx - 9]
                / (2.0 * kk * denom)
            )

        # --- c[k-8] term ---
        if k_val >= 8:
            kk = k_val
            val += (
                a**4
                * omega**7
                * (
                    48.0j * (159.0 - 36.0 * kk + 2.0 * kk * kk) * M**3
                    + 8.0
                    * a
                    * m
                    * M * M
                    * (
                        kk * (525.0 - 34.0 * lam)
                        + 2.0 * kk * kk * (-15.0 + lam)
                        + 3.0 * (-751.0 + 47.0 * lam)
                    )
                    - 4.0 * a**5 * m * (51.0 * kk - 2.0 * (249.0 + lam)) * omega * omega
                    + 12.0 * a**6 * (-88.0 + 9.0 * kk) * omega**3
                    + 4.0
                    * a**3
                    * m
                    * (
                        -12.0
                        - 15.0 * lam
                        - lam_sq
                        + m2 * (12.0 + lam)
                        + 6.0j * M * omega
                        - 18.0j * M * lam * omega
                        + 2.0 * (-8.0 + kk) ** 2 * (-9.0 + 2.0 * lam + 39.0j * M * omega)
                        + (-8.0 + kk) * (36.0 + 1.0j * M * (-141.0 + 4.0 * lam) * omega)
                    )
                    + a**4
                    * omega
                    * (
                        -8.0 * (-8.0 + kk) ** 2 * (-15.0 + 2.0 * lam + 21.0j * M * omega)
                        + 4.0
                        * (
                            9.0
                            + lam_sq
                            - 3.0 * m2 * (19.0 + lam)
                            - 54.0j * M * omega
                            + 2.0 * lam * (6.0 + 5.0j * M * omega)
                        )
                        + (-8.0 + kk)
                        * (
                            -264.0
                            + 96.0 * m2
                            - lam_sq
                            + 456.0j * M * omega
                            + lam * (-2.0 - 16.0j * M * omega)
                        )
                    )
                    - 1.0j
                    * a * a
                    * M
                    * (
                        2.0
                        * (-8.0 + kk) ** 2
                        * (
                            -108.0
                            + 72.0 * m2
                            - lam_sq
                            + 156.0j * M * omega
                            + lam * (-2.0 - 8.0j * M * omega)
                        )
                        - (-8.0 + kk)
                        * (
                            -432.0
                            + 168.0 * m2
                            + lam_sq
                            + 540.0j * M * omega
                            + 2.0 * lam * (1.0 - 8.0j * M * omega)
                        )
                        - 4.0
                        * (
                            m2 * (30.0 + 8.0 * lam)
                            - 3.0
                            * (
                                lam_sq
                                - 18.0j * M * omega
                                + lam * (2.0 + 2.0j * M * omega)
                            )
                        )
                    )
                )
                * c[idx - 8]
                / (kk * denom)
            )

        # --- c[k-7] term ---
        if k_val >= 7:
            kk = k_val
            val += (
                1.0j
                * a**3
                * omega**6
                * (
                    -192.0j * (54.0 - 15.0 * kk + kk * kk) * m * M**3
                    - 168.0 * a**6 * m * omega**3
                    + 4.0
                    * a**5
                    * omega * omega
                    * (
                        3459.0
                        + 48.0 * kk * kk
                        + 93.0 * m2
                        - 149.0 * lam
                        - 732.0j * M * omega
                        + 4.0 * kk * (-204.0 + 4.0 * lam + 21.0j * M * omega)
                    )
                    - 2.0
                    * a**4
                    * m
                    * omega
                    * (
                        10086.0
                        + 168.0 * kk * kk
                        + 126.0 * m2
                        - 348.0 * lam
                        - lam_sq
                        - 2772.0j * M * omega
                        + 8.0 * kk * (-327.0 + 4.0 * lam + 39.0j * M * omega)
                    )
                    + 8.0j
                    * a * a
                    * m
                    * M
                    * (
                        3.0
                        + 18.0 * m2
                        - 37.0 * lam
                        - 2.0 * lam_sq
                        - 24.0j * M * omega
                        + 6.0 * (-7.0 + kk) ** 2 * (-9.0 + 2.0 * lam + 9.0j * M * omega)
                        - 2.0 * (-7.0 + kk) * (-42.0 + lam + 42.0j * M * omega)
                    )
                    + 4.0
                    * a
                    * M * M
                    * (
                        -12.0 * (-7.0 + kk) * (-15.0 + 4.0 * m2 + 4.0j * M * omega)
                        + 4.0
                        * (9.0 - 12.0 * m2 + 2.0 * lam + lam_sq - 36.0j * M * omega)
                        + (-7.0 + kk) ** 2
                        * (
                            -108.0
                            + 48.0 * m2
                            - 2.0 * lam
                            - lam_sq
                            + 60.0j * M * omega
                        )
                    )
                    + a**3
                    * (
                        12.0
                        + 48.0 * m2 * m2
                        + 54.0 * lam
                        + 29.0 * lam_sq
                        + lam_cu
                        - 420.0j * M * omega
                        + 236.0j * M * lam * omega
                        + 16.0j * M * lam_sq * omega
                        - m2 * (156.0 + 98.0 * lam + lam_sq + 756.0j * M * omega)
                        + 4.0
                        * (-7.0 + kk) ** 2
                        * (
                            -30.0
                            + 36.0 * m2
                            - 2.0 * lam
                            - lam_sq
                            + 192.0j * M * omega
                            - 24.0j * M * lam * omega
                            + 60.0 * M * M * omega * omega
                        )
                        + 4.0
                        * (-7.0 + kk)
                        * (
                            lam_sq * (-1.0 - 1.0j * M * omega)
                            + lam * (-2.0 + 2.0j * M * omega)
                            + m2 * (-36.0 + 72.0j * M * omega)
                            - 6.0 * (-9.0 + 52.0j * M * omega + 22.0 * M * M * omega * omega)
                        )
                    )
                )
                * c[idx - 7]
                / (2.0 * kk * denom)
            )

        # --- c[k-6] term ---
        if k_val >= 6:
            kk = k_val
            val += (
                a * a
                * omega**5
                * (
                    48.0j * (-7.0 + kk) * (-6.0 + kk) * M**3
                    + 8.0
                    * a
                    * m
                    * M * M
                    * (
                        21.0
                        + 21.0 * (-6.0 + kk)
                        + 4.0 * (-6.0 + kk) ** 2 * (-6.0 + lam)
                        - 12.0 * lam
                    )
                    + 24.0 * a**5 * m * (105.0 - 14.0 * kk + lam) * omega * omega
                    + 96.0 * a**6 * (-15.0 + 2.0 * kk) * omega**3
                    + 4.0
                    * a**3
                    * m
                    * (
                        -15.0
                        - 20.0 * lam
                        - 3.0 * lam_sq
                        + m2 * (15.0 + 2.0 * lam)
                        - 9.0j * M * omega
                        - 43.0j * M * lam * omega
                        + 6.0 * (-6.0 + kk) ** 2 * (-2.0 + lam + 15.0j * M * omega)
                        + 3.0 * (-6.0 + kk) * (8.0 + 1.0j * M * (-39.0 + 4.0 * lam) * omega)
                    )
                    + 4.0
                    * a**4
                    * omega
                    * (
                        -3.0
                        + 11.0 * lam
                        + 3.0 * lam_sq
                        - m2 * (81.0 + 8.0 * lam)
                        - 57.0j * M * omega
                        + 27.0j * M * lam * omega
                        - 6.0 * (-6.0 + kk) ** 2 * (-5.0 + lam + 9.0j * M * omega)
                        + (-6.0 + kk)
                        * (
                            -54.0
                            + 36.0 * m2
                            - lam_sq
                            + 135.0j * M * omega
                            + lam * (-2.0 - 12.0j * M * omega)
                        )
                    )
                    - 1.0j
                    * a * a
                    * M
                    * (
                        36.0
                        + 66.0 * lam
                        + 35.0 * lam_sq
                        + lam_cu
                        - 4.0 * m2 * (27.0 + 16.0 * lam)
                        - 636.0j * M * omega
                        + 84.0j * M * lam * omega
                        - (-6.0 + kk)
                        * (120.0 * m2 + 10.0 * lam + 5.0 * lam_sq + 36.0 * (-6.0 + 7.0j * M * omega))
                        + 2.0
                        * (-6.0 + kk) ** 2
                        * (
                            -72.0
                            + 72.0 * m2
                            - 3.0 * lam_sq
                            + 180.0j * M * omega
                            + lam * (-6.0 - 16.0j * M * omega)
                        )
                    )
                )
                * c[idx - 6]
                / (kk * denom)
            )

        # --- c[k-5] term ---
        if k_val >= 5:
            kk = k_val
            val += (
                -1.0j
                * a
                * omega**4
                * (
                    96.0j * (-1.0 + (-5.0 + kk) ** 2) * m * M**3
                    + 216.0 * a**6 * m * omega**3
                    - 8.0j
                    * a * a
                    * m
                    * M
                    * (
                        -6.0
                        + 9.0 * m2
                        - 33.0 * lam
                        - 4.0 * lam_sq
                        + 12.0 * (-5.0 + kk) ** 2 * (-2.0 + lam + 3.0j * M * omega)
                        + 2.0 * (-5.0 + kk) * (15.0 + lam - 12.0j * M * omega)
                    )
                    + 12.0
                    * a**5
                    * omega * omega
                    * (
                        -590.0
                        - 14.0 * kk * kk
                        - 34.0 * m2
                        + 57.0 * lam
                        + 228.0j * M * omega
                        - 2.0 * kk * (-91.0 + 4.0 * lam + 18.0j * M * omega)
                    )
                    + 8.0
                    * a
                    * M * M
                    * (
                        -10.0 * lam
                        - 5.0 * lam_sq
                        + 72.0j * M * omega
                        + (-5.0 + kk) * (-18.0 + 2.0 * lam + lam_sq - 12.0j * M * omega)
                        + (-5.0 + kk) ** 2
                        * (18.0 - 12.0 * m2 + 2.0 * lam + lam_sq - 24.0j * M * omega)
                    )
                    + 6.0
                    * a**4
                    * m
                    * omega
                    * (
                        1360.0
                        + 44.0 * kk * kk
                        + 36.0 * m2
                        - 132.0 * lam
                        - lam_sq
                        - 772.0j * M * omega
                        + 4.0 * kk * (-123.0 + 4.0 * lam + 30.0j * M * omega)
                    )
                    + a**3
                    * (
                        -12.0
                        - 24.0 * m2 * m2
                        - 64.0 * lam
                        - 38.0 * lam_sq
                        - 3.0 * lam_cu
                        + 552.0j * M * omega
                        - 132.0j * M * lam * omega
                        - 32.0j * M * lam_sq * omega
                        + 2.0 * m2 * (30.0 + 50.0 * lam + lam_sq + 288.0j * M * omega)
                        - 6.0
                        * (-5.0 + kk) ** 2
                        * (
                            -10.0
                            + 16.0 * m2
                            - lam_sq
                            + 92.0j * M * omega
                            + 32.0 * M * M * omega * omega
                            + lam * (-2.0 - 16.0j * M * omega)
                        )
                        + 2.0
                        * (-5.0 + kk)
                        * (
                            48.0 * m2 * (1.0 - 3.0j * M * omega)
                            + lam_sq * (3.0 + 6.0j * M * omega)
                            + lam * (6.0 + 20.0j * M * omega)
                            + 42.0 * (-1.0 + 6.0j * M * omega + 4.0 * M * M * omega * omega)
                        )
                    )
                )
                * c[idx - 5]
                / (2.0 * kk * denom)
            )

        # --- c[k-4] term ---
        if k_val >= 4:
            kk = k_val
            val += (
                a
                * omega**3
                * (
                    8.0
                    * m
                    * M * M
                    * (kk * (51.0 - 14.0 * lam) + 21.0 * (-5.0 + lam) + 2.0 * kk * kk * (-3.0 + lam))
                    + 24.0 * a**4 * m * (57.0 - 11.0 * kk + lam) * omega * omega
                    + 24.0 * a**5 * (-36.0 + 7.0 * kk) * omega**3
                    + 2.0
                    * a**3
                    * omega
                    * (
                        kk * kk * (30.0 - 8.0 * lam - 60.0j * M * omega)
                        + kk
                        * (
                            -282.0
                            + 48.0 * m2
                            + 58.0 * lam
                            - 3.0 * lam_sq
                            + 630.0j * M * omega
                            - 24.0j * M * lam * omega
                        )
                        - 2.0
                        * (
                            -321.0
                            + 50.0 * lam
                            - 9.0 * lam_sq
                            + 7.0 * m2 * (21.0 + lam)
                            + 816.0j * M * omega
                            - 72.0j * M * lam * omega
                        )
                    )
                    + 4.0
                    * a * a
                    * m
                    * (
                        -6.0
                        - 11.0 * lam
                        - 3.0 * lam_sq
                        + m2 * (6.0 + lam)
                        - 32.0j * M * lam * omega
                        + (-4.0 + kk) ** 2 * (-3.0 + 4.0 * lam + 42.0j * M * omega)
                        + 3.0 * (-4.0 + kk) * (2.0 + 1.0j * M * (-9.0 + 4.0 * lam) * omega)
                    )
                    - 1.0j
                    * a
                    * M
                    * (
                        2.0
                        * (
                            (26.0 - 16.0 * m2) * lam
                            + 15.0 * lam_sq
                            + lam_cu
                            - 168.0j * M * omega
                        )
                        + 2.0
                        * (-4.0 + kk) ** 2
                        * (
                            -18.0
                            + 24.0 * m2
                            - 3.0 * lam_sq
                            + 84.0j * M * omega
                            + lam * (-6.0 - 8.0j * M * omega)
                        )
                        - (-4.0 + kk)
                        * (
                            -36.0
                            + 24.0 * m2
                            + 7.0 * lam_sq
                            - 60.0j * M * omega
                            + 2.0 * lam * (7.0 + 8.0j * M * omega)
                        )
                    )
                )
                * c[idx - 4]
                / (kk * denom)
            )

        # --- c[k-3] term ---
        if k_val >= 3:
            kk = k_val
            val += (
                -1.0j
                * omega * omega
                * (
                    120.0 * a**5 * m * omega**3
                    + 4.0 * (-4.0 + kk) * kk * M * M * (2.0 * lam + lam_sq - 12.0j * M * omega)
                    - 8.0j
                    * a
                    * m
                    * M
                    * (
                        -7.0 * lam
                        - 2.0 * lam_sq
                        + (-3.0 + kk) * (3.0 + 2.0 * lam)
                        + 6.0j * M * omega
                        + (-3.0 + kk) ** 2 * (-3.0 + 4.0 * lam + 6.0j * M * omega)
                    )
                    + 4.0
                    * a**4
                    * omega * omega
                    * (
                        -363.0
                        - 18.0 * kk * kk
                        - 45.0 * m2
                        + 79.0 * lam
                        + 228.0j * M * omega
                        - 2.0 * kk * (-81.0 + 8.0 * lam + 30.0j * M * omega)
                    )
                    + 2.0
                    * a**3
                    * m
                    * omega
                    * (
                        48.0 * kk * kk
                        + 30.0 * m2
                        + 8.0 * kk * (-39.0 + 4.0 * lam + 21.0j * M * omega)
                        - 3.0 * (-162.0 + 60.0 * lam + lam_sq + 212.0j * M * omega)
                    )
                    + a * a
                    * (
                        -3.0 * lam_cu
                        + 12.0j * M * omega * (15.0 + 11.0 * m2 + 4.0j * M * omega)
                        + lam_sq * (-21.0 + m2 - 16.0j * M * omega)
                        + lam * (-30.0 + 34.0 * m2 + 28.0j * M * omega)
                        + 4.0
                        * (-3.0 + kk)
                        * (
                            -3.0
                            + 6.0j * M * omega
                            + 36.0 * M * M * omega * omega
                            + lam_sq * (1.0 + 3.0j * M * omega)
                            + 6.0 * m2 * (1.0 - 4.0j * M * omega)
                            + 2.0 * lam * (1.0 + 5.0j * M * omega)
                        )
                        - 4.0
                        * (-3.0 + kk) ** 2
                        * (
                            -3.0
                            + 6.0 * m2
                            - lam_sq
                            + 42.0j * M * omega
                            + 12.0 * M * M * omega * omega
                            + lam * (-2.0 - 8.0j * M * omega)
                        )
                    )
                )
                * c[idx - 3]
                / (2.0 * kk * denom)
            )

        # --- c[k-2] term ---
        if k_val >= 2:
            kk = k_val
            val += (
                -omega
                * (
                    8.0 * a**3 * m * (-33.0 + 12.0 * kk - lam) * omega * omega
                    - 24.0 * a**4 * (-8.0 + 3.0 * kk) * omega**3
                    + M
                    * (3.0 + 5.0 * kk - 2.0 * kk * kk + lam)
                    * (2.0j * lam + 1.0j * lam_sq + 12.0 * M * omega)
                    + 4.0
                    * a * a
                    * omega
                    * (
                        -18.0
                        + lam
                        - 3.0 * lam_sq
                        + 2.0 * m2 * (12.0 + lam)
                        + 75.0j * M * omega
                        - 15.0j * M * lam * omega
                        + kk * kk * (-3.0 + lam + 6.0j * M * omega)
                        + kk
                        * (
                            15.0
                            - 6.0 * m2
                            - 2.0 * lam
                            + lam_sq
                            - 45.0j * M * omega
                            + 4.0j * M * lam * omega
                        )
                    )
                    + 4.0
                    * a
                    * m
                    * (
                        lam_sq
                        - 3.0j * (5.0 - 7.0 * kk + 2.0 * kk * kk) * M * omega
                        - lam * (2.0 + kk * kk - 15.0j * M * omega + kk * (-4.0 + 4.0j * M * omega))
                    )
                )
                * c[idx - 2]
                / (kk * denom)
            )

        # --- c[k-1] term ---
        if k_val >= 1:
            kk = k_val
            val += (
                -1.0j
                * (
                    -lam_cu
                    + lam_sq
                    * (kk * kk - 2.0 * (2.0 + a * m * omega + 2.0j * M * omega) + kk * (-1.0 + 4.0j * M * omega))
                    + 12.0
                    * omega
                    * (
                        2.0 * a**3 * m * omega * omega
                        + a * a
                        * omega
                        * (
                            -6.0
                            + 5.0 * kk
                            - kk * kk
                            - 2.0 * m2
                            + 4.0j * M * omega
                            - 4.0j * kk * M * omega
                        )
                        + a
                        * m
                        * (-2.0 - kk + kk * kk - 2.0j * M * omega + 4.0j * kk * M * omega)
                        + M * (2.0j + 1.0j * kk - 1.0j * kk * kk - 4.0 * M * omega + 4.0 * kk * M * omega)
                    )
                    + 2.0
                    * lam
                    * (
                        -2.0
                        + kk * kk
                        - 24.0 * a * m * omega
                        + 2.0j * M * omega
                        + 22.0 * a * a * omega * omega
                        + kk * (-1.0 + 8.0 * a * m * omega + 4.0j * M * omega - 8.0 * a * a * omega * omega)
                    )
                )
                * c[idx - 1]
                / (2.0 * kk * denom)
            )

        c[idx] = val

    return [c[offset + j] for j in range(n_coeff + 1)]


def _sn_outer_boundary(
    lam: complex, m: int, a: float, omega: complex, rout: float, n_coeff: int = 20
) -> tuple[complex, complex, complex]:
    """Compute {X(rout), X'(rout), X''(rout)} for the SN Up solution.

    Uses the asymptotic expansion X(r) = exp(i ω r*) Σ c[j] (ω r)^{-j}.
    """
    c_coeffs = _sn_outer_boundary_coefficients(lam, m, a, omega, n_coeff)
    r_star = _tortoise(rout, a)
    phase = 1.0j * omega * r_star

    # X(r) = exp(i ω r*) Σ c[j] (ω r)^{-j}
    inv_or = 1.0 / (omega * rout)
    X_val = 0.0j
    dX_val = 0.0j
    d2X_val = 0.0j

    for j, cj in enumerate(c_coeffs):
        term = cj * (inv_or ** j)
        X_val += term
        # d/dr of (ω r)^{-j} = -j ω (ω r)^{-j-1}
        dX_val += cj * (-j) * omega * (inv_or ** (j + 1))
        # d²/dr² of (ω r)^{-j} = j(j+1) ω² (ω r)^{-j-2}
        d2X_val += cj * j * (j + 1) * omega * omega * (inv_or ** (j + 2))

    exp_phase = np.exp(phase)
    # X(r) = exp(i ω r*) * sum
    # X' = exp(i ω r*) * (sum' + i ω (dr*/dr) * sum)
    drstar_dr = rout * rout / ((rout - _rp(a)) * (rout - _rm(a))) if rout > _rp(a) else 1.0
    # Actually, for large r, dr*/dr ≈ r²/Δ(r)

    X_val *= exp_phase
    dX_val = exp_phase * (dX_val + 1.0j * omega * drstar_dr * (X_val / exp_phase))
    d2X_val = exp_phase * (
        d2X_val
        + 2.0j * omega * drstar_dr * (dX_val / exp_phase)
        + (1.0j * omega * _d2rstar_dr2(rout, a) - omega * omega * drstar_dr * drstar_dr)
        * (X_val / exp_phase)
    )

    return X_val, dX_val, d2X_val


def _d2rstar_dr2(r: float, a: float, mass: float = 1.0) -> float:
    """Second derivative of tortoise coordinate r* with respect to r."""
    delta = r * r - 2.0 * mass * r + a * a
    ddelta = 2.0 * (r - mass)
    return -ddelta / (delta * delta)


def _sn_equation_rhs(
    r: float, y: np.ndarray, lam: complex, m: int, a: float, omega: complex
) -> np.ndarray:
    """Right-hand side of the Sasaki-Nakamura equation as a first-order ODE system.

    y = [X, X']

    The SN equation is: f² X'' + f (f' - F) X' - U X = 0

    where:
        f = Δ / (r² + a²)
        F = (η'/η) Δ / (r² + a²)
        U = (Δ U1)/(r² + a²)² + G² + Δ G'/(r² + a²) - F G
    """
    M = 1.0
    delta = r * r - 2.0 * M * r + a * a
    r2_p_a2 = r * r + a * a
    f_val = delta / r2_p_a2

    # η coefficients
    c0 = -12.0j * omega * M + lam * (lam + 2.0) - 12.0 * a * omega * (a * omega - m)
    c1 = 8.0j * a * (3.0 * a * omega - lam * (a * omega - m))
    c2 = -24.0j * a * M * (a * omega - m) + 12.0 * a * a * (1.0 - 2.0 * (a * omega - m) ** 2)
    c3 = 24.0j * a**3 * (a * omega - m) - 24.0 * M * a * a
    c4 = 12.0 * a**4

    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    eta = c0 + c1 * inv_r + c2 * inv_r2 + c3 * inv_r2 * inv_r + c4 * inv_r2 * inv_r2
    deta_dr = (
        -c1 * inv_r2
        - 2.0 * c2 * inv_r2 * inv_r
        - 3.0 * c3 * inv_r2 * inv_r2
        - 4.0 * c4 * inv_r2 * inv_r2 * inv_r
    )

    # K and derivatives
    K_val = r2_p_a2 * omega - m * a
    dK_dr = 2.0 * r * omega

    # β and α
    two_delta_over_r = 2.0 * delta * inv_r
    beta = 2.0 * delta * (-1.0j * K_val + r - M - two_delta_over_r)
    alpha = (
        -1.0j * K_val * beta / (delta * delta)
        + 3.0j * dK_dr
        + lam
        + 6.0 * delta * inv_r2
    )

    # V
    V_val = (
        -(K_val * K_val + 4.0j * (r - M) * K_val) / delta
        + 8.0j * omega * r
        + lam
    )

    # U1
    dbeta_dr = _dbeta_dr(r, a, omega, m)
    U1 = (
        V_val
        + delta * delta / beta
        * (
            _d_dr_2alpha_plus_dbeta_over_delta(r, a, omega, m, lam, alpha, beta, delta)
            - deta_dr / eta * (alpha + dbeta_dr / delta)
        )
    )

    # F and G
    F_val = deta_dr / eta * delta / r2_p_a2
    G_val = -2.0 * (r - M) / r2_p_a2 + r * delta / (r2_p_a2 * r2_p_a2)

    # U
    dG_dr = _dG_dr(r, a)
    U_val = (
        delta * U1 / (r2_p_a2 * r2_p_a2)
        + G_val * G_val
        + delta * dG_dr / r2_p_a2
        - F_val * G_val
    )

    # f' = dΔ/dr / (r²+a²) - Δ * 2r / (r²+a²)²
    df_dr = 2.0 * (r - M) / r2_p_a2 - delta * 2.0 * r / (r2_p_a2 * r2_p_a2)

    X = y[0]
    dX = y[1]

    # f² X'' = -f(f' - F) X' + U X
    d2X = (-f_val * (df_dr - F_val) * dX + U_val * X) / (f_val * f_val)

    return np.array([dX, d2X], dtype=np.complex128)


def _dbeta_dr(r: float, a: float, omega: complex, m: int) -> complex:
    """Derivative of β with respect to r."""
    M = 1.0
    delta = r * r - 2.0 * M * r + a * a
    ddelta = 2.0 * (r - M)
    K_val = (r * r + a * a) * omega - m * a
    dK_dr = 2.0 * r * omega

    two_delta_over_r = 2.0 * delta / r
    d_two_delta_over_r = 2.0 * (ddelta / r - delta / (r * r))

    return 2.0 * ddelta * (-1.0j * K_val + r - M - two_delta_over_r) + 2.0 * delta * (
        -1.0j * dK_dr + 1.0 - d_two_delta_over_r
    )


def _d_dr_2alpha_plus_dbeta_over_delta(
    r: float, a: float, omega: complex, m: int, lam: complex,
    alpha: complex, beta: complex, delta: float,
) -> complex:
    """Derivative d/dr of (2α + β'/Δ)."""
    M = 1.0
    ddelta = 2.0 * (r - M)
    dbeta_dr = _dbeta_dr(r, a, omega, m)

    # dα/dr = d/dr(-iKβ/Δ² + 3iK' + λ + 6Δ/r²)
    K_val = (r * r + a * a) * omega - m * a
    dK_dr = 2.0 * r * omega
    d2K_dr2 = 2.0 * omega

    # d/dr(-iKβ/Δ²) = -i(K'β + Kβ')/Δ² + 2iKβΔ'/Δ³
    term1 = (
        -1.0j * (dK_dr * beta + K_val * dbeta_dr) / (delta * delta)
        + 2.0j * K_val * beta * ddelta / (delta**3)
    )

    # d/dr(3iK') = 3i K''
    term2 = 3.0j * d2K_dr2

    # d/dr(6Δ/r²) = 6(dΔ/dr / r² - 2Δ/r³) = 6(2(r-M)/r² - 2Δ/r³)
    term3 = 6.0 * (ddelta / (r * r) - 2.0 * delta / (r**3))

    dalpha_dr = term1 + term2 + term3

    # d/dr(β'/Δ) = β''/Δ - β'Δ'/Δ²
    d2beta_dr2 = _d2beta_dr2(r, a, omega, m)
    d_beta_over_delta = d2beta_dr2 / delta - dbeta_dr * ddelta / (delta * delta)

    return 2.0 * dalpha_dr + d_beta_over_delta


def _d2beta_dr2(r: float, a: float, omega: complex, m: int) -> complex:
    """Second derivative of β with respect to r."""
    M = 1.0
    delta = r * r - 2.0 * M * r + a * a
    ddelta = 2.0 * (r - M)
    d2delta = 2.0
    K_val = (r * r + a * a) * omega - m * a
    dK_dr = 2.0 * r * omega
    d2K_dr2 = 2.0 * omega

    two_delta_over_r = 2.0 * delta / r
    d_two_delta_over_r = 2.0 * (ddelta / r - delta / (r * r))
    d2_two_delta_over_r = 2.0 * (
        d2delta / r - 2.0 * ddelta / (r * r) + 2.0 * delta / (r**3)
    )

    return (
        2.0 * d2delta * (-1.0j * K_val + r - M - two_delta_over_r)
        + 4.0 * ddelta * (-1.0j * dK_dr + 1.0 - d_two_delta_over_r)
        + 2.0 * delta * (-1.0j * d2K_dr2 - d2_two_delta_over_r)
    )


def _dG_dr(r: float, a: float) -> float:
    """Derivative of G with respect to r."""
    M = 1.0
    r2_p_a2 = r * r + a * a
    delta = r * r - 2.0 * M * r + a * a
    ddelta = 2.0 * (r - M)

    return (
        -2.0 / r2_p_a2
        + 4.0 * (r - M) * r / (r2_p_a2 * r2_p_a2)
        + (ddelta * r + delta) / (r2_p_a2 * r2_p_a2)
        - 4.0 * r * r * delta / (r2_p_a2**3)
    )


def _sn_radial_up(
    lam: complex, m: int, a: float, omega: complex, r_min: float, rout: float = 1000.0
):
    """Integrate the SN equation for the Up solution from rout down to r_min."""
    X0, dX0, _ = _sn_outer_boundary(lam, m, a, omega, rout)

    sol = solve_ivp(
        lambda r, y: _sn_equation_rhs(r, y, lam, m, a, omega),
        (rout, r_min),
        np.array([X0, dX0], dtype=np.complex128),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
        max_step=100.0,
    )
    if not sol.success:
        raise RuntimeError(f"SN integration failed: {sol.message}")

    return sol.sol


def _sn_to_teukolsky_up_value(
    r: float,
    X: complex,
    dX: complex,
    lam: complex,
    m: int,
    a: float,
    omega: complex,
) -> complex:
    """Convert SN solution X(r) to Teukolsky R_up(r) (value only, no derivative).

    The conversion formula (s=-2):

        χ = X Δ / √(r² + a²)
        R_up = (1/η) [(α + β'/Δ) χ - (β/Δ) χ']
    """
    M = 1.0
    delta = r * r - 2.0 * M * r + a * a
    r2_p_a2 = r * r + a * a

    c0 = -12.0j * omega * M + lam * (lam + 2.0) - 12.0 * a * omega * (a * omega - m)
    c1 = 8.0j * a * (3.0 * a * omega - lam * (a * omega - m))
    c2 = -24.0j * a * M * (a * omega - m) + 12.0 * a * a * (1.0 - 2.0 * (a * omega - m) ** 2)
    c3 = 24.0j * a**3 * (a * omega - m) - 24.0 * M * a * a
    c4 = 12.0 * a**4

    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    eta = c0 + c1 * inv_r + c2 * inv_r2 + c3 * inv_r2 * inv_r + c4 * inv_r2 * inv_r2

    K_val = r2_p_a2 * omega - m * a
    ddelta = 2.0 * (r - M)

    two_delta_over_r = 2.0 * delta * inv_r
    beta = 2.0 * delta * (-1.0j * K_val + r - M - two_delta_over_r)
    alpha = (
        -1.0j * K_val * beta / (delta * delta)
        + 3.0j * 2.0 * r * omega
        + lam
        + 6.0 * delta * inv_r2
    )

    dbeta_dr = _dbeta_dr(r, a, omega, m)
    sqrt_r2_p_a2 = math.sqrt(r2_p_a2)

    chi = X * delta / sqrt_r2_p_a2
    dchi = (
        dX * delta / sqrt_r2_p_a2
        + X * ddelta / sqrt_r2_p_a2
        - X * delta * r / (sqrt_r2_p_a2 * r2_p_a2)
    )

    alpha_plus_dbeta = alpha + dbeta_dr / delta
    return (alpha_plus_dbeta * chi - beta / delta * dchi) / eta


def _sn_to_teukolsky_up(
    r: float,
    X: complex,
    dX: complex,
    lam: complex,
    m: int,
    a: float,
    omega: complex,
) -> tuple[complex, complex]:
    """Convert SN solution X(r) back to Teukolsky function R_up(r).

    The conversion is (for s=-2):

        χ = X Δ / √(r² + a²)
        R_up = (1/η) [(α + β'/Δ) χ - (β/Δ) χ']

    Returns (R_up, dR_up/dr).
    """
    M = 1.0
    delta = r * r - 2.0 * M * r + a * a
    r2_p_a2 = r * r + a * a

    # η coefficients
    c0 = -12.0j * omega * M + lam * (lam + 2.0) - 12.0 * a * omega * (a * omega - m)
    c1 = 8.0j * a * (3.0 * a * omega - lam * (a * omega - m))
    c2 = -24.0j * a * M * (a * omega - m) + 12.0 * a * a * (1.0 - 2.0 * (a * omega - m) ** 2)
    c3 = 24.0j * a**3 * (a * omega - m) - 24.0 * M * a * a
    c4 = 12.0 * a**4

    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    eta = c0 + c1 * inv_r + c2 * inv_r2 + c3 * inv_r2 * inv_r + c4 * inv_r2 * inv_r2

    K_val = r2_p_a2 * omega - m * a

    two_delta_over_r = 2.0 * delta * inv_r
    beta = 2.0 * delta * (-1.0j * K_val + r - M - two_delta_over_r)
    alpha = (
        -1.0j * K_val * beta / (delta * delta)
        + 3.0j * 2.0 * r * omega
        + lam
        + 6.0 * delta * inv_r2
    )

    dbeta_dr = _dbeta_dr(r, a, omega, m)
    ddelta = 2.0 * (r - M)

    # χ = X Δ / √(r²+a²)
    sqrt_r2_p_a2 = math.sqrt(r2_p_a2)
    chi = X * delta / sqrt_r2_p_a2

    # χ' = X' Δ/√ + X Δ'/√ - X Δ r / (r²+a²)^(3/2)
    dchi = (
        dX * delta / sqrt_r2_p_a2
        + X * ddelta / sqrt_r2_p_a2
        - X * delta * r / (sqrt_r2_p_a2 * r2_p_a2)
    )

    # R_up = (1/η) [(α + β'/Δ) χ - (β/Δ) χ']
    alpha_plus_dbeta = alpha + dbeta_dr / delta
    R_up = (alpha_plus_dbeta * chi - beta / delta * dchi) / eta

    # dR_up/dr — compute via finite difference for simplicity
    # (could be done analytically but the expressions are very long)
    dr_step = 1e-5
    r_plus = r + dr_step
    r_minus = r - dr_step

    delta_p = r_plus * r_plus - 2.0 * M * r_plus + a * a
    delta_m = r_minus * r_minus - 2.0 * M * r_minus + a * a
    r2_p_a2_p = r_plus * r_plus + a * a
    r2_p_a2_m = r_minus * r_minus + a * a

    # Use the X solution at nearby points to get derivatives
    # For dR_up/dr we compute R_up at r±dr and differentiate
    inv_r_p = 1.0 / r_plus
    inv_r_m = 1.0 / r_minus
    eta_p = c0 + c1 * inv_r_p + c2 * inv_r_p * inv_r_p + c3 * inv_r_p**3 + c4 * inv_r_p**4
    eta_m = c0 + c1 * inv_r_m + c2 * inv_r_m * inv_r_m + c3 * inv_r_m**3 + c4 * inv_r_m**4

    K_p = r2_p_a2_p * omega - m * a
    K_m = r2_p_a2_m * omega - m * a
    beta_p = 2.0 * delta_p * (-1.0j * K_p + r_plus - M - 2.0 * delta_p / r_plus)
    beta_m = 2.0 * delta_m * (-1.0j * K_m + r_minus - M - 2.0 * delta_m / r_minus)
    alpha_p = (
        -1.0j * K_p * beta_p / (delta_p * delta_p)
        + 3.0j * 2.0 * r_plus * omega
        + lam
        + 6.0 * delta_p / (r_plus * r_plus)
    )
    alpha_m = (
        -1.0j * K_m * beta_m / (delta_m * delta_m)
        + 3.0j * 2.0 * r_minus * omega
        + lam
        + 6.0 * delta_m / (r_minus * r_minus)
    )

    # Approximate X at r±dr using X(r) and X'(r)
    X_p = X + dX * dr_step
    X_m = X - dX * dr_step

    chi_p = X_p * delta_p / math.sqrt(r2_p_a2_p)
    chi_m = X_m * delta_m / math.sqrt(r2_p_a2_m)
    dchi_p = (
        dX * delta_p / math.sqrt(r2_p_a2_p)
        + X_p * 2.0 * (r_plus - M) / math.sqrt(r2_p_a2_p)
        - X_p * delta_p * r_plus / (math.sqrt(r2_p_a2_p) * r2_p_a2_p)
    )
    dchi_m = (
        dX * delta_m / math.sqrt(r2_p_a2_m)
        + X_m * 2.0 * (r_minus - M) / math.sqrt(r2_p_a2_m)
        - X_m * delta_m * r_minus / (math.sqrt(r2_p_a2_m) * r2_p_a2_m)
    )

    dbeta_p = _dbeta_dr(r_plus, a, omega, m)
    dbeta_m = _dbeta_dr(r_minus, a, omega, m)

    R_p = ((alpha_p + dbeta_p / delta_p) * chi_p - beta_p / delta_p * dchi_p) / eta_p
    R_m = ((alpha_m + dbeta_m / delta_m) * chi_m - beta_m / delta_m * dchi_m) / eta_m

    dR_up = (R_p - R_m) / (2.0 * dr_step)

    return R_up, dR_up


def sn_radial_up_solution(
    s: int, ell: int, m: int, a: float, omega: complex, lam: complex
) -> RadialSolution:
    """Compute the s=-2 Teukolsky Up radial solution via the Sasaki-Nakamura method.

    This implements the SN transformation method from Gralla-Porfyriadis-Warburton
    (2015, arXiv:1506.08496).  The SN equation has a short-range potential,
    making the outer boundary condition and numerical integration more stable
    than direct Teukolsky integration.

    Only s = -2 is supported by the SN transformation.
    """
    if s != -2:
        raise ValueError(f"Sasaki-Nakamura method only supports s = -2, got s = {s}")

    rout = 1000.0
    r_min = _rp(a) + 1e-5

    X_sol = _sn_radial_up(lam, m, a, omega, r_min, rout)

    def _radial(r: float) -> complex:
        if r < r_min:
            r = r_min + 1e-5
        if r > rout:
            r = rout - 1e-5
        X_val = complex(X_sol(r)[0])
        dX_val = complex(X_sol(r)[1])
        R_val = _sn_to_teukolsky_up_value(r, X_val, dX_val, lam, m, a, omega)
        return R_val

    def _derivative(order: int, r: float) -> complex:
        if order == 1:
            step = 1e-4
            return (_radial(r + step) - _radial(r - step)) / (2.0 * step)
        if order == 2:
            step = 1e-3
            return (_radial(r + step) - 2.0 * _radial(r) + _radial(r - step)) / (step * step)
        if order == 4:
            step = 1e-2
            coeffs = np.array([1, -4, 6, -4, 1], dtype=float) / (step**4)
            points = np.array([r - 2 * step, r - step, r, r + step, r + 2 * step])
            values = np.array([_radial(p) for p in points])
            return complex(np.dot(coeffs, values))
        raise ValueError(f"SN derivatives implemented for orders 1, 2, and 4, got {order}")

    return RadialSolution(
        s=s,
        l=ell,
        m=m,
        a=a,
        omega=omega,
        eigenvalue=lam,
        renormalized_angular_momentum=complex(0),
        method="SasakiNakamura",
        boundary_conditions="Up",
        amplitudes={
            "Transmission": 1.0 + 0.0j,
            "Incidence": 1.0 + 0.0j,
            "Reflection": 0.0 + 0.0j,
        },
        unscaled_amplitudes={
            "Transmission": 1.0 + 0.0j,
            "Incidence": 1.0 + 0.0j,
            "Reflection": 0.0 + 0.0j,
        },
        domain=(float(r_min), float(rout)),
        radial_function=_radial,
        derivative_function=_derivative,
        method_options=(),
    )


# ---------------------------------------------------------------------------
#  SN In solution -- integrate from the horizon outward
# ---------------------------------------------------------------------------


def _sn_horizon_ics(
    lam: complex, m: int, a: float, omega: complex, h: float,
) -> tuple[complex, complex]:
    """Initial conditions for the SN In solution near the horizon.

    Near the horizon the SN potential vanishes, reducing the equation to

        d²X/dr*² + ω² X = 0 .

    The In solution (purely ingoing at the horizon) is the downgoing wave

        X(r) ~ exp(-i ω r*)   (r → r_+).

    Converting back to the Schwarzschild coordinate r:

        X(r_+ + h) = exp(-i ω r*(r_+ + h))

        dX/dr|_(r_+ + h) = -i ω * (dr*/dr) * X

    where h is a small offset from the horizon (typically ~ 1e-5).

    These ICs are valid for |ω| > 0.  For ω = 0 the static solution
    should be used instead of the dynamical SN integration.
    """
    rp = _rp(a)
    r_h = rp + h

    r_star = _tortoise(r_h, a)

    # X = exp(-i ω r*)  (In / downgoing)
    X0 = np.exp(-1j * omega * r_star)

    # dX/dr = dX/dr* · dr*/dr  =  -i ω · (r² + a²)/Δ · X
    delta = r_h * r_h - 2.0 * r_h + a * a
    drstar_dr = (r_h * r_h + a * a) / delta
    dX0 = -1j * omega * drstar_dr * X0

    return complex(X0), complex(dX0)


def _sn_radial_in(
    lam: complex, m: int, a: float, omega: complex, r_min: float, rout: float = 1000.0,
):
    """Integrate the SN equation from r_min (near horizon) outward to rout.

    Returns a dense-output OdeSolution for [X(r), X'(r)] at any r in
    [r_min, rout].
    """
    h = r_min - _rp(a)
    X0, dX0 = _sn_horizon_ics(lam, m, a, omega, h)

    sol = solve_ivp(
        lambda r, y: _sn_equation_rhs(r, y, lam, m, a, omega),
        (r_min, rout),
        np.array([X0, dX0], dtype=np.complex128),
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
        max_step=100.0,
    )
    if not sol.success:
        raise RuntimeError(f"SN In integration failed: {sol.message}")

    return sol.sol


def sn_radial_in_solution(
    s: int, ell: int, m: int, a: float, omega: complex, lam: complex,
) -> RadialSolution:
    """Compute the s=-2 Teukolsky In radial solution via the Sasaki-Nakamura method.

    Integrates the SN equation from just outside the horizon outward, then
    converts the SN function X(r) back to the Teukolsky function R(r).

    The returned solution has arbitrary normalisation; the caller must fix
    the physical normalisation (transmission amplitude = 1 at the horizon)
    by fitting infinity amplitudes and applying the transmission-normalisation
    factor.
    """
    if s != -2:
        raise ValueError(f"Sasaki-Nakamura method only supports s = -2, got s = {s}")

    rout = 1000.0
    r_min = _rp(a) + 1e-5

    X_sol = _sn_radial_in(lam, m, a, omega, r_min, rout)

    def _radial(r: float) -> complex:
        if r < r_min:
            r = r_min + 1e-5
        if r > rout:
            r = rout - 1e-5
        X_val = complex(X_sol(r)[0])
        dX_val = complex(X_sol(r)[1])
        R_val = _sn_to_teukolsky_up_value(r, X_val, dX_val, lam, m, a, omega)
        return R_val

    def _derivative(order: int, r: float) -> complex:
        if order == 1:
            step = 1e-4
            return (_radial(r + step) - _radial(r - step)) / (2.0 * step)
        if order == 2:
            step = 1e-3
            return (_radial(r + step) - 2.0 * _radial(r) + _radial(r - step)) / (step * step)
        if order == 4:
            step = 1e-2
            coeffs = np.array([1, -4, 6, -4, 1], dtype=float) / (step**4)
            points = np.array([r - 2 * step, r - step, r, r + step, r + 2 * step])
            values = np.array([_radial(p) for p in points])
            return complex(np.dot(coeffs, values))
        raise ValueError(f"SN derivatives implemented for orders 1, 2, and 4, got {order}")

    return RadialSolution(
        s=s,
        l=ell,
        m=m,
        a=a,
        omega=omega,
        eigenvalue=lam,
        renormalized_angular_momentum=complex(0),
        method="SasakiNakamura",
        boundary_conditions="In",
        amplitudes={
            "Transmission": 1.0 + 0.0j,
            "Incidence": 1.0 + 0.0j,
            "Reflection": 0.0 + 0.0j,
        },
        unscaled_amplitudes={
            "Transmission": 1.0 + 0.0j,
            "Incidence": 1.0 + 0.0j,
            "Reflection": 0.0 + 0.0j,
        },
        domain=(float(r_min), float(rout)),
        radial_function=_radial,
        derivative_function=_derivative,
        method_options=(),
    )

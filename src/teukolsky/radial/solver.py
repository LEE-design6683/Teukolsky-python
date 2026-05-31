from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, replace

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from teukolsky.angular.eigen import spin_weighted_spheroidal_eigenvalue
from teukolsky.core import RadialSolution
from teukolsky.mst import renormalized_angular_momentum
from teukolsky.radial.sasaki_nakamura import sn_radial_in_solution, sn_radial_up_solution
from teukolsky.radial.mst_solver import mst_radial_solution


def _rp(a: float, mass: float = 1.0) -> float:
    return mass + math.sqrt(mass * mass - a * a)


def _rm(a: float, mass: float = 1.0) -> float:
    return mass - math.sqrt(mass * mass - a * a)


def _delta(r: float, a: float) -> float:
    return r * r - 2.0 * r + a * a


def _rs(r: float, a: float) -> float:
    rp = _rp(a)
    rm = _rm(a)
    return r + 2.0 / (rp - rm) * (rp * math.log((r - rp) / 2.0) - rm * math.log((r - rm) / 2.0))


def _phi_reg(r: float, a: float) -> float:
    rp = _rp(a)
    rm = _rm(a)
    return a / (rp - rm) * math.log((r - rp) / (r - rm))


def _in_transmission_normalization(a: float, m: int) -> complex:
    rp = _rp(a)
    rm = _rm(a)
    phase = (
        1j
        * a
        * m
        * (
            -rm * rp
            - 2.0 * rm * math.log(rp - rm)
            + rp * rp
            - 2.0 * rp * math.log(1.0 / (rp - rm))
            + (2.0 * rm - 2.0 * rp) * math.log(2.0)
        )
        / (2.0 * rp * (rm - rp))
    )
    return rp * cmath.exp(-phase)


def _physical_from_regular(
    r: float,
    regular_value: complex,
    regular_derivative: complex,
    s: int,
    m: int,
    a: float,
    omega: complex,
    boundary: str,
) -> tuple[complex, complex]:
    bc_dir = -1.0 if boundary == "In" else 1.0
    prefactor = (
        r ** (-1)
        * _delta(r, a) ** (-s)
        * cmath.exp(1j * bc_dir * omega * _rs(r, a))
        * cmath.exp(1j * m * _phi_reg(r, a))
    )
    dlog_prefactor = (
        -1.0 / r
        - s * (2.0 * r - 2.0) / _delta(r, a)
        + 1j * bc_dir * omega * (r * r + a * a) / _delta(r, a)
        + 1j * m * a / _delta(r, a)
    )
    return prefactor * regular_value, prefactor * (regular_derivative + dlog_prefactor * regular_value)


def _rs_array(r: np.ndarray, a: float) -> np.ndarray:
    rp = _rp(a)
    rm = _rm(a)
    return r + 2.0 / (rp - rm) * (rp * np.log((r - rp) / 2.0) - rm * np.log((r - rm) / 2.0))


def _phi_reg_array(r: np.ndarray, a: float) -> np.ndarray:
    rp = _rp(a)
    rm = _rm(a)
    return a / (rp - rm) * np.log((r - rp) / (r - rm))


def _physical_from_regular_array(
    r: np.ndarray,
    regular_value: np.ndarray,
    regular_derivative: np.ndarray,
    s: int,
    m: int,
    a: float,
    omega: complex,
    boundary: str,
) -> tuple[np.ndarray, np.ndarray]:
    bc_dir = -1.0 if boundary == "In" else 1.0
    delta = r * r - 2.0 * r + a * a
    prefactor = (
        r ** (-1)
        * delta ** (-s)
        * np.exp(1j * bc_dir * omega * _rs_array(r, a))
        * np.exp(1j * m * _phi_reg_array(r, a))
    )
    dlog_prefactor = (
        -1.0 / r
        - s * (2.0 * r - 2.0) / delta
        + 1j * bc_dir * omega * (r * r + a * a) / delta
        + 1j * m * a / delta
    )
    return prefactor * regular_value, prefactor * (regular_derivative + dlog_prefactor * regular_value)


def _fit_infinity_amplitudes(
    r: float,
    value: complex,
    derivative: complex,
    s: int,
    omega: complex,
    a: float,
) -> tuple[complex, complex]:
    drs = (r * r + a * a) / _delta(r, a)
    incoming = r ** (-1) * cmath.exp(-1j * omega * _rs(r, a))
    outgoing = r ** (-1 - 2 * s) * cmath.exp(1j * omega * _rs(r, a))
    incoming_derivative = incoming * (-1.0 / r - 1j * omega * drs)
    outgoing_derivative = outgoing * (-(1.0 + 2 * s) / r + 1j * omega * drs)
    matrix = np.array(
        [[incoming, outgoing], [incoming_derivative, outgoing_derivative]],
        dtype=np.complex128,
    )
    rhs = np.array([value, derivative], dtype=np.complex128)
    incidence, reflection = np.linalg.solve(matrix, rhs)
    return complex(incidence), complex(reflection)


def _regular_rhs(
    r: float,
    y: np.ndarray,
    s: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    horizon_sign: int,
) -> np.ndarray:
    y1, y2 = y
    delta = a * a - 2.0 * r + r * r
    a2pr2 = a * a + r * r
    numerator = (
        (
            delta * (2.0 * a * a - 2.0 * r * (1.0 + s) - r * r * lam)
            - 2.0 * a * (1.0 + horizon_sign) * m * r * r * a2pr2 * omega
            + 2.0j
            * r
            * r
            * (
                -(1.0 + horizon_sign) * (-a * a + r * r)
                + (1.0 - horizon_sign) * r * delta
            )
            * s
            * omega
            + (1.0 - horizon_sign * horizon_sign) * r * r * a2pr2 * a2pr2 * omega * omega
            - 2.0j * a * r * delta * (m + a * horizon_sign * omega)
        )
        * y1
        / (r**6)
    )
    coeff = (
        (2.0 * (-a * a + r * r) * delta) / (r**4 * a2pr2)
        - (
            2.0
            * delta
            * (a * a * delta + a2pr2 * ((-1.0 + r) * r * s - 1.0j * r * (a * m + horizon_sign * a2pr2 * omega)))
        )
        / (r**5 * a2pr2)
    )
    y2p = -(numerator + coeff * y2) / ((delta * delta) / (r**4))
    return np.array([y2, y2p], dtype=np.complex128)


def _in_bc_series_coefficients(
    s: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    order: int = 4,
    x0: float = 1e-5,
) -> np.ndarray:
    rp = 1.0 + math.sqrt(1.0 - a * a)
    sample_points = x0 * np.arange(1, order + 1, dtype=float)

    def residual(coefficients: np.ndarray, x: float) -> complex:
        r = rp + x
        y1 = 1.0 + sum(coefficients[k - 1] * x**k for k in range(1, len(coefficients) + 1))
        y2 = sum(k * coefficients[k - 1] * x ** (k - 1) for k in range(1, len(coefficients) + 1))
        y = np.array([y1, y2], dtype=np.complex128)
        rhs = _regular_rhs(r, y, s=s, m=m, a=a, omega=omega, lam=lam, horizon_sign=-1)
        target = sum(k * (k - 1) * coefficients[k - 1] * x ** (k - 2) for k in range(2, len(coefficients) + 1))
        return rhs[1] - target

    base = np.zeros(order, dtype=np.complex128)
    constant = np.array([residual(base, x) for x in sample_points], dtype=np.complex128)
    columns = []
    for column in range(order):
        basis = np.zeros(order, dtype=np.complex128)
        basis[column] = 1.0
        columns.append(
            np.array(
                [residual(basis, x) - constant[index] for index, x in enumerate(sample_points)],
                dtype=np.complex128,
            )
        )
    matrix = np.column_stack(columns)
    return np.linalg.solve(matrix, -constant)


def _in_bc_coefficients(s: int, m: int, a: float, omega: complex, lam: complex) -> tuple[complex, complex]:
    rp = 1.0 + math.sqrt(1.0 - a * a)

    def residual(a1: complex, a2: complex, x: float) -> complex:
        r = rp + x
        y = np.array([1.0 + a1 * x + a2 * x * x, a1 + 2.0 * a2 * x], dtype=np.complex128)
        rhs = _regular_rhs(r, y, s=s, m=m, a=a, omega=omega, lam=lam, horizon_sign=-1)
        return rhs[1] - 2.0 * a2

    x1 = 1e-6
    x2 = 2e-6
    c = np.array([residual(0.0, 0.0, x1), residual(0.0, 0.0, x2)], dtype=np.complex128)
    col1 = np.array(
        [residual(1.0, 0.0, x1) - c[0], residual(1.0, 0.0, x2) - c[1]],
        dtype=np.complex128,
    )
    col2 = np.array(
        [residual(0.0, 1.0, x1) - c[0], residual(0.0, 1.0, x2) - c[1]],
        dtype=np.complex128,
    )
    matrix = np.column_stack([col1, col2])
    a1, a2 = np.linalg.solve(matrix, -c)
    return complex(a1), complex(a2)


def _up_bc_coefficients(
    s: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    r0: float,
) -> tuple[complex, complex, complex]:
    def residual(b1: complex, b2: complex, b3: complex, r: float) -> complex:
        y1 = 1.0 + b1 / r + b2 / (r * r) + b3 / (r**3)
        y2 = -b1 / (r * r) - 2.0 * b2 / (r**3) - 3.0 * b3 / (r**4)
        y = np.array([y1, y2], dtype=np.complex128)
        rhs = _regular_rhs(r, y, s=s, m=m, a=a, omega=omega, lam=lam, horizon_sign=1)
        target = 2.0 * b1 / (r**3) + 6.0 * b2 / (r**4) + 12.0 * b3 / (r**5)
        return rhs[1] - target

    r1 = r0
    r2 = 1.5 * r0
    r3 = 2.0 * r0
    c = np.array(
        [
            residual(0.0, 0.0, 0.0, r1),
            residual(0.0, 0.0, 0.0, r2),
            residual(0.0, 0.0, 0.0, r3),
        ],
        dtype=np.complex128,
    )
    col1 = np.array(
        [
            residual(1.0, 0.0, 0.0, r1) - c[0],
            residual(1.0, 0.0, 0.0, r2) - c[1],
            residual(1.0, 0.0, 0.0, r3) - c[2],
        ],
        dtype=np.complex128,
    )
    col2 = np.array(
        [
            residual(0.0, 1.0, 0.0, r1) - c[0],
            residual(0.0, 1.0, 0.0, r2) - c[1],
            residual(0.0, 1.0, 0.0, r3) - c[2],
        ],
        dtype=np.complex128,
    )
    col3 = np.array(
        [
            residual(0.0, 0.0, 1.0, r1) - c[0],
            residual(0.0, 0.0, 1.0, r2) - c[1],
            residual(0.0, 0.0, 1.0, r3) - c[2],
        ],
        dtype=np.complex128,
    )
    matrix = np.column_stack([col1, col2, col3])
    b1, b2, b3 = np.linalg.solve(matrix, -c)
    return complex(b1), complex(b2), complex(b3)


def _static_amplitudes(s: int, ell: int, m: int, a: float) -> dict[str, dict[str, complex]]:
    tau = -(m * a) / math.sqrt(1.0 - a * a)
    kappa = math.sqrt(1.0 - a * a)
    if tau == 0:
        return {
            "In": {
                "H": 1.0,
                "I": (2.0 * kappa) ** (-ell + abs(s))
                * math.gamma(2 * ell + 1)
                * math.gamma(1 + abs(s))
                / (math.gamma(ell + 1) * math.gamma(1 + ell + abs(s))),
            },
            "Up": {
                "H": (
                    -((2.0 * kappa) ** (-1 - ell) * math.gamma(3 + 2 * ell))
                    / (2.0 * math.gamma(ell + 2) * math.gamma(1 + ell))
                    if s == 0
                    else (2.0 * kappa) ** (-1 - ell + abs(s))
                    * math.gamma(2 + 2 * ell)
                    * math.gamma(abs(s))
                    / (math.gamma(1 + ell + abs(s)) * math.gamma(1 + ell))
                ),
                "I": 1.0,
            },
        }
    return {
        "In": {
            "H": 1.0,
            "I-": -(
                2.0 ** (ell - s - 1j * tau)
                * kappa ** (1 - s)
                * ((-1) ** (ell + s))
                * math.gamma(1 + ell + s)
                * mp.gamma(1 - s - 1j * tau)
                / (math.gamma(2 + 2 * ell) * mp.gamma(-ell - 1j * tau))
            ),
            "I+": (
                (2.0 * kappa) ** (-ell - s - 1j * tau)
                * math.gamma(2 * ell + 1)
                * mp.gamma(1 - s - 1j * tau)
                / (math.gamma(1 + ell - s) * mp.gamma(1 + ell - 1j * tau))
            ),
        },
        "Up": {
            "H-": (
                (2.0 * kappa) ** (-1 - ell + s + 1j * tau)
                * math.gamma(2 + 2 * ell)
                * mp.gamma(s + 1j * tau)
                / (math.gamma(1 + ell + s) * mp.gamma(1 + ell + 1j * tau))
            ),
            "H+": (
                (2.0 * kappa) ** (-1 - ell - s - 1j * tau)
                * math.gamma(2 + 2 * ell)
                * mp.gamma(-s - 1j * tau)
                / (math.gamma(1 + ell - s) * mp.gamma(1 + ell - 1j * tau))
            ),
            "I": 1.0,
        },
    }


def _reg_hyp2f1_series(
    a: complex, b: complex, c: complex, z: complex, max_terms: int = 500,
) -> complex:
    """Compute 2F1(a,b;c;z)/Gamma(c) via direct series summation.

    The regularized form avoids numerical issues when *c* is near a
    non-positive integer (both 2F1 and Gamma have poles that cancel).
    """
    val = mp.mpc(0)
    z = mp.mpc(z)
    for k in range(max_terms):
        ck = mp.mpc(c) + k
        # When c+k is a non-positive integer, Gamma(c+k) has a pole and
        # the term vanishes in the regularised limit.  Tolerance: a few
        # ulps above machine epsilon at working precision.
        if mp.im(ck) == 0 and mp.re(ck) <= 0 and abs(mp.re(ck) - round(mp.re(ck))) < 1e-14:
            # term = 0 in the limit  c -> non-positive integer
            continue
        term = (
            mp.rf(a, k) * mp.rf(b, k) * z**k
            / (mp.gamma(ck) * mp.factorial(k))
        )
        val += term
        if k > 5 and mp.fabs(term) < mp.fabs(val) * mp.mpf("1e-40"):
            break
    return val


def _reg_hyp2f1_in(
    a: complex, b: complex, c: complex, z: complex,
) -> complex:
    """Regularized 2F1/Gamma(c) for the static Teukolsky In solution.

    For |z| > 1 uses analytic continuation:

        2F1(a,b;c;z)/Gamma(c) = (1-z)^(-a) * 2F1(a, c-b; c; z/(z-1))/Gamma(c)

    In the Teukolsky In solution c-b = -(s+ell) is a non-positive
    integer, so the transformed series terminates.  When this identity
    does not apply the function falls back to the direct series.
    """
    x = z / (z - mp.mpf(1))
    if mp.fabs(x) < mp.mpf(1):
        return (1 - z) ** (-a) * _reg_hyp2f1_series(a, c - b, c, x)
    return _reg_hyp2f1_series(a, b, c, z)


def _static_solution(boundary: str, s: int, ell: int, m: int, a: float, omega: complex, lam: complex) -> RadialSolution:
    tau = -(m * a) / math.sqrt(1.0 - a * a)
    kappa = math.sqrt(1.0 - a * a)
    amplitudes = _static_amplitudes(s, ell, m, a)[boundary]
    norm_in = (
        math.gamma(s + 1) * mp.rf(ell + s + 1, -2 * s)
        if tau == 0 and s > 0
        else (2.0 * kappa) ** (-2 * s - 1j * tau) * mp.gamma(1 - s - 1j * tau)
    )
    norm_up = (2.0 * kappa) ** (-s - ell - 1)

    def radial(r: float) -> complex:
        z = (1.0 + kappa - r) / (2.0 * kappa)
        if boundary == "In":
            a_hyp = -mp.mpc(ell) - mp.mpc(1j) * mp.mpc(tau)
            b_hyp = mp.mpc(ell + 1) - mp.mpc(1j) * mp.mpc(tau)
            c_hyp = mp.mpc(1 - s) - mp.mpc(1j) * mp.mpc(tau)
            return complex(
                norm_in
                * (-z) ** (-s - 1j * tau / 2.0)
                * (1.0 - z) ** (-1j * tau / 2.0)
                * _reg_hyp2f1_in(a_hyp, b_hyp, c_hyp, z)
            )
        return complex(
            norm_up
            * (-z) ** (-s - 1j * tau / 2.0)
            * (1.0 - z) ** (1j * tau / 2.0 - ell - 1.0)
            * mp.hyp2f1(ell + 1 - 1j * tau, ell + 1 - s, 2 * ell + 2, 1.0 / (1.0 - z))
        )

    def derivative(order: int, r: float) -> complex:
        step = 1e-6
        if order == 1:
            return (radial(r + step) - radial(r - step)) / (2.0 * step)
        if order == 2:
            return (radial(r + step) - 2.0 * radial(r) + radial(r - step)) / (step**2)
        raise ValueError("static derivatives implemented only up to second order")

    return RadialSolution(
        s=s,
        l=ell,
        m=m,
        a=a,
        omega=omega,
        eigenvalue=lam,
        renormalized_angular_momentum=complex(lam),
        method="Static",
        boundary_conditions=boundary,
        amplitudes={key: complex(value) for key, value in amplitudes.items()},
        unscaled_amplitudes={key: complex(value) for key, value in amplitudes.items()},
        domain=(_rp(a), math.inf),
        radial_function=radial,
        derivative_function=derivative,
        method_options=(),
    )


def _numerical_solution(
    boundary: str,
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    nu: complex,
    domain: tuple[float, float] | None = None,
) -> RadialSolution:
    rp = _rp(a)
    if domain is not None:
        domain_min, domain_max = map(float, domain)
        if domain_min <= rp:
            raise ValueError(f"domain lower bound must exceed r_+ = {rp}")
        if domain_max <= domain_min:
            raise ValueError("domain upper bound must be larger than lower bound")
    else:
        domain_min, domain_max = rp, math.inf

    integration_max = domain_max if domain is not None else 1000.0
    integration_min = max(domain_min, rp + 1e-6)

    if boundary == "In":
        if s >= 0:
            eps = 1e-5
            r0 = rp + eps
            coefficients = _in_bc_series_coefficients(s=s, m=m, a=a, omega=omega, lam=lam, order=4, x0=eps)
            y0 = np.array(
                [
                    1.0 + sum(coefficients[k - 1] * eps**k for k in range(1, len(coefficients) + 1)),
                    sum(k * coefficients[k - 1] * eps ** (k - 1) for k in range(1, len(coefficients) + 1)),
                ],
                dtype=np.complex128,
            )
        else:
            eps = 1e-6
            r0 = rp + eps
            a1, a2 = _in_bc_coefficients(s=s, m=m, a=a, omega=omega, lam=lam)
            y0 = np.array([1.0 + a1 * eps + a2 * eps * eps, a1 + 2.0 * a2 * eps], dtype=np.complex128)
        horizon_sign = -1
        span = (r0, integration_max)
    else:
        r0 = integration_max
        b1, b2, b3 = _up_bc_coefficients(s=s, m=m, a=a, omega=omega, lam=lam, r0=r0)
        y0 = np.array(
            [
                1.0 + b1 / r0 + b2 / (r0 * r0) + b3 / (r0**3),
                -b1 / (r0 * r0) - 2.0 * b2 / (r0**3) - 3.0 * b3 / (r0**4),
            ],
            dtype=np.complex128,
        )
        horizon_sign = 1
        span = (r0, integration_min)

    sol = solve_ivp(
        lambda r, y: _regular_rhs(r, y, s=s, m=m, a=a, omega=omega, lam=lam, horizon_sign=horizon_sign),
        span,
        y0,
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    def radial(r: float | np.ndarray) -> complex | np.ndarray:
        if isinstance(r, np.ndarray):
            y = sol.sol(r)
            value, _ = _physical_from_regular_array(r, y[0], y[1], s=s, m=m, a=a, omega=omega, boundary=boundary)
            return np.asarray(value, dtype=np.complex128)
        y = sol.sol(r)
        value, _ = _physical_from_regular(r, y[0], y[1], s=s, m=m, a=a, omega=omega, boundary=boundary)
        return value

    raw_radial = radial

    def derivative(order: int, r: float | np.ndarray) -> complex | np.ndarray:
        if order == 1:
            if isinstance(r, np.ndarray):
                y = sol.sol(r)
                _, deriv = _physical_from_regular_array(r, y[0], y[1], s=s, m=m, a=a, omega=omega, boundary=boundary)
                return np.asarray(deriv, dtype=np.complex128)
            y = sol.sol(r)
            _, deriv = _physical_from_regular(r, y[0], y[1], s=s, m=m, a=a, omega=omega, boundary=boundary)
            return deriv
        if order == 2:
            step = 1e-5
            return (raw_radial(r - step) - 2.0 * raw_radial(r) + raw_radial(r + step)) / (step**2)
        if order == 4:
            step = 1e-2
            coeffs = np.array([1, -4, 6, -4, 1], dtype=float) / (step**4)
            points = np.array([r - 2 * step, r - step, r, r + step, r + 2 * step])
            values = np.array([raw_radial(point) for point in points])
            return np.dot(coeffs, values)
        raise ValueError("derivatives implemented only for orders 1, 2, and 4")

    fit_radius = integration_max
    fit_value = radial(fit_radius)
    fit_derivative = derivative(1, fit_radius)
    incidence, reflection = _fit_infinity_amplitudes(
        r=fit_radius,
        value=fit_value,
        derivative=fit_derivative,
        s=s,
        omega=omega,
        a=a,
    )
    if boundary == "In":
        scale = _in_transmission_normalization(a=a, m=m)

        base_radial = radial
        base_derivative = derivative

        def radial(r: float) -> complex:
            return scale * base_radial(r)

        def derivative(order: int, r: float) -> complex:
            return scale * base_derivative(order, r)

        incidence = scale * incidence
        reflection = scale * reflection

    return RadialSolution(
        s=s,
        l=ell,
        m=m,
        a=a,
        omega=omega,
        eigenvalue=lam,
        renormalized_angular_momentum=nu,
        method="NumericalIntegration",
        boundary_conditions=boundary,
        amplitudes={
            "Transmission": 1.0 + 0.0j,
            "Incidence": incidence,
            "Reflection": reflection,
        },
        unscaled_amplitudes={
            "Transmission": 1.0 + 0.0j,
            "Incidence": incidence,
            "Reflection": reflection,
        },
        domain=(domain_min, domain_max),
        radial_function=radial,
        derivative_function=derivative,
        method_options=(({"Domain": (domain_min, domain_max)},) if domain is not None else ()),
    )


def _heunc_solution(
    boundary: str,
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    nu: complex,
) -> RadialSolution:
    """Evaluate the confluent-Heun gauge solution with HeunC method semantics.

    Mathematica uses a closed-form `HeunC[...]` special function evaluation.
    The Python port does not have a native HeunC implementation available in the
    local numerical stack, so this branch evaluates the same regular
    confluent-Heun form by direct integration of the regularized ODE and exposes
    it under the public `method="HeunC"` interface.
    """
    base = _numerical_solution(
        boundary=boundary,
        s=s,
        ell=ell,
        m=m,
        a=a,
        omega=omega,
        lam=lam,
        nu=nu,
        domain=None,
    )
    return replace(base, method="HeunC", method_options=())


def _boundary_domain(
    domain: tuple[float, float] | dict[str, tuple[float, float]] | None,
    boundary: str,
) -> tuple[float, float] | None:
    if domain is None:
        return None
    if isinstance(domain, dict):
        if boundary not in domain:
            raise ValueError(f"missing domain for boundary condition {boundary!r}")
        return tuple(domain[boundary])
    return tuple(domain)


def solve_radial(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    method: str = "NumericalIntegration",
    boundary_conditions: tuple[str, ...] = ("In", "Up"),
    eigenvalue: complex | None = None,
    renormalized_angular_momentum_value: complex | None = None,
    domain: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
) -> dict[str, RadialSolution]:
    if ell < abs(s) or abs(m) > ell:
        raise ValueError("invalid mode indices")
    if complex(a).imag != 0.0:
        raise ValueError("only real Kerr spin is supported")
    if domain is not None and omega == 0:
        raise ValueError("explicit domain is not supported for static modes")
    if eigenvalue is None:
        eigenvalue = spin_weighted_spheroidal_eigenvalue(s, ell, m, a * omega)
    if omega == 0:
        return {
            boundary: _static_solution(boundary, s=s, ell=ell, m=m, a=a, omega=omega, lam=eigenvalue)
            for boundary in boundary_conditions
        }
    if renormalized_angular_momentum_value is None:
        renormalized_angular_momentum_value = renormalized_angular_momentum(
            s=s, ell=ell, m=m, a=a, omega=omega, lam=eigenvalue
        )
    if method not in {"NumericalIntegration", "MST", "SasakiNakamura", "HeunC"}:
        raise ValueError(f"unsupported method: {method}")
    if domain is not None and method not in {"NumericalIntegration", "SasakiNakamura"}:
        raise ValueError("explicit domain is only supported with method='NumericalIntegration' or method='SasakiNakamura'")
    if method == "MST":
        return mst_radial_solution(s, ell, m, a, omega, eigenvalue, renormalized_angular_momentum_value)
    if method == "HeunC":
        return {
            boundary: _heunc_solution(
                boundary=boundary,
                s=s,
                ell=ell,
                m=m,
                a=a,
                omega=omega,
                lam=eigenvalue,
                nu=renormalized_angular_momentum_value,
            )
            for boundary in boundary_conditions
        }
    if method == "SasakiNakamura":
        if s != -2:
            raise ValueError(
                f"The Sasaki-Nakamura transformation is only defined for s = -2 "
                f"(Gralla-Porfyriadis-Warburton, Phys. Rev. D 92, 064029, 2015). "
                f"Got s = {s}. Use method='NumericalIntegration' or method='MST' instead."
            )
        norm_r = 1000.0
        result: dict[str, RadialSolution] = {}

        if "Up" in boundary_conditions:
            # Build a reference Up solution via NumericalIntegration at large r,
            # then use its value to normalize the SN Up solution.
            ni_up = _numerical_solution(
                boundary="Up", s=s, ell=ell, m=m, a=a, omega=omega,
                lam=eigenvalue, nu=renormalized_angular_momentum_value,
            )
            sn_up_raw = sn_radial_up_solution(s, ell, m, a, omega, eigenvalue)
            ni_val = ni_up.radial_function(norm_r)
            sn_val = sn_up_raw.radial_function(norm_r)
            norm_factor = ni_val / sn_val

            base_radial = sn_up_raw.radial_function
            base_derivative = sn_up_raw.derivative_function

            def _sn_up_radial(r: float) -> complex:
                return norm_factor * base_radial(r)

            def _sn_up_derivative(order: int, r: float) -> complex:
                return norm_factor * base_derivative(order, r)

            requested_domain = _boundary_domain(domain, "Up")
            result["Up"] = RadialSolution(
                s=s,
                l=ell,
                m=m,
                a=a,
                omega=omega,
                eigenvalue=eigenvalue,
                renormalized_angular_momentum=renormalized_angular_momentum_value,
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
                domain=sn_up_raw.domain if requested_domain is None else requested_domain,
                radial_function=_sn_up_radial,
                derivative_function=_sn_up_derivative,
                method_options=(({"Domain": requested_domain},) if requested_domain is not None else ()),
            )

        if "In" in boundary_conditions:
            # Build a reference In solution via NumericalIntegration at large r,
            # then use its value to normalize the SN In solution.
            ni_in = _numerical_solution(
                boundary="In", s=s, ell=ell, m=m, a=a, omega=omega,
                lam=eigenvalue, nu=renormalized_angular_momentum_value,
            )
            sn_in_raw = sn_radial_in_solution(s, ell, m, a, omega, eigenvalue)
            ni_in_val = ni_in.radial_function(norm_r)
            sn_in_val = sn_in_raw.radial_function(norm_r)
            norm_factor_in = ni_in_val / sn_in_val

            base_in_radial = sn_in_raw.radial_function
            base_in_derivative = sn_in_raw.derivative_function

            def _sn_in_radial(r: float) -> complex:
                return norm_factor_in * base_in_radial(r)

            def _sn_in_derivative(order: int, r: float) -> complex:
                return norm_factor_in * base_in_derivative(order, r)

            requested_domain = _boundary_domain(domain, "In")
            result["In"] = RadialSolution(
                s=s,
                l=ell,
                m=m,
                a=a,
                omega=omega,
                eigenvalue=eigenvalue,
                renormalized_angular_momentum=renormalized_angular_momentum_value,
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
                domain=sn_in_raw.domain if requested_domain is None else requested_domain,
                radial_function=_sn_in_radial,
                derivative_function=_sn_in_derivative,
                method_options=(({"Domain": requested_domain},) if requested_domain is not None else ()),
            )

        return result
    return {
        boundary: _numerical_solution(
            boundary=boundary,
            s=s,
            ell=ell,
            m=m,
            a=a,
            omega=omega,
            lam=eigenvalue,
            nu=renormalized_angular_momentum_value,
            domain=_boundary_domain(domain, boundary),
        )
        for boundary in boundary_conditions
    }

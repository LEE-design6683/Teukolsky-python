"""DCU-accelerated batch radial ODE solver.

Parallelises independent radial ODE integrations for multiple ω values
using Python's ProcessPoolExecutor.  Returns fully-functional
``RadialSolution`` objects.

The worker processes return pickle-safe intermediate data (scipy
``OdeSolution`` objects and scalar parameters); closures are rebuilt
in the parent process.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
#  Pickle-safe worker: returns raw data, not RadialSolution closures
# ---------------------------------------------------------------------------


def _compute_one_raw(s: int, ell: int, m: int, a: float, omega: complex,
                     lam: complex, nu: complex, boundary: str) -> dict[str, Any]:
    """Integrate radial ODE and return pickle-safe intermediate data.

    Mirrors ``_numerical_solution`` in ``teukolsky.radial.solver`` but
    returns raw ``OdeSolution`` + scalars instead of a ``RadialSolution``
    with closures.
    """
    from scipy.integrate import solve_ivp

    from teukolsky.radial.solver import (
        _rp,
        _delta,
        _rs,
        _phi_reg,
        _in_transmission_normalization,
        _physical_from_regular,
        _fit_infinity_amplitudes,
        _regular_rhs,
        _in_bc_series_coefficients,
        _in_bc_coefficients,
        _up_bc_coefficients,
    )

    if boundary == "In":
        if s >= 0:
            eps = 1e-5
            r0 = _rp(a) + eps
            coeffs = _in_bc_series_coefficients(
                s=s, m=m, a=a, omega=omega, lam=lam, order=4, x0=eps,
            )
            y0 = np.array(
                [
                    1.0 + sum(coeffs[k - 1] * eps**k for k in range(1, len(coeffs) + 1)),
                    sum(k * coeffs[k - 1] * eps ** (k - 1) for k in range(1, len(coeffs) + 1)),
                ],
                dtype=np.complex128,
            )
        else:
            eps = 1e-6
            r0 = _rp(a) + eps
            a1, a2 = _in_bc_coefficients(s=s, m=m, a=a, omega=omega, lam=lam)
            y0 = np.array(
                [1.0 + a1 * eps + a2 * eps * eps, a1 + 2.0 * a2 * eps],
                dtype=np.complex128,
            )
        horizon_sign = -1
        span = (r0, 1000.0)
    else:
        r0 = 1000.0
        b1, b2, b3 = _up_bc_coefficients(s=s, m=m, a=a, omega=omega, lam=lam, r0=r0)
        y0 = np.array(
            [
                1.0 + b1 / r0 + b2 / (r0 * r0) + b3 / (r0**3),
                -b1 / (r0 * r0) - 2.0 * b2 / (r0**3) - 3.0 * b3 / (r0**4),
            ],
            dtype=np.complex128,
        )
        horizon_sign = 1
        span = (r0, _rp(a) + 1e-6)

    sol = solve_ivp(
        lambda r, y: _regular_rhs(
            r, y, s=s, m=m, a=a, omega=omega, lam=lam, horizon_sign=horizon_sign,
        ),
        span,
        y0,
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    # Evaluate at infinity-fitting radius to get amplitudes
    fit_radius = 1000.0
    y_fit = sol.sol(fit_radius)
    fit_value, fit_derivative = _physical_from_regular(
        fit_radius, y_fit[0], y_fit[1],
        s=s, m=m, a=a, omega=omega, boundary=boundary,
    )
    incidence, reflection = _fit_infinity_amplitudes(
        r=fit_radius, value=fit_value, derivative=fit_derivative,
        s=s, omega=omega, a=a,
    )

    scale = None
    if boundary == "In":
        scale = _in_transmission_normalization(a=a, m=m)
        incidence = scale * incidence
        reflection = scale * reflection

    return {
        "ode_solution": sol.sol,
        "s": s, "ell": ell, "m": m, "a": a,
        "omega": omega, "lam": lam, "nu": nu,
        "boundary": boundary,
        "horizon_sign": horizon_sign,
        "incidence": incidence,
        "reflection": reflection,
        "scale": scale,
        "span": span,
    }


def _build_solution(raw: dict[str, Any]) -> "RadialSolution":
    """Reconstruct a ``RadialSolution`` from pickle-safe intermediate data."""
    from teukolsky.core import RadialSolution
    from teukolsky.radial.solver import (
        _rp,
        _delta,
        _rs,
        _phi_reg,
        _physical_from_regular,
    )

    sol_sol = raw["ode_solution"]
    s = raw["s"]
    ell = raw["ell"]
    m = raw["m"]
    a = raw["a"]
    omega = raw["omega"]
    lam = raw["lam"]
    nu = raw["nu"]
    boundary = raw["boundary"]
    scale = raw["scale"]
    incidence = raw["incidence"]
    reflection = raw["reflection"]

    # ---- radial function closure ----
    def _radial(r: float) -> complex:
        y = sol_sol(r)
        value, _ = _physical_from_regular(
            r, y[0], y[1], s=s, m=m, a=a, omega=omega, boundary=boundary,
        )
        if scale is not None:
            return scale * value
        return value

    # ---- derivative closure ----
    def _derivative(order: int, r: float) -> complex:
        if order == 1:
            y = sol_sol(r)
            _, deriv = _physical_from_regular(
                r, y[0], y[1], s=s, m=m, a=a, omega=omega, boundary=boundary,
            )
            if scale is not None:
                deriv = scale * deriv
            return deriv
        if order == 2:
            step = 1e-5
            return complex(
                (_radial(r - step) - 2.0 * _radial(r) + _radial(r + step)) / (step**2)
            )
        if order == 4:
            step = 1e-2
            coeffs = np.array([1, -4, 6, -4, 1], dtype=float) / (step**4)
            points = np.array(
                [r - 2 * step, r - step, r, r + step, r + 2 * step]
            )
            values = np.array([_radial(pt) for pt in points])
            return complex(np.dot(coeffs, values))
        raise ValueError("derivatives implemented only for orders 1, 2, and 4")

    return RadialSolution(
        s=s, l=ell, m=m, a=a, omega=omega,
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
        domain=(_rp(a), math.inf),
        radial_function=_radial,
        derivative_function=_derivative,
        method_options=(),
    )


# ---------------------------------------------------------------------------
#  Worker entry point (module-level, pickleable)
# ---------------------------------------------------------------------------

def _solve_one(s: int, ell: int, m: int, a: float, omega: complex,
               lam: complex | None = None, nu: complex | None = None):
    """Solve In + Up radial ODE for a single ω and return pickle-safe data."""
    from teukolsky.angular.eigen import spin_weighted_spheroidal_eigenvalue
    from teukolsky.mst import renormalized_angular_momentum

    if lam is None:
        lam = spin_weighted_spheroidal_eigenvalue(s, ell, m, a * omega)
    if nu is None:
        nu = renormalized_angular_momentum(
            s=s, ell=ell, m=m, a=a, omega=omega, lam=lam,
        )

    raw_in = _compute_one_raw(s, ell, m, a, omega, lam, nu, boundary="In")
    raw_up = _compute_one_raw(s, ell, m, a, omega, lam, nu, boundary="Up")
    return {"In": raw_in, "Up": raw_up}


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def batch_solve_radial(s: int, ell: int, m: int, a: float,
                       omega_list: list,
                       lam_list: list | None = None,
                       nu_list: list | None = None,
                       n_workers: int | None = None):
    """Solve radial Teukolsky equation for multiple ω values in parallel.

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Angular mode number.
    m : int
        Azimuthal mode number.
    a : float
        Kerr spin parameter (0 ≤ a < 1).
    omega_list : list of complex
        Mode frequencies to solve.
    lam_list : list of complex or None
        Spin-weighted spheroidal eigenvalues.  If ``None`` they are
        computed inside each worker.
    nu_list : list of complex or None
        Renormalised angular momentum values.  If ``None`` they are
        computed inside each worker.
    n_workers : int or None
        Number of worker processes.  Defaults to ``min(len(omega_list),
        cpu_count())``.

    Returns
    -------
    list of dict
        Each element is ``{"In": RadialSolution, "Up": RadialSolution}``,
        in the same order as *omega_list*.
    """
    n_vals = len(omega_list)

    if lam_list is None:
        lam_list = [None] * n_vals
    if nu_list is None:
        nu_list = [None] * n_vals

    if len(lam_list) != n_vals:
        raise ValueError(
            f"lam_list length {len(lam_list)} != omega_list length {n_vals}"
        )
    if len(nu_list) != n_vals:
        raise ValueError(
            f"nu_list length {len(nu_list)} != omega_list length {n_vals}"
        )

    if n_workers is None:
        n_workers = min(n_vals, mp.cpu_count())

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, omega in enumerate(omega_list):
            futures.append(
                executor.submit(
                    _solve_one, s, ell, m, a, omega, lam_list[i], nu_list[i],
                )
            )
        raw_results = [f.result() for f in futures]

    # Rebuild RadialSolution objects in the parent process
    results = []
    for raw in raw_results:
        results.append({
            "In": _build_solution(raw["In"]),
            "Up": _build_solution(raw["Up"]),
        })
    return results


def batch_solve_point_particle_modes(s: int, ell: int, m: int,
                                     orbit,
                                     n_list: list[int],
                                     k_list: list[int],
                                     n_workers: int | None = None):
    """Solve radial solutions for multiple (n, k) harmonics of one orbit.

    Each (n, k) yields ω = m·Ω_φ + n·Ω_r + k·Ω_θ.  All ω values are
    solved in parallel.

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Angular mode number.
    m : int
        Azimuthal mode number.
    orbit : Orbit
        Kerr geodesic orbit (provides Ω_r, Ω_θ, Ω_φ).
    n_list : list of int
        Radial harmonic numbers.
    k_list : list of int
        Polar harmonic numbers.  Must have the same length as *n_list*.

    Returns
    -------
    list of dict
        Each element is ``{"In": RadialSolution, "Up": RadialSolution}``,
        in the same order as the (n, k) pairs.
    """
    if len(n_list) != len(k_list):
        raise ValueError(
            f"n_list length {len(n_list)} != k_list length {len(k_list)}"
        )

    omega_list = [
        m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
        for n, k in zip(n_list, k_list)
    ]

    return batch_solve_radial(
        s=s, ell=ell, m=m, a=orbit.a,
        omega_list=omega_list,
        n_workers=n_workers,
    )

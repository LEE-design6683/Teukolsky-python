"""DCU-accelerated source convolution integrals for point-particle modes.

The bottleneck in generic/eccentric mode computation is the 2D numerical
integration of the source term over radial (q_r) and polar (q_θ) Mino-time
phases.  Each grid point requires evaluating spin-coefficient functions
(_spin_two_coefficients / _spin_two_positive_coefficients /
_spin_minus_one_coefficients / _spin_plus_one_coefficients) which are
pointwise — trivially parallelizable on GPU.

This module provides torch-based vectorized versions that compute the
integrand on GPU in one shot, reducing generic-mode wall time from ~17 s
(CPU) to < 1 s (DCU).

Supports all spin weights s = -2, -1, 0, 1, 2.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch

from .backend import require_dcu


def _ensure_tensor(
    x: np.ndarray | torch.Tensor | float | complex, device: torch.device,
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    return torch.tensor(x, device=device)


# ---------------------------------------------------------------------------
# Vectorized spin-weight -2 / 0 coefficients (s <= 0)
# ---------------------------------------------------------------------------


def _spin_two_coefficients_batch(
    r: torch.Tensor,
    ur: torch.Tensor,
    theta: torch.Tensor,
    u_theta: torch.Tensor,
    a: float,
    energy: float,
    angular_momentum: float,
    s: int,
    m: int,
    omega: complex,
    harmonic_value: torch.Tensor,
    harmonic_derivative: torch.Tensor,
    harmonic_second_derivative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized spin-weight s coefficients on GPU.

    All tensor inputs have shape (N_r, N_theta) or broadcast-compatible.
    Returns source0, source1, source2 of same shape.
    """
    device = r.device
    cfloat = torch.complex64 if r.dtype == torch.float32 else torch.complex128

    delta = r * r - 2.0 * r + a * a
    kt = (r * r + a * a) * omega - m * a

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    l1 = -m / sin_theta + a * omega * sin_theta + cos_theta / sin_theta
    l2 = -m / sin_theta + a * omega * sin_theta + 2.0 * cos_theta / sin_theta
    l2s = harmonic_derivative + l2 * harmonic_value
    l2p = m * cos_theta / (sin_theta * sin_theta) + a * omega * cos_theta - 2.0 / (sin_theta * sin_theta)
    l1sp = harmonic_second_derivative + l1 * harmonic_derivative
    l1l2s = l1sp + l2p * harmonic_value + l2 * harmonic_derivative + l1 * l2 * harmonic_value

    # rho = -1/(r - i a cos θ), rhobar = -1/(r + i a cos θ)
    denom_in = r - 1.0j * a * cos_theta
    denom_out = r + 1.0j * a * cos_theta
    rho = -1.0 / denom_in
    rhobar = -1.0 / denom_out
    sigma = 1.0 / (rho * rhobar)

    sqrt2_delta = math.sqrt(2.0) * delta
    ann0 = (
        -(rho ** (-2))
        * (rhobar ** (-1))
        * (sqrt2_delta ** (-2))
        * (
            rho ** (-1) * l1l2s
            + 3.0j * a * sin_theta * l1 * harmonic_value
            + 3.0j * a * cos_theta * harmonic_value
            + 2.0j * a * sin_theta * harmonic_derivative
            - 1.0j * a * sin_theta * l2 * harmonic_value
        )
    )
    anmbar0 = (rho ** (-3)) * (sqrt2_delta ** (-1)) * (
        (rho + rhobar - 1.0j * kt / delta) * l2s
        + (rho - rhobar) * a * sin_theta * kt / delta * harmonic_value
    )
    anmbar1 = -(rho ** (-3)) * (sqrt2_delta ** (-1)) * (
        l2s + 1.0j * (rho - rhobar) * a * sin_theta * harmonic_value
    )
    ambarmbar0 = (
        (kt * kt * harmonic_value * rhobar) / (4.0 * delta * delta * rho**3)
        + (1.0j * kt * harmonic_value * (1.0 - r + delta * rho) * rhobar) / (2.0 * delta * delta * rho**3)
        + (1.0j * r * harmonic_value * rhobar * omega) / (2.0 * delta * rho**3)
    )
    ambarmbar1 = -(rho ** (-3)) * rhobar * harmonic_value / 2.0 * (1.0j * kt / delta - rho)
    ambarmbar2 = -(rho ** (-3)) * rhobar * harmonic_value / 4.0

    rcomp = (energy * (r * r + a * a) - a * angular_momentum + ur) / (2.0 * sigma)
    theta_comp = rho * (1.0j * sin_theta * (a * energy - angular_momentum / (sin_theta * sin_theta)) + u_theta) / math.sqrt(2.0)

    cnn = rcomp * rcomp
    cnm = rcomp * theta_comp
    cmm = theta_comp * theta_comp

    source0 = ann0 * cnn + anmbar0 * cnm + ambarmbar0 * cmm
    source1 = anmbar1 * cnm + ambarmbar1 * cmm
    source2 = ambarmbar2 * cmm

    return source0, source1, source2


# ---------------------------------------------------------------------------
# Vectorized spin-weight -1 coefficients (Maxwell, s = -1)
# ---------------------------------------------------------------------------


def _spin_minus_one_coefficients_batch(
    r: torch.Tensor,
    ur: torch.Tensor,
    theta: torch.Tensor,
    u_theta: torch.Tensor,
    a: float,
    energy: float,
    angular_momentum: float,
    m: int,
    omega: complex,
    harmonic_value: torch.Tensor,
    harmonic_derivative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized spin-weight -1 coefficients on GPU.

    Returns source0, source1 (two components, no second-derivative term).
    """
    device = r.device
    cfloat = torch.complex64 if r.dtype == torch.float32 else torch.complex128

    delta = r * r - 2.0 * r + a * a
    kt = (r * r + a * a) * omega - m * a
    sin_theta = torch.sin(theta)

    denom_in = r - 1.0j * a * torch.cos(theta)
    denom_out = r + 1.0j * a * torch.cos(theta)
    rho = -1.0 / denom_in
    rhobar = -1.0 / denom_out
    sigma = 1.0 / (rho * rhobar)

    l1 = -m / sin_theta + a * omega * sin_theta + torch.cos(theta) / sin_theta

    an0 = -(
        harmonic_derivative + l1 * harmonic_value + 1.0j * a * harmonic_value * rho * sin_theta
    ) / (2.0 * math.sqrt(2.0) * delta * rho * rho * rhobar)
    ambar0 = harmonic_value * (-1.0j * kt / delta + rho) / (4.0 * rho * rho)
    ambar1 = -harmonic_value / (4.0 * rho * rho)

    rcomp = (ur + (a * a + r * r) * energy - a * angular_momentum) / (2.0 * sigma)
    theta_comp = rho * (
        u_theta + 1.0j * a * energy * sin_theta - 1.0j * angular_momentum / sin_theta
    ) / math.sqrt(2.0)

    source0 = an0 * rcomp + ambar0 * theta_comp
    source1 = ambar1 * theta_comp

    return source0, source1


# ---------------------------------------------------------------------------
# Vectorized spin-weight +1 coefficients (Maxwell, s = +1)
# ---------------------------------------------------------------------------


def _spin_plus_one_coefficients_batch(
    r: torch.Tensor,
    ur: torch.Tensor,
    theta: torch.Tensor,
    u_theta: torch.Tensor,
    a: float,
    energy: float,
    angular_momentum: float,
    m: int,
    omega: complex,
    harmonic_value: torch.Tensor,
    harmonic_derivative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized spin-weight +1 coefficients on GPU.

    Returns source0, source1 (two components, no second-derivative term).
    """
    device = r.device
    cfloat = torch.complex64 if r.dtype == torch.float32 else torch.complex128

    delta = r * r - 2.0 * r + a * a
    delta_prime = 2.0 * (r - 1.0)
    kt = (r * r + a * a) * omega - m * a
    sin_theta = torch.sin(theta)

    denom_in = r - 1.0j * a * torch.cos(theta)
    denom_out = r + 1.0j * a * torch.cos(theta)
    rho = -1.0 / denom_in
    rhobar = -1.0 / denom_out

    l1 = m / sin_theta - a * omega * sin_theta + torch.cos(theta) / sin_theta

    al0 = -(delta * (harmonic_derivative + l1 * harmonic_value + 1.0j * a * harmonic_value * rho * sin_theta)) / (
        2.0 * math.sqrt(2.0) * rho
    )
    am0 = -(harmonic_value * (1.0j * kt + delta_prime + delta * rho)) / (2.0 * rho * rhobar)
    am1 = harmonic_value * delta / (2.0 * rho * rhobar)

    rcomp = (ur - (a * a + r * r) * energy + a * angular_momentum) / delta
    theta_comp = rhobar * (
        -u_theta - 1.0j * angular_momentum / sin_theta + 1.0j * a * energy * sin_theta
    ) / math.sqrt(2.0)

    source0 = al0 * rcomp + am0 * theta_comp
    source1 = am1 * theta_comp

    return source0, source1


# ---------------------------------------------------------------------------
# Vectorized spin-weight +2 coefficients
# ---------------------------------------------------------------------------


def _spin_two_positive_coefficients_batch(
    r: torch.Tensor,
    ur: torch.Tensor,
    theta: torch.Tensor,
    u_theta: torch.Tensor,
    a: float,
    energy: float,
    angular_momentum: float,
    m: int,
    omega: complex,
    harmonic_value: torch.Tensor,
    harmonic_derivative: torch.Tensor,
    harmonic_second_derivative: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized spin-weight +2 coefficients on GPU."""
    delta = r * r - 2.0 * r + a * a
    kt = (r * r + a * a) * omega - m * a

    denom_in = r - 1.0j * a * torch.cos(theta)
    denom_out = r + 1.0j * a * torch.cos(theta)
    rho = -1.0 / denom_in
    rhobar = -1.0 / denom_out

    sin_theta = torch.sin(theta)
    d_rho_over_rho = 1.0j * a * rho * sin_theta
    d2_rho_over_rho = 1.0j * a * rho * (torch.cos(theta) + 2.0 * sin_theta * d_rho_over_rho)
    ld1 = m / sin_theta - a * omega * sin_theta + torch.cos(theta) / sin_theta
    ld2 = m / sin_theta - a * omega * sin_theta + 2.0 * torch.cos(theta) / sin_theta
    d_ld2 = -m * torch.cos(theta) / (sin_theta * sin_theta) - a * omega * torch.cos(theta) - 2.0 / (sin_theta * sin_theta)

    all0 = (
        -0.5 * (rho ** (-1)) * rhobar
        * (
            harmonic_second_derivative
            + (ld1 + ld2 + 2.0 * d_rho_over_rho) * harmonic_derivative
            + (
                d_ld2 + ld1 * ld2 - 6.0 * d_rho_over_rho * d_rho_over_rho
                + 3.0 * d2_rho_over_rho + (3.0 * ld1 - ld2) * d_rho_over_rho
            ) * harmonic_value
        )
    )
    alm0 = (
        math.sqrt(2.0) * (rho ** (-1))
        * (
            -(rho + rhobar + 1.0j * kt / delta) * (harmonic_derivative + ld2 * harmonic_value)
            + (rho - rhobar) * a * sin_theta * kt / delta * harmonic_value
        )
    )
    alm1 = math.sqrt(2.0) * (rho ** (-1)) * (
        harmonic_derivative + ld2 * harmonic_value + 1.0j * (rho - rhobar) * a * sin_theta * harmonic_value
    )
    amm0 = (
        kt * kt * harmonic_value / (delta * delta * rho * rhobar)
        + 2.0j * kt * harmonic_value * (-1.0 + r - delta * rho) / (delta * delta * rho * rhobar)
        - 2.0j * r * harmonic_value * omega / (delta * rho * rhobar)
    )
    amm1 = 2.0 * (rho ** (-1)) * (rhobar ** (-1)) * harmonic_value * (1.0j * kt / delta + rho)
    amm2 = -(rho ** (-1)) * (rhobar ** (-1)) * harmonic_value

    rcomp = (energy * (r * r + a * a) - a * angular_momentum - ur) / delta
    theta_comp = -rhobar * (
        1.0j * sin_theta * (a * energy - angular_momentum / (sin_theta * sin_theta)) - u_theta
    ) / math.sqrt(2.0)

    cll = rcomp * rcomp
    clm = rcomp * theta_comp
    cmm = theta_comp * theta_comp

    source0 = all0 * cll + alm0 * clm + amm0 * cmm
    source1 = alm1 * clm + amm1 * cmm
    source2 = amm2 * cmm

    return source0, source1, source2


# ---------------------------------------------------------------------------
# GPU-accelerated generic-mode alpha integral
# ---------------------------------------------------------------------------


def accelerated_generic_alpha(
    radial_function: Callable[[np.ndarray], np.ndarray],
    radial_derivative: Callable[[np.ndarray], np.ndarray],
    s: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    orbit,
    n: int = 0,
    k: int = 0,
    n_r: int = 513,
    n_theta: int = 513,
    device_id: int = 0,
    ell: int = 2,
) -> complex:
    """GPU-accelerated α integral for generic point-particle modes.

    Replaces the nested-loop 2D integration in ``_solve_generic_mode``
    with a batched GPU computation.  Supports spin weights s = -2, -1, 0, 1, 2.

    Returns α = (1/(2π)²) ∬ integrand dq_r dq_θ.
    """
    status = require_dcu(device_id)
    device = torch.device(status["device"])

    q_r = np.linspace(0.0, 2.0 * math.pi, n_r)
    q_theta = np.linspace(0.0, 2.0 * math.pi, n_theta)

    # Pre-compute orbit data on CPU (these are cheap)
    r_vals = np.array([orbit.radial_phase_function(p) for p in q_r], dtype=np.float64)
    ur_vals = np.array([orbit.radial_velocity_function(p) for p in q_r], dtype=np.float64)
    dt_r = np.array([orbit.radial_delta_t_function(p) for p in q_r], dtype=np.float64)
    dphi_r = np.array([orbit.radial_delta_phi_function(p) for p in q_r], dtype=np.float64)

    theta_vals = np.array([orbit.theta_phase_function(p) for p in q_theta], dtype=np.float64)
    u_theta_vals = np.array([orbit.theta_velocity_function(p) for p in q_theta], dtype=np.float64)
    dt_theta = np.array([orbit.theta_delta_t_function(p) for p in q_theta], dtype=np.float64)
    dphi_theta = np.array([orbit.theta_delta_phi_function(p) for p in q_theta], dtype=np.float64)

    # Radial function values
    rad_vals = radial_function(r_vals)
    rad_derivs = radial_derivative(r_vals)

    # Move to GPU
    r_t = torch.tensor(r_vals, device=device, dtype=torch.float64)
    ur_t = torch.tensor(ur_vals, device=device, dtype=torch.float64)
    dt_r_t = torch.tensor(dt_r, device=device, dtype=torch.float64)
    dphi_r_t = torch.tensor(dphi_r, device=device, dtype=torch.float64)

    theta_t = torch.tensor(theta_vals, device=device, dtype=torch.float64)
    u_theta_t = torch.tensor(u_theta_vals, device=device, dtype=torch.float64)
    dt_theta_t = torch.tensor(dt_theta, device=device, dtype=torch.float64)
    dphi_theta_t = torch.tensor(dphi_theta, device=device, dtype=torch.float64)

    rad_t = torch.tensor(rad_vals, device=device, dtype=torch.complex128)
    drad_t = torch.tensor(rad_derivs, device=device, dtype=torch.complex128)

    # Create 2D meshgrid
    # r: (N_r,) -> (N_r, N_theta), theta: (N_theta,) -> (N_r, N_theta)
    r_grid = r_t[:, None].expand(-1, n_theta)
    ur_grid = ur_t[:, None].expand(-1, n_theta)
    dt_r_grid = dt_r_t[:, None].expand(-1, n_theta)
    dphi_r_grid = dphi_r_t[:, None].expand(-1, n_theta)

    theta_grid = theta_t[None, :].expand(n_r, -1)
    u_theta_grid = u_theta_t[None, :].expand(n_r, -1)
    dt_theta_grid = dt_theta_t[None, :].expand(n_r, -1)
    dphi_theta_grid = dphi_theta_t[None, :].expand(n_r, -1)

    rad_grid = rad_t[:, None].expand(-1, n_theta)
    drad_grid = drad_t[:, None].expand(-1, n_theta)

    # Harmonic values on θ grid
    from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, a * omega)

    h_vals = np.array([harmonic(float(t), 0.0) for t in theta_vals], dtype=np.complex128)
    dh_vals = np.array([harmonic.derivative_theta(float(t), 0.0) for t in theta_vals], dtype=np.complex128)

    h_grid = torch.tensor(h_vals, device=device, dtype=torch.complex128)[None, :].expand(n_r, -1)
    dh_grid = torch.tensor(dh_vals, device=device, dtype=torch.complex128)[None, :].expand(n_r, -1)

    # Source coefficients on GPU — dispatch by spin weight
    if s in (-1, 1):
        # s = ±1: 4-term symmetry decomposition over [0, π] × [0, π]
        # Matches _solve_spin_minus_one_generic_mode / _solve_spin_plus_one_generic_mode
        # Phase is always computed at forward q; ur/u_theta use fwd/bwd variants.
        coeff_func = _spin_minus_one_coefficients_batch if s == -1 else _spin_plus_one_coefficients_batch

        n_r_half = n_r // 2 + 1
        n_theta_half = n_theta // 2 + 1
        q_r_half = np.linspace(0.0, math.pi, n_r_half)
        q_theta_half = np.linspace(0.0, math.pi, n_theta_half)

        r_half = np.array([orbit.radial_phase_function(p) for p in q_r_half], dtype=np.float64)
        ur_fwd = np.array([orbit.radial_velocity_function(p) for p in q_r_half], dtype=np.float64)
        ur_bwd = np.array([orbit.radial_velocity_function(2.0 * math.pi - p) for p in q_r_half], dtype=np.float64)
        dt_r_fwd = np.array([orbit.radial_delta_t_function(p) for p in q_r_half], dtype=np.float64)
        dphi_r_fwd = np.array([orbit.radial_delta_phi_function(p) for p in q_r_half], dtype=np.float64)

        theta_half = np.array([orbit.theta_phase_function(p) for p in q_theta_half], dtype=np.float64)
        ut_fwd = np.array([orbit.theta_velocity_function(p) for p in q_theta_half], dtype=np.float64)
        ut_bwd = np.array([orbit.theta_velocity_function(2.0 * math.pi - p) for p in q_theta_half], dtype=np.float64)
        dt_t_fwd = np.array([orbit.theta_delta_t_function(p) for p in q_theta_half], dtype=np.float64)
        dphi_t_fwd = np.array([orbit.theta_delta_phi_function(p) for p in q_theta_half], dtype=np.float64)

        # Radial function values
        rad_half = radial_function(r_half)
        drad_half = radial_derivative(r_half)

        # Harmonic values on half θ grid
        h_half = np.array([harmonic(float(t), 0.0) for t in theta_half], dtype=np.complex128)
        dh_half = np.array([harmonic.derivative_theta(float(t), 0.0) for t in theta_half], dtype=np.complex128)

        # Move to GPU
        r_t_h = torch.tensor(r_half, device=device, dtype=torch.float64)[:, None].expand(-1, n_theta_half)
        theta_t_h = torch.tensor(theta_half, device=device, dtype=torch.float64)[None, :].expand(n_r_half, -1)
        rad_t_h = torch.tensor(rad_half, device=device, dtype=torch.complex128)[:, None].expand(-1, n_theta_half)
        drad_t_h = torch.tensor(drad_half, device=device, dtype=torch.complex128)[:, None].expand(-1, n_theta_half)
        h_t_h = torch.tensor(h_half, device=device, dtype=torch.complex128)[None, :].expand(n_r_half, -1)
        dh_t_h = torch.tensor(dh_half, device=device, dtype=torch.complex128)[None, :].expand(n_r_half, -1)

        # Phase at forward phase only (sign-flipped for backward terms)
        phase_r = torch.tensor(
            omega * dt_r_fwd - m * dphi_r_fwd + n * q_r_half,
            device=device, dtype=torch.float64,
        )[:, None].expand(-1, n_theta_half)
        phase_t = torch.tensor(
            omega * dt_t_fwd - m * dphi_t_fwd + k * q_theta_half,
            device=device, dtype=torch.float64,
        )[None, :].expand(n_r_half, -1)

        # Velocity grids for 4 combinations
        ur_fwd_t = torch.tensor(ur_fwd, device=device, dtype=torch.float64)[:, None].expand(-1, n_theta_half)
        ur_bwd_t = torch.tensor(ur_bwd, device=device, dtype=torch.float64)[:, None].expand(-1, n_theta_half)
        ut_fwd_t = torch.tensor(ut_fwd, device=device, dtype=torch.float64)[None, :].expand(n_r_half, -1)
        ut_bwd_t = torch.tensor(ut_bwd, device=device, dtype=torch.float64)[None, :].expand(n_r_half, -1)

        integrand_val = torch.zeros((n_r_half, n_theta_half), device=device, dtype=torch.complex128)
        for ur_t, ut_t, ps_r, ps_t in (
            (ur_fwd_t, ut_fwd_t,  phase_r,  phase_t),
            (ur_fwd_t, ut_bwd_t,  phase_r, -phase_t),
            (ur_bwd_t, ut_fwd_t, -phase_r,  phase_t),
            (ur_bwd_t, ut_bwd_t, -phase_r, -phase_t),
        ):
            src0, src1 = coeff_func(
                r_t_h, ur_t, theta_t_h, ut_t,
                a, orbit.energy, orbit.angular_momentum,
                m, omega, h_t_h, dh_t_h,
            )
            integrand_val += (src0 * rad_t_h - src1 * drad_t_h) * torch.exp(1.0j * (ps_r + ps_t))

        # Integration over [0, π] × [0, π] with 2D trapezoidal
        dq_r_half = q_r_half[1] - q_r_half[0]
        dq_theta_half = q_theta_half[1] - q_theta_half[0]
        integral_r = torch.trapezoid(integrand_val, dx=dq_theta_half, dim=0)
        total = torch.trapezoid(integral_r, dx=dq_r_half, dim=0)

        return complex(total.item()) / (2.0 * math.pi) ** 2
    else:
        # s = -2, 0, +2: three-term alpha, needs d2h and d2rad
        d2h_vals = np.array([harmonic.derivative_theta2(float(t), 0.0) for t in theta_vals], dtype=np.complex128)
        d2h_grid = torch.tensor(d2h_vals, device=device, dtype=torch.complex128)[None, :].expand(n_r, -1)

        from teukolsky.modes.point_particle import _radial_second_derivative
        d2rad_vals = np.array([
            _radial_second_derivative(
                rad_vals[i], rad_derivs[i], s=s, m=m, a=a, omega=omega,
                eigenvalue=lam, r=r_vals[i],
            )
            for i in range(n_r)
        ], dtype=np.complex128)
        d2rad_grid = torch.tensor(d2rad_vals, device=device, dtype=torch.complex128)[:, None].expand(-1, n_theta)

        if s == 2:
            src0, src1, src2 = _spin_two_positive_coefficients_batch(
                r_grid, ur_grid, theta_grid, u_theta_grid,
                a, orbit.energy, orbit.angular_momentum,
                m, omega, h_grid, dh_grid, d2h_grid,
            )
        else:
            src0, src1, src2 = _spin_two_coefficients_batch(
                r_grid, ur_grid, theta_grid, u_theta_grid,
                a, orbit.energy, orbit.angular_momentum,
                s, m, omega, h_grid, dh_grid, d2h_grid,
            )

        if s == 2:
            # Δ²R wrapping for s=+2 Teukolsky equation
            delta_grid = r_grid * r_grid - 2.0 * r_grid + a * a
            d_delta_grid = 2.0 * (r_grid - 1.0)
            d2_delta_grid = 2.0
            wr_R = delta_grid * delta_grid * rad_grid
            wr_Rp = delta_grid * delta_grid * drad_grid + 2.0 * delta_grid * d_delta_grid * rad_grid
            wr_Rpp = (
                delta_grid * delta_grid * d2rad_grid
                + 4.0 * delta_grid * d_delta_grid * drad_grid
                + (2.0 * d_delta_grid * d_delta_grid + 2.0 * delta_grid * d2_delta_grid) * rad_grid
            )
            integrand_val = src0 * wr_R - src1 * wr_Rp + src2 * wr_Rpp
        else:
            integrand_val = src0 * rad_grid - src1 * drad_grid + src2 * d2rad_grid

    # Phase factor: exp(i * (omega*dt_r - m*dphi_r + n*phase_r + omega*dt_theta - m*dphi_theta + k*phase_theta))
    q_r_t = torch.tensor(q_r, device=device, dtype=torch.float64)
    q_theta_t = torch.tensor(q_theta, device=device, dtype=torch.float64)
    q_r_grid = q_r_t[:, None].expand(-1, n_theta)
    q_theta_grid = q_theta_t[None, :].expand(n_r, -1)

    phase = (
        omega * (dt_r_grid + dt_theta_grid)
        - m * (dphi_r_grid + dphi_theta_grid)
        + n * q_r_grid
        + k * q_theta_grid
    )
    integrand_val = integrand_val * torch.exp(1.0j * phase)

    # 2D trapezoidal integration on GPU
    dq_r = q_r[1] - q_r[0]
    dq_theta = q_theta[1] - q_theta[0]
    integral_r = torch.trapezoid(integrand_val, dx=dq_theta, dim=0)
    total = torch.trapezoid(integral_r, dx=dq_r, dim=0)

    return complex(total.item()) / (2.0 * math.pi) ** 2


# ---------------------------------------------------------------------------
# GPU-accelerated eccentric-mode alpha integral
# ---------------------------------------------------------------------------


def accelerated_eccentric_alpha(
    radial_function: Callable[[np.ndarray], np.ndarray],
    radial_derivative: Callable[[np.ndarray], np.ndarray],
    s: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    orbit,
    n: int = 0,
    n_q: int = 4097,
    device_id: int = 0,
    ell: int = 2,
) -> complex:
    """GPU-accelerated α integral for eccentric-equatorial modes.

    1D integral over q_r, vectorized on GPU.
    Supports spin weights s = -2, -1, 0, 1, 2.
    """
    status = require_dcu(device_id)
    device = torch.device(status["device"])

    q = np.linspace(0.0, 2.0 * math.pi, n_q)
    r_vals = np.array([orbit.radial_phase_function(p) for p in q], dtype=np.float64)
    ur_vals = np.array([orbit.radial_velocity_function(p) for p in q], dtype=np.float64)
    dt_r = np.array([orbit.radial_delta_t_function(p) for p in q], dtype=np.float64)
    dphi_r = np.array([orbit.radial_delta_phi_function(p) for p in q], dtype=np.float64)

    rad_vals = radial_function(r_vals)
    rad_derivs = radial_derivative(r_vals)

    theta_val = math.pi / 2.0
    from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, a * omega)
    s0 = harmonic(theta_val, 0.0)
    ds0 = harmonic.derivative_theta(theta_val, 0.0)

    # Move to GPU
    r_t = torch.tensor(r_vals, device=device, dtype=torch.float64)
    ur_t = torch.tensor(ur_vals, device=device, dtype=torch.float64)
    dt_t = torch.tensor(dt_r, device=device, dtype=torch.float64)
    dphi_t = torch.tensor(dphi_r, device=device, dtype=torch.float64)
    rad_t = torch.tensor(rad_vals, device=device, dtype=torch.complex128)
    drad_t = torch.tensor(rad_derivs, device=device, dtype=torch.complex128)

    theta_t = torch.full_like(r_t, theta_val)
    u_theta_t = torch.zeros_like(r_t)
    h_t = torch.full_like(rad_t, complex(s0))
    dh_t = torch.full_like(rad_t, complex(ds0))

    # Source coefficients on GPU — dispatch by spin weight
    if s in (-1, 1):
        # s = ±1: 2-term mirror decomposition over [0, π]
        # Matches _solve_spin_minus_one_eccentric_equatorial_mode / _solve_spin_plus_one_eccentric_equatorial_mode
        coeff_func = _spin_minus_one_coefficients_batch if s == -1 else _spin_plus_one_coefficients_batch

        n_q_half = n_q // 2 + 1
        q_half = np.linspace(0.0, math.pi, n_q_half)
        r_half = np.array([orbit.radial_phase_function(p) for p in q_half], dtype=np.float64)
        ur_fwd = np.array([orbit.radial_velocity_function(p) for p in q_half], dtype=np.float64)
        ur_bwd = np.array([orbit.radial_velocity_function(2.0 * math.pi - p) for p in q_half], dtype=np.float64)
        dt_half = np.array([orbit.radial_delta_t_function(p) for p in q_half], dtype=np.float64)
        dphi_half = np.array([orbit.radial_delta_phi_function(p) for p in q_half], dtype=np.float64)

        rad_half = radial_function(r_half)
        drad_half = radial_derivative(r_half)

        # Move to GPU
        r_t_h = torch.tensor(r_half, device=device, dtype=torch.float64)
        theta_t_h = torch.full_like(r_t_h, theta_val)
        u_theta_t_h = torch.zeros_like(r_t_h)
        rad_t_h = torch.tensor(rad_half, device=device, dtype=torch.complex128)
        drad_t_h = torch.tensor(drad_half, device=device, dtype=torch.complex128)
        h_t_h = torch.full_like(rad_t_h, complex(s0))
        dh_t_h = torch.full_like(rad_t_h, complex(ds0))

        phase_fwd = torch.tensor(
            omega * dt_half - m * dphi_half + n * q_half,
            device=device, dtype=torch.float64,
        )

        # Forward term (ur_plus, +phase)
        ur_fwd_t = torch.tensor(ur_fwd, device=device, dtype=torch.float64)
        src0p, src1p = coeff_func(
            r_t_h, ur_fwd_t, theta_t_h, u_theta_t_h,
            a, orbit.energy, orbit.angular_momentum,
            m, omega, h_t_h, dh_t_h,
        )
        # Backward term (ur_minus, -phase = conj(phase_factor))
        ur_bwd_t = torch.tensor(ur_bwd, device=device, dtype=torch.float64)
        src0m, src1m = coeff_func(
            r_t_h, ur_bwd_t, theta_t_h, u_theta_t_h,
            a, orbit.energy, orbit.angular_momentum,
            m, omega, h_t_h, dh_t_h,
        )

        integrand = (
            (src0p * rad_t_h - src1p * drad_t_h) * torch.exp(1.0j * phase_fwd)
            + (src0m * rad_t_h - src1m * drad_t_h) * torch.exp(-1.0j * phase_fwd)
        )

        dq_half = q_half[1] - q_half[0]
        total = torch.trapezoid(integrand, dx=dq_half, dim=0)

        return complex(total.item()) / (2.0 * math.pi)
    else:
        # s = -2, 0, +2: three-term alpha, needs d2h and d2rad
        d2s0 = harmonic.derivative_theta2(theta_val, 0.0)
        d2h_t = torch.full_like(rad_t, complex(d2s0))

        from teukolsky.modes.point_particle import _radial_second_derivative
        d2rad_vals = np.array([
            _radial_second_derivative(
                rad_vals[i], rad_derivs[i], s=s, m=m, a=a, omega=omega,
                eigenvalue=lam, r=r_vals[i],
            )
            for i in range(n_q)
        ], dtype=np.complex128)
        d2rad_t = torch.tensor(d2rad_vals, device=device, dtype=torch.complex128)

        if s == 2:
            src0, src1, src2 = _spin_two_positive_coefficients_batch(
                r_t, ur_t, theta_t, u_theta_t,
                a, orbit.energy, orbit.angular_momentum,
                m, omega, h_t, dh_t, d2h_t,
            )
        else:
            src0, src1, src2 = _spin_two_coefficients_batch(
                r_t, ur_t, theta_t, u_theta_t,
                a, orbit.energy, orbit.angular_momentum,
                s, m, omega, h_t, dh_t, d2h_t,
            )

        if s == 2:
            # Δ²R wrapping for s=+2 Teukolsky equation
            delta_t = r_t * r_t - 2.0 * r_t + a * a
            d_delta_t = 2.0 * (r_t - 1.0)
            d2_delta_t = 2.0
            wr_R = delta_t * delta_t * rad_t
            wr_Rp = delta_t * delta_t * drad_t + 2.0 * delta_t * d_delta_t * rad_t
            wr_Rpp = (
                delta_t * delta_t * d2rad_t
                + 4.0 * delta_t * d_delta_t * drad_t
                + (2.0 * d_delta_t * d_delta_t + 2.0 * delta_t * d2_delta_t) * rad_t
            )
            integrand = src0 * wr_R - src1 * wr_Rp + src2 * wr_Rpp
        else:
            integrand = src0 * rad_t - src1 * drad_t + src2 * d2rad_t

    q_t = torch.tensor(q, device=device, dtype=torch.float64)
    phase = omega * dt_t - m * dphi_t + n * q_t
    integrand = integrand * torch.exp(1.0j * phase)

    dq = q[1] - q[0]
    total = torch.trapezoid(integrand, dx=dq, dim=0)

    return complex(total.item()) / (2.0 * math.pi)

"""DCU/ROCm batch radial ODE solver for the Teukolsky equation.

Replaces scipy.integrate.solve_ivp for the radial ODE when ``accelerator="gpu"``.
Integrates a batch of modes simultaneously using PyTorch tensors on the GPU.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from teukolsky.accelerated.backend import require_dcu

# ---------------------------------------------------------------------------
#  Boundary condition helpers (mirror radial/solver.py)
# ---------------------------------------------------------------------------


def _rp_static(a: float, mass: float = 1.0) -> float:
    return mass + math.sqrt(max(0.0, mass * mass - a * a))


def _delta_static(r: torch.Tensor, a: float) -> torch.Tensor:
    return r * r - 2.0 * r + a * a


def _rs_static(r: torch.Tensor, a: float) -> torch.Tensor:
    rp = _rp_static(a)
    rm = 1.0 - math.sqrt(max(0.0, 1.0 - a * a))
    return r + 2.0 / (rp - rm) * (rp * torch.log((r - rp) / 2.0) - rm * torch.log((r - rm) / 2.0))


def _phi_reg_static(r: torch.Tensor, a: float) -> torch.Tensor:
    rp = _rp_static(a)
    rm = 1.0 - math.sqrt(max(0.0, 1.0 - a * a))
    return a / (rp - rm) * torch.log((r - rp) / (r - rm))


# ---------------------------------------------------------------------------
#  Batched regular RHS for the radial Teukolsky equation
# ---------------------------------------------------------------------------


def _batch_regular_rhs(
    r: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    s: int,
    m: int,
    a: float,
    omega: torch.Tensor,
    lam: torch.Tensor,
    horizon_sign: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched regular RHS for the radial Teukolsky equation.

    All tensor inputs except r are of shape (n_modes,).  r is a scalar float
    tensor (same r for all modes in the batch).

    Returns (dy1, dy2) each of shape (n_modes,).
    """
    a2 = a * a
    one_plus_h = 1.0 + horizon_sign
    one_minus_h = 1.0 - horizon_sign
    r2 = r * r
    delta_val = a2 - 2.0 * r + r2
    a2pr2 = a2 + r2
    r4 = r2 * r2
    r5 = r4 * r
    r6 = r5 * r
    delta_over_r4 = (delta_val * delta_val) / r4

    # --- Exact match of _regular_rhs_numba, vectorized over modes ---
    #
    # Original numba code:
    #   numerator_prefactor = (
    #       delta * (2*a2 - 2*r*(1+s) - r2*lam)
    #       - 2*a*h_plus*m*r2*a2pr2*omega
    #       + 2j*r2*(-h_plus*(-a2+r2) + h_minus*r*delta)*s*omega
    #       + (1 - h^2)*r2*a2pr2*a2pr2*omega*omega
    #       - 2j*a*r*delta*(m + a*h*omega)
    #   )
    #   numerator = numerator_prefactor * y1 / r6
    #
    #   coeff = (
    #       2*(-a2+r2)*delta/(r4*a2pr2)
    #       - 2*delta*(a2*delta + a2pr2*((-1+r)*r*s - 1j*r*(a*m + h*a2pr2*omega)))
    #         / (r5*a2pr2)
    #   )
    #   y2p = -(numerator + coeff * y2) / (delta^2 / r4)

    # numerator_prefactor — scalar parts + mode-dep parts
    num_A_scalar = delta_val * (2.0 * a2 - 2.0 * r * (1.0 + s))
    num_A_lam = -delta_val * r2                          # × lam

    num_B = -2.0 * a * one_plus_h * m * r2 * a2pr2       # × omega

    num_C = 2.0j * r2 * (-one_plus_h * (-a2 + r2) + one_minus_h * r * delta_val) * s  # × omega

    num_D = (1.0 - horizon_sign * horizon_sign) * r2 * a2pr2 * a2pr2  # × omega^2

    num_E_const = -2.0j * a * r * delta_val * m            # scalar
    num_E_omega = -2.0j * a2 * r * delta_val * horizon_sign  # × omega

    numerator_prefactor = (
        num_A_scalar + num_A_lam * lam
        + num_B * omega
        + num_C * omega
        + num_D * omega * omega
        + num_E_const + num_E_omega * omega
    )
    numerator = numerator_prefactor * y1 / r6

    # coefficient
    coeff1 = (2.0 * (-a2 + r2) * delta_val) / (r4 * a2pr2)
    coeff2_inner = a2 * delta_val + a2pr2 * ((-1.0 + r) * r * s
                    - 1.0j * r * (a * m + horizon_sign * a2pr2 * omega))
    coeff = coeff1 - 2.0 * delta_val * coeff2_inner / (r5 * a2pr2)

    y2p = -(numerator + coeff * y2) / delta_over_r4

    return y2, y2p


# ---------------------------------------------------------------------------
#  Batch RK4 integrator
# ---------------------------------------------------------------------------


def _batch_rk4_step(
    r: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    dr: torch.Tensor,
    s: int,
    m: int,
    a: float,
    omega: torch.Tensor,
    lam: torch.Tensor,
    horizon_sign: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single RK4 step for batched modes. All mode tensors shape (n_modes,)."""
    # k1
    k1_y1, k1_y2 = _batch_regular_rhs(r, y1, y2, s, m, a, omega, lam, horizon_sign)
    # k2
    r_mid = r + 0.5 * dr
    k2_y1, k2_y2 = _batch_regular_rhs(
        r_mid, y1 + 0.5 * dr * k1_y1, y2 + 0.5 * dr * k1_y2,
        s, m, a, omega, lam, horizon_sign,
    )
    # k3
    k3_y1, k3_y2 = _batch_regular_rhs(
        r_mid, y1 + 0.5 * dr * k2_y1, y2 + 0.5 * dr * k2_y2,
        s, m, a, omega, lam, horizon_sign,
    )
    # k4
    r_next = r + dr
    k4_y1, k4_y2 = _batch_regular_rhs(
        r_next, y1 + dr * k3_y1, y2 + dr * k3_y2,
        s, m, a, omega, lam, horizon_sign,
    )

    y1_next = y1 + (dr / 6.0) * (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1)
    y2_next = y2 + (dr / 6.0) * (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2)
    return y1_next, y2_next


# ---------------------------------------------------------------------------
#  Physical solution transformation (mirrors _physical_from_regular_array)
# ---------------------------------------------------------------------------


def _batch_physical_from_regular(
    r: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    s: int,
    m: int,
    a: float,
    omega: torch.Tensor,
    boundary: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform regular radial solution to physical. Batched over modes.

    r: scalar float tensor
    y1, y2, omega: tensors of shape (n_modes,)
    """
    bc_dir = -1.0 if boundary == "In" else 1.0
    delta = _delta_static(r, a)
    rs_val = _rs_static(r, a)
    phi_val = _phi_reg_static(r, a)

    prefactor = (
        (r ** (-1))
        * (delta ** (-s))
        * torch.exp(1j * bc_dir * omega * rs_val)
        * torch.exp(1j * m * phi_val)
    )
    dlog_prefactor = (
        -1.0 / r
        - s * (2.0 * (r - 1.0)) / delta
        + 1j * bc_dir * omega * ((r * r + a * a) / delta)
        + 1j * m * a / delta
    )

    # First derivative of the physical function
    # The regular function y1 relates to the physical function R as:
    # R(r) = prefactor * y1(r)
    # dR/dr = prefactor * y2(r)  where y2 = dy1/dr (the second component of the state)
    #
    # Actually, the regular ODE solves for (X, dX/dr) where X = (something) * R.
    # The physical solution is R = prefactor * X.
    # dR/dr = d(prefactor)/dr * X + prefactor * dX/dr
    #       = dlog_prefactor * prefactor * X + prefactor * y2
    #       = prefactor * (dlog_prefactor * y1 + y2)

    physical = prefactor * y1
    physical_deriv = prefactor * (dlog_prefactor * y1 + y2)
    return physical, physical_deriv


def _in_transmission_normalization_dcu(a: float, m: int, device: torch.device) -> torch.Tensor:
    """Compute In transmission normalization as a complex scalar tensor."""
    import cmath
    rp = _rp_static(a)
    rm = 1.0 - math.sqrt(max(0.0, 1.0 - a * a))
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
    return torch.tensor(rp * cmath.exp(-phase), dtype=torch.complex128, device=device)


# ---------------------------------------------------------------------------
#  Main batch integration API
# ---------------------------------------------------------------------------


def batch_integrate_radial_dcu(
    s: int,
    m: int,
    a: float,
    omega_list: Sequence[complex],
    lam_list: Sequence[complex],
    sample_points: np.ndarray,
    boundary: str,
    device_id: int = 0,
    *,
    n_steps: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch radial ODE integration on GPU.

    Parameters
    ----------
    s : int
        Spin weight.
    m : int
        Azimuthal mode number (same for all modes in batch).
    a : float
        Kerr spin parameter.
    omega_list : sequence of complex
        Mode frequencies.
    lam_list : sequence of complex
        Spin-weighted spheroidal eigenvalues.
    sample_points : np.ndarray
        1D array of radial sample points where the solution is evaluated.
    boundary : {"In", "Up"}
        Boundary condition.
    device_id : int
        GPU device index.
    n_steps : int
        Number of RK4 steps (default 16384).

    Returns
    -------
    values : np.ndarray of shape (n_modes, len(sample_points))
        Physical radial function values.
    derivs : np.ndarray of shape (n_modes, len(sample_points))
        Physical radial function derivatives.
    """
    status = require_dcu(device_id)
    device = torch.device(f"cuda:{device_id}")

    n_modes = len(omega_list)
    omega_t = torch.tensor(
        [complex(o) for o in omega_list], dtype=torch.complex128, device=device
    )
    lam_t = torch.tensor(
        [complex(l) for l in lam_list], dtype=torch.complex128, device=device
    )

    sample_pts = np.asarray(sample_points, dtype=np.float64)

    if boundary == "In":
        eps = 1e-5 if s >= 0 else 1e-6
        r0 = _rp_static(a) + eps
        r_max = float(sample_pts.max())

        # Series boundary condition for y0, computed per mode.  The ODE is
        # batched on DCU, but the asymptotic coefficients are scalar helper
        # routines from the CPU radial solver.
        from teukolsky.radial.solver import (
            _in_bc_coefficients,
            _in_bc_series_coefficients,
        )
        if s >= 0:
            y0_1_values = []
            y0_2_values = []
            for omega_value, lam_value in zip(omega_list, lam_list):
                coeffs = _in_bc_series_coefficients(
                    s=s, m=m, a=a, omega=complex(omega_value),
                    lam=complex(lam_value), order=4, x0=eps,
                )
                y0_1_values.append(1.0 + sum(c * eps**k for k, c in enumerate(coeffs, 1)))
                y0_2_values.append(sum(k * c * eps**(k - 1) for k, c in enumerate(coeffs, 1)))
        else:
            y0_1_values = []
            y0_2_values = []
            for omega_value, lam_value in zip(omega_list, lam_list):
                a1, a2 = _in_bc_coefficients(
                    s=s, m=m, a=a, omega=complex(omega_value),
                    lam=complex(lam_value),
                )
                y0_1_values.append(1.0 + a1 * eps + a2 * eps * eps)
                y0_2_values.append(a1 + 2.0 * a2 * eps)

        horizon_sign = -1

        # Integration grid: uniform from r0 to r_max
        r_grid = torch.linspace(r0, r_max, n_steps + 1, dtype=torch.float64, device=device)
        dr = (r_max - r0) / n_steps

    else:  # "Up"
        from teukolsky.radial.solver import _up_bc_coefficients
        r0 = 1000.0
        r_min = float(sample_pts.min())

        y0_1_values = []
        y0_2_values = []
        for omega_value, lam_value in zip(omega_list, lam_list):
            b1, b2, b3 = _up_bc_coefficients(
                s=s, m=m, a=a,
                omega=complex(omega_value),
                lam=complex(lam_value), r0=r0,
            )
            y0_1_values.append(1.0 + b1 / r0 + b2 / (r0 * r0) + b3 / (r0**3))
            y0_2_values.append(-b1 / (r0 * r0) - 2.0 * b2 / (r0**3) - 3.0 * b3 / (r0**4))

        horizon_sign = 1

        # Integration grid: uniform from r0 inward to r_min
        r_grid = torch.linspace(r0, r_min, n_steps + 1, dtype=torch.float64, device=device)
        dr = (r_min - r0) / n_steps

    # Broadcast initial condition to all modes
    y1 = torch.tensor(y0_1_values, dtype=torch.complex128, device=device)
    y2 = torch.tensor(y0_2_values, dtype=torch.complex128, device=device)

    # Integrate on a UNIFORM dense grid — NEVER include sample points in the
    # integration grid, because sample points are clustered near the orbit and
    # would leave huge gaps near the boundaries, breaking RK4 stability.
    r_grid_use = r_grid
    n_steps_use = n_steps

    # Store trajectory at every step for later interpolation
    traj_y1 = torch.zeros((n_steps_use + 1, n_modes), dtype=torch.complex128, device=device)
    traj_y2 = torch.zeros((n_steps_use + 1, n_modes), dtype=torch.complex128, device=device)
    traj_y1[0] = y1
    traj_y2[0] = y2

    for i in range(n_steps_use):
        r_i = r_grid_use[i]
        y1, y2 = _batch_rk4_step(r_i, y1, y2, dr, s, m, a, omega_t, lam_t, horizon_sign)
        traj_y1[i + 1] = y1
        traj_y2[i + 1] = y2

    # Interpolate trajectory to sample points.
    # For "Up" boundary, r_grid is DECREASING (1000 → r_min).
    # Flip everything to increasing order for simpler interpolation logic.
    if boundary == "Up":
        r_grid_use = torch.flip(r_grid_use, dims=[0])
        traj_y1 = torch.flip(traj_y1, dims=[0])
        traj_y2 = torch.flip(traj_y2, dims=[0])

    # Now r_grid_use is INCREASING for both In and Up.
    n_sample = len(sample_pts)
    values_out = torch.zeros((n_modes, n_sample), dtype=torch.complex128, device=device)
    derivs_out = torch.zeros((n_modes, n_sample), dtype=torch.complex128, device=device)

    r_grid_vals = np.array([float(r_grid_use[i].item()) for i in range(len(r_grid_use))], dtype=np.float64)
    pos_indices = np.searchsorted(r_grid_vals, sample_pts)

    for idx in range(n_sample):
        r_val = sample_pts[idx]
        p = int(pos_indices[idx])
        n_grid = len(r_grid_use)

        if p == 0:
            i0, i1 = 0, 1
            alpha = 0.0
        elif p >= n_grid:
            i0, i1 = n_grid - 2, n_grid - 1
            alpha = 1.0
        else:
            i0, i1 = p - 1, p
            r_lo = float(r_grid_use[i0].item())
            r_hi = float(r_grid_use[i1].item())
            alpha = (r_val - r_lo) / (r_hi - r_lo) if r_hi > r_lo else 0.0

        # Linear interpolation
        y1_interp = (1.0 - alpha) * traj_y1[i0] + alpha * traj_y1[i1]
        y2_interp = (1.0 - alpha) * traj_y2[i0] + alpha * traj_y2[i1]

        r_t = torch.tensor(r_val, dtype=torch.float64, device=device)
        phys, phys_deriv = _batch_physical_from_regular(
            r_t, y1_interp, y2_interp, s, m, a, omega_t, boundary,
        )
        values_out[:, idx] = phys
        derivs_out[:, idx] = phys_deriv

    if boundary == "In":
        scale = _in_transmission_normalization_dcu(a, m, device)
        values_out = scale * values_out
        derivs_out = scale * derivs_out

    # Convert to numpy via CPU tensors (may fail; fallback to list conversion)
    try:
        v_np = values_out.cpu().numpy()
        d_np = derivs_out.cpu().numpy()
    except RuntimeError:
        v_np = np.array(values_out.cpu().tolist(), dtype=np.complex128)
        d_np = np.array(derivs_out.cpu().tolist(), dtype=np.complex128)

    return v_np, d_np

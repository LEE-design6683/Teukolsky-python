from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root

from teukolsky.core import Orbit


def _carter_from_inclination(angular_momentum: float, inclination: float) -> float:
    if abs(inclination) >= 1.0:
        return 0.0
    return angular_momentum * angular_momentum * (1.0 - inclination * inclination) / (inclination * inclination)


def _radial_potential(r: float, a: float, energy: float, angular_momentum: float, carter_constant: float = 0.0) -> float:
    delta = r * r - 2.0 * r + a * a
    pterm = energy * (r * r + a * a) - a * angular_momentum
    return pterm * pterm - delta * (r * r + (angular_momentum - a * energy) ** 2 + carter_constant)


def _polar_potential(theta: float, a: float, energy: float, angular_momentum: float, carter_constant: float) -> float:
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    return carter_constant - cos_theta * cos_theta * (
        a * a * (1.0 - energy * energy) + angular_momentum * angular_momentum / (sin_theta * sin_theta)
    )


def _equatorial_constants(a: float, p: float, e: float, inclination: float) -> tuple[float, float]:
    if abs(inclination) != 1.0:
        raise ValueError("equatorial orbit requires inclination = ±1")
    rp = p / (1.0 + e)
    ra = p / (1.0 - e)
    schwarzschild_energy = math.sqrt(((p - 2.0) ** 2 - 4.0 * e * e) / (p * (p - 3.0 - e * e)))
    schwarzschild_angular_momentum = p / math.sqrt(p - 3.0 - e * e)
    circular = circular_orbit(a, p)
    guess = np.array(
        [
            0.5 * (schwarzschild_energy + circular.energy),
            math.copysign(0.5 * (schwarzschild_angular_momentum + abs(circular.angular_momentum)), inclination),
        ],
        dtype=float,
    )

    def equations(values: np.ndarray) -> np.ndarray:
        energy, angular_momentum = values
        return np.array(
            [
                _radial_potential(rp, a, energy, angular_momentum),
                _radial_potential(ra, a, energy, angular_momentum),
            ],
            dtype=float,
        )

    solution = root(equations, guess, method="hybr")
    if not solution.success:
        raise RuntimeError(f"failed to solve equatorial eccentric constants: {solution.message}")
    energy = float(solution.x[0])
    angular_momentum = float(solution.x[1])
    return energy, angular_momentum


def _spherical_constants(a: float, radius: float, inclination: float) -> tuple[float, float]:
    if not (0.0 < abs(inclination) < 1.0):
        raise ValueError("spherical orbit requires 0 < |inclination| < 1")
    circular = circular_orbit(a, radius)
    guess = np.array([circular.energy, inclination * abs(circular.angular_momentum)], dtype=float)

    def equations(values: np.ndarray) -> np.ndarray:
        energy, angular_momentum = values
        carter_constant = _carter_from_inclination(angular_momentum, inclination)
        pterm = energy * (radius * radius + a * a) - a * angular_momentum
        delta = radius * radius - 2.0 * radius + a * a
        radial = _radial_potential(radius, a, energy, angular_momentum, carter_constant)
        radial_derivative = (
            4.0 * radius * energy * pterm
            - (2.0 * radius - 2.0) * (radius * radius + (angular_momentum - a * energy) ** 2 + carter_constant)
            - 2.0 * radius * delta
        )
        return np.array([radial, radial_derivative], dtype=float)

    solution = root(equations, guess, method="hybr")
    if not solution.success:
        raise RuntimeError(f"failed to solve spherical constants: {solution.message}")
    energy = float(solution.x[0])
    angular_momentum = float(solution.x[1])
    return energy, angular_momentum


def _generic_constants(a: float, p: float, e: float, inclination: float) -> tuple[float, float]:
    if not (0.0 < e < 1.0):
        raise ValueError("generic orbit requires 0 < e < 1")
    if not (0.0 < abs(inclination) < 1.0):
        raise ValueError("generic orbit requires 0 < |inclination| < 1")
    rp = p / (1.0 + e)
    ra = p / (1.0 - e)
    equatorial = equatorial_eccentric_orbit(a, p, e, math.copysign(1.0, inclination))
    spherical = spherical_orbit(a, p, inclination)
    guess = np.array(
        [
            0.5 * (equatorial.energy + spherical.energy),
            0.5 * (equatorial.angular_momentum + spherical.angular_momentum),
        ],
        dtype=float,
    )

    def equations(values: np.ndarray) -> np.ndarray:
        energy, angular_momentum = values
        carter_constant = _carter_from_inclination(angular_momentum, inclination)
        return np.array(
            [
                _radial_potential(rp, a, energy, angular_momentum, carter_constant),
                _radial_potential(ra, a, energy, angular_momentum, carter_constant),
            ],
            dtype=float,
        )

    solution = root(equations, guess, method="hybr")
    if not solution.success:
        raise RuntimeError(f"failed to solve generic constants: {solution.message}")
    energy = float(solution.x[0])
    angular_momentum = float(solution.x[1])
    return energy, angular_momentum


def _dt_dlambda(r: np.ndarray, a: float, energy: float, angular_momentum: float) -> np.ndarray:
    delta = r * r - 2.0 * r + a * a
    pterm = energy * (r * r + a * a) - a * angular_momentum
    return a * (angular_momentum - a * energy) + (r * r + a * a) * pterm / delta


def _dphi_dlambda(r: np.ndarray, a: float, energy: float, angular_momentum: float) -> np.ndarray:
    delta = r * r - 2.0 * r + a * a
    pterm = energy * (r * r + a * a) - a * angular_momentum
    return angular_momentum - a * energy + a * pterm / delta


def _dt_dlambda_radial(r: np.ndarray, a: float, energy: float, angular_momentum: float) -> np.ndarray:
    delta = r * r - 2.0 * r + a * a
    pterm = energy * (r * r + a * a) - a * angular_momentum
    return (r * r + a * a) * pterm / delta


def _dphi_dlambda_radial(r: np.ndarray, a: float, energy: float, angular_momentum: float) -> np.ndarray:
    delta = r * r - 2.0 * r + a * a
    pterm = energy * (r * r + a * a) - a * angular_momentum
    return a * pterm / delta


def _dt_dlambda_polar(theta: np.ndarray, a: float, energy: float, angular_momentum: float) -> np.ndarray:
    return a * (angular_momentum - a * energy * np.sin(theta) ** 2)


def _dphi_dlambda_polar(theta: np.ndarray, a: float, energy: float, angular_momentum: float) -> np.ndarray:
    return angular_momentum / np.maximum(np.sin(theta), 1e-14) ** 2 - a * energy


def _cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    steps = np.diff(x)
    averages = 0.5 * (y[1:] + y[:-1])
    return np.concatenate(([0.0], np.cumsum(steps * averages)))


def _build_periodic_interpolant(grid: np.ndarray, values: np.ndarray):
    period = float(grid[-1] - grid[0])
    periodic_values = np.array(values, dtype=float, copy=True)
    periodic_values[-1] = periodic_values[0]
    spline = CubicSpline(grid, periodic_values, bc_type="periodic")

    def interpolant(argument: float) -> float:
        wrapped = ((argument - grid[0]) % period) + grid[0]
        return float(spline(wrapped))

    return interpolant


def circular_orbit(a: float, radius: float) -> Orbit:
    sqrt_r = math.sqrt(radius)
    r32 = radius * sqrt_r
    stability = math.sqrt(1.0 - 3.0 / radius + 2.0 * a / r32)
    energy = (1.0 - 2.0 / radius + a / r32) / stability
    angular_momentum = sqrt_r * (1.0 - 2.0 * a / r32 + a * a / (radius * radius)) / stability
    omega_phi = 1.0 / (r32 + a)
    delta = radius * radius - 2.0 * radius + a * a
    pterm = energy * (radius * radius + a * a) - a * angular_momentum
    upsilon_t = -a * (a * energy - angular_momentum) + (radius * radius + a * a) * pterm / delta
    return Orbit(
        a=a,
        p=radius,
        e=0.0,
        inclination=1.0,
        energy=energy,
        angular_momentum=angular_momentum,
        omega_r=0.0,
        omega_theta=0.0,
        omega_phi=omega_phi,
        upsilon_r=0.0,
        upsilon_theta=0.0,
        upsilon_t=upsilon_t,
        kind="circular-equatorial",
    )


def equatorial_eccentric_orbit(a: float, p: float, e: float, inclination: float = 1.0, samples: int = 8193) -> Orbit:
    if not (0.0 < e < 1.0):
        raise ValueError("eccentric equatorial orbit requires 0 < e < 1")
    if abs(inclination) != 1.0:
        raise ValueError("equatorial orbit requires inclination = ±1")
    if samples < 1025 or samples % 2 == 0:
        raise ValueError("samples must be an odd integer >= 1025")

    rp = p / (1.0 + e)
    horizon = 1.0 + math.sqrt(1.0 - a * a)
    if rp <= horizon:
        raise ValueError("periastron must lie outside the event horizon")

    energy, angular_momentum = _equatorial_constants(a, p, e, inclination)

    eps = 1e-8
    chi_inner = np.linspace(eps, 2.0 * math.pi - eps, samples - 2)
    chi = np.concatenate(([0.0], chi_inner, [2.0 * math.pi]))
    r = p / (1.0 + e * np.cos(chi))
    dr_dchi = p * e * np.sin(chi) / (1.0 + e * np.cos(chi)) ** 2

    radial_potential = np.array([_radial_potential(value, a, energy, angular_momentum) for value in r], dtype=float)
    radial_potential = np.maximum(radial_potential, 0.0)
    dr_dlambda_abs = np.sqrt(radial_potential)
    dlambda_dchi = np.empty_like(chi)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlambda_dchi = np.abs(dr_dchi) / dr_dlambda_abs
    invalid = ~np.isfinite(dlambda_dchi)
    valid = ~invalid
    if not np.any(valid):
        raise RuntimeError("failed to construct eccentric orbit phase map")
    dlambda_dchi[invalid] = np.interp(chi[invalid], chi[valid], dlambda_dchi[valid])

    lambda_values = _cumulative_trapezoid(chi, dlambda_dchi)
    lambda_period = float(lambda_values[-1])
    upsilon_r = 2.0 * math.pi / lambda_period
    q_r = upsilon_r * lambda_values

    dt_dlambda = _dt_dlambda(r, a, energy, angular_momentum)
    dphi_dlambda = _dphi_dlambda(r, a, energy, angular_momentum)
    dt_dchi = dt_dlambda * dlambda_dchi
    dphi_dchi = dphi_dlambda * dlambda_dchi
    t_values = _cumulative_trapezoid(chi, dt_dchi)
    phi_values = _cumulative_trapezoid(chi, dphi_dchi)

    upsilon_t = float(t_values[-1] / lambda_period)
    upsilon_phi = float(phi_values[-1] / lambda_period)
    omega_r = upsilon_r / upsilon_t
    omega_phi = upsilon_phi / upsilon_t

    ur_mino = np.sign(np.sin(chi)) * dr_dlambda_abs
    ur_mino[0] = 0.0
    ur_mino[-1] = 0.0

    delta_t = t_values - upsilon_t * lambda_values
    delta_phi = phi_values - upsilon_phi * lambda_values
    delta_t[0] = 0.0
    delta_t[-1] = 0.0
    delta_phi[0] = 0.0
    delta_phi[-1] = 0.0

    q_grid = np.linspace(0.0, 2.0 * math.pi, samples)
    r_q = np.interp(q_grid, q_r, r)
    ur_q = np.interp(q_grid, q_r, ur_mino)
    delta_t_q = np.interp(q_grid, q_r, delta_t)
    delta_phi_q = np.interp(q_grid, q_r, delta_phi)
    r_q[0] = rp
    r_q[-1] = rp
    ur_q[0] = 0.0
    ur_q[-1] = 0.0
    delta_t_q[0] = 0.0
    delta_t_q[-1] = 0.0
    delta_phi_q[0] = 0.0
    delta_phi_q[-1] = 0.0

    return Orbit(
        a=a,
        p=p,
        e=e,
        inclination=inclination,
        energy=energy,
        angular_momentum=angular_momentum,
        omega_r=omega_r,
        omega_theta=0.0,
        omega_phi=omega_phi,
        upsilon_r=upsilon_r,
        upsilon_theta=0.0,
        upsilon_t=upsilon_t,
        kind="eccentric-equatorial",
        radial_phase_function=_build_periodic_interpolant(q_grid, r_q),
        radial_velocity_function=_build_periodic_interpolant(q_grid, ur_q),
        radial_delta_t_function=_build_periodic_interpolant(q_grid, delta_t_q),
        radial_delta_phi_function=_build_periodic_interpolant(q_grid, delta_phi_q),
    )


def spherical_orbit(a: float, radius: float, inclination: float, samples: int = 8193) -> Orbit:
    if not (0.0 < abs(inclination) < 1.0):
        raise ValueError("spherical orbit requires 0 < |inclination| < 1")
    if samples < 1025 or samples % 2 == 0:
        raise ValueError("samples must be an odd integer >= 1025")

    energy, angular_momentum = _spherical_constants(a, radius, inclination)
    carter_constant = _carter_from_inclination(angular_momentum, inclination)
    if a == 0.0:
        z_scale = abs(inclination)
    else:
        z_scale_sq = (
            (a * a * (1.0 - energy * energy) + angular_momentum * angular_momentum + carter_constant)
            - math.sqrt(
                (a * a * (1.0 - energy * energy) + angular_momentum * angular_momentum + carter_constant) ** 2
                - 4.0 * a * a * (1.0 - energy * energy) * carter_constant
            )
        ) / (2.0 * a * a * (1.0 - energy * energy))
        z_scale = math.sqrt(max(z_scale_sq, 0.0))
    eps = 1e-8
    chi_inner = np.linspace(eps, 2.0 * math.pi - eps, samples - 2)
    chi = np.concatenate(([0.0], chi_inner, [2.0 * math.pi]))
    cos_theta = z_scale * np.cos(chi)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    dtheta_dchi = z_scale * np.sin(chi) / np.maximum(np.sin(theta), 1e-14)
    polar_potential = np.array(
        [_polar_potential(float(value), a, energy, angular_momentum, carter_constant) for value in theta],
        dtype=float,
    )
    polar_potential = np.maximum(polar_potential, 0.0)
    dtheta_dlambda_abs = np.sqrt(polar_potential)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlambda_dchi = np.abs(dtheta_dchi) / dtheta_dlambda_abs
    invalid = ~np.isfinite(dlambda_dchi)
    valid = ~invalid
    if not np.any(valid):
        raise RuntimeError("failed to construct spherical orbit phase map")
    dlambda_dchi[invalid] = np.interp(chi[invalid], chi[valid], dlambda_dchi[valid])

    lambda_values = _cumulative_trapezoid(chi, dlambda_dchi)
    lambda_period = float(lambda_values[-1])
    upsilon_theta = 2.0 * math.pi / lambda_period
    q_theta = upsilon_theta * lambda_values

    pterm = energy * (radius * radius + a * a) - a * angular_momentum
    delta = radius * radius - 2.0 * radius + a * a
    dt_dlambda = a * (angular_momentum - a * energy * np.sin(theta) ** 2) + (radius * radius + a * a) * pterm / delta
    dphi_dlambda = angular_momentum / np.maximum(np.sin(theta), 1e-14) ** 2 - a * energy + a * pterm / delta
    dt_dchi = dt_dlambda * dlambda_dchi
    dphi_dchi = dphi_dlambda * dlambda_dchi
    t_values = _cumulative_trapezoid(chi, dt_dchi)
    phi_values = _cumulative_trapezoid(chi, dphi_dchi)

    upsilon_t = float(t_values[-1] / lambda_period)
    upsilon_phi = float(phi_values[-1] / lambda_period)
    omega_theta = upsilon_theta / upsilon_t
    omega_phi = upsilon_phi / upsilon_t

    u_theta_mino = np.sign(np.sin(chi)) * dtheta_dlambda_abs
    u_theta_mino[0] = 0.0
    u_theta_mino[-1] = 0.0
    delta_t = t_values - upsilon_t * lambda_values
    delta_phi = phi_values - upsilon_phi * lambda_values
    delta_t[0] = 0.0
    delta_t[-1] = 0.0
    delta_phi[0] = 0.0
    delta_phi[-1] = 0.0

    q_grid = np.linspace(0.0, 2.0 * math.pi, samples)
    theta_q = np.interp(q_grid, q_theta, theta)
    u_theta_q = np.interp(q_grid, q_theta, u_theta_mino)
    delta_t_q = np.interp(q_grid, q_theta, delta_t)
    delta_phi_q = np.interp(q_grid, q_theta, delta_phi)
    theta_q[0] = theta_q[-1] = math.acos(z_scale)
    u_theta_q[0] = u_theta_q[-1] = 0.0
    delta_t_q[0] = delta_t_q[-1] = 0.0
    delta_phi_q[0] = delta_phi_q[-1] = 0.0

    return Orbit(
        a=a,
        p=radius,
        e=0.0,
        inclination=inclination,
        energy=energy,
        angular_momentum=angular_momentum,
        omega_r=0.0,
        omega_theta=omega_theta,
        omega_phi=omega_phi,
        upsilon_r=0.0,
        upsilon_theta=upsilon_theta,
        upsilon_t=upsilon_t,
        kind="spherical",
        theta_phase_function=_build_periodic_interpolant(q_grid, theta_q),
        theta_velocity_function=_build_periodic_interpolant(q_grid, u_theta_q),
        theta_delta_t_function=_build_periodic_interpolant(q_grid, delta_t_q),
        theta_delta_phi_function=_build_periodic_interpolant(q_grid, delta_phi_q),
    )


def generic_orbit(a: float, p: float, e: float, inclination: float, samples: int = 8193) -> Orbit:
    if not (0.0 < e < 1.0):
        raise ValueError("generic orbit requires 0 < e < 1")
    if not (0.0 < abs(inclination) < 1.0):
        raise ValueError("generic orbit requires 0 < |inclination| < 1")
    if samples < 1025 or samples % 2 == 0:
        raise ValueError("samples must be an odd integer >= 1025")

    rp = p / (1.0 + e)
    horizon = 1.0 + math.sqrt(1.0 - a * a)
    if rp <= horizon:
        raise ValueError("periastron must lie outside the event horizon")

    energy, angular_momentum = _generic_constants(a, p, e, inclination)
    carter_constant = _carter_from_inclination(angular_momentum, inclination)

    eps = 1e-8
    chi_r_inner = np.linspace(eps, 2.0 * math.pi - eps, samples - 2)
    chi_r = np.concatenate(([0.0], chi_r_inner, [2.0 * math.pi]))
    r = p / (1.0 + e * np.cos(chi_r))
    dr_dchi = p * e * np.sin(chi_r) / (1.0 + e * np.cos(chi_r)) ** 2
    radial_potential = np.array(
        [_radial_potential(value, a, energy, angular_momentum, carter_constant) for value in r],
        dtype=float,
    )
    radial_potential = np.maximum(radial_potential, 0.0)
    dr_dlambda_abs = np.sqrt(radial_potential)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlambda_dchi_r = np.abs(dr_dchi) / dr_dlambda_abs
    invalid = ~np.isfinite(dlambda_dchi_r)
    valid = ~invalid
    if not np.any(valid):
        raise RuntimeError("failed to construct generic radial phase map")
    dlambda_dchi_r[invalid] = np.interp(chi_r[invalid], chi_r[valid], dlambda_dchi_r[valid])
    lambda_r = _cumulative_trapezoid(chi_r, dlambda_dchi_r)
    lambda_r_period = float(lambda_r[-1])
    upsilon_r = 2.0 * math.pi / lambda_r_period
    q_r = upsilon_r * lambda_r
    ur_mino = np.sign(np.sin(chi_r)) * dr_dlambda_abs
    ur_mino[0] = 0.0
    ur_mino[-1] = 0.0
    t_r = _cumulative_trapezoid(chi_r, _dt_dlambda_radial(r, a, energy, angular_momentum) * dlambda_dchi_r)
    phi_r = _cumulative_trapezoid(chi_r, _dphi_dlambda_radial(r, a, energy, angular_momentum) * dlambda_dchi_r)
    upsilon_t_r = float(t_r[-1] / lambda_r_period)
    upsilon_phi_r = float(phi_r[-1] / lambda_r_period)
    delta_t_r = t_r - upsilon_t_r * lambda_r
    delta_phi_r = phi_r - upsilon_phi_r * lambda_r
    delta_t_r[0] = delta_t_r[-1] = 0.0
    delta_phi_r[0] = delta_phi_r[-1] = 0.0
    q_r_grid = np.linspace(0.0, 2.0 * math.pi, samples)
    r_q = np.interp(q_r_grid, q_r, r)
    ur_q = np.interp(q_r_grid, q_r, ur_mino)
    delta_t_r_q = np.interp(q_r_grid, q_r, delta_t_r)
    delta_phi_r_q = np.interp(q_r_grid, q_r, delta_phi_r)
    r_q[0] = r_q[-1] = rp
    ur_q[0] = ur_q[-1] = 0.0
    delta_t_r_q[0] = delta_t_r_q[-1] = 0.0
    delta_phi_r_q[0] = delta_phi_r_q[-1] = 0.0

    if a == 0.0:
        z_scale = abs(inclination)
    else:
        z_scale_sq = (
            (a * a * (1.0 - energy * energy) + angular_momentum * angular_momentum + carter_constant)
            - math.sqrt(
                (a * a * (1.0 - energy * energy) + angular_momentum * angular_momentum + carter_constant) ** 2
                - 4.0 * a * a * (1.0 - energy * energy) * carter_constant
            )
        ) / (2.0 * a * a * (1.0 - energy * energy))
        z_scale = math.sqrt(max(z_scale_sq, 0.0))
    eps = 1e-8
    chi_theta_inner = np.linspace(eps, 2.0 * math.pi - eps, samples - 2)
    chi_theta = np.concatenate(([0.0], chi_theta_inner, [2.0 * math.pi]))
    cos_theta = z_scale * np.cos(chi_theta)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    dtheta_dchi = z_scale * np.sin(chi_theta) / np.maximum(np.sin(theta), 1e-14)
    polar_potential = np.array(
        [_polar_potential(float(value), a, energy, angular_momentum, carter_constant) for value in theta],
        dtype=float,
    )
    polar_potential = np.maximum(polar_potential, 0.0)
    dtheta_dlambda_abs = np.sqrt(polar_potential)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlambda_dchi_theta = np.abs(dtheta_dchi) / dtheta_dlambda_abs
    invalid = ~np.isfinite(dlambda_dchi_theta)
    valid = ~invalid
    if not np.any(valid):
        raise RuntimeError("failed to construct generic polar phase map")
    dlambda_dchi_theta[invalid] = np.interp(chi_theta[invalid], chi_theta[valid], dlambda_dchi_theta[valid])
    lambda_theta = _cumulative_trapezoid(chi_theta, dlambda_dchi_theta)
    lambda_theta_period = float(lambda_theta[-1])
    upsilon_theta = 2.0 * math.pi / lambda_theta_period
    q_theta = upsilon_theta * lambda_theta
    u_theta_mino = np.sign(np.sin(chi_theta)) * dtheta_dlambda_abs
    u_theta_mino[0] = 0.0
    u_theta_mino[-1] = 0.0
    t_theta = _cumulative_trapezoid(chi_theta, _dt_dlambda_polar(theta, a, energy, angular_momentum) * dlambda_dchi_theta)
    phi_theta = _cumulative_trapezoid(chi_theta, _dphi_dlambda_polar(theta, a, energy, angular_momentum) * dlambda_dchi_theta)
    upsilon_t_theta = float(t_theta[-1] / lambda_theta_period)
    upsilon_phi_theta = float(phi_theta[-1] / lambda_theta_period)
    delta_t_theta = t_theta - upsilon_t_theta * lambda_theta
    delta_phi_theta = phi_theta - upsilon_phi_theta * lambda_theta
    delta_t_theta[0] = delta_t_theta[-1] = 0.0
    delta_phi_theta[0] = delta_phi_theta[-1] = 0.0
    q_theta_grid = np.linspace(0.0, 2.0 * math.pi, samples)
    theta_q = np.interp(q_theta_grid, q_theta, theta)
    u_theta_q = np.interp(q_theta_grid, q_theta, u_theta_mino)
    delta_t_theta_q = np.interp(q_theta_grid, q_theta, delta_t_theta)
    delta_phi_theta_q = np.interp(q_theta_grid, q_theta, delta_phi_theta)
    theta_q[0] = theta_q[-1] = math.acos(z_scale)
    u_theta_q[0] = u_theta_q[-1] = 0.0
    delta_t_theta_q[0] = delta_t_theta_q[-1] = 0.0
    delta_phi_theta_q[0] = delta_phi_theta_q[-1] = 0.0

    upsilon_t = upsilon_t_r + upsilon_t_theta
    upsilon_phi = upsilon_phi_r + upsilon_phi_theta
    omega_r = upsilon_r / upsilon_t
    omega_theta = upsilon_theta / upsilon_t
    omega_phi = upsilon_phi / upsilon_t

    return Orbit(
        a=a,
        p=p,
        e=e,
        inclination=inclination,
        energy=energy,
        angular_momentum=angular_momentum,
        omega_r=omega_r,
        omega_theta=omega_theta,
        omega_phi=omega_phi,
        upsilon_r=upsilon_r,
        upsilon_theta=upsilon_theta,
        upsilon_t=upsilon_t,
        kind="generic",
        radial_phase_function=_build_periodic_interpolant(q_r_grid, r_q),
        radial_velocity_function=_build_periodic_interpolant(q_r_grid, ur_q),
        radial_delta_t_function=_build_periodic_interpolant(q_r_grid, delta_t_r_q),
        radial_delta_phi_function=_build_periodic_interpolant(q_r_grid, delta_phi_r_q),
        theta_phase_function=_build_periodic_interpolant(q_theta_grid, theta_q),
        theta_velocity_function=_build_periodic_interpolant(q_theta_grid, u_theta_q),
        theta_delta_t_function=_build_periodic_interpolant(q_theta_grid, delta_t_theta_q),
        theta_delta_phi_function=_build_periodic_interpolant(q_theta_grid, delta_phi_theta_q),
    )

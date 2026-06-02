from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from teukolsky.api import KerrGeoOrbit, TeukolskyPointParticleMode
from teukolsky.core import ModeSolution, Orbit


GMSUN_SEC = 4.92549095e-6
MRSUN_SI = 1476.6250380501249
GPC_SI = 3.0856775814913673e25


@dataclass(frozen=True)
class WaveformPolarizations:
    time: np.ndarray
    h_plus: np.ndarray
    h_cross: np.ndarray
    complex_strain: np.ndarray
    theta: float
    phi: float
    radius: float
    modes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class AdiabaticTrajectory:
    time: np.ndarray
    p: np.ndarray
    e: np.ndarray
    energy: np.ndarray
    angular_momentum: np.ndarray
    pdot: np.ndarray
    edot: np.ndarray
    edot_energy: np.ndarray
    edot_angular_momentum: np.ndarray


@dataclass(frozen=True)
class AdiabaticWaveform:
    waveform: WaveformPolarizations
    trajectory: AdiabaticTrajectory


def mode_frequency(orbit: Orbit, m: int, n: int = 0, k: int = 0) -> float:
    return m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta


def _seconds_per_mass(M: float) -> float:
    if M <= 0.0:
        raise ValueError("M must be > 0")
    return M * GMSUN_SEC


def source_frame_radius(distance_gpc: float, secondary_mass: float) -> float:
    if distance_gpc <= 0.0:
        raise ValueError("distance_gpc must be > 0")
    if secondary_mass <= 0.0:
        raise ValueError("secondary_mass must be > 0")
    return (distance_gpc * GPC_SI) / (secondary_mass * MRSUN_SI)


def _cumulative_trapezoid(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    steps = np.diff(time)
    averages = 0.5 * (values[1:] + values[:-1])
    return np.concatenate(([0.0], np.cumsum(steps * averages)))


def _interpolate_real_series(source_t: np.ndarray, source_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if len(source_t) == 1:
        return np.full_like(target_t, float(source_y[0]), dtype=float)
    if len(source_t) < 4:
        return np.interp(target_t, source_t, source_y)
    spline = CubicSpline(source_t, source_y, bc_type="natural")
    return np.asarray(spline(target_t), dtype=float)


def _interpolate_complex_series(source_t: np.ndarray, source_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    real = _interpolate_real_series(source_t, np.asarray(source_y.real, dtype=float), target_t)
    imag = _interpolate_real_series(source_t, np.asarray(source_y.imag, dtype=float), target_t)
    return real + 1j * imag


def _sum_adiabatic_mode_series(
    orbit_nodes: list[Orbit],
    sparse_time: np.ndarray,
    dense_time: np.ndarray,
    mode_indices: list[tuple[int, int, int, int]],
    *,
    theta: float,
    phi: float,
    radius: float,
    mass_seconds: float,
    accelerator: str,
    device_id: int,
    accelerator_resolution: int | None,
    omega_tol: float,
) -> tuple[np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    complex_strain = np.zeros_like(dense_time, dtype=np.complex128)
    used_modes: list[tuple[int, int, int, int]] = []

    for ell, m, n, k in mode_indices:
        sparse_omega_geom = np.array([mode_frequency(orbit, m, n, k) for orbit in orbit_nodes], dtype=float)
        if np.all(np.abs(sparse_omega_geom) <= omega_tol):
            continue
        sparse_omega = sparse_omega_geom / mass_seconds

        sparse_amplitude = np.empty(len(orbit_nodes), dtype=np.complex128)
        sparse_angular = np.empty(len(orbit_nodes), dtype=np.complex128)
        for i, orbit in enumerate(orbit_nodes):
            mode = TeukolskyPointParticleMode(
                -2,
                ell,
                m,
                n,
                k,
                orbit,
                accelerator=accelerator,
                device_id=device_id,
                accelerator_resolution=accelerator_resolution,
            )
            sparse_amplitude[i] = mode["Amplitudes"]["I"]
            sparse_angular[i] = mode["AngularFunction"].evaluate(theta, phi)

        dense_omega = _interpolate_real_series(sparse_time, sparse_omega, dense_time)
        dense_omega_geom = _interpolate_real_series(sparse_time, sparse_omega_geom, dense_time)
        nonzero = np.abs(dense_omega) > omega_tol
        if not np.any(nonzero):
            continue

        dense_amplitude = _interpolate_complex_series(sparse_time, sparse_amplitude, dense_time)
        dense_angular = _interpolate_complex_series(sparse_time, sparse_angular, dense_time)
        phase = _cumulative_trapezoid(dense_time, dense_omega)
        mode_strain_values = np.zeros_like(complex_strain)
        mode_strain_values[nonzero] = (
            -2.0
            * dense_amplitude[nonzero]
            * dense_angular[nonzero]
            * np.exp(-1j * phase[nonzero])
            / (radius * dense_omega_geom[nonzero] * dense_omega_geom[nonzero])
        )
        complex_strain = complex_strain + mode_strain_values
        used_modes.append((ell, m, n, k))

    return complex_strain, tuple(used_modes)




def carter_constant(angular_momentum: float, inclination: float) -> float:
    if abs(inclination) >= 1.0:
        return 0.0
    return angular_momentum * angular_momentum * (1.0 - inclination * inclination) / (inclination * inclination)


def finite_difference_jacobian(p: float, e: float) -> np.ndarray:
    return finite_difference_jacobian_equatorial(0.0, p, e, 1.0)


def finite_difference_jacobian_equatorial(a: float, p: float, e: float, x: float) -> np.ndarray:
    dp = max(1e-6, 1e-5 * abs(p))
    de = max(1e-8, 1e-5 * max(abs(e), 1e-3))

    def orbit_quantities(pp: float, ee: float) -> tuple[float, float]:
        orb = KerrGeoOrbit(float(a), float(pp), float(ee), float(x))
        return float(orb.energy), float(orb.angular_momentum)

    e_minus = max(0.0, e - de)
    e_plus = min(0.95, e + de)

    e_p_plus, l_p_plus = orbit_quantities(p + dp, e)
    e_p_minus, l_p_minus = orbit_quantities(p - dp, e)
    dE_dp = (e_p_plus - e_p_minus) / (2.0 * dp)
    dL_dp = (l_p_plus - l_p_minus) / (2.0 * dp)

    if e_plus > e_minus:
        e_e_plus, l_e_plus = orbit_quantities(p, e_plus)
        e_e_minus, l_e_minus = orbit_quantities(p, e_minus)
        dE_de = (e_e_plus - e_e_minus) / (e_plus - e_minus)
        dL_de = (l_e_plus - l_e_minus) / (e_plus - e_minus)
    else:
        dE_de = 0.0
        dL_de = 0.0

    return np.array([[dE_dp, dE_de], [dL_dp, dL_de]], dtype=float)


def finite_difference_jacobian_generic(a: float, p: float, e: float, x: float) -> np.ndarray:
    dp = max(1e-6, 1e-5 * abs(p))
    de = max(1e-8, 1e-5 * max(abs(e), 1e-3))
    dx = max(1e-6, 1e-5 * max(abs(x), 1e-3))

    def orbit_quantities(pp: float, ee: float, xx: float) -> tuple[float, float, float, float]:
        orb = KerrGeoOrbit(float(a), float(pp), float(ee), float(xx))
        E_val = float(orb.energy)
        Lz_val = float(orb.angular_momentum)
        Q_val = carter_constant(Lz_val, float(xx))
        omega_theta = float(orb.omega_theta)
        return E_val, Lz_val, Q_val, omega_theta

    # p derivatives (central difference)
    E_pp, Lz_pp, Q_pp, _ = orbit_quantities(p + dp, e, x)
    E_pm, Lz_pm, Q_pm, _ = orbit_quantities(p - dp, e, x)
    dE_dp = (E_pp - E_pm) / (2.0 * dp)
    dLz_dp = (Lz_pp - Lz_pm) / (2.0 * dp)
    dQ_dp = (Q_pp - Q_pm) / (2.0 * dp)

    # e derivatives (forward/backward near boundaries)
    e_minus = max(0.0, e - de)
    e_plus = min(0.95, e + de)
    if e_plus > e_minus:
        E_ep, Lz_ep, Q_ep, _ = orbit_quantities(p, e_plus, x)
        E_em, Lz_em, Q_em, _ = orbit_quantities(p, e_minus, x)
        dE_de = (E_ep - E_em) / (e_plus - e_minus)
        dLz_de = (Lz_ep - Lz_em) / (e_plus - e_minus)
        dQ_de = (Q_ep - Q_em) / (e_plus - e_minus)
    else:
        dE_de = dLz_de = dQ_de = 0.0

    # x derivatives (forward/backward near boundaries)
    x_minus = max(0.05, x - dx)
    x_plus = min(0.95, x + dx)
    if x_plus > x_minus:
        E_xp, Lz_xp, Q_xp, _ = orbit_quantities(p, e, x_plus)
        E_xm, Lz_xm, Q_xm, _ = orbit_quantities(p, e, x_minus)
        dE_dx = (E_xp - E_xm) / (x_plus - x_minus)
        dLz_dx = (Lz_xp - Lz_xm) / (x_plus - x_minus)
        dQ_dx = (Q_xp - Q_xm) / (x_plus - x_minus)
    else:
        dE_dx = dLz_dx = dQ_dx = 0.0

    return np.array(
        [[dE_dp, dE_de, dE_dx], [dLz_dp, dLz_de, dLz_dx], [dQ_dp, dQ_de, dQ_dx]],
        dtype=float,
    )


def equatorial_total_fluxes(
    a: float,
    p: float,
    e: float,
    x: float,
    *,
    ell_max: int,
    n_max: int,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
) -> tuple[float, float]:
    orbit = KerrGeoOrbit(float(a), float(p), float(e), float(x))
    energy_flux = 0.0
    angular_flux = 0.0
    if a == 0.0 and abs(x) == 1.0:
        return schwarzschild_total_fluxes(
            p,
            e,
            ell_max=ell_max,
            n_max=n_max,
            accelerator=accelerator,
            device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            if m == 0:
                continue
            for n in range(-n_max, n_max + 1):
                mode = TeukolskyPointParticleMode(
                    -2,
                    ell,
                    m,
                    n,
                    0,
                    orbit,
                    accelerator=accelerator,
                    device_id=device_id,
                    accelerator_resolution=accelerator_resolution,
                )
                energy_flux += float(mode.fluxes.energy.real)
                angular_flux += float(mode.fluxes.angular_momentum.real)
    return energy_flux, angular_flux


def schwarzschild_total_fluxes(
    p: float,
    e: float,
    *,
    ell_max: int,
    n_max: int,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
) -> tuple[float, float]:
    orbit = KerrGeoOrbit(0.0, float(p), float(e), 1.0)
    energy_flux = 0.0
    angular_flux = 0.0
    for ell in range(2, ell_max + 1):
        for m in range(1, ell + 1):
            for n in range(-n_max, n_max + 1):
                mode = TeukolskyPointParticleMode(
                    -2,
                    ell,
                    m,
                    n,
                    0,
                    orbit,
                    accelerator=accelerator,
                    device_id=device_id,
                    accelerator_resolution=accelerator_resolution,
                )
                energy_flux += 2.0 * float(mode.fluxes.energy.real)
                angular_flux += 2.0 * float(mode.fluxes.angular_momentum.real)
    return energy_flux, angular_flux


def generic_total_fluxes(
    a: float,
    p: float,
    e: float,
    x: float,
    *,
    ell_max: int,
    n_max: int,
    k_max: int = 3,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
) -> tuple[float, float, float]:
    r"""Orbit-averaged fluxes :math:`(\langle\dot{E}\rangle, \langle\dot{L_z}\rangle, \langle\dot{Q}\rangle)`.

    Supported cases:

    - equatorial Kerr (:math:`|x|=1`): delegates to
      :func:`equatorial_total_fluxes` and returns :math:`\dot{Q}=0`
    - inclined Schwarzschild (:math:`a=0`, :math:`|x|<1`): uses the exact
      symmetry relation with :math:`\dot{x}=0`

    Kerr non-equatorial (:math:`a\neq0`, :math:`|x|\neq1`) is not
    implemented and raises :class:`NotImplementedError`.
    """
    a_val = float(a)
    if abs(x) == 1.0:
        energy_flux, angular_flux = equatorial_total_fluxes(
            a_val, p, e, x,
            ell_max=ell_max, n_max=n_max,
            accelerator=accelerator, device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        return energy_flux, angular_flux, 0.0

    if a_val == 0.0:
        # Schwarzschild: Edot is inclination-independent, Lzdot ∝ x,
        # Qdot = 2 Lz (1-x²)/x² Lzdot (exact, x is conserved).
        energy_flux, angular_flux = equatorial_total_fluxes(
            0.0, p, e, 1.0,
            ell_max=ell_max, n_max=n_max,
            accelerator=accelerator, device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        orb = KerrGeoOrbit(0.0, float(p), float(e), float(x))
        Lz = float(orb.angular_momentum)
        carter_flux = 2.0 * Lz * (1.0 - x * x) / (x * x) * angular_flux
        return energy_flux, angular_flux, carter_flux

    raise NotImplementedError(
        "Carter-constant flux for Kerr non-equatorial orbits "
        "(a≠0, |x|≠1) requires angular projection of the Carter-constant "
        "operator onto spheroidal harmonics.  This is not implemented in "
        "the current Teukolsky Python port.  "
        "The Mathematica BHPT Teukolsky package includes this computation; "
        "porting it requires modifying ``_fluxes_s_minus_2`` and related "
        "functions in ``src/teukolsky/modes/point_particle.py``."
    )


def schwarzschild_eccentric_rhs(
    t: float,
    y: np.ndarray,
    *,
    M: float,
    mu: float,
    ell_max: int,
    n_max: int,
    accelerator: str,
    device_id: int,
    accelerator_resolution: int | None,
) -> np.ndarray:
    del t
    p = float(y[0])
    e = float(y[1])
    M_sec = _seconds_per_mass(M)
    scale = (mu / M) / M_sec
    energy_flux_raw, angular_flux_raw = schwarzschild_total_fluxes(
        p,
        e,
        ell_max=ell_max,
        n_max=n_max,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )
    Edot = -energy_flux_raw * scale
    Lzdot = -angular_flux_raw * scale
    jacobian = finite_difference_jacobian(p, e)
    rhs = np.array([Edot, Lzdot], dtype=float)
    try:
        pdot, edot = np.linalg.solve(jacobian, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"singular Jacobian at p={p}, e={e}") from exc
    return np.array([pdot, edot], dtype=float)


def equatorial_eccentric_rhs(
    t: float,
    y: np.ndarray,
    *,
    a: float,
    x: float,
    M: float,
    mu: float,
    ell_max: int,
    n_max: int,
    accelerator: str,
    device_id: int,
    accelerator_resolution: int | None,
) -> np.ndarray:
    del t
    p = float(y[0])
    e = float(y[1])
    M_sec = _seconds_per_mass(M)
    scale = (mu / M) / M_sec
    energy_flux_raw, angular_flux_raw = equatorial_total_fluxes(
        a,
        p,
        e,
        x,
        ell_max=ell_max,
        n_max=n_max,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )
    Edot = -energy_flux_raw * scale
    Lzdot = -angular_flux_raw * scale
    jacobian = finite_difference_jacobian_equatorial(a, p, e, x)
    rhs = np.array([Edot, Lzdot], dtype=float)
    try:
        pdot, edot = np.linalg.solve(jacobian, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"singular Jacobian at a={a}, p={p}, e={e}, x={x}") from exc
    return np.array([pdot, edot], dtype=float)


def generic_eccentric_rhs(
    t: float,
    y: np.ndarray,
    *,
    a: float,
    M: float,
    mu: float,
    ell_max: int,
    n_max: int,
    k_max: int = 3,
    accelerator: str,
    device_id: int,
    accelerator_resolution: int | None,
) -> np.ndarray:
    r"""Right-hand side for generic Kerr eccentric inspiral.

    .. warning::

       Only implemented for **equatorial** (:math:`|x| = 1`, any *a*)
       and **Schwarzschild** (:math:`a = 0`, any *x*) where
       :math:`\dot{x} = 0` by symmetry.  Raises :class:`NotImplementedError`
       for Kerr non-equatorial orbits.
    """
    del t
    p = float(y[0])
    e = float(y[1])
    x = float(y[2])
    M_sec = _seconds_per_mass(M)
    scale = (mu / M) / M_sec
    a_val = float(a)

    if abs(x) == 1.0 or a_val == 0.0:
        # equatorial or Schwarzschild: 2-DOF, xdot = 0
        energy_flux_raw, angular_flux_raw = equatorial_total_fluxes(
            a_val, p, e, 1.0,
            ell_max=ell_max, n_max=n_max,
            accelerator=accelerator, device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        Edot = -energy_flux_raw * scale
        Lzdot = -angular_flux_raw * scale
        jacobian = finite_difference_jacobian_equatorial(a_val, p, e, x)
        rhs_vec = np.array([Edot, Lzdot], dtype=float)
        try:
            pdot, edot = np.linalg.solve(jacobian, rhs_vec)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"singular Jacobian at a={a_val}, p={p}, e={e}, x={x}") from exc
        return np.array([pdot, edot, 0.0], dtype=float)

    raise NotImplementedError(
        "generic_eccentric_rhs for Kerr non-equatorial (a≠0, |x|≠1) "
        "requires Qdot which is not available.  See generic_total_fluxes."
    )


def integrate_schwarzschild_eccentric_inspiral(
    M: float,
    mu: float,
    p0: float,
    e0: float,
    *,
    t_end: float,
    trajectory_dt: float,
    ell_max: int = 4,
    n_max: int = 3,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
) -> AdiabaticTrajectory:
    if t_end <= 0.0:
        raise ValueError("t_end must be > 0")
    if trajectory_dt <= 0.0:
        raise ValueError("trajectory_dt must be > 0")
    if p0 <= 6.0:
        raise ValueError("p0 must be > 6 for Schwarzschild bound inspiral")
    if not (0.0 <= e0 < 1.0):
        raise ValueError("e0 must satisfy 0 <= e0 < 1")

    grid = np.arange(0.0, t_end + 0.5 * trajectory_dt, trajectory_dt, dtype=float)
    grid = grid[grid <= t_end]
    if grid.size == 0 or abs(grid[-1] - t_end) > 1e-12:
        grid = np.append(grid, t_end)

    M_sec = _seconds_per_mass(M)
    scale = (mu / M) / M_sec
    flux_cache: dict[tuple[float, float], tuple[float, float]] = {}
    jacobian_cache: dict[tuple[float, float], np.ndarray] = {}
    rhs_cache: dict[tuple[float, float], np.ndarray] = {}

    def fluxes_for_state(p: float, e: float) -> tuple[float, float]:
        key = (float(p), float(e))
        cached = flux_cache.get(key)
        if cached is not None:
            return cached
        value = schwarzschild_total_fluxes(
            p,
            e,
            ell_max=ell_max,
            n_max=n_max,
            accelerator=accelerator,
            device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        flux_cache[key] = value
        return value

    def jacobian_for_state(p: float, e: float) -> np.ndarray:
        key = (float(p), float(e))
        cached = jacobian_cache.get(key)
        if cached is not None:
            return cached
        value = finite_difference_jacobian(p, e)
        jacobian_cache[key] = value
        return value

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        del t
        p = float(y[0])
        e = float(y[1])
        key = (p, e)
        cached = rhs_cache.get(key)
        if cached is not None:
            return cached.copy()
        energy_flux_raw, angular_flux_raw = fluxes_for_state(p, e)
        jacobian = jacobian_for_state(p, e)
        rhs_flux = np.array([-energy_flux_raw * scale, -angular_flux_raw * scale], dtype=float)
        try:
            deriv = np.linalg.solve(jacobian, rhs_flux)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"singular Jacobian at p={p}, e={e}") from exc
        rhs_cache[key] = deriv
        return deriv.copy()

    solution = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        np.array([p0, e0], dtype=float),
        t_eval=grid,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    p = np.asarray(solution.y[0], dtype=float)
    e = np.asarray(solution.y[1], dtype=float)
    energy = np.empty_like(p)
    angular_momentum = np.empty_like(p)
    pdot = np.empty_like(p)
    edot = np.empty_like(p)
    edot_energy = np.empty_like(p)
    edot_angular_momentum = np.empty_like(p)

    for i, (ti, pi, ei) in enumerate(zip(solution.t, p, e)):
        orbit = KerrGeoOrbit(0.0, float(pi), float(ei), 1.0)
        energy[i] = float(orbit.energy)
        angular_momentum[i] = float(orbit.angular_momentum)
        energy_flux_raw, angular_flux_raw = fluxes_for_state(float(pi), float(ei))
        edot_energy[i] = -energy_flux_raw * scale
        edot_angular_momentum[i] = -angular_flux_raw * scale
        deriv = rhs(float(ti), np.array([pi, ei], dtype=float))
        pdot[i] = float(deriv[0])
        edot[i] = float(deriv[1])

    return AdiabaticTrajectory(
        time=np.asarray(solution.t, dtype=float),
        p=p,
        e=e,
        energy=energy,
        angular_momentum=angular_momentum,
        pdot=pdot,
        edot=edot,
        edot_energy=edot_energy,
        edot_angular_momentum=edot_angular_momentum,
    )


def integrate_equatorial_eccentric_inspiral(
    M: float,
    mu: float,
    a: float,
    p0: float,
    e0: float,
    x: float,
    *,
    t_end: float,
    trajectory_dt: float,
    ell_max: int = 4,
    n_max: int = 3,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
) -> AdiabaticTrajectory:
    if abs(x) != 1.0:
        raise ValueError("integrate_equatorial_eccentric_inspiral requires |x| = 1")
    if t_end <= 0.0:
        raise ValueError("t_end must be > 0")
    if trajectory_dt <= 0.0:
        raise ValueError("trajectory_dt must be > 0")
    if not (0.0 <= e0 < 1.0):
        raise ValueError("e0 must satisfy 0 <= e0 < 1")

    grid = np.arange(0.0, t_end + 0.5 * trajectory_dt, trajectory_dt, dtype=float)
    grid = grid[grid <= t_end]
    if grid.size == 0 or abs(grid[-1] - t_end) > 1e-12:
        grid = np.append(grid, t_end)

    M_sec = _seconds_per_mass(M)
    scale = (mu / M) / M_sec
    flux_cache: dict[tuple[float, float], tuple[float, float]] = {}
    jacobian_cache: dict[tuple[float, float], np.ndarray] = {}
    rhs_cache: dict[tuple[float, float], np.ndarray] = {}

    def fluxes_for_state(p: float, e: float) -> tuple[float, float]:
        key = (float(p), float(e))
        cached = flux_cache.get(key)
        if cached is not None:
            return cached
        value = equatorial_total_fluxes(
            a,
            p,
            e,
            x,
            ell_max=ell_max,
            n_max=n_max,
            accelerator=accelerator,
            device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        flux_cache[key] = value
        return value

    def jacobian_for_state(p: float, e: float) -> np.ndarray:
        key = (float(p), float(e))
        cached = jacobian_cache.get(key)
        if cached is not None:
            return cached
        value = finite_difference_jacobian_equatorial(a, p, e, x)
        jacobian_cache[key] = value
        return value

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        del t
        p = float(y[0])
        e = float(y[1])
        key = (p, e)
        cached = rhs_cache.get(key)
        if cached is not None:
            return cached.copy()
        energy_flux_raw, angular_flux_raw = fluxes_for_state(p, e)
        jacobian = jacobian_for_state(p, e)
        rhs_flux = np.array([-energy_flux_raw * scale, -angular_flux_raw * scale], dtype=float)
        try:
            deriv = np.linalg.solve(jacobian, rhs_flux)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"singular Jacobian at a={a}, p={p}, e={e}, x={x}") from exc
        rhs_cache[key] = deriv
        return deriv.copy()

    solution = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        np.array([p0, e0], dtype=float),
        t_eval=grid,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    p = np.asarray(solution.y[0], dtype=float)
    e = np.asarray(solution.y[1], dtype=float)
    energy = np.empty_like(p)
    angular_momentum = np.empty_like(p)
    pdot = np.empty_like(p)
    edot = np.empty_like(p)
    edot_energy = np.empty_like(p)
    edot_angular_momentum = np.empty_like(p)

    for i, (ti, pi, ei) in enumerate(zip(solution.t, p, e)):
        orbit = KerrGeoOrbit(float(a), float(pi), float(ei), float(x))
        energy[i] = float(orbit.energy)
        angular_momentum[i] = float(orbit.angular_momentum)
        energy_flux_raw, angular_flux_raw = fluxes_for_state(float(pi), float(ei))
        edot_energy[i] = -energy_flux_raw * scale
        edot_angular_momentum[i] = -angular_flux_raw * scale
        deriv = rhs(float(ti), np.array([pi, ei], dtype=float))
        pdot[i] = float(deriv[0])
        edot[i] = float(deriv[1])

    return AdiabaticTrajectory(
        time=np.asarray(solution.t, dtype=float),
        p=p,
        e=e,
        energy=energy,
        angular_momentum=angular_momentum,
        pdot=pdot,
        edot=edot,
        edot_energy=edot_energy,
        edot_angular_momentum=edot_angular_momentum,
    )


@dataclass(frozen=True)
class AdiabaticTrajectoryGeneric:
    time: np.ndarray
    p: np.ndarray
    e: np.ndarray
    x: np.ndarray
    energy: np.ndarray
    angular_momentum: np.ndarray
    carter_constant: np.ndarray
    pdot: np.ndarray
    edot: np.ndarray
    xdot: np.ndarray
    edot_energy: np.ndarray
    edot_angular_momentum: np.ndarray
    edot_carter: np.ndarray


def integrate_generic_eccentric_inspiral(
    M: float,
    mu: float,
    a: float,
    p0: float,
    e0: float,
    x0: float,
    *,
    t_end: float,
    trajectory_dt: float,
    ell_max: int = 4,
    n_max: int = 3,
    k_max: int = 3,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
) -> AdiabaticTrajectoryGeneric:
    if t_end <= 0.0:
        raise ValueError("t_end must be > 0")
    if trajectory_dt <= 0.0:
        raise ValueError("trajectory_dt must be > 0")
    if not (0.0 <= e0 < 1.0):
        raise ValueError("e0 must satisfy 0 <= e0 < 1")
    if not (0.05 <= abs(x0) <= 1.0):
        raise ValueError("x0 must satisfy 0.05 <= |x0| <= 1.0")

    a_val = float(a)
    x0_val = float(x0)
    if a_val != 0.0 and abs(x0_val) != 1.0:
        raise NotImplementedError(
            "integrate_generic_eccentric_inspiral does not support Kerr "
            "non-equatorial inspirals (a≠0, |x|≠1).  The current Teukolsky "
            "Python port lacks a validated third flux / Qdot.  Use a=0 "
            "(Schwarzschild inclined) or |x|=1 (equatorial Kerr) instead."
        )

    grid = np.arange(0.0, t_end + 0.5 * trajectory_dt, trajectory_dt, dtype=float)
    grid = grid[grid <= t_end]
    if grid.size == 0 or abs(grid[-1] - t_end) > 1e-12:
        grid = np.append(grid, t_end)

    M_sec = _seconds_per_mass(M)
    scale = (mu / M) / M_sec

    flux_cache_2d: dict[tuple[float, float], tuple[float, float]] = {}
    jac_cache_2d: dict[tuple[float, float], np.ndarray] = {}
    rhs_cache_2d: dict[tuple[float, float], np.ndarray] = {}

    def fluxes_2d(p: float, e: float) -> tuple[float, float]:
        key = (float(p), float(e))
        if key in flux_cache_2d:
            return flux_cache_2d[key]
        val = equatorial_total_fluxes(
            a_val, p, e, 1.0,
            ell_max=ell_max, n_max=n_max,
            accelerator=accelerator, device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        flux_cache_2d[key] = val
        return val

    def jac_2d(p: float, e: float) -> np.ndarray:
        key = (float(p), float(e))
        if key in jac_cache_2d:
            return jac_cache_2d[key]
        val = finite_difference_jacobian_equatorial(
            a_val, p, e, x0_val if abs(x0_val) == 1.0 else 1.0
        )
        jac_cache_2d[key] = val
        return val

    def rhs_2d(t: float, y: np.ndarray) -> np.ndarray:
        del t
        p = float(y[0]); e = float(y[1])
        key = (p, e)
        if key in rhs_cache_2d:
            return rhs_cache_2d[key].copy()
        ef, af = fluxes_2d(p, e)
        jac = jac_2d(p, e)
        vec = np.array([-ef * scale, -af * scale], dtype=float)
        deriv = np.linalg.solve(jac, vec)
        rhs_cache_2d[key] = deriv
        return deriv.copy()

    solution = solve_ivp(
        rhs_2d, (0.0, float(t_end)),
        np.array([p0, e0], dtype=float),
        t_eval=grid, method="DOP853", rtol=1e-8, atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    p = np.asarray(solution.y[0], dtype=float)
    e = np.asarray(solution.y[1], dtype=float)
    x_arr = np.full_like(p, x0_val)
    pdot_arr = np.empty_like(p)
    edot_arr = np.empty_like(p)
    xdot_arr = np.zeros_like(p)
    for i, (ti, pi, ei) in enumerate(zip(solution.t, p, e)):
        d = rhs_2d(float(ti), np.array([pi, ei], dtype=float))
        pdot_arr[i] = float(d[0]); edot_arr[i] = float(d[1])

    # --- unified post-processing for both 2‑DOF and 3‑DOF ---
    energy = np.empty_like(p)
    angular_momentum = np.empty_like(p)
    carter_arr = np.empty_like(p)
    edot_energy_arr = np.empty_like(p)
    edot_angular_momentum_arr = np.empty_like(p)
    edot_carter_arr = np.empty_like(p)

    for i in range(len(solution.t)):
        pi, ei, xi = float(p[i]), float(e[i]), float(x_arr[i])
        orb = KerrGeoOrbit(a_val, pi, ei, xi)
        energy[i] = float(orb.energy)
        Lz_val = float(orb.angular_momentum)
        angular_momentum[i] = Lz_val
        carter_arr[i] = carter_constant(Lz_val, xi)
        ef, af = fluxes_2d(pi, ei)
        edot_energy_arr[i] = -ef * scale
        edot_angular_momentum_arr[i] = -af * scale
        edot_carter_arr[i] = 2.0 * Lz_val * (1.0 - xi * xi) / (xi * xi) * edot_angular_momentum_arr[i]

    return AdiabaticTrajectoryGeneric(
        time=np.asarray(solution.t, dtype=float),
        p=p,
        e=e,
        x=np.asarray(x_arr, dtype=float),
        energy=energy,
        angular_momentum=angular_momentum,
        carter_constant=carter_arr,
        pdot=pdot_arr,
        edot=edot_arr,
        xdot=xdot_arr,
        edot_energy=edot_energy_arr,
        edot_angular_momentum=edot_angular_momentum_arr,
        edot_carter=edot_carter_arr,
    )


def enumerate_mode_indices(
    orbit: Orbit,
    *,
    ell_min: int = 2,
    ell_max: int = 10,
    n_max: int = 30,
    k_max: int = 20,
    include_m_zero: bool = True,
) -> list[tuple[int, int, int, int]]:
    if ell_min < 2:
        raise ValueError("ell_min must be >= 2")
    if ell_max < ell_min:
        raise ValueError("ell_max must be >= ell_min")
    if n_max < 0:
        raise ValueError("n_max must be >= 0")
    if k_max < 0:
        raise ValueError("k_max must be >= 0")

    if orbit.kind == "circular-equatorial":
        n_values = (0,)
        k_values = (0,)
    elif orbit.kind == "spherical":
        n_values = (0,)
        k_values = tuple(range(-k_max, k_max + 1))
    elif orbit.kind == "eccentric-equatorial":
        n_values = tuple(range(-n_max, n_max + 1))
        k_values = (0,)
    elif orbit.kind == "generic":
        n_values = tuple(range(-n_max, n_max + 1))
        k_values = tuple(range(-k_max, k_max + 1))
    else:
        raise ValueError(f"unsupported orbit kind: {orbit.kind}")

    indices: list[tuple[int, int, int, int]] = []
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):
            if not include_m_zero and m == 0:
                continue
            for n in n_values:
                for k in k_values:
                    indices.append((ell, m, n, k))
    return indices


def mode_strain(
    mode: ModeSolution,
    time: float | np.ndarray,
    *,
    theta: float,
    phi: float = 0.0,
    radius: float = 1.0,
    mass_seconds: float | None = None,
    omega_tol: float = 1e-12,
) -> complex | np.ndarray:
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    omega_geom = complex(mode.omega)
    if abs(omega_geom) <= omega_tol:
        raise ValueError("mode_strain is undefined for zero-frequency modes")
    omega_phase = omega_geom if mass_seconds is None else omega_geom / mass_seconds

    time_array = np.asarray(time, dtype=float)
    angular = mode["AngularFunction"].evaluate(theta, phi)
    psi4 = mode["Amplitudes"]["I"] * angular * np.exp(-1j * omega_phase * time_array) / radius
    strain = -2.0 * psi4 / (omega_geom * omega_geom)
    if time_array.ndim == 0:
        return complex(np.asarray(strain).item())
    return np.asarray(strain, dtype=np.complex128)


def generate_fixed_orbit_waveform(
    orbit: Orbit,
    time: np.ndarray | list[float] | tuple[float, ...],
    *,
    theta: float,
    phi: float = 0.0,
    radius: float = 1.0,
    mass_seconds: float | None = None,
    mode_indices: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...] | None = None,
    ell_min: int = 2,
    ell_max: int = 10,
    n_max: int = 30,
    k_max: int = 20,
    include_m_zero: bool = True,
    s: int = -2,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
    omega_tol: float = 1e-12,
) -> WaveformPolarizations:
    if s != -2:
        raise ValueError("generate_fixed_orbit_waveform currently supports only s = -2")

    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    if radius <= 0.0:
        raise ValueError("radius must be > 0")

    if mode_indices is None:
        mode_indices = enumerate_mode_indices(
            orbit,
            ell_min=ell_min,
            ell_max=ell_max,
            n_max=n_max,
            k_max=k_max,
            include_m_zero=include_m_zero,
        )

    complex_strain = np.zeros_like(time_array, dtype=np.complex128)
    used_modes: list[tuple[int, int, int, int]] = []

    for ell, m, n, k in mode_indices:
        omega = mode_frequency(orbit, m, n, k)
        if abs(omega) <= omega_tol:
            continue
        mode = TeukolskyPointParticleMode(
            s,
            ell,
            m,
            n,
            k,
            orbit,
            accelerator=accelerator,
            device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        complex_strain = complex_strain + mode_strain(
            mode,
            time_array,
            theta=theta,
            phi=phi,
            radius=radius,
            mass_seconds=mass_seconds,
            omega_tol=omega_tol,
        )
        used_modes.append((ell, m, n, k))

    h_plus = np.asarray(complex_strain.real, dtype=float)
    h_cross = np.asarray(-complex_strain.imag, dtype=float)
    return WaveformPolarizations(
        time=np.asarray(time_array, dtype=float),
        h_plus=h_plus,
        h_cross=h_cross,
        complex_strain=np.asarray(complex_strain, dtype=np.complex128),
        theta=float(theta),
        phi=float(phi),
        radius=float(radius),
        modes=tuple(used_modes),
    )


def generate_sparse_trajectory_waveform(
    M: float,
    a: float,
    time: np.ndarray | list[float] | tuple[float, ...],
    p: np.ndarray | list[float] | tuple[float, ...],
    e: np.ndarray | list[float] | tuple[float, ...],
    x: np.ndarray | list[float] | tuple[float, ...],
    *,
    evaluation_time: np.ndarray | list[float] | tuple[float, ...] | None = None,
    theta: float,
    phi: float = 0.0,
    radius: float = 1.0,
    mode_indices: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...] | None = None,
    waveform_ell_max: int = 4,
    waveform_n_max: int = 3,
    waveform_k_max: int = 3,
    include_m_zero: bool = True,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
    omega_tol: float = 1e-12,
) -> WaveformPolarizations:
    time_array = np.asarray(time, dtype=float)
    dense_time_array = time_array if evaluation_time is None else np.asarray(evaluation_time, dtype=float)
    p_array = np.asarray(p, dtype=float)
    e_array = np.asarray(e, dtype=float)
    x_array = np.asarray(x, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    if dense_time_array.ndim != 1:
        raise ValueError("evaluation_time must be a one-dimensional array")
    if p_array.ndim != 1 or e_array.ndim != 1 or x_array.ndim != 1:
        raise ValueError("p, e, and x must be one-dimensional arrays")
    if not (len(time_array) == len(p_array) == len(e_array) == len(x_array)):
        raise ValueError("time, p, e, and x must have the same length")
    if len(time_array) == 0:
        raise ValueError("time must not be empty")
    if len(dense_time_array) == 0:
        raise ValueError("evaluation_time must not be empty")
    if np.any(np.diff(time_array) < 0.0):
        raise ValueError("time must be monotonically nondecreasing")
    if np.any(np.diff(dense_time_array) < 0.0):
        raise ValueError("evaluation_time must be monotonically nondecreasing")
    if radius <= 0.0:
        raise ValueError("radius must be > 0")
    if M <= 0.0:
        raise ValueError("M must be > 0")

    orbit_nodes = [
        KerrGeoOrbit(float(a), float(pi), float(ei), float(xi))
        for pi, ei, xi in zip(p_array, e_array, x_array)
    ]
    if mode_indices is None:
        mode_indices = enumerate_mode_indices(
            orbit_nodes[0],
            ell_min=2,
            ell_max=waveform_ell_max,
            n_max=waveform_n_max,
            k_max=waveform_k_max,
            include_m_zero=include_m_zero,
        )

    complex_strain, used_modes = _sum_adiabatic_mode_series(
        orbit_nodes,
        time_array,
        dense_time_array,
        mode_indices,
        theta=theta,
        phi=phi,
        radius=radius,
        mass_seconds=_seconds_per_mass(M),
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
        omega_tol=omega_tol,
    )

    return WaveformPolarizations(
        time=np.asarray(dense_time_array, dtype=float),
        h_plus=np.asarray(complex_strain.real, dtype=float),
        h_cross=np.asarray(-complex_strain.imag, dtype=float),
        complex_strain=np.asarray(complex_strain, dtype=np.complex128),
        theta=float(theta),
        phi=float(phi),
        radius=float(radius),
        modes=used_modes,
    )


def generate_schwarzschild_eccentric_adiabatic_waveform(
    M: float,
    mu: float,
    p0: float,
    e0: float,
    time: np.ndarray | list[float] | tuple[float, ...],
    *,
    theta: float,
    phi: float = 0.0,
    radius: float = 1.0,
    trajectory_dt: float,
    mode_indices: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...] | None = None,
    trajectory_ell_max: int = 4,
    trajectory_n_max: int = 3,
    waveform_ell_max: int = 4,
    waveform_n_max: int = 3,
    include_m_zero: bool = True,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
    omega_tol: float = 1e-12,
) -> AdiabaticWaveform:
    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    if len(time_array) == 0:
        raise ValueError("time must not be empty")

    trajectory = integrate_schwarzschild_eccentric_inspiral(
        M,
        mu,
        p0,
        e0,
        t_end=float(time_array[-1]),
        trajectory_dt=trajectory_dt,
        ell_max=trajectory_ell_max,
        n_max=trajectory_n_max,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )

    if mode_indices is None:
        mode_indices = enumerate_mode_indices(
            KerrGeoOrbit(0.0, float(p0), float(e0), 1.0),
            ell_min=2,
            ell_max=waveform_ell_max,
            n_max=waveform_n_max,
            k_max=0,
            include_m_zero=include_m_zero,
        )
    sparse_time = trajectory.time
    orbit_nodes = [KerrGeoOrbit(0.0, float(pi), float(ei), 1.0) for pi, ei in zip(trajectory.p, trajectory.e)]
    M_sec = _seconds_per_mass(M)
    complex_strain, used_modes = _sum_adiabatic_mode_series(
        orbit_nodes,
        sparse_time,
        time_array,
        mode_indices,
        theta=theta,
        phi=phi,
        radius=radius,
        mass_seconds=M_sec,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
        omega_tol=omega_tol,
    )

    waveform = WaveformPolarizations(
        time=np.asarray(time_array, dtype=float),
        h_plus=np.asarray(complex_strain.real, dtype=float),
        h_cross=np.asarray(-complex_strain.imag, dtype=float),
        complex_strain=np.asarray(complex_strain, dtype=np.complex128),
        theta=float(theta),
        phi=float(phi),
        radius=float(radius),
        modes=used_modes,
    )
    return AdiabaticWaveform(waveform=waveform, trajectory=trajectory)


@dataclass(frozen=True)
class AdiabaticWaveformGeneric:
    waveform: WaveformPolarizations
    trajectory: AdiabaticTrajectoryGeneric


def generate_generic_eccentric_adiabatic_waveform(
    M: float,
    mu: float,
    a: float,
    p0: float,
    e0: float,
    x0: float,
    time: np.ndarray | list[float] | tuple[float, ...],
    *,
    theta: float,
    phi: float = 0.0,
    radius: float = 1.0,
    trajectory_dt: float,
    mode_indices: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...] | None = None,
    trajectory_ell_max: int = 4,
    trajectory_n_max: int = 3,
    trajectory_k_max: int = 3,
    waveform_ell_max: int = 4,
    waveform_n_max: int = 3,
    waveform_k_max: int = 3,
    include_m_zero: bool = True,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
    omega_tol: float = 1e-12,
) -> AdiabaticWaveformGeneric:
    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    if len(time_array) == 0:
        raise ValueError("time must not be empty")

    trajectory = integrate_generic_eccentric_inspiral(
        M, mu, a, p0, e0, x0,
        t_end=float(time_array[-1]),
        trajectory_dt=trajectory_dt,
        ell_max=trajectory_ell_max,
        n_max=trajectory_n_max,
        k_max=trajectory_k_max,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )

    if mode_indices is None:
        mode_indices = enumerate_mode_indices(
            KerrGeoOrbit(float(a), float(p0), float(e0), float(x0)),
            ell_min=2,
            ell_max=waveform_ell_max,
            n_max=waveform_n_max,
            k_max=waveform_k_max,
            include_m_zero=include_m_zero,
        )
    sparse_time = trajectory.time
    orbit_nodes = [
        KerrGeoOrbit(float(a), float(pi), float(ei), float(xi))
        for pi, ei, xi in zip(trajectory.p, trajectory.e, trajectory.x)
    ]
    M_sec = _seconds_per_mass(M)
    complex_strain, used_modes = _sum_adiabatic_mode_series(
        orbit_nodes,
        sparse_time,
        time_array,
        mode_indices,
        theta=theta,
        phi=phi,
        radius=radius,
        mass_seconds=M_sec,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
        omega_tol=omega_tol,
    )

    result_waveform = WaveformPolarizations(
        time=np.asarray(time_array, dtype=float),
        h_plus=np.asarray(complex_strain.real, dtype=float),
        h_cross=np.asarray(-complex_strain.imag, dtype=float),
        complex_strain=np.asarray(complex_strain, dtype=np.complex128),
        theta=float(theta),
        phi=float(phi),
        radius=float(radius),
        modes=used_modes,
    )
    return AdiabaticWaveformGeneric(waveform=result_waveform, trajectory=trajectory)


def generate_equatorial_eccentric_adiabatic_waveform(
    M: float,
    mu: float,
    a: float,
    p0: float,
    e0: float,
    x: float,
    time: np.ndarray | list[float] | tuple[float, ...],
    *,
    theta: float,
    phi: float = 0.0,
    radius: float = 1.0,
    trajectory_dt: float,
    mode_indices: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...] | None = None,
    trajectory_ell_max: int = 4,
    trajectory_n_max: int = 3,
    waveform_ell_max: int = 4,
    waveform_n_max: int = 3,
    include_m_zero: bool = True,
    accelerator: str = "cpu",
    device_id: int = 0,
    accelerator_resolution: int | None = None,
    omega_tol: float = 1e-12,
) -> AdiabaticWaveform:
    if abs(x) != 1.0:
        raise ValueError("generate_equatorial_eccentric_adiabatic_waveform requires |x| = 1")
    time_array = np.asarray(time, dtype=float)
    if time_array.ndim != 1:
        raise ValueError("time must be a one-dimensional array")
    if len(time_array) == 0:
        raise ValueError("time must not be empty")

    trajectory = integrate_equatorial_eccentric_inspiral(
        M,
        mu,
        a,
        p0,
        e0,
        x,
        t_end=float(time_array[-1]),
        trajectory_dt=trajectory_dt,
        ell_max=trajectory_ell_max,
        n_max=trajectory_n_max,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )

    if mode_indices is None:
        mode_indices = enumerate_mode_indices(
            KerrGeoOrbit(float(a), float(p0), float(e0), float(x)),
            ell_min=2,
            ell_max=waveform_ell_max,
            n_max=waveform_n_max,
            k_max=0,
            include_m_zero=include_m_zero,
        )
    sparse_time = trajectory.time
    orbit_nodes = [KerrGeoOrbit(float(a), float(pi), float(ei), float(x)) for pi, ei in zip(trajectory.p, trajectory.e)]
    M_sec = _seconds_per_mass(M)
    complex_strain, used_modes = _sum_adiabatic_mode_series(
        orbit_nodes,
        sparse_time,
        time_array,
        mode_indices,
        theta=theta,
        phi=phi,
        radius=radius,
        mass_seconds=M_sec,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
        omega_tol=omega_tol,
    )

    waveform = WaveformPolarizations(
        time=np.asarray(time_array, dtype=float),
        h_plus=np.asarray(complex_strain.real, dtype=float),
        h_cross=np.asarray(-complex_strain.imag, dtype=float),
        complex_strain=np.asarray(complex_strain, dtype=np.complex128),
        theta=float(theta),
        phi=float(phi),
        radius=float(radius),
        modes=used_modes,
    )
    return AdiabaticWaveform(waveform=waveform, trajectory=trajectory)

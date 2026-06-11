from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os

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


def _load_few_pn5():
    try:
        from few.trajectory.ode.pn5 import PN5
    except Exception as exc:
        raise ImportError(
            "Kerr non-equatorial adiabatic inspiral requires the optional "
            "'few' package with PN5 trajectory support."
        ) from exc
    return PN5


@lru_cache(maxsize=16)
def _pn5_generic_ode(a: float):
    PN5 = _load_few_pn5()
    ode = PN5()
    ode.add_fixed_parameters(1.0, 1.0, float(a), additional_args=[])
    return ode


def _pn5_internal_generic_rhs(a: float, p: float, e: float, x: float) -> np.ndarray:
    ode = _pn5_generic_ode(float(a))
    y = np.array([float(p), float(e), float(x), 0.0, 0.0, 0.0], dtype=float)
    return np.asarray(ode(y)[:3], dtype=float)


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

    if a == 0.0 and abs(x) == 1.0:
        return schwarzschild_total_fluxes(
            p, e,
            ell_max=ell_max, n_max=n_max,
            accelerator=accelerator, device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )

    energy_flux = 0.0
    angular_flux = 0.0
    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            if m == 0:
                continue
            for n in range(-n_max, n_max + 1):
                mode = TeukolskyPointParticleMode(
                    -2, ell, m, n, 0, orbit,
                    accelerator=accelerator, device_id=device_id,
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
                    -2, ell, m, n, 0, orbit,
                    accelerator=accelerator, device_id=device_id,
                    accelerator_resolution=accelerator_resolution,
                )
                energy_flux += 2.0 * float(mode.fluxes.energy.real)
                angular_flux += 2.0 * float(mode.fluxes.angular_momentum.real)
    return energy_flux, angular_flux


def _generic_mode_flux_sums(
    orbit: Orbit,
    *,
    ell_max: int,
    n_max: int,
    k_max: int,
    accelerator: str,
    device_id: int,
    accelerator_resolution: int | None,
) -> tuple[float, float, float, float]:
    energy_flux = 0.0
    angular_flux = 0.0
    radial_action_flux = 0.0
    polar_action_flux = 0.0

    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            for n in range(-n_max, n_max + 1):
                for k in range(-k_max, k_max + 1):
                    if m == 0 and n == 0 and k == 0:
                        continue
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
                    energy_mode = float(mode.fluxes.energy.real)
                    angular_mode = float(mode.fluxes.angular_momentum.real)
                    energy_flux += energy_mode
                    angular_flux += angular_mode
                    omega = float(mode.omega.real)
                    if abs(omega) <= 1e-14:
                        continue
                    radial_action_flux += float(n) * energy_mode / omega
                    polar_action_flux += float(k) * energy_mode / omega
    return energy_flux, angular_flux, radial_action_flux, polar_action_flux


# ---------------------------------------------------------------------------
#  Job-based parallel mode-sum (CPU or GPU)
# ---------------------------------------------------------------------------

_GENERIC_MODE_JOB_FIELDS = (
    "a", "p", "e", "x",
    "ell", "m", "n", "k",
    "accelerator", "device_id", "accelerator_resolution",
)


def _enumerate_generic_mode_jobs(
    a: float,
    p: float,
    e: float,
    x: float,
    *,
    ell_max: int,
    n_max: int,
    k_max: int,
    accelerator: str = "cpu",
    device_ids: list[int] | None = None,
    accelerator_resolution: int | None = None,
) -> list[dict]:
    """Return a flat list of job dicts for every (ell, m, n, k) mode.

    Each job is a plain dict — pickle-safe, no orbit / RadialSolution objects.
    """
    if device_ids is None:
        device_ids = [0]
    jobs: list[dict] = []
    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            for n in range(-n_max, n_max + 1):
                for k in range(-k_max, k_max + 1):
                    if m == 0 and n == 0 and k == 0:
                        continue
                    jobs.append({
                        "a": a, "p": p, "e": e, "x": x,
                        "ell": ell, "m": m, "n": n, "k": k,
                        "accelerator": accelerator,
                        "device_id": device_ids[len(jobs) % len(device_ids)],
                        "accelerator_resolution": accelerator_resolution,
                    })
    return jobs


def _compute_generic_mode_flux_job(job: dict) -> dict:
    """Worker entry point — pickle-safe, rebuilds orbit internally."""
    a = float(job["a"])
    p = float(job["p"])
    e = float(job["e"])
    x = float(job["x"])
    ell = int(job["ell"])
    m = int(job["m"])
    n = int(job["n"])
    k = int(job["k"])
    accelerator = str(job["accelerator"])
    device_id = int(job["device_id"])
    accelerator_resolution = job.get("accelerator_resolution")

    orbit = KerrGeoOrbit(a, p, e, x)
    mode = TeukolskyPointParticleMode(
        -2, ell, m, n, k, orbit,
        accelerator=accelerator, device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )
    energy_mode = float(mode.fluxes.energy.real)
    angular_mode = float(mode.fluxes.angular_momentum.real)
    omega = float(mode.omega.real)

    radial_contrib = 0.0
    polar_contrib = 0.0
    if abs(omega) > 1e-14:
        radial_contrib = float(n) * energy_mode / omega
        polar_contrib = float(k) * energy_mode / omega

    return {
        "ell": ell, "m": m, "n": n, "k": k,
        "omega": omega,
        "energy_flux": energy_mode,
        "angular_flux": angular_mode,
        "radial_action_contribution": radial_contrib,
        "polar_action_contribution": polar_contrib,
    }


def parallel_generic_action_fluxes_cpu(
    a: float,
    p: float,
    e: float,
    x: float,
    *,
    ell_max: int,
    n_max: int,
    k_max: int,
    workers: int | None = None,
    accelerator: str = "cpu",
    device_ids: list[int] | None = None,
    accelerator_resolution: int | None = None,
    chunksize: int = 1,
) -> tuple[float, float, float, float]:
    """CPU-parallel Teukolsky mode-sum for non-equatorial Kerr.

    Enumerates all :math:`(\\ell,m,n,k)` modes and distributes them across
    *workers* processes using :class:`concurrent.futures.ProcessPoolExecutor`.
    Each worker reconstructs the orbit internally and computes a single mode
    via :func:`TeukolskyPointParticleMode`.

    Parameters
    ----------
    a : float
        Kerr spin parameter (must satisfy ``a != 0``).
    p : float
        Semi-latus rectum.
    e : float
        Eccentricity.
    x : float
        Inclination parameter (must satisfy ``|x| != 1``).
    ell_max : int
    n_max : int
    k_max : int
    workers : int or None
        Number of worker processes.  Defaults to :func:`os.cpu_count`.
    accelerator : str
        ``"cpu"`` (default) or ``"gpu"``.  When ``"gpu"``, each worker
        uses a GPU device from *device_ids* assigned round-robin.
    device_ids : list of int or None
        GPU device indices for ``accelerator="gpu"``.  Required when using
        GPU acceleration.
    accelerator_resolution : int or None
        Radial sampling resolution for the GPU-accelerated path.
    chunksize : int
        Chunksize for :meth:`~concurrent.futures.Executor.map`.

    Returns
    -------
    tuple[float, float, float, float]
        ``(Edot, Lzdot, Jrdot, Jthetadot)``.
    """
    if a == 0.0 or abs(x) == 1.0:
        raise ValueError(
            "parallel_generic_action_fluxes_cpu is only defined for "
            "non-equatorial Kerr (a != 0 and |x| != 1)"
        )
    if accelerator == "gpu" and device_ids is None:
        raise ValueError("device_ids is required when accelerator='gpu'")

    import os as _os
    from concurrent.futures import ProcessPoolExecutor

    if workers is None:
        workers = _os.cpu_count() or 4

    jobs = _enumerate_generic_mode_jobs(
        a, p, e, x,
        ell_max=ell_max, n_max=n_max, k_max=k_max,
        accelerator=accelerator, device_ids=device_ids or [0],
        accelerator_resolution=accelerator_resolution,
    )

    energy_flux = 0.0
    angular_flux = 0.0
    radial_action_flux = 0.0
    polar_action_flux = 0.0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(_compute_generic_mode_flux_job, jobs, chunksize=chunksize):
            energy_flux += float(result["energy_flux"])
            angular_flux += float(result["angular_flux"])
            radial_action_flux += float(result["radial_action_contribution"])
            polar_action_flux += float(result["polar_action_contribution"])

    return energy_flux, angular_flux, radial_action_flux, polar_action_flux


def _radial_action(a_val: float, energy: float, angular_momentum: float, carter_constant_val: float) -> float:
    r"""Compute the radial action :math:`J_r` for a bound Kerr geodesic.

    The radial action is defined as

    .. math::

       J_r = \frac{1}{2\pi} \oint p_r \, dr
           = \frac{1}{\pi} \int_{r_{\min}}^{r_{\max}}
             \frac{\sqrt{R(r)}}{\Delta} \, dr

    where :math:`R(r) = [E(r^2+a^2)-aL_z]^2 - \Delta[r^2+(L_z-aE)^2+Q]`
    and :math:`\Delta = r^2 - 2r + a^2`.
    """
    delta_fn = lambda r: r * r - 2.0 * r + a_val * a_val

    # Radial potential coefficients
    # R(r) = (E²-1) r⁴ + 2 r³ + [a²(E²-1) - Lz² - Q] r² + 2[(aE-Lz)²+Q] r - a²Q
    E2m1 = energy * energy - 1.0
    a2 = a_val * a_val
    coeffs = [
        E2m1,
        2.0,
        a2 * E2m1 - angular_momentum * angular_momentum - carter_constant_val,
        2.0 * ((a_val * energy - angular_momentum) ** 2 + carter_constant_val),
        -a2 * carter_constant_val,
    ]
    roots = np.roots(coeffs)
    real_roots = sorted([float(r.real) for r in roots if abs(r.imag) < 1e-12 and float(r.real) > 0.0])

    if len(real_roots) < 2:
        raise ValueError(
            f"Could not find two positive real roots of R(r) for "
            f"a={a_val}, E={energy}, Lz={angular_momentum}, Q={carter_constant_val}"
        )

    # For a bound orbit, the two largest positive roots are r_min and r_max
    r_min = real_roots[-2]
    r_max = real_roots[-1]

    # Gauss-Legendre integration
    n_pts = 128
    xi, wi = np.polynomial.legendre.leggauss(n_pts)
    r_mid = 0.5 * (r_max + r_min)
    r_half = 0.5 * (r_max - r_min)
    r_pts = r_mid + r_half * xi

    integrand = np.zeros(n_pts, dtype=float)
    for i in range(n_pts):
        r = float(r_pts[i])
        delta = delta_fn(r)
        if delta <= 0.0:
            continue
        p_term = energy * (r * r + a2) - a_val * angular_momentum
        radicand = p_term * p_term - delta * (r * r + (angular_momentum - a_val * energy) ** 2 + carter_constant_val)
        if radicand <= 0.0:
            continue
        integrand[i] = np.sqrt(float(radicand)) / delta

    return float(np.sum(wi * integrand) * r_half / math.pi)


def _generic_jacobian_with_jr(a_val: float, p: float, e: float, x: float) -> np.ndarray:
    r"""Compute :math:`\partial(E, L_z, J_r)/\partial(p, e, x)` by finite differences.

    This replaces :func:`finite_difference_jacobian_generic` which computes
    :math:`\partial(E, L_z, Q)/\partial(p, e, x)`.  Using :math:`J_r` instead
    of :math:`Q` avoids the need to compute :math:`\dot{Q}` from the Teukolsky
    mode sum — the mode sum naturally provides :math:`\dot{J}_r`.
    """
    dp = max(1e-6, 1e-5 * abs(p))
    de_val = max(1e-8, 1e-5 * max(abs(e), 1e-3))
    dx_val = max(1e-6, 1e-5 * max(abs(x), 1e-3))

    def orbit_and_jr(pp, ee, xx):
        orb = KerrGeoOrbit(a_val, pp, ee, xx)
        E_val = float(orb.energy)
        Lz_val = float(orb.angular_momentum)
        Q_val = carter_constant(Lz_val, float(xx))
        Jr_val = _radial_action(a_val, E_val, Lz_val, Q_val)
        return E_val, Lz_val, Jr_val

    # p derivatives
    E_p, Lz_p, Jr_p = orbit_and_jr(p + dp, e, x)
    E_m, Lz_m, Jr_m = orbit_and_jr(p - dp, e, x)
    dE_dp = (E_p - E_m) / (2.0 * dp)
    dLz_dp = (Lz_p - Lz_m) / (2.0 * dp)
    dJr_dp = (Jr_p - Jr_m) / (2.0 * dp)

    # e derivatives
    e_minus = max(0.0, e - de_val)
    e_plus = min(0.95, e + de_val)
    if e_plus > e_minus:
        E_p, Lz_p, Jr_p = orbit_and_jr(p, e_plus, x)
        E_m, Lz_m, Jr_m = orbit_and_jr(p, e_minus, x)
        h_e = e_plus - e_minus
        dE_de = (E_p - E_m) / h_e
        dLz_de = (Lz_p - Lz_m) / h_e
        dJr_de = (Jr_p - Jr_m) / h_e
    else:
        dE_de = dLz_de = dJr_de = 0.0

    # x derivatives
    x_minus = max(0.05, x - dx_val)
    x_plus = min(0.95, x + dx_val)
    if x_plus > x_minus:
        E_p, Lz_p, Jr_p = orbit_and_jr(p, e, x_plus)
        E_m, Lz_m, Jr_m = orbit_and_jr(p, e, x_minus)
        h_x = x_plus - x_minus
        dE_dx = (E_p - E_m) / h_x
        dLz_dx = (Lz_p - Lz_m) / h_x
        dJr_dx = (Jr_p - Jr_m) / h_x
    else:
        dE_dx = dLz_dx = dJr_dx = 0.0

    return np.array(
        [[dE_dp, dE_de, dE_dx], [dLz_dp, dLz_de, dLz_dx], [dJr_dp, dJr_de, dJr_dx]],
        dtype=float,
    )


def _radial_balance_averages(orbit: Orbit) -> tuple[float, float]:
    if orbit.radial_phase_function is None:
        raise ValueError("generic orbit is missing radial phase data")
    q_r = np.linspace(0.0, 2.0 * math.pi, 513)
    r_values = np.asarray(orbit.radial_phase_function(q_r), dtype=float)
    delta = r_values * r_values - 2.0 * r_values + orbit.a * orbit.a
    p_hat = orbit.energy * (r_values * r_values + orbit.a * orbit.a) - orbit.a * orbit.angular_momentum
    average_p_over_delta = np.trapz(((r_values * r_values + orbit.a * orbit.a) / delta) * p_hat, q_r) / (
        2.0 * math.pi
    )
    average_ap_over_delta = np.trapz((orbit.a / delta) * p_hat, q_r) / (2.0 * math.pi)
    return float(average_p_over_delta), float(average_ap_over_delta)


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
    num_gpus: int = 0,
) -> tuple[float, float, float]:
    r"""Orbit-averaged fluxes ``(Edot, Lzdot, Qdot)``.

    Always returns a 3-tuple.

    - equatorial Kerr (:math:`|x|=1`): delegates to
      :func:`equatorial_total_fluxes`, returns ``(Edot, Lzdot, 0.0)``
    - inclined Schwarzschild (:math:`a=0`, :math:`|x|<1`): uses the exact
      symmetry relation with :math:`\dot{x}=0`, returns ``(Edot, Lzdot, Qdot)``
    - non-equatorial Kerr (:math:`a\\neq0`, :math:`|x|<1`): **not supported**
      — raises ``RuntimeError``.  Use :func:`generic_action_fluxes` to obtain
      the Teukolsky mode-sum action fluxes, or use
      :func:`generic_eccentric_rhs` to obtain :math:`(\dot{p},\dot{e},\dot{x})`
      directly.
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
        # Schwarzschild: Edot is inclination-independent, while the
        # azimuthal flux scales with the conserved direction cosine x:
        #   Lzdot = x * Ldot_eq
        # and
        #   Qdot = 2 Lz (1-x²)/x² Lzdot
        energy_flux, angular_flux_equatorial = equatorial_total_fluxes(
            0.0, p, e, 1.0,
            ell_max=ell_max, n_max=n_max,
            accelerator=accelerator, device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        angular_flux = float(x) * angular_flux_equatorial
        orb = KerrGeoOrbit(0.0, float(p), float(e), float(x))
        Lz = float(orb.angular_momentum)
        carter_flux = 2.0 * Lz * (1.0 - x * x) / (x * x) * angular_flux
        return energy_flux, angular_flux, carter_flux

    # Non-equatorial Kerr: Qdot is not available from the pure Teukolsky
    # mode sum.  Use generic_action_fluxes for the action fluxes, or
    # generic_eccentric_rhs to evolve (p, e, x) directly.
    raise RuntimeError(
        "generic_total_fluxes cannot return Qdot for non-equatorial Kerr "
        "(a != 0 and |x| != 1).  Use generic_action_fluxes to obtain "
        "(Edot, Lzdot, Jrdot, Jthetadot) from the pure Teukolsky mode sum, "
        "or generic_eccentric_rhs to obtain (pdot, edot, xdot) directly."
    )


def generic_action_fluxes(
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
) -> tuple[float, float, float, float]:
    r"""Pure Teukolsky mode-sum action fluxes for non-equatorial Kerr.

    Returns ``(Edot, Lzdot, Jrdot, Jthetadot)`` — the four action-flux
    components obtained by summing :math:`(\ell,m,n,k)` Teukolsky modes.

    This function is the building block for :func:`generic_eccentric_rhs`,
    which uses :math:`\dot{J}_r` together with a
    :math:`\partial(E,L_z,J_r)/\partial(p,e,x)` Jacobian to obtain
    :math:`(\dot{p},\dot{e},\dot{x})` without a Carter-flux formula.

    Parameters
    ----------
    a : float
        Kerr spin parameter.
    p : float
        Semi-latus rectum.
    e : float
        Eccentricity.
    x : float
        Orbital inclination parameter :math:`x = \cos\iota`.
    ell_max : int
        Maximum :math:`\ell` mode.
    n_max : int
        Maximum :math:`|n|` harmonic.
    k_max : int
        Maximum :math:`|k|` harmonic.
    accelerator : str
        ``"cpu"`` or ``"gpu"`` / ``"dcu"``.
    device_id : int
        GPU device index (used when *accelerator* is ``"gpu"``).
    accelerator_resolution : int or None
        Radial sampling resolution for the GPU-accelerated path.

    Returns
    -------
    tuple[float, float, float, float]
        ``(energy_flux, angular_flux, radial_action_flux, polar_action_flux)``.
    """
    if abs(x) == 1.0 or a == 0.0:
        raise ValueError(
            "generic_action_fluxes is only defined for non-equatorial Kerr "
            "(a != 0 and |x| != 1).  Use generic_total_fluxes for equatorial "
            "or Schwarzschild-inclined cases."
        )
    a_val = float(a)
    orbit = KerrGeoOrbit(a_val, float(p), float(e), float(x))
    return _generic_mode_flux_sums(
        orbit,
        ell_max=ell_max,
        n_max=n_max,
        k_max=k_max,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
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

       Equatorial Kerr (:math:`|x| = 1`) and inclined Schwarzschild
       (:math:`a = 0`) reduce to a 2-DOF system with :math:`\dot{x}=0`.
       Kerr non-equatorial is evolved from generic Teukolsky mode-sum
       fluxes over :math:`(\ell,m,n,k)`.
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
        if a_val == 0.0 and abs(x) != 1.0:
            angular_flux_raw = float(x) * angular_flux_raw
        Edot = -energy_flux_raw * scale
        Lzdot = -angular_flux_raw * scale
        jacobian = finite_difference_jacobian_equatorial(a_val, p, e, x)
        rhs_vec = np.array([Edot, Lzdot], dtype=float)
        try:
            pdot, edot = np.linalg.solve(jacobian, rhs_vec)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"singular Jacobian at a={a_val}, p={p}, e={e}, x={x}") from exc
        return np.array([pdot, edot, 0.0], dtype=float)

    # Non-equatorial Kerr: use action-Jacobian approach.
    # Compute (Edot, Lzdot, Jrdot, Jthetadot) from pure Teukolsky mode sum,
    # then use ∂(E,Lz,Jr)/∂(p,e,x) Jacobian to obtain (pdot,edot,xdot).
    # This avoids the need for a Carter-flux formula.
    orbit = KerrGeoOrbit(a_val, float(p), float(e), float(x))
    energy_flux_raw, angular_flux_raw, radial_action_flux_raw, _ = _generic_mode_flux_sums(
        orbit,
        ell_max=ell_max,
        n_max=n_max,
        k_max=k_max,
        accelerator=accelerator,
        device_id=device_id,
        accelerator_resolution=accelerator_resolution,
    )
    jacobian = _generic_jacobian_with_jr(a_val, p, e, x)
    rhs_vec = -np.array([energy_flux_raw, angular_flux_raw, radial_action_flux_raw], dtype=float) * scale
    try:
        pdot, edot, xdot = np.linalg.solve(jacobian, rhs_vec)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"singular generic Jacobian (Jr-based) at a={a_val}, p={p}, e={e}, x={x}"
        ) from exc
    return np.array([pdot, edot, xdot], dtype=float)


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

    grid = np.arange(0.0, t_end + 0.5 * trajectory_dt, trajectory_dt, dtype=float)
    grid = grid[grid <= t_end]
    if grid.size == 0 or abs(grid[-1] - t_end) > 1e-12:
        grid = np.append(grid, t_end)

    M_sec = _seconds_per_mass(M)
    scale = (mu / M) / M_sec

    use_full_3dof = a_val != 0.0 and abs(x0_val) != 1.0

    flux_cache_2d: dict[tuple[float, float], tuple[float, float]] = {}
    jac_cache_2d: dict[tuple[float, float], np.ndarray] = {}
    rhs_cache_2d: dict[tuple[float, float], np.ndarray] = {}

    def fluxes_2d(p: float, e: float) -> tuple[float, float]:
        key = (float(p), float(e))
        if key in flux_cache_2d:
            return flux_cache_2d[key]
        energy_flux_raw, angular_flux_raw = equatorial_total_fluxes(
            a_val, p, e, 1.0,
            ell_max=ell_max, n_max=n_max,
            accelerator=accelerator, device_id=device_id,
            accelerator_resolution=accelerator_resolution,
        )
        if a_val == 0.0 and abs(x0_val) != 1.0:
            angular_flux_raw = x0_val * angular_flux_raw
        val = (energy_flux_raw, angular_flux_raw)
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

    if use_full_3dof:
        rhs_cache_3d: dict[tuple[float, float, float], np.ndarray] = {}

        def rhs_3d(t: float, y: np.ndarray) -> np.ndarray:
            del t
            p = float(y[0]); e = float(y[1]); x = float(y[2])
            key = (p, e, x)
            if key in rhs_cache_3d:
                return rhs_cache_3d[key].copy()
            deriv = generic_eccentric_rhs(
                0.0,
                np.array([p, e, x], dtype=float),
                a=a_val,
                M=M,
                mu=mu,
                ell_max=ell_max,
                n_max=n_max,
                k_max=k_max,
                accelerator=accelerator,
                device_id=device_id,
                accelerator_resolution=accelerator_resolution,
            )
            rhs_cache_3d[key] = deriv
            return deriv.copy()

        solution = solve_ivp(
            rhs_3d, (0.0, float(t_end)),
            np.array([p0, e0, x0_val], dtype=float),
            t_eval=grid, method="DOP853", rtol=1e-8, atol=1e-10,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        p = np.asarray(solution.y[0], dtype=float)
        e = np.asarray(solution.y[1], dtype=float)
        x_arr = np.asarray(solution.y[2], dtype=float)
        pdot_arr = np.empty_like(p)
        edot_arr = np.empty_like(p)
        xdot_arr = np.empty_like(p)
        for i in range(len(solution.t)):
            d = rhs_3d(float(solution.t[i]), np.array([p[i], e[i], x_arr[i]], dtype=float))
            pdot_arr[i] = float(d[0]); edot_arr[i] = float(d[1]); xdot_arr[i] = float(d[2])
    else:
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
        for i in range(len(solution.t)):
            d = rhs_2d(float(solution.t[i]), np.array([p[i], e[i]], dtype=float))
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
        if use_full_3dof:
            jac = finite_difference_jacobian_generic(a_val, pi, ei, xi)
            flux_deriv = jac @ np.array([pdot_arr[i], edot_arr[i], xdot_arr[i]], dtype=float)
            edot_energy_arr[i] = float(flux_deriv[0])
            edot_angular_momentum_arr[i] = float(flux_deriv[1])
            edot_carter_arr[i] = float(flux_deriv[2])
        else:
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

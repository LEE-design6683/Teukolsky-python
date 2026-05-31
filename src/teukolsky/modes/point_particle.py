from __future__ import annotations

import math

import numpy as np

from teukolsky.accelerated.backend import require_dcu
from teukolsky.accelerated.convolution import accelerated_eccentric_alpha, accelerated_generic_alpha
from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic
from teukolsky.core import Fluxes, ModeSolution, Orbit, RadialSolution
from teukolsky.geodesics.kerr import spherical_orbit
from teukolsky.radial import solve_radial

_MODE_OPTION_CONTEXT: dict[str, object] = {
    "source_type": "Automatic",
    "domain": "Automatic",
    "accelerator": "cpu",
    "device_id": 0,
}


def _default_source_type(s: int) -> str:
    if s in {-2, 2}:
        return "Weyl"
    if s in {-1, 1}:
        return "Maxwell"
    return "Scalar"


def _resolve_source_type(s: int, source_type: str | None) -> str:
    if source_type in {None, "Automatic"}:
        return _default_source_type(s)
    return str(source_type)


def _resolve_mode_domain(domain: tuple[float, float] | str | None) -> tuple[float, float] | str:
    if domain in {None, "Automatic"}:
        return "Automatic"
    if len(domain) != 2:
        raise ValueError("domain must be a pair (rmin, rmax)")
    return float(domain[0]), float(domain[1])


def _resolve_accelerator(accelerator: str | None) -> str:
    if accelerator in {None, "cpu"}:
        return "cpu"
    if accelerator == "dcu":
        return "dcu"
    raise ValueError(f"unsupported accelerator: {accelerator!r}")


def _mode_frequency(orbit: Orbit, m: int, n: int, k: int) -> complex:
    return m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta


def _validate_point_particle_request(
    *,
    s: int,
    m: int,
    orbit: Orbit,
    n: int,
    k: int,
    domain: tuple[float, float] | str,
) -> None:
    if orbit["Parametrization"] != "Mino":
        raise ValueError(f"unsupported orbit parametrization: {orbit['Parametrization']}")
    if orbit.kind == "circular-equatorial" and (n != 0 or k != 0):
        raise ValueError("circular equatorial orbits only support n = k = 0")
    if orbit.kind == "spherical" and n != 0:
        raise ValueError("spherical orbits only support n = 0")
    if orbit.kind == "eccentric-equatorial" and k != 0:
        raise ValueError("eccentric equatorial orbits only support k = 0")
    if domain != "Automatic" and _mode_frequency(orbit, m=m, n=n, k=k) == 0:
        raise ValueError("Domain option is not supported for static point-particle modes")


def _point_particle_radial_solutions(
    *,
    s: int,
    ell: int,
    m: int,
    orbit: Orbit,
    omega: complex,
) -> dict[str, RadialSolution]:
    domain = _MODE_OPTION_CONTEXT["domain"]
    if domain == "Automatic":
        return solve_radial(s=s, ell=ell, m=m, a=orbit.a, omega=omega)
    return solve_radial(
        s=s,
        ell=ell,
        m=m,
        a=orbit.a,
        omega=omega,
        method="NumericalIntegration",
        domain=domain,
    )


def _restrict_radial_domain(radial: RadialSolution, domain: tuple[float, float]) -> RadialSolution:
    return RadialSolution(
        s=radial.s,
        l=radial.l,
        m=radial.m,
        a=radial.a,
        omega=radial.omega,
        eigenvalue=radial.eigenvalue,
        renormalized_angular_momentum=radial.renormalized_angular_momentum,
        method=radial.method,
        boundary_conditions=radial.boundary_conditions,
        amplitudes=radial.amplitudes,
        unscaled_amplitudes=radial.unscaled_amplitudes,
        domain=domain,
        radial_function=radial.radial_function,
        derivative_function=radial.derivative_function,
        method_options=radial.method_options,
    )


def _mode_radial_solutions(
    *,
    s: int,
    ell: int,
    m: int,
    orbit: Orbit,
    omega: complex,
) -> tuple[dict[str, RadialSolution], dict[str, RadialSolution]]:
    user_radial = _point_particle_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    if orbit.kind not in {"eccentric-equatorial", "generic"} or omega == 0:
        return user_radial, user_radial
    eps = 1e-12
    rmin = orbit.p / (1.0 + orbit.e)
    rmax = orbit.p / (1.0 - orbit.e)
    internal_domain = ((1.0 - eps) * rmin, (1.0 + eps) * rmax)
    if _MODE_OPTION_CONTEXT["domain"] == "Automatic":
        base_radial = user_radial
    else:
        base_radial = solve_radial(
            s=s,
            ell=ell,
            m=m,
            a=orbit.a,
            omega=omega,
            eigenvalue=user_radial["In"].eigenvalue,
            renormalized_angular_momentum_value=user_radial["In"].renormalized_angular_momentum,
        )
    compute_radial = {
        "In": _restrict_radial_domain(base_radial["In"], internal_domain),
        "Up": _restrict_radial_domain(base_radial["Up"], internal_domain),
    }
    return user_radial, compute_radial


def _fluxes_s_minus_2(
    *,
    a: float,
    m: int,
    omega: complex,
    eigenvalue: complex,
    z_i: complex,
    z_h: complex,
) -> Fluxes:
    mass = 1.0
    rh = mass + math.sqrt(mass * mass - a * a)
    omega_h = a / (2.0 * mass * rh)
    kappa = omega - m * omega_h
    epsilon = math.sqrt(mass * mass - a * a) / (4.0 * mass * rh)
    flux_inf = abs(z_i) ** 2 / (4.0 * math.pi * omega * omega)
    abs_c_sq = (
        ((eigenvalue + 2.0) ** 2 + 4.0 * a * m * omega - 4.0 * a * a * omega * omega)
        * (eigenvalue * eigenvalue + 36.0 * m * a * omega - 36.0 * a * a * omega * omega)
        + (2.0 * eigenvalue + 3.0) * (96.0 * a * a * omega * omega - 48.0 * m * a * omega)
        + 144.0 * omega * omega * (mass * mass - a * a)
    )
    alpha = (
        256.0
        * (2.0 * mass * rh) ** 5
        * kappa
        * (kappa * kappa + 4.0 * epsilon * epsilon)
        * (kappa * kappa + 16.0 * epsilon * epsilon)
        * omega**3
        / abs_c_sq
    )
    flux_h = alpha * abs(z_h) ** 2 / (4.0 * math.pi * omega * omega)
    return Fluxes(
        energy_infinity=complex(flux_inf),
        energy_horizon=complex(flux_h),
        angular_momentum_infinity=complex(flux_inf * m / omega),
        angular_momentum_horizon=complex(flux_h * m / omega),
    )


def _fluxes_s_zero(
    *,
    a: float,
    m: int,
    omega: complex,
    z_i: complex,
    z_h: complex,
) -> Fluxes:
    if omega == 0:
        return Fluxes(
            energy_infinity=0.0 + 0.0j,
            energy_horizon=0.0 + 0.0j,
            angular_momentum_infinity=0.0 + 0.0j,
            angular_momentum_horizon=0.0 + 0.0j,
        )
    mass = 1.0
    rh = mass + math.sqrt(mass * mass - a * a)
    omega_h = a / (2.0 * mass * rh)
    flux_inf = abs(z_i) ** 2 * omega * omega / (4.0 * math.pi)
    flux_h = rh * omega * (omega - m * omega_h) * abs(z_h) ** 2 / (2.0 * math.pi)
    return Fluxes(
        energy_infinity=complex(flux_inf),
        energy_horizon=complex(flux_h),
        angular_momentum_infinity=complex(flux_inf * m / omega),
        angular_momentum_horizon=complex(flux_h * m / omega),
    )


def _fluxes_s_plus_2(
    *,
    a: float,
    m: int,
    omega: complex,
    eigenvalue: complex,
    z_i: complex,
    z_h: complex,
) -> Fluxes:
    if omega == 0:
        return Fluxes(
            energy_infinity=0.0 + 0.0j,
            energy_horizon=0.0 + 0.0j,
            angular_momentum_infinity=0.0 + 0.0j,
            angular_momentum_horizon=0.0 + 0.0j,
        )
    mass = 1.0
    rh = mass + math.sqrt(mass * mass - a * a)
    omega_h = a / (2.0 * mass * rh)
    kappa = omega - m * omega_h
    epsilon = math.sqrt(mass * mass - a * a) / (4.0 * mass * rh)
    abs_c_sq = (
        (eigenvalue + 4.0) ** 2 * (eigenvalue + 6.0) ** 2
        + 144.0 * mass * mass * omega * omega
        + 8.0 * a * (eigenvalue + 4.0) * (-4.0 + 5.0 * (eigenvalue + 6.0)) * omega * (m - a * omega)
        + 48.0 * a * a * omega * omega * (2.0 * (eigenvalue + 4.0) + 3.0 * (m - a * omega) ** 2)
    )
    flux_inf = 16.0 * omega**6 * abs(z_i) ** 2 / abs_c_sq / (4.0 * math.pi)
    flux_h = omega * abs(z_h) ** 2 / (512.0 * math.pi * rh**3 * kappa * (kappa * kappa + 4.0 * epsilon * epsilon))
    return Fluxes(
        energy_infinity=complex(flux_inf),
        energy_horizon=complex(flux_h),
        angular_momentum_infinity=complex(flux_inf * m / omega),
        angular_momentum_horizon=complex(flux_h * m / omega),
    )


def _fluxes_s_minus_1(
    *,
    a: float,
    m: int,
    omega: complex,
    eigenvalue: complex,
    z_i: complex,
    z_h: complex,
) -> Fluxes:
    if omega == 0:
        return Fluxes(0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
    mass = 1.0
    rh = mass + math.sqrt(mass * mass - a * a)
    omega_h = a / (2.0 * mass * rh)
    kappa = omega - m * omega_h
    p = eigenvalue * eigenvalue + 4.0 * a * omega * (m - a * omega)
    flux_inf = abs(z_i) ** 2 / (2.0 * math.pi)
    flux_h = 16.0 * omega * rh * kappa * ((2.0 * rh * kappa) ** 2 + (mass * mass - a * a)) * abs(z_h) ** 2 / (p * math.pi)
    return Fluxes(
        energy_infinity=complex(flux_inf),
        energy_horizon=complex(flux_h),
        angular_momentum_infinity=complex(flux_inf * m / omega),
        angular_momentum_horizon=complex(flux_h * m / omega),
    )


def _fluxes_s_plus_1(
    *,
    a: float,
    m: int,
    omega: complex,
    eigenvalue: complex,
    z_i: complex,
    z_h: complex,
) -> Fluxes:
    if omega == 0:
        return Fluxes(0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
    mass = 1.0
    rh = mass + math.sqrt(mass * mass - a * a)
    omega_h = a / (2.0 * mass * rh)
    kappa = omega - m * omega_h
    abs_c_sq = (eigenvalue + 2.0) ** 2 + 4.0 * a * omega * (m - a * omega)
    flux_inf = 8.0 * omega**4 * abs(z_i) ** 2 / abs_c_sq / (4.0 * math.pi)
    flux_h = omega * abs(z_h) ** 2 / (16.0 * math.pi * kappa * rh)
    return Fluxes(
        energy_infinity=complex(flux_inf),
        energy_horizon=complex(flux_h),
        angular_momentum_infinity=complex(flux_inf * m / omega),
        angular_momentum_horizon=complex(flux_h * m / omega),
    )


def _radial_second_derivative(
    radial_value: complex,
    radial_derivative: complex,
    *,
    s: int,
    m: int,
    a: float,
    omega: complex,
    eigenvalue: complex,
    r: float,
) -> complex:
    delta = r * r - 2.0 * r + a * a
    common = -(
        -eigenvalue
        + 4.0j * r * s * omega
        + (
            -2.0j * (r - 1.0) * s * (-a * m + (a * a + r * r) * omega)
            + (-a * m + (a * a + r * r) * omega) ** 2
        )
        / delta
    )
    return (common * radial_value - 2.0 * (r - 1.0) * (1.0 + s) * radial_derivative) / delta


def _spin_two_coefficients(
    *,
    r: float,
    ur: float,
    theta: float,
    u_theta: float,
    a: float,
    energy: float,
    angular_momentum: float,
    s: int,
    m: int,
    omega: complex,
    harmonic_value: complex,
    harmonic_derivative: complex,
    harmonic_second_derivative: complex,
) -> tuple[complex, complex, complex]:
    delta = r * r - 2.0 * r + a * a
    kt = (r * r + a * a) * omega - m * a
    l1 = -m / math.sin(theta) + a * omega * math.sin(theta) + math.cos(theta) / math.sin(theta)
    l2 = -m / math.sin(theta) + a * omega * math.sin(theta) + 2.0 * math.cos(theta) / math.sin(theta)
    l2s = harmonic_derivative + l2 * harmonic_value
    l2p = m * math.cos(theta) / math.sin(theta) ** 2 + a * omega * math.cos(theta) - 2.0 / math.sin(theta) ** 2
    l1sp = harmonic_second_derivative + l1 * harmonic_derivative
    l1l2s = l1sp + l2p * harmonic_value + l2 * harmonic_derivative + l1 * l2 * harmonic_value

    rho = -1.0 / complex(r, -a * math.cos(theta))
    rhobar = -1.0 / complex(r, a * math.cos(theta))
    sigma = 1.0 / (rho * rhobar)
    ann0 = (
        -(rho**-2)
        * (rhobar**-1)
        * (math.sqrt(2.0) * delta) ** (-2)
        * (
            rho**-1 * l1l2s
            + 3.0j * a * math.sin(theta) * l1 * harmonic_value
            + 3.0j * a * math.cos(theta) * harmonic_value
            + 2.0j * a * math.sin(theta) * harmonic_derivative
            - 1.0j * a * math.sin(theta) * l2 * harmonic_value
        )
    )
    anmbar0 = rho**-3 * (math.sqrt(2.0) * delta) ** (-1) * (
        (rho + rhobar - 1.0j * kt / delta) * l2s + (rho - rhobar) * a * math.sin(theta) * kt / delta * harmonic_value
    )
    anmbar1 = -(rho**-3) * (math.sqrt(2.0) * delta) ** (-1) * (
        l2s + 1.0j * (rho - rhobar) * a * math.sin(theta) * harmonic_value
    )
    ambarmbar0 = (
        (kt * kt * harmonic_value * rhobar) / (4.0 * delta * delta * rho**3)
        + (1.0j * kt * harmonic_value * (1.0 - r + delta * rho) * rhobar) / (2.0 * delta * delta * rho**3)
        + (1.0j * r * harmonic_value * rhobar * omega) / (2.0 * delta * rho**3)
    )
    ambarmbar1 = -(rho**-3) * rhobar * harmonic_value / 2.0 * (1.0j * kt / delta - rho)
    ambarmbar2 = -(rho**-3) * rhobar * harmonic_value / 4.0

    rcomp = (energy * (r * r + a * a) - a * angular_momentum + ur) / (2.0 * sigma)
    theta_comp = rho * (1.0j * math.sin(theta) * (a * energy - angular_momentum / math.sin(theta) ** 2) + u_theta) / math.sqrt(2.0)
    cnn = rcomp * rcomp
    cnm = rcomp * theta_comp
    cmm = theta_comp * theta_comp
    source0 = ann0 * cnn + anmbar0 * cnm + ambarmbar0 * cmm
    source1 = anmbar1 * cnm + ambarmbar1 * cmm
    source2 = ambarmbar2 * cmm
    return source0, source1, source2


def _spin_two_positive_coefficients(
    *,
    r: float,
    ur: float,
    theta: float,
    u_theta: float,
    a: float,
    energy: float,
    angular_momentum: float,
    m: int,
    omega: complex,
    harmonic_value: complex,
    harmonic_derivative: complex,
    harmonic_second_derivative: complex,
) -> tuple[complex, complex, complex]:
    delta = r * r - 2.0 * r + a * a
    kt = (r * r + a * a) * omega - m * a
    rho = -1.0 / complex(r, -a * math.cos(theta))
    rhobar = -1.0 / complex(r, a * math.cos(theta))
    d_rho_over_rho = 1.0j * a * rho * math.sin(theta)
    d2_rho_over_rho = 1.0j * a * rho * (math.cos(theta) + 2.0 * math.sin(theta) * d_rho_over_rho)
    ld1 = m / math.sin(theta) - a * omega * math.sin(theta) + math.cos(theta) / math.sin(theta)
    ld2 = m / math.sin(theta) - a * omega * math.sin(theta) + 2.0 * math.cos(theta) / math.sin(theta)
    d_ld2 = -m * math.cos(theta) / math.sin(theta) ** 2 - a * omega * math.cos(theta) - 2.0 / math.sin(theta) ** 2
    all0 = (
        -0.5
        * rho**-1
        * rhobar
        * (
            harmonic_second_derivative
            + (ld1 + ld2 + 2.0 * d_rho_over_rho) * harmonic_derivative
            + (
                d_ld2
                + ld1 * ld2
                - 6.0 * d_rho_over_rho**2
                + 3.0 * d2_rho_over_rho
                + (3.0 * ld1 - ld2) * d_rho_over_rho
            )
            * harmonic_value
        )
    )
    alm0 = (
        math.sqrt(2.0)
        * rho**-1
        * (
            -(rho + rhobar + 1.0j * kt / delta) * (harmonic_derivative + ld2 * harmonic_value)
            + (rho - rhobar) * a * math.sin(theta) * kt / delta * harmonic_value
        )
    )
    alm1 = math.sqrt(2.0) * rho**-1 * (
        harmonic_derivative + ld2 * harmonic_value + 1.0j * (rho - rhobar) * a * math.sin(theta) * harmonic_value
    )
    amm0 = (
        kt * kt * harmonic_value / (delta * delta * rho * rhobar)
        + 2.0j * kt * harmonic_value * (-1.0 + r - delta * rho) / (delta * delta * rho * rhobar)
        - 2.0j * r * harmonic_value * omega / (delta * rho * rhobar)
    )
    amm1 = 2.0 * rho**-1 * rhobar**-1 * harmonic_value * (1.0j * kt / delta + rho)
    amm2 = -rho**-1 * rhobar**-1 * harmonic_value
    rcomp = (energy * (r * r + a * a) - a * angular_momentum - ur) / delta
    theta_comp = -rhobar * (1.0j * math.sin(theta) * (a * energy - angular_momentum / math.sin(theta) ** 2) - u_theta) / math.sqrt(2.0)
    cll = rcomp * rcomp
    clm = rcomp * theta_comp
    cmm = theta_comp * theta_comp
    source0 = all0 * cll + alm0 * clm + amm0 * cmm
    source1 = alm1 * clm + amm1 * cmm
    source2 = amm2 * cmm
    return source0, source1, source2


def _spin_minus_one_coefficients(
    *,
    r: float,
    theta: float,
    a: float,
    energy: float,
    angular_momentum: float,
    ur: float,
    u_theta: float,
    m: int,
    omega: complex,
    harmonic_value: complex,
    harmonic_derivative: complex,
) -> tuple[complex, complex]:
    delta = r * r - 2.0 * r + a * a
    kt = (r * r + a * a) * omega - m * a
    sin_theta = math.sin(theta)
    rho = -1.0 / complex(r, -a * math.cos(theta))
    rhobar = -1.0 / complex(r, a * math.cos(theta))
    sigma = 1.0 / (rho * rhobar)
    l1 = -m / sin_theta + a * omega * sin_theta + math.cos(theta) / sin_theta
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


def _spin_plus_one_coefficients(
    *,
    r: float,
    theta: float,
    a: float,
    energy: float,
    angular_momentum: float,
    ur: float,
    u_theta: float,
    m: int,
    omega: complex,
    harmonic_value: complex,
    harmonic_derivative: complex,
) -> tuple[complex, complex]:
    delta = r * r - 2.0 * r + a * a
    delta_prime = 2.0 * (r - 1.0)
    kt = (r * r + a * a) * omega - m * a
    sin_theta = math.sin(theta)
    rho = -1.0 / complex(r, -a * math.cos(theta))
    rhobar = -1.0 / complex(r, a * math.cos(theta))
    l1 = m / sin_theta - a * omega * sin_theta + math.cos(theta) / sin_theta
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


def _wronskian(rin: RadialSolution, rup: RadialSolution, s: int, r: float) -> complex:
    delta = r * r - 2.0 * r + rin.a * rin.a
    return delta ** (s + 1) * (rup.derivative(1, r) * rin(r) - rin.derivative(1, r) * rup(r))


def _mode_from_amplitudes(
    *,
    s: int,
    ell: int,
    m: int,
    n: int,
    k: int,
    orbit: Orbit,
    omega: complex,
    rin: RadialSolution,
    rup: RadialSolution,
    z_i: complex,
    z_h: complex,
    acceleration: dict[str, object] | None = None,
) -> ModeSolution:
    source_type = str(_MODE_OPTION_CONTEXT["source_type"])
    domain = _MODE_OPTION_CONTEXT["domain"]
    if s == -2:
        fluxes = _fluxes_s_minus_2(
            a=orbit.a,
            m=m,
            omega=omega,
            eigenvalue=rin.eigenvalue,
            z_i=z_i,
            z_h=z_h,
        )
    elif s == 0:
        fluxes = _fluxes_s_zero(
            a=orbit.a,
            m=m,
            omega=omega,
            z_i=z_i,
            z_h=z_h,
        )
    elif s == 2:
        fluxes = _fluxes_s_plus_2(
            a=orbit.a,
            m=m,
            omega=omega,
            eigenvalue=rin.eigenvalue,
            z_i=z_i,
            z_h=z_h,
        )
    elif s == -1:
        fluxes = _fluxes_s_minus_1(
            a=orbit.a,
            m=m,
            omega=omega,
            eigenvalue=rin.eigenvalue,
            z_i=z_i,
            z_h=z_h,
        )
    elif s == 1:
        fluxes = _fluxes_s_plus_1(
            a=orbit.a,
            m=m,
            omega=omega,
            eigenvalue=rin.eigenvalue,
            z_i=z_i,
            z_h=z_h,
        )
    else:
        raise NotImplementedError(f"flux formula not implemented for spin weight s = {s}")
    return ModeSolution(
        s=s,
        l=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        eigenvalue=rin.eigenvalue,
        radial_in=rin,
        radial_up=rup,
        amplitudes={"I": z_i, "H": z_h},
        fluxes=fluxes,
        source_type=source_type,
        domain=domain,
        acceleration=acceleration,
    )


def _dcu_supported_orbit_kind(kind: str) -> bool:
    return kind in {"eccentric-equatorial", "generic"}


def _dcu_mode_metadata(device_id: int, orbit_kind: str) -> dict[str, object]:
    status = require_dcu(device_id)
    return {
        "Backend": "DCU",
        "Device": status["device"],
        "DeviceName": status["device_name"],
        "DeviceID": device_id,
        "OrbitKind": orbit_kind,
        "Used": True,
    }


def _solve_eccentric_equatorial_mode_dcu(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if k != 0:
        raise ValueError("eccentric equatorial orbits only support k = 0")
    device_id = int(_MODE_OPTION_CONTEXT["device_id"])
    acceleration = _dcu_mode_metadata(device_id, orbit.kind)
    omega = m * orbit.omega_phi + n * orbit.omega_r
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    rin = radial["In"]
    rup = radial["Up"]
    alpha_in = accelerated_eccentric_alpha(
        lambda rr: np.array([rin(float(r)) for r in rr], dtype=np.complex128),
        lambda rr: np.array([rin.derivative(1, float(r)) for r in rr], dtype=np.complex128),
        s=s, m=m, a=orbit.a, omega=omega, lam=rin.eigenvalue, orbit=orbit, n=n, device_id=device_id, ell=ell,
    )
    alpha_up = accelerated_eccentric_alpha(
        lambda rr: np.array([rup(float(r)) for r in rr], dtype=np.complex128),
        lambda rr: np.array([rup.derivative(1, float(r)) for r in rr], dtype=np.complex128),
        s=s, m=m, a=orbit.a, omega=omega, lam=rup.eigenvalue, orbit=orbit, n=n, device_id=device_id, ell=ell,
    )
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    sign = 1.0 if s in (-1, 1) else -1.0
    prefactor = sign * 8.0 * math.pi / w / orbit.upsilon_t
    z_h = prefactor * alpha_up
    z_i = prefactor * alpha_in
    return _mode_from_amplitudes(
        s=s, ell=ell, m=m, n=n, k=k, orbit=orbit, omega=omega,
        rin=user_radial["In"], rup=user_radial["Up"], z_i=z_i, z_h=z_h,
        acceleration=acceleration,
    )


def _solve_generic_mode_dcu(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    device_id = int(_MODE_OPTION_CONTEXT["device_id"])
    acceleration = _dcu_mode_metadata(device_id, orbit.kind)
    omega = m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    rin = radial["In"]
    rup = radial["Up"]
    alpha_in = accelerated_generic_alpha(
        lambda rr: np.array([rin(float(r)) for r in rr], dtype=np.complex128),
        lambda rr: np.array([rin.derivative(1, float(r)) for r in rr], dtype=np.complex128),
        s=s, m=m, a=orbit.a, omega=omega, lam=rin.eigenvalue, orbit=orbit, n=n, k=k, device_id=device_id, ell=ell,
    )
    alpha_up = accelerated_generic_alpha(
        lambda rr: np.array([rup(float(r)) for r in rr], dtype=np.complex128),
        lambda rr: np.array([rup.derivative(1, float(r)) for r in rr], dtype=np.complex128),
        s=s, m=m, a=orbit.a, omega=omega, lam=rup.eigenvalue, orbit=orbit, n=n, k=k, device_id=device_id, ell=ell,
    )
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    sign = 1.0 if s in (-1, 1) else -1.0
    prefactor = sign * 8.0 * math.pi / w / orbit.upsilon_t
    z_h = prefactor * alpha_up
    z_i = prefactor * alpha_in
    return _mode_from_amplitudes(
        s=s, ell=ell, m=m, n=n, k=k, orbit=orbit, omega=omega,
        rin=user_radial["In"], rup=user_radial["Up"], z_i=z_i, z_h=z_h,
        acceleration=acceleration,
    )


def _refined_spherical_orbit(orbit: Orbit, k: int) -> Orbit:
    if orbit.kind != "spherical" or k == 0:
        return orbit
    return spherical_orbit(orbit.a, orbit.p, orbit.inclination, samples=131073)


def _spheroidal_value(harmonic, theta: float) -> complex:
    return harmonic(theta, 0.0)


def _scalar_phase_average(phases: np.ndarray, amplitudes: np.ndarray) -> complex:
    return np.trapz(amplitudes * np.cos(phases), dx=(phases[-1] - phases[0]) / (len(phases) - 1))


def _solve_circular_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if n != 0 or k != 0:
        raise ValueError("circular equatorial orbits only support n = k = 0")
    omega = m * orbit.omega_phi
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    s0 = harmonic(math.pi / 2.0, 0.0)
    ds0 = harmonic.derivative_theta(math.pi / 2.0, 0.0)
    d2s0 = harmonic.derivative_theta2(math.pi / 2.0, 0.0)
    rin0 = rin(r0)
    rup0 = rup(r0)
    drin = rin.derivative(1, r0)
    drup = rup.derivative(1, r0)
    d2rin = _radial_second_derivative(rin0, drin, s=s, m=m, a=orbit.a, omega=omega, eigenvalue=rin.eigenvalue, r=r0)
    d2rup = _radial_second_derivative(rup0, drup, s=s, m=m, a=orbit.a, omega=omega, eigenvalue=rin.eigenvalue, r=r0)
    source0, source1, source2 = _spin_two_coefficients(
        r=r0,
        ur=0.0,
        theta=math.pi / 2.0,
        u_theta=0.0,
        a=orbit.a,
        energy=orbit.energy,
        angular_momentum=orbit.angular_momentum,
        s=s,
        m=m,
        omega=omega,
        harmonic_value=s0,
        harmonic_derivative=ds0,
        harmonic_second_derivative=d2s0,
    )
    alpha_in = source0 * rin0 - source1 * drin + source2 * d2rin
    alpha_up = source0 * rup0 - source1 * drup + source2 * d2rup
    w = _wronskian(rin, rup, s, r0)
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_scalar_circular_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if n != 0 or k != 0:
        raise ValueError("circular equatorial orbits only support n = k = 0")
    omega = m * orbit.omega_phi
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    source = -4.0 * math.pi * r0 * r0 * _spheroidal_value(harmonic, math.pi / 2.0)
    w = _wronskian(rin, rup, s, r0)
    z_h = source * rup(r0) / w / orbit.upsilon_t
    z_i = source * rin(r0) / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _delta_squared_wrap(
    radial_value: complex,
    radial_derivative: complex,
    radial_second_derivative: complex,
    delta: float,
    d_delta: float,
    d2_delta: float,
) -> tuple[complex, complex, complex]:
    """Compute Δ²R, d(Δ²R)/dr, d²(Δ²R)/dr² for the spin-weight s=+2 Teukolsky equation.

    The s=+2 radial function R_{+2} is related to the s=-2 function by
    R_{+2} = Δ² R_{-2}.  The source convolution for s=+2 is expressed in
    terms of Δ²R rather than R itself.
    """
    d2r = delta * delta * radial_value
    d_d2r = delta * delta * radial_derivative + 2.0 * delta * d_delta * radial_value
    d2_d2r = (
        delta * delta * radial_second_derivative
        + 4.0 * delta * d_delta * radial_derivative
        + (2.0 * d_delta * d_delta + 2.0 * delta * d2_delta) * radial_value
    )
    return d2r, d_d2r, d2_d2r


def _solve_spin_plus_two_circular_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if n != 0 or k != 0:
        raise ValueError("circular equatorial orbits only support n = k = 0")
    omega = m * orbit.omega_phi
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    delta = r0 * r0 - 2.0 * r0 + orbit.a * orbit.a
    d_delta = 2.0 * (r0 - 1.0)
    d2_delta = 2.0
    rin0 = rin(r0)
    rup0 = rup(r0)
    drin = rin.derivative(1, r0)
    drup = rup.derivative(1, r0)
    d2rin = _radial_second_derivative(rin0, drin, s=s, m=m, a=orbit.a, omega=omega, eigenvalue=rin.eigenvalue, r=r0)
    d2rup = _radial_second_derivative(rup0, drup, s=s, m=m, a=orbit.a, omega=omega, eigenvalue=rin.eigenvalue, r=r0)
    delta2_rin = delta * delta * rin0
    d_delta2_rin = delta * delta * drin + 2.0 * delta * d_delta * rin0
    d2_delta2_rin = delta * delta * d2rin + 4.0 * delta * d_delta * drin + (2.0 * d_delta * d_delta + 2.0 * delta * d2_delta) * rin0
    delta2_rup = delta * delta * rup0
    d_delta2_rup = delta * delta * drup + 2.0 * delta * d_delta * rup0
    d2_delta2_rup = delta * delta * d2rup + 4.0 * delta * d_delta * drup + (2.0 * d_delta * d_delta + 2.0 * delta * d2_delta) * rup0
    s0 = harmonic(math.pi / 2.0, 0.0)
    ds0 = harmonic.derivative_theta(math.pi / 2.0, 0.0)
    d2s0 = harmonic.derivative_theta2(math.pi / 2.0, 0.0)
    source0, source1, source2 = _spin_two_positive_coefficients(
        r=r0,
        ur=0.0,
        theta=math.pi / 2.0,
        u_theta=0.0,
        a=orbit.a,
        energy=orbit.energy,
        angular_momentum=orbit.angular_momentum,
        m=m,
        omega=omega,
        harmonic_value=s0,
        harmonic_derivative=ds0,
        harmonic_second_derivative=d2s0,
    )
    alpha_in = source0 * delta2_rin - source1 * d_delta2_rin + source2 * d2_delta2_rin
    alpha_up = source0 * delta2_rup - source1 * d_delta2_rup + source2 * d2_delta2_rup
    w = _wronskian(rin, rup, s, r0)
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_plus_two_spherical_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    orbit = _refined_spherical_orbit(orbit, k)
    if n != 0:
        raise ValueError("spherical orbits only support n = 0")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("spherical orbit is missing polar phase data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("spherical orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p

    delta = r0 * r0 - 2.0 * r0 + orbit.a * orbit.a
    d_delta = 2.0 * (r0 - 1.0)
    d2_delta = 2.0

    rin0 = rin(r0)
    rup0 = rup(r0)
    drin = rin.derivative(1, r0)
    drup = rup.derivative(1, r0)
    d2rin = _radial_second_derivative(rin0, drin, s=s, m=m, a=orbit.a, omega=omega, eigenvalue=rin.eigenvalue, r=r0)
    d2rup = _radial_second_derivative(rup0, drup, s=s, m=m, a=orbit.a, omega=omega, eigenvalue=rin.eigenvalue, r=r0)

    delta2_rin, d_delta2_rin, d2_delta2_rin = _delta_squared_wrap(rin0, drin, d2rin, delta, d_delta, d2_delta)
    delta2_rup, d_delta2_rup, d2_delta2_rup = _delta_squared_wrap(rup0, drup, d2rup, delta, d_delta, d2_delta)

    q = np.linspace(0.0, 2.0 * math.pi, 2049)
    theta_values = np.array([orbit.theta_phase_function(phase) for phase in q], dtype=float)
    u_theta_values = np.array([orbit.theta_velocity_function(phase) for phase in q], dtype=float)
    delta_t_values = np.array([orbit.theta_delta_t_function(phase) for phase in q], dtype=float)
    delta_phi_values = np.array([orbit.theta_delta_phi_function(phase) for phase in q], dtype=float)
    harmonic_values = np.array([harmonic(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_derivatives = np.array([harmonic.derivative_theta(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_second_derivatives = np.array(
        [harmonic.derivative_theta2(theta, 0.0) for theta in theta_values],
        dtype=np.complex128,
    )

    def alpha(delta2_R: complex, d_delta2_R: complex, d2_delta2_R: complex) -> complex:
        values = []
        for index, phase in enumerate(q):
            source0, source1, source2 = _spin_two_positive_coefficients(
                r=r0,
                ur=0.0,
                theta=theta_values[index],
                u_theta=u_theta_values[index],
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                m=m,
                omega=omega,
                harmonic_value=harmonic_values[index],
                harmonic_derivative=harmonic_derivatives[index],
                harmonic_second_derivative=harmonic_second_derivatives[index],
            )
            phase_factor = np.exp(
                1.0j
                * (
                    omega * delta_t_values[index]
                    - m * delta_phi_values[index]
                    + k * phase
                )
            )
            values.append((source0 * delta2_R - source1 * d_delta2_R + source2 * d2_delta2_R) * phase_factor)
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(delta2_rin, d_delta2_rin, d2_delta2_rin)
    alpha_up = alpha(delta2_rup, d_delta2_rup, d2_delta2_rup)
    w = _wronskian(rin, rup, s, r0)
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_plus_two_eccentric_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if k != 0:
        raise ValueError("eccentric equatorial orbits only support k = 0")
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("eccentric equatorial orbit is missing radial phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("eccentric equatorial orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    theta = math.pi / 2.0
    s0 = harmonic(theta, 0.0)
    ds0 = harmonic.derivative_theta(theta, 0.0)
    d2s0 = harmonic.derivative_theta2(theta, 0.0)
    q = np.linspace(0.0, 2.0 * math.pi, 4097)

    def alpha(radial_function: RadialSolution) -> complex:
        values = []
        for phase in q:
            r0 = orbit.radial_phase_function(phase)
            ur0 = orbit.radial_velocity_function(phase)
            radial_value = radial_function(r0)
            radial_derivative = radial_function.derivative(1, r0)
            radial_second_derivative = _radial_second_derivative(
                radial_value,
                radial_derivative,
                s=s,
                m=m,
                a=orbit.a,
                omega=omega,
                eigenvalue=rin.eigenvalue,
                r=r0,
            )
            delta_val = r0 * r0 - 2.0 * r0 + orbit.a * orbit.a
            d_delta_val = 2.0 * (r0 - 1.0)
            d2_delta_val = 2.0
            delta2_R, d_delta2_R, d2_delta2_R = _delta_squared_wrap(
                radial_value, radial_derivative, radial_second_derivative,
                delta_val, d_delta_val, d2_delta_val,
            )
            source0, source1, source2 = _spin_two_positive_coefficients(
                r=r0,
                ur=ur0,
                theta=theta,
                u_theta=0.0,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                m=m,
                omega=omega,
                harmonic_value=s0,
                harmonic_derivative=ds0,
                harmonic_second_derivative=d2s0,
            )
            phase_factor = np.exp(
                1.0j
                * (
                    omega * orbit.radial_delta_t_function(phase)
                    - m * orbit.radial_delta_phi_function(phase)
                    + n * phase
                )
            )
            values.append((source0 * delta2_R - source1 * d_delta2_R + source2 * d2_delta2_R) * phase_factor)
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_plus_two_generic_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("generic orbit is missing radial phase data")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("generic orbit is missing polar phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("generic orbit is missing radial trajectory delta data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("generic orbit is missing polar trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]

    q_r = np.linspace(0.0, 2.0 * math.pi, 513)
    q_theta = np.linspace(0.0, 2.0 * math.pi, 513)
    r_values = np.array([orbit.radial_phase_function(phase) for phase in q_r], dtype=float)
    ur_values = np.array([orbit.radial_velocity_function(phase) for phase in q_r], dtype=float)
    delta_t_r_values = np.array([orbit.radial_delta_t_function(phase) for phase in q_r], dtype=float)
    delta_phi_r_values = np.array([orbit.radial_delta_phi_function(phase) for phase in q_r], dtype=float)
    theta_values = np.array([orbit.theta_phase_function(phase) for phase in q_theta], dtype=float)
    u_theta_values = np.array([orbit.theta_velocity_function(phase) for phase in q_theta], dtype=float)
    delta_t_theta_values = np.array([orbit.theta_delta_t_function(phase) for phase in q_theta], dtype=float)
    delta_phi_theta_values = np.array([orbit.theta_delta_phi_function(phase) for phase in q_theta], dtype=float)
    harmonic_values = np.array([harmonic(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_derivatives = np.array([harmonic.derivative_theta(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_second_derivatives = np.array(
        [harmonic.derivative_theta2(theta, 0.0) for theta in theta_values],
        dtype=np.complex128,
    )

    def alpha(radial_function: RadialSolution) -> complex:
        radial_values = np.array([radial_function(r0) for r0 in r_values], dtype=np.complex128)
        radial_derivatives = np.array([radial_function.derivative(1, r0) for r0 in r_values], dtype=np.complex128)
        radial_second_derivatives = np.array(
            [
                _radial_second_derivative(
                    radial_values[index],
                    radial_derivatives[index],
                    s=s,
                    m=m,
                    a=orbit.a,
                    omega=omega,
                    eigenvalue=rin.eigenvalue,
                    r=r_values[index],
                )
                for index in range(len(q_r))
            ],
            dtype=np.complex128,
        )
        integrand = np.empty((len(q_r), len(q_theta)), dtype=np.complex128)
        for i, phase_r in enumerate(q_r):
            r0 = r_values[i]
            ur0 = ur_values[i]
            radial_value = radial_values[i]
            radial_derivative = radial_derivatives[i]
            radial_second_derivative = radial_second_derivatives[i]
            delta_val = r0 * r0 - 2.0 * r0 + orbit.a * orbit.a
            d_delta_val = 2.0 * (r0 - 1.0)
            d2_delta_val = 2.0
            delta2_R, d_delta2_R, d2_delta2_R = _delta_squared_wrap(
                radial_value, radial_derivative, radial_second_derivative,
                delta_val, d_delta_val, d2_delta_val,
            )
            radial_phase = omega * delta_t_r_values[i] - m * delta_phi_r_values[i] + n * phase_r
            for j, phase_theta in enumerate(q_theta):
                source0, source1, source2 = _spin_two_positive_coefficients(
                    r=r0,
                    ur=ur0,
                    theta=theta_values[j],
                    u_theta=u_theta_values[j],
                    a=orbit.a,
                    energy=orbit.energy,
                    angular_momentum=orbit.angular_momentum,
                    m=m,
                    omega=omega,
                    harmonic_value=harmonic_values[j],
                    harmonic_derivative=harmonic_derivatives[j],
                    harmonic_second_derivative=harmonic_second_derivatives[j],
                )
                theta_phase = omega * delta_t_theta_values[j] - m * delta_phi_theta_values[j] + k * phase_theta
                integrand[i, j] = (
                    source0 * delta2_R - source1 * d_delta2_R + source2 * d2_delta2_R
                ) * np.exp(1.0j * (radial_phase + theta_phase))
        return np.trapz(np.trapz(integrand, q_theta, axis=1), q_r) / (2.0 * math.pi) ** 2

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_one_circular_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if n != 0 or k != 0:
        raise ValueError("circular equatorial orbits only support n = k = 0")
    omega = m * orbit.omega_phi
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    delta = r0 * r0 - 2.0 * r0 + orbit.a * orbit.a
    delta_p = 2.0 * r0 - 2.0
    spheroidal = harmonic(math.pi / 2.0, 0.0)
    dspheroidal = harmonic.derivative_theta(math.pi / 2.0, 0.0)
    p_in = rin(r0)
    p_up = rup(r0)
    dp_in = rin.derivative(1, r0)
    dp_up = rup.derivative(1, r0)
    if s == 1:
        p_in = delta * p_in
        p_up = delta * p_up
        dp_in = delta * dp_in + delta_p * rin(r0)
        dp_up = delta * dp_up + delta_p * rup(r0)
    W = _wronskian(rin, rup, s, r0)
    orbital_omega = 1.0 / (math.sqrt(r0**3) + orbit.a)
    script_s = (4.0 * math.pi) / (math.sqrt(2.0) * r0) * (0.5 if s == -1 else 1.0)
    script_b = delta * ((r0 * r0 + orbit.a * orbit.a) * orbital_omega - orbit.a)
    ar = r0 * (r0 * ((r0 * r0 + orbit.a * orbit.a) * orbital_omega**2 - 1.0) + 2.0 * (1.0 - orbit.a * orbital_omega) ** 2)
    ati = r0 * delta * orbital_omega
    c = -delta * (1.0 - orbit.a * orbital_omega)
    A = script_s * ((m * ar + s * 1.0j * ati) * spheroidal + s * c * dspheroidal)
    B = s * 1.0j * script_s * script_b * spheroidal
    z_h = (p_up * A - dp_up * B) / (delta * W)
    z_i = (p_in * A - dp_in * B) / (delta * W)
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_minus_one_eccentric_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if k != 0:
        raise ValueError("eccentric equatorial orbits only support k = 0")
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("eccentric equatorial orbit is missing radial phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("eccentric equatorial orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    theta = math.pi / 2.0
    harmonic_value = harmonic(theta, 0.0)
    harmonic_derivative = harmonic.derivative_theta(theta, 0.0)
    q = np.linspace(0.0, math.pi, 1025)

    def alpha(radial_function: RadialSolution) -> complex:
        values = []
        for phase in q:
            r0 = orbit.radial_phase_function(phase)
            radial_value = radial_function(r0)
            radial_derivative = radial_function.derivative(1, r0)
            phase_factor = np.exp(
                1.0j
                * (
                    omega * orbit.radial_delta_t_function(phase)
                    - m * orbit.radial_delta_phi_function(phase)
                    + n * phase
                )
            )
            phase_mirror = np.conjugate(phase_factor)
            source0, source1 = _spin_minus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=orbit.radial_velocity_function(phase),
                u_theta=0.0,
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            source0_mirror, source1_mirror = _spin_minus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=orbit.radial_velocity_function(2.0 * math.pi - phase),
                u_theta=0.0,
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            values.append(
                (source0 * radial_value - source1 * radial_derivative) * phase_factor
                + (source0_mirror * radial_value - source1_mirror * radial_derivative) * phase_mirror
            )
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = 8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = 8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_plus_one_eccentric_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if k != 0:
        raise ValueError("eccentric equatorial orbits only support k = 0")
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("eccentric equatorial orbit is missing radial phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("eccentric equatorial orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    theta = math.pi / 2.0
    harmonic_value = harmonic(theta, 0.0)
    harmonic_derivative = harmonic.derivative_theta(theta, 0.0)
    q = np.linspace(0.0, math.pi, 1025)

    def alpha(radial_function: RadialSolution) -> complex:
        values = []
        for phase in q:
            r0 = orbit.radial_phase_function(phase)
            radial_value = radial_function(r0)
            radial_derivative = radial_function.derivative(1, r0)
            phase_factor = np.exp(
                1.0j
                * (
                    omega * orbit.radial_delta_t_function(phase)
                    - m * orbit.radial_delta_phi_function(phase)
                    + n * phase
                )
            )
            phase_mirror = np.conjugate(phase_factor)
            source0, source1 = _spin_plus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=orbit.radial_velocity_function(phase),
                u_theta=0.0,
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            source0_mirror, source1_mirror = _spin_plus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=orbit.radial_velocity_function(2.0 * math.pi - phase),
                u_theta=0.0,
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            values.append(
                (source0 * radial_value - source1 * radial_derivative) * phase_factor
                + (source0_mirror * radial_value - source1_mirror * radial_derivative) * phase_mirror
            )
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = 8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = 8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_eccentric_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if k != 0:
        raise ValueError("eccentric equatorial orbits only support k = 0")
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("eccentric equatorial orbit is missing radial phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("eccentric equatorial orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    theta = math.pi / 2.0
    s0 = harmonic(theta, 0.0)
    ds0 = harmonic.derivative_theta(theta, 0.0)
    d2s0 = harmonic.derivative_theta2(theta, 0.0)
    q = np.linspace(0.0, 2.0 * math.pi, 4097)

    def alpha(radial_function: RadialSolution) -> complex:
        values = []
        for phase in q:
            r0 = orbit.radial_phase_function(phase)
            ur0 = orbit.radial_velocity_function(phase)
            radial_value = radial_function(r0)
            radial_derivative = radial_function.derivative(1, r0)
            radial_second_derivative = _radial_second_derivative(
                radial_value,
                radial_derivative,
                s=s,
                m=m,
                a=orbit.a,
                omega=omega,
                eigenvalue=rin.eigenvalue,
                r=r0,
            )
            source0, source1, source2 = _spin_two_coefficients(
                r=r0,
                ur=ur0,
                theta=theta,
                u_theta=0.0,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                s=s,
                m=m,
                omega=omega,
                harmonic_value=s0,
                harmonic_derivative=ds0,
                harmonic_second_derivative=d2s0,
            )
            phase_factor = np.exp(
                1.0j
                * (
                    omega * orbit.radial_delta_t_function(phase)
                    - m * orbit.radial_delta_phi_function(phase)
                    + n * phase
                )
            )
            values.append((source0 * radial_value - source1 * radial_derivative + source2 * radial_second_derivative) * phase_factor)
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_scalar_eccentric_equatorial_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if k != 0:
        raise ValueError("eccentric equatorial orbits only support k = 0")
    if orbit.radial_phase_function is None or orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("eccentric equatorial orbit is missing radial data")
    omega = m * orbit.omega_phi + n * orbit.omega_r
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    s0 = _spheroidal_value(harmonic, math.pi / 2.0)
    q = np.linspace(0.0, math.pi, 1025)
    r_values = np.array([orbit.radial_phase_function(phase) for phase in q], dtype=float)
    phases = np.array(
        [omega * orbit.radial_delta_t_function(phase) - m * orbit.radial_delta_phi_function(phase) + n * phase for phase in q],
        dtype=float,
    )

    def alpha(radial_function: RadialSolution) -> complex:
        amplitudes = np.array([-8.0 * math.pi * (r * r) * radial_function(r) for r in r_values], dtype=np.complex128)
        return s0 * np.trapz(amplitudes * np.cos(phases), q) / (2.0 * math.pi)

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = alpha_up / w / orbit.upsilon_t
    z_i = alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_minus_one_spherical_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    orbit = _refined_spherical_orbit(orbit, k)
    if n != 0:
        raise ValueError("spherical orbits only support n = 0")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("spherical orbit is missing polar phase data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("spherical orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    radial_in = rin(r0)
    radial_up = rup(r0)
    derivative_in = rin.derivative(1, r0)
    derivative_up = rup.derivative(1, r0)
    q = np.linspace(0.0, math.pi, 2049)

    def alpha(radial_value: complex, radial_derivative: complex) -> complex:
        values = []
        for phase in q:
            theta = orbit.theta_phase_function(phase)
            phase_factor = np.exp(
                1.0j
                * (
                    omega * orbit.theta_delta_t_function(phase)
                    - m * orbit.theta_delta_phi_function(phase)
                    + k * phase
                )
            )
            phase_mirror = np.conjugate(phase_factor)
            harmonic_value = harmonic(theta, 0.0)
            harmonic_derivative = harmonic.derivative_theta(theta, 0.0)
            source0, source1 = _spin_minus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=0.0,
                u_theta=orbit.theta_velocity_function(phase),
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            source0_mirror, source1_mirror = _spin_minus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=0.0,
                u_theta=orbit.theta_velocity_function(2.0 * math.pi - phase),
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            values.append(
                (source0 * radial_value - source1 * radial_derivative) * phase_factor
                + (source0_mirror * radial_value - source1_mirror * radial_derivative) * phase_mirror
            )
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(radial_in, derivative_in)
    alpha_up = alpha(radial_up, derivative_up)
    w = _wronskian(rin, rup, s, r0)
    z_h = 8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = 8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_plus_one_spherical_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    orbit = _refined_spherical_orbit(orbit, k)
    if n != 0:
        raise ValueError("spherical orbits only support n = 0")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("spherical orbit is missing polar phase data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("spherical orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    radial_in = rin(r0)
    radial_up = rup(r0)
    derivative_in = rin.derivative(1, r0)
    derivative_up = rup.derivative(1, r0)
    q = np.linspace(0.0, math.pi, 1025)

    def alpha(radial_value: complex, radial_derivative: complex) -> complex:
        values = []
        for phase in q:
            theta = orbit.theta_phase_function(phase)
            phase_factor = np.exp(
                1.0j
                * (
                    omega * orbit.theta_delta_t_function(phase)
                    - m * orbit.theta_delta_phi_function(phase)
                    + k * phase
                )
            )
            phase_mirror = np.conjugate(phase_factor)
            harmonic_value = harmonic(theta, 0.0)
            harmonic_derivative = harmonic.derivative_theta(theta, 0.0)
            source0, source1 = _spin_plus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=0.0,
                u_theta=orbit.theta_velocity_function(phase),
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            source0_mirror, source1_mirror = _spin_plus_one_coefficients(
                r=r0,
                theta=theta,
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                ur=0.0,
                u_theta=orbit.theta_velocity_function(2.0 * math.pi - phase),
                m=m,
                omega=omega,
                harmonic_value=harmonic_value,
                harmonic_derivative=harmonic_derivative,
            )
            values.append(
                (source0 * radial_value - source1 * radial_derivative) * phase_factor
                + (source0_mirror * radial_value - source1_mirror * radial_derivative) * phase_mirror
            )
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(radial_in, derivative_in)
    alpha_up = alpha(radial_up, derivative_up)
    w = _wronskian(rin, rup, s, r0)
    z_h = 8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = 8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spherical_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    orbit = _refined_spherical_orbit(orbit, k)
    if n != 0:
        raise ValueError("spherical orbits only support n = 0")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("spherical orbit is missing polar phase data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("spherical orbit is missing trajectory delta data")

    omega = m * orbit.omega_phi + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    q = np.linspace(0.0, math.pi, 2049)
    theta_values = np.array([orbit.theta_phase_function(phase) for phase in q], dtype=float)
    u_theta_values = np.array([orbit.theta_velocity_function(phase) for phase in q], dtype=float)
    u_theta_mirror_values = np.array([orbit.theta_velocity_function(2.0 * math.pi - phase) for phase in q], dtype=float)
    delta_t_values = np.array([orbit.theta_delta_t_function(phase) for phase in q], dtype=float)
    delta_phi_values = np.array([orbit.theta_delta_phi_function(phase) for phase in q], dtype=float)
    harmonic_values = np.array([harmonic(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_derivatives = np.array([harmonic.derivative_theta(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_second_derivatives = np.array(
        [harmonic.derivative_theta2(theta, 0.0) for theta in theta_values],
        dtype=np.complex128,
    )

    def alpha(radial_function: RadialSolution) -> complex:
        radial_value = radial_function(r0)
        radial_derivative = radial_function.derivative(1, r0)
        radial_second_derivative = _radial_second_derivative(
            radial_value,
            radial_derivative,
            s=s,
            m=m,
            a=orbit.a,
            omega=omega,
            eigenvalue=rin.eigenvalue,
            r=r0,
        )
        values = []
        for index, phase in enumerate(q):
            theta = theta_values[index]
            source0_p, source1_p, source2_p = _spin_two_coefficients(
                r=r0,
                ur=0.0,
                theta=theta,
                u_theta=u_theta_values[index],
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                s=s,
                m=m,
                omega=omega,
                harmonic_value=harmonic_values[index],
                harmonic_derivative=harmonic_derivatives[index],
                harmonic_second_derivative=harmonic_second_derivatives[index],
            )
            source0_m, source1_m, source2_m = _spin_two_coefficients(
                r=r0,
                ur=0.0,
                theta=theta,
                u_theta=u_theta_mirror_values[index],
                a=orbit.a,
                energy=orbit.energy,
                angular_momentum=orbit.angular_momentum,
                s=s,
                m=m,
                omega=omega,
                harmonic_value=harmonic_values[index],
                harmonic_derivative=harmonic_derivatives[index],
                harmonic_second_derivative=harmonic_second_derivatives[index],
            )
            theta_phase = omega * delta_t_values[index] - m * delta_phi_values[index] + k * phase
            phase_factor_p = np.exp(1.0j * theta_phase)
            phase_factor_m = np.exp(-1.0j * theta_phase)
            values.append(
                (source0_p * radial_value - source1_p * radial_derivative + source2_p * radial_second_derivative)
                * phase_factor_p
                + (source0_m * radial_value - source1_m * radial_derivative + source2_m * radial_second_derivative)
                * phase_factor_m
            )
        return np.trapz(np.asarray(values, dtype=np.complex128), q) / (2.0 * math.pi)

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, r0)
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_scalar_spherical_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    orbit = _refined_spherical_orbit(orbit, k)
    if n != 0:
        raise ValueError("spherical orbits only support n = 0")
    if orbit.theta_phase_function is None or orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("spherical orbit is missing polar data")
    omega = m * orbit.omega_phi + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    r0 = orbit.p
    if orbit.a != 0.0 and (ell + m + k) % 2 == 1:
        return _mode_from_amplitudes(
            s=s,
            ell=ell,
            m=m,
            n=n,
            k=k,
            orbit=orbit,
            omega=omega,
            rin=rin,
            rup=rup,
            z_i=0.0 + 0.0j,
            z_h=0.0 + 0.0j,
        )
    if orbit.a == 0.0 and (((ell + m + k) % 2 == 1) or (abs(m + k) > ell)):
        return _mode_from_amplitudes(
            s=s,
            ell=ell,
            m=m,
            n=n,
            k=k,
            orbit=orbit,
            omega=omega,
            rin=rin,
            rup=rup,
            z_i=0.0 + 0.0j,
            z_h=0.0 + 0.0j,
        )
    q = np.linspace(0.0, math.pi, 2049)
    theta_values = np.array([orbit.theta_phase_function(phase) for phase in q], dtype=float)
    phase_values = np.array(
        [omega * orbit.theta_delta_t_function(phase) - m * orbit.theta_delta_phi_function(phase) + k * phase for phase in q],
        dtype=float,
    )
    harmonic_values = np.array([_spheroidal_value(harmonic, theta) for theta in theta_values], dtype=np.complex128)
    alpha2 = np.trapz(-8.0 * math.pi * harmonic_values * np.cos(phase_values), q) / (2.0 * math.pi)
    alpha4 = 0.0 + 0.0j
    if orbit.a != 0.0:
        alpha4 = np.trapz(
            -8.0 * math.pi * orbit.a * orbit.a * np.cos(theta_values) ** 2 * harmonic_values * np.cos(phase_values),
            q,
        ) / (2.0 * math.pi)
    w = _wronskian(rin, rup, s, r0)
    z_h = (r0 * r0 * rup(r0) * alpha2 + rup(r0) * alpha4) / w / orbit.upsilon_t
    z_i = (r0 * r0 * rin(r0) * alpha2 + rin(r0) * alpha4) / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_minus_one_generic_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("generic orbit is missing radial phase data")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("generic orbit is missing polar phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("generic orbit is missing radial trajectory delta data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("generic orbit is missing polar trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    q_r = np.linspace(0.0, math.pi, 129)
    q_theta = np.linspace(0.0, math.pi, 129)

    def alpha(radial_function: RadialSolution) -> complex:
        integrand = np.empty((len(q_r), len(q_theta)), dtype=np.complex128)
        for i, phase_r in enumerate(q_r):
            r0 = orbit.radial_phase_function(phase_r)
            radial_value = radial_function(r0)
            radial_derivative = radial_function.derivative(1, r0)
            radial_phase = omega * orbit.radial_delta_t_function(phase_r) - m * orbit.radial_delta_phi_function(phase_r) + n * phase_r
            ur_plus = orbit.radial_velocity_function(phase_r)
            ur_minus = orbit.radial_velocity_function(2.0 * math.pi - phase_r)
            for j, phase_theta in enumerate(q_theta):
                theta = orbit.theta_phase_function(phase_theta)
                harmonic_value = harmonic(theta, 0.0)
                harmonic_derivative = harmonic.derivative_theta(theta, 0.0)
                theta_phase = (
                    omega * orbit.theta_delta_t_function(phase_theta)
                    - m * orbit.theta_delta_phi_function(phase_theta)
                    + k * phase_theta
                )
                u_theta_plus = orbit.theta_velocity_function(phase_theta)
                u_theta_minus = orbit.theta_velocity_function(2.0 * math.pi - phase_theta)
                terms = []
                for ur0, u_theta0, phase_sign_r, phase_sign_theta in (
                    (ur_plus, u_theta_plus, 1.0, 1.0),
                    (ur_plus, u_theta_minus, 1.0, -1.0),
                    (ur_minus, u_theta_plus, -1.0, 1.0),
                    (ur_minus, u_theta_minus, -1.0, -1.0),
                ):
                    source0, source1 = _spin_minus_one_coefficients(
                        r=r0,
                        theta=theta,
                        a=orbit.a,
                        energy=orbit.energy,
                        angular_momentum=orbit.angular_momentum,
                        ur=ur0,
                        u_theta=u_theta0,
                        m=m,
                        omega=omega,
                        harmonic_value=harmonic_value,
                        harmonic_derivative=harmonic_derivative,
                    )
                    phase_total = phase_sign_r * radial_phase + phase_sign_theta * theta_phase
                    terms.append((source0 * radial_value - source1 * radial_derivative) * np.exp(1.0j * phase_total))
                integrand[i, j] = sum(terms)
        return np.trapz(np.trapz(integrand, q_theta, axis=1), q_r) / (2.0 * math.pi) ** 2

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = 8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = 8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_spin_plus_one_generic_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("generic orbit is missing radial phase data")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("generic orbit is missing polar phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("generic orbit is missing radial trajectory delta data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("generic orbit is missing polar trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    q_r = np.linspace(0.0, math.pi, 129)
    q_theta = np.linspace(0.0, math.pi, 129)

    def alpha(radial_function: RadialSolution) -> complex:
        integrand = np.empty((len(q_r), len(q_theta)), dtype=np.complex128)
        for i, phase_r in enumerate(q_r):
            r0 = orbit.radial_phase_function(phase_r)
            radial_value = radial_function(r0)
            radial_derivative = radial_function.derivative(1, r0)
            radial_phase = omega * orbit.radial_delta_t_function(phase_r) - m * orbit.radial_delta_phi_function(phase_r) + n * phase_r
            ur_plus = orbit.radial_velocity_function(phase_r)
            ur_minus = orbit.radial_velocity_function(2.0 * math.pi - phase_r)
            for j, phase_theta in enumerate(q_theta):
                theta = orbit.theta_phase_function(phase_theta)
                harmonic_value = harmonic(theta, 0.0)
                harmonic_derivative = harmonic.derivative_theta(theta, 0.0)
                theta_phase = (
                    omega * orbit.theta_delta_t_function(phase_theta)
                    - m * orbit.theta_delta_phi_function(phase_theta)
                    + k * phase_theta
                )
                u_theta_plus = orbit.theta_velocity_function(phase_theta)
                u_theta_minus = orbit.theta_velocity_function(2.0 * math.pi - phase_theta)
                terms = []
                for ur0, u_theta0, phase_sign_r, phase_sign_theta in (
                    (ur_plus, u_theta_plus, 1.0, 1.0),
                    (ur_plus, u_theta_minus, 1.0, -1.0),
                    (ur_minus, u_theta_plus, -1.0, 1.0),
                    (ur_minus, u_theta_minus, -1.0, -1.0),
                ):
                    source0, source1 = _spin_plus_one_coefficients(
                        r=r0,
                        theta=theta,
                        a=orbit.a,
                        energy=orbit.energy,
                        angular_momentum=orbit.angular_momentum,
                        ur=ur0,
                        u_theta=u_theta0,
                        m=m,
                        omega=omega,
                        harmonic_value=harmonic_value,
                        harmonic_derivative=harmonic_derivative,
                    )
                    phase_total = phase_sign_r * radial_phase + phase_sign_theta * theta_phase
                    terms.append((source0 * radial_value - source1 * radial_derivative) * np.exp(1.0j * phase_total))
                integrand[i, j] = sum(terms)
        return np.trapz(np.trapz(integrand, q_theta, axis=1), q_r) / (2.0 * math.pi) ** 2

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = 8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = 8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_generic_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if orbit.radial_phase_function is None or orbit.radial_velocity_function is None:
        raise ValueError("generic orbit is missing radial phase data")
    if orbit.theta_phase_function is None or orbit.theta_velocity_function is None:
        raise ValueError("generic orbit is missing polar phase data")
    if orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("generic orbit is missing radial trajectory delta data")
    if orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("generic orbit is missing polar trajectory delta data")

    omega = m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]

    q_r = np.linspace(0.0, 2.0 * math.pi, 513)
    q_theta = np.linspace(0.0, 2.0 * math.pi, 513)
    r_values = np.array([orbit.radial_phase_function(phase) for phase in q_r], dtype=float)
    ur_values = np.array([orbit.radial_velocity_function(phase) for phase in q_r], dtype=float)
    delta_t_r_values = np.array([orbit.radial_delta_t_function(phase) for phase in q_r], dtype=float)
    delta_phi_r_values = np.array([orbit.radial_delta_phi_function(phase) for phase in q_r], dtype=float)
    theta_values = np.array([orbit.theta_phase_function(phase) for phase in q_theta], dtype=float)
    u_theta_values = np.array([orbit.theta_velocity_function(phase) for phase in q_theta], dtype=float)
    delta_t_theta_values = np.array([orbit.theta_delta_t_function(phase) for phase in q_theta], dtype=float)
    delta_phi_theta_values = np.array([orbit.theta_delta_phi_function(phase) for phase in q_theta], dtype=float)
    harmonic_values = np.array([harmonic(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_derivatives = np.array([harmonic.derivative_theta(theta, 0.0) for theta in theta_values], dtype=np.complex128)
    harmonic_second_derivatives = np.array(
        [harmonic.derivative_theta2(theta, 0.0) for theta in theta_values],
        dtype=np.complex128,
    )

    def alpha(radial_function: RadialSolution) -> complex:
        radial_values = np.array([radial_function(r0) for r0 in r_values], dtype=np.complex128)
        radial_derivatives = np.array([radial_function.derivative(1, r0) for r0 in r_values], dtype=np.complex128)
        radial_second_derivatives = np.array(
            [
                _radial_second_derivative(
                    radial_values[index],
                    radial_derivatives[index],
                    s=s,
                    m=m,
                    a=orbit.a,
                    omega=omega,
                    eigenvalue=rin.eigenvalue,
                    r=r_values[index],
                )
                for index in range(len(q_r))
            ],
            dtype=np.complex128,
        )
        integrand = np.empty((len(q_r), len(q_theta)), dtype=np.complex128)
        for i, phase_r in enumerate(q_r):
            r0 = r_values[i]
            ur0 = ur_values[i]
            radial_value = radial_values[i]
            radial_derivative = radial_derivatives[i]
            radial_second_derivative = radial_second_derivatives[i]
            radial_phase = omega * delta_t_r_values[i] - m * delta_phi_r_values[i] + n * phase_r
            for j, phase_theta in enumerate(q_theta):
                theta = theta_values[j]
                source0, source1, source2 = _spin_two_coefficients(
                    r=r0,
                    ur=ur0,
                    theta=theta,
                    u_theta=u_theta_values[j],
                    a=orbit.a,
                    energy=orbit.energy,
                    angular_momentum=orbit.angular_momentum,
                    s=s,
                    m=m,
                    omega=omega,
                    harmonic_value=harmonic_values[j],
                    harmonic_derivative=harmonic_derivatives[j],
                    harmonic_second_derivative=harmonic_second_derivatives[j],
                )
                theta_phase = omega * delta_t_theta_values[j] - m * delta_phi_theta_values[j] + k * phase_theta
                integrand[i, j] = (
                    source0 * radial_value - source1 * radial_derivative + source2 * radial_second_derivative
                ) * np.exp(1.0j * (radial_phase + theta_phase))
        return np.trapz(np.trapz(integrand, q_theta, axis=1), q_r) / (2.0 * math.pi) ** 2

    alpha_in = alpha(rin)
    alpha_up = alpha(rup)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = -8.0 * math.pi * alpha_up / w / orbit.upsilon_t
    z_i = -8.0 * math.pi * alpha_in / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def _solve_scalar_generic_mode(s: int, ell: int, m: int, orbit: Orbit, n: int, k: int) -> ModeSolution:
    if orbit.radial_phase_function is None or orbit.radial_delta_t_function is None or orbit.radial_delta_phi_function is None:
        raise ValueError("generic orbit is missing radial data")
    if orbit.theta_phase_function is None or orbit.theta_delta_t_function is None or orbit.theta_delta_phi_function is None:
        raise ValueError("generic orbit is missing polar data")
    omega = m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
    user_radial, radial = _mode_radial_solutions(s=s, ell=ell, m=m, orbit=orbit, omega=omega)
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    rin = radial["In"]
    rup = radial["Up"]
    q_r = np.linspace(0.0, math.pi, 1025)
    q_theta = np.linspace(0.0, math.pi, 1025)
    r_values = np.array([orbit.radial_phase_function(phase) for phase in q_r], dtype=float)
    radial_phases = np.array(
        [omega * orbit.radial_delta_t_function(phase) - m * orbit.radial_delta_phi_function(phase) + n * phase for phase in q_r],
        dtype=float,
    )
    theta_values = np.array([orbit.theta_phase_function(phase) for phase in q_theta], dtype=float)
    theta_phases = np.array(
        [omega * orbit.theta_delta_t_function(phase) - m * orbit.theta_delta_phi_function(phase) + k * phase for phase in q_theta],
        dtype=float,
    )
    harmonic_values = np.array([_spheroidal_value(harmonic, theta) for theta in theta_values], dtype=np.complex128)

    def radial_average(radial_function: RadialSolution, power: int) -> complex:
        amplitudes = np.array(
            [2.0 * (r ** power) * radial_function(r) for r in r_values],
            dtype=np.complex128,
        )
        return np.trapz(amplitudes * np.cos(radial_phases), q_r) / (2.0 * math.pi)

    alpha1_in = radial_average(rin, 2)
    alpha1_up = radial_average(rup, 2)
    alpha2 = np.trapz(-8.0 * math.pi * harmonic_values * np.cos(theta_phases), q_theta) / (2.0 * math.pi)
    alpha3_in = 0.0 + 0.0j
    alpha3_up = 0.0 + 0.0j
    alpha4 = 0.0 + 0.0j
    if orbit.a != 0.0:
        alpha3_in = radial_average(rin, 0)
        alpha3_up = radial_average(rup, 0)
        alpha4 = np.trapz(
            -8.0 * math.pi * orbit.a * orbit.a * np.cos(theta_values) ** 2 * harmonic_values * np.cos(theta_phases),
            q_theta,
        ) / (2.0 * math.pi)
    w = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    z_h = (alpha1_up * alpha2 + alpha3_up * alpha4) / w / orbit.upsilon_t
    z_i = (alpha1_in * alpha2 + alpha3_in * alpha4) / w / orbit.upsilon_t
    return _mode_from_amplitudes(
        s=s,
        ell=ell,
        m=m,
        n=n,
        k=k,
        orbit=orbit,
        omega=omega,
        rin=user_radial["In"],
        rup=user_radial["Up"],
        z_i=z_i,
        z_h=z_h,
    )


def solve_point_particle_mode(
    s: int,
    ell: int,
    m: int,
    orbit: Orbit,
    n: int = 0,
    k: int = 0,
    domain: tuple[float, float] | str | None = "Automatic",
    source_type: str | None = "Automatic",
    accelerator: str | None = "cpu",
    device_id: int = 0,
) -> ModeSolution:
    resolved_source_type = _resolve_source_type(s, source_type)
    resolved_domain = _resolve_mode_domain(domain)
    resolved_accelerator = _resolve_accelerator(accelerator)
    _validate_point_particle_request(s=s, m=m, orbit=orbit, n=n, k=k, domain=resolved_domain)
    if resolved_accelerator == "dcu" and not _dcu_supported_orbit_kind(orbit.kind):
        raise ValueError("DCU acceleration is currently implemented only for eccentric-equatorial and generic orbits")
    previous_source_type = _MODE_OPTION_CONTEXT["source_type"]
    previous_domain = _MODE_OPTION_CONTEXT["domain"]
    previous_accelerator = _MODE_OPTION_CONTEXT["accelerator"]
    previous_device_id = _MODE_OPTION_CONTEXT["device_id"]
    _MODE_OPTION_CONTEXT["source_type"] = resolved_source_type
    _MODE_OPTION_CONTEXT["domain"] = resolved_domain
    _MODE_OPTION_CONTEXT["accelerator"] = resolved_accelerator
    _MODE_OPTION_CONTEXT["device_id"] = int(device_id)
    try:
        if s not in {-2, -1, 0, 1, 2}:
            raise NotImplementedError("point-particle source is currently implemented only for spin weights s = -2, -1, 0, 1, and 2")
        if s == -2:
            if orbit.kind == "circular-equatorial":
                return _solve_circular_equatorial_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "spherical":
                return _solve_spherical_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "eccentric-equatorial":
                if resolved_accelerator == "dcu":
                    return _solve_eccentric_equatorial_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_eccentric_equatorial_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "generic":
                if resolved_accelerator == "dcu":
                    return _solve_generic_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_generic_mode(s, ell, m, orbit, n, k)
        if s == 0:
            if orbit.kind == "circular-equatorial":
                return _solve_scalar_circular_equatorial_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "spherical":
                return _solve_scalar_spherical_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "eccentric-equatorial":
                if resolved_accelerator == "dcu":
                    return _solve_eccentric_equatorial_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_scalar_eccentric_equatorial_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "generic":
                if resolved_accelerator == "dcu":
                    return _solve_generic_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_scalar_generic_mode(s, ell, m, orbit, n, k)
        if s in {-1, 1}:
            if orbit.kind == "circular-equatorial":
                return _solve_spin_one_circular_equatorial_mode(s, ell, m, orbit, n, k)
            if s == -1 and orbit.kind == "spherical":
                return _solve_spin_minus_one_spherical_mode(s, ell, m, orbit, n, k)
            if s == -1 and orbit.kind == "eccentric-equatorial":
                if resolved_accelerator == "dcu":
                    return _solve_eccentric_equatorial_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_spin_minus_one_eccentric_equatorial_mode(s, ell, m, orbit, n, k)
            if s == -1 and orbit.kind == "generic":
                if resolved_accelerator == "dcu":
                    return _solve_generic_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_spin_minus_one_generic_mode(s, ell, m, orbit, n, k)
            if s == 1 and orbit.kind == "spherical":
                return _solve_spin_plus_one_spherical_mode(s, ell, m, orbit, n, k)
            if s == 1 and orbit.kind == "eccentric-equatorial":
                if resolved_accelerator == "dcu":
                    return _solve_eccentric_equatorial_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_spin_plus_one_eccentric_equatorial_mode(s, ell, m, orbit, n, k)
            if s == 1 and orbit.kind == "generic":
                if resolved_accelerator == "dcu":
                    return _solve_generic_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_spin_plus_one_generic_mode(s, ell, m, orbit, n, k)
        if s == 2:
            if orbit.kind == "circular-equatorial":
                return _solve_spin_plus_two_circular_equatorial_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "spherical":
                return _solve_spin_plus_two_spherical_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "eccentric-equatorial":
                if resolved_accelerator == "dcu":
                    return _solve_eccentric_equatorial_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_spin_plus_two_eccentric_equatorial_mode(s, ell, m, orbit, n, k)
            if orbit.kind == "generic":
                if resolved_accelerator == "dcu":
                    return _solve_generic_mode_dcu(s, ell, m, orbit, n, k)
                return _solve_spin_plus_two_generic_mode(s, ell, m, orbit, n, k)
        raise NotImplementedError("only circular, spherical, eccentric equatorial, and generic bound orbits are implemented in this revision")
    finally:
        _MODE_OPTION_CONTEXT["source_type"] = previous_source_type
        _MODE_OPTION_CONTEXT["domain"] = previous_domain
        _MODE_OPTION_CONTEXT["accelerator"] = previous_accelerator
        _MODE_OPTION_CONTEXT["device_id"] = previous_device_id

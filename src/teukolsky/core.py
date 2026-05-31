from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable


ComplexFn = Callable[[float], complex]
RealFn = Callable[[float], float]


def _is_numeric_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _map_numeric_sequence(values, fn):
    return [fn(value) for value in values]


def _finite_difference_derivative(fn: ComplexFn, order: int, r: float, step: float = 1e-4) -> complex:
    if order == 1:
        return (fn(r + step) - fn(r - step)) / (2.0 * step)
    if order == 2:
        return (fn(r + step) - 2.0 * fn(r) + fn(r - step)) / (step * step)
    if order == 4:
        return (
            fn(r - 2.0 * step)
            - 4.0 * fn(r - step)
            + 6.0 * fn(r)
            - 4.0 * fn(r + step)
            + fn(r + 2.0 * step)
        ) / (step ** 4)
    raise ValueError(f"derivatives implemented only for orders 1, 2, and 4, got {order}")


@dataclass(frozen=True)
class Missing:
    head: str
    key: str

    def __iter__(self):
        yield self.head
        yield self.key


@dataclass(frozen=True)
class RadialSolution:
    s: int
    l: int
    m: int
    a: float
    omega: complex
    eigenvalue: complex
    renormalized_angular_momentum: complex
    method: str
    boundary_conditions: str
    amplitudes: dict[str, complex]
    unscaled_amplitudes: dict[str, complex]
    domain: tuple[float, float]
    radial_function: ComplexFn
    derivative_function: Callable[[int, float], complex]
    method_options: tuple[object, ...] = ()

    def __call__(self, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, self.__call__)
        rmin, rmax = self.domain
        if r < rmin or (not math.isinf(rmax) and r > rmax):
            raise ValueError(f"radial solution is undefined outside domain {self.domain}: r = {r}")
        return self.radial_function(r)

    def derivative(self, order: int, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, lambda value: self.derivative(order, value))
        rmin, rmax = self.domain
        if r < rmin or (not math.isinf(rmax) and r > rmax):
            raise ValueError(f"radial derivative is undefined outside domain {self.domain}: r = {r}")
        return self.derivative_function(order, r)

    def __getitem__(self, key: str):
        method_value = [self.method, *self.method_options]
        mapping = {
            "s": self.s,
            "l": self.l,
            "m": self.m,
            "a": self.a,
            "Omega": self.omega,
            "omega": self.omega,
            "Eigenvalue": self.eigenvalue,
            "RenormalizedAngularMomentum": self.renormalized_angular_momentum,
            "Method": method_value,
            "BoundaryConditions": self.boundary_conditions,
            "Amplitudes": self.amplitudes,
            "UnscaledAmplitudes": self.unscaled_amplitudes,
            "Domain": self.domain,
            "RadialFunction": self.radial_function,
        }
        return mapping.get(key, Missing("KeyAbsent", key))

    def keys(self) -> list[str]:
        return [
            "s",
            "l",
            "m",
            "a",
            "Omega",
            "omega",
            "Eigenvalue",
            "RenormalizedAngularMomentum",
            "Method",
            "BoundaryConditions",
            "Amplitudes",
            "UnscaledAmplitudes",
            "Domain",
        ]


@dataclass(frozen=True)
class Fluxes:
    energy_infinity: complex
    energy_horizon: complex
    angular_momentum_infinity: complex
    angular_momentum_horizon: complex

    @property
    def energy(self) -> complex:
        return self.energy_infinity + self.energy_horizon

    @property
    def angular_momentum(self) -> complex:
        return self.angular_momentum_infinity + self.angular_momentum_horizon


@dataclass(frozen=True)
class Orbit:
    a: float
    p: float
    e: float
    inclination: float
    energy: float
    angular_momentum: float
    omega_r: float
    omega_theta: float
    omega_phi: float
    upsilon_r: float
    upsilon_theta: float
    upsilon_t: float
    kind: str
    radial_phase_function: RealFn | None = None
    radial_velocity_function: RealFn | None = None
    radial_delta_t_function: RealFn | None = None
    radial_delta_phi_function: RealFn | None = None
    theta_phase_function: RealFn | None = None
    theta_velocity_function: RealFn | None = None
    theta_delta_t_function: RealFn | None = None
    theta_delta_phi_function: RealFn | None = None

    def __getitem__(self, key: str):
        orbit_type = {
            "circular-equatorial": ("Bound", "Circular", "Equatorial"),
            "spherical": ("Bound", "Spherical"),
            "eccentric-equatorial": ("Bound", "Eccentric", "Equatorial"),
            "generic": ("Bound", "Generic"),
        }.get(self.kind, (self.kind,))
        frequencies = {
            "Omega_r": self.omega_r,
            "Omega_theta": self.omega_theta,
            "Omega_phi": self.omega_phi,
            "Upsilon_r": self.upsilon_r,
            "Upsilon_theta": self.upsilon_theta,
            "Upsilon_t": self.upsilon_t,
        }
        trajectory_deltas = {
            "Delta_tr": self.radial_delta_t_function,
            "Delta_ttheta": self.theta_delta_t_function,
            "Delta_phir": self.radial_delta_phi_function,
            "Delta_phitheta": self.theta_delta_phi_function,
        }
        initial_phases = {
            "qt0": 0.0,
            "qr0": 0.0,
            "qtheta0": 0.0,
            "qphi0": 0.0,
        }
        mapping = {
            "a": self.a,
            "p": self.p,
            "e": self.e,
            "Inclination": self.inclination,
            "Energy": self.energy,
            "AngularMomentum": self.angular_momentum,
            "Parametrization": "Mino",
            "Frequencies": frequencies,
            "TrajectoryDeltas": trajectory_deltas,
            "InitialPhases": initial_phases,
            "Type": orbit_type,
        }
        return mapping.get(key, Missing("KeyAbsent", key))

    def keys(self) -> list[str]:
        return [
            "a",
            "p",
            "e",
            "Inclination",
            "Energy",
            "AngularMomentum",
            "Parametrization",
            "Frequencies",
            "TrajectoryDeltas",
            "InitialPhases",
            "Type",
        ]


@dataclass(frozen=True)
class ExtendedHomogeneousSolution:
    radial: RadialSolution
    amplitude: complex

    def __call__(self, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, self.__call__)
        return self.amplitude * self.radial(r)

    def derivative(self, order: int, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, lambda value: self.derivative(order, value))
        return self.amplitude * self.radial.derivative(order, r)


@dataclass(frozen=True)
class PNRadialSolution:
    s: int
    l: int
    m: int
    a: float
    omega: complex
    pn: tuple[object, int]
    boundary_condition: str
    series_min_order: int
    term_count: int
    normalization: str
    simplify: bool
    amplitudes_bool: bool
    amplitudes: dict[str, object]
    radial_function: ComplexFn
    leading_order_function: ComplexFn

    def __call__(self, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, self.__call__)
        return self.radial_function(r)

    def derivative(self, order: int, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, lambda value: self.derivative(order, value))
        return _finite_difference_derivative(self.radial_function, order, r)

    def __getitem__(self, key: str):
        mapping = {
            "s": self.s,
            "l": self.l,
            "m": self.m,
            "a": self.a,
            "Omega": self.omega,
            "omega": self.omega,
            "PN": self.pn,
            "BoundaryCondition": self.boundary_condition,
            "BoundaryConditions": self.boundary_condition,
            "SeriesMinOrder": self.series_min_order,
            "LeadingOrder": self.leading_order_function,
            "TermCount": self.term_count,
            "Normalization": self.normalization,
            "Amplitudes": self.amplitudes,
            "Simplify": self.simplify,
            "RadialFunction": self.radial_function,
            "AmplitudesBool": self.amplitudes_bool,
        }
        return mapping.get(key, Missing("KeyAbsent", key))

    def keys(self) -> list[str]:
        return [
            "s",
            "l",
            "m",
            "a",
            "Omega",
            "omega",
            "PN",
            "BoundaryCondition",
            "SeriesMinOrder",
            "LeadingOrder",
            "TermCount",
            "Normalization",
            "Amplitudes",
            "Simplify",
        ]


@dataclass(frozen=True)
class PNEExtendedHomogeneousSolution:
    radial: PNRadialSolution
    amplitude: complex

    def __call__(self, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, self.__call__)
        return self.amplitude * self.radial(r)

    def derivative(self, order: int, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, lambda value: self.derivative(order, value))
        return self.amplitude * self.radial.derivative(order, r)


@dataclass(frozen=True)
class PNModeSolution:
    s: int
    l: int
    m: int
    orbit: Orbit
    pn: tuple[object, int]
    series_min_order: int
    normalization: str
    simplify: bool
    radial_in: PNRadialSolution
    radial_up: PNRadialSolution
    amplitudes: dict[str, complex]
    wronskian: complex
    source_function: Callable[[object], object]
    coefficient_list_function: Callable[[object], object]
    delta_coefficient: object

    def __call__(self, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, self.__call__)
        if math.isclose(r, self.orbit.p, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"mode is undefined at the particle radius r = {r}")
        if r < self.orbit.p:
            return self.amplitudes["H"] * self.radial_in(r)
        return self.amplitudes["I"] * self.radial_up(r)

    def derivative(self, order: int, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, lambda value: self.derivative(order, value))
        if math.isclose(r, self.orbit.p, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"mode derivative is undefined at the particle radius r = {r}")
        if r < self.orbit.p:
            return self.amplitudes["H"] * self.radial_in.derivative(order, r)
        return self.amplitudes["I"] * self.radial_up.derivative(order, r)

    def __getitem__(self, key):
        if key in {("ExtendedHomogeneous", "H"), "ExtendedHomogeneous:H"}:
            return PNEExtendedHomogeneousSolution(self.radial_in, self.amplitudes["H"])
        if key in {("ExtendedHomogeneous", "I"), "ExtendedHomogeneous:I"}:
            return PNEExtendedHomogeneousSolution(self.radial_up, self.amplitudes["I"])
        mapping = {
            "s": self.s,
            "l": self.l,
            "m": self.m,
            "a": self.orbit.a,
            "r0": self.orbit.p,
            "PN": self.pn,
            "Amplitudes": self.amplitudes,
            "Wronskian": self.wronskian,
            "Normalization": self.normalization,
            "Simplify": self.simplify,
            "RadialFunctions": {"In": self.radial_in, "Up": self.radial_up},
            "Source": self.source_function,
            "CoefficientList": self.coefficient_list_function,
            "SeriesMinOrder": self.series_min_order,
            "Delta": self.delta_coefficient,
            "Δ": self.delta_coefficient,
            "Orbit": self.orbit,
            "Type": ("PointParticleCircular", {"Radius": self.orbit.p}),
        }
        return mapping.get(key, Missing("KeyAbsent", str(key)))

    def keys(self) -> list[object]:
        return [
            "s",
            "l",
            "m",
            "a",
            "r0",
            "PN",
            "Amplitudes",
            "Wronskian",
            "Normalization",
            "Simplify",
            "RadialFunctions",
            "Source",
            "CoefficientList",
            "Delta",
            "Orbit",
            "Type",
            ("ExtendedHomogeneous", "H"),
            ("ExtendedHomogeneous", "I"),
        ]


@dataclass(frozen=True)
class ModeSolution:
    s: int
    l: int
    m: int
    n: int
    k: int
    orbit: Orbit
    omega: complex
    eigenvalue: complex
    radial_in: RadialSolution
    radial_up: RadialSolution
    amplitudes: dict[str, complex]
    fluxes: Fluxes
    source_type: str
    domain: tuple[float, float] | str
    acceleration: dict[str, object] | None = None

    def _radius_bounds(self) -> tuple[float, float]:
        if self.orbit.kind in {"circular-equatorial", "spherical"}:
            return self.orbit.p, self.orbit.p
        return self.orbit.p / (1.0 + self.orbit.e), self.orbit.p / (1.0 - self.orbit.e)

    def _type(self):
        if self.orbit.kind == "circular-equatorial":
            return ("PointParticleCircular", {"Radius": self.orbit.p})
        if self.orbit.kind == "spherical":
            return ("PointParticleSpherical", {"Radius": self.orbit.p, "Inclination": self.orbit.inclination})
        if self.orbit.kind == "eccentric-equatorial":
            return ("PointParticleEccentric", {"Semi-latus Rectum": self.orbit.p, "Eccentricity": self.orbit.e})
        return (
            "PointParticleGeneric",
            {"Semi-latus Rectum": self.orbit.p, "Eccentricity": self.orbit.e, "Inclination": self.orbit.inclination},
        )

    def _angular_function(self):
        from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic

        return spin_weighted_spheroidal_harmonic(self.s, self.l, self.m, self.orbit.a * self.omega)

    def _energy_flux_dict(self) -> dict[str, complex]:
        return {"I": self.fluxes.energy_infinity, "H": self.fluxes.energy_horizon}

    def _angular_momentum_flux_dict(self) -> dict[str, complex]:
        return {
            "I": self.fluxes.angular_momentum_infinity,
            "H": self.fluxes.angular_momentum_horizon,
        }

    def __call__(self, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, self.__call__)
        rmin, rmax = self._radius_bounds()
        if r < rmin:
            return self.amplitudes["H"] * self.radial_in(r)
        if r > rmax:
            return self.amplitudes["I"] * self.radial_up(r)
        if rmin == rmax:
            raise ValueError(f"mode is undefined at the particle radius r = {r}")
        raise ValueError(f"mode is undefined inside the libration region {rmin} <= r <= {rmax}")

    def derivative(self, order: int, r: float | Sequence[float]):
        if _is_numeric_sequence(r):
            return _map_numeric_sequence(r, lambda value: self.derivative(order, value))
        rmin, rmax = self._radius_bounds()
        if r < rmin:
            return self.amplitudes["H"] * self.radial_in.derivative(order, r)
        if r > rmax:
            return self.amplitudes["I"] * self.radial_up.derivative(order, r)
        if rmin == rmax:
            raise ValueError(f"mode derivative is undefined at the particle radius r = {r}")
        raise ValueError(f"mode derivative is undefined inside the libration region {rmin} <= r <= {rmax}")

    def __getitem__(self, key):
        rmin, rmax = self._radius_bounds()
        if key in {("ExtendedHomogeneous", "H"), "ExtendedHomogeneous:H"}:
            return ExtendedHomogeneousSolution(self.radial_in, self.amplitudes["H"])
        if key in {("ExtendedHomogeneous", "I"), "ExtendedHomogeneous:I"}:
            return ExtendedHomogeneousSolution(self.radial_up, self.amplitudes["I"])
        mapping = {
            "s": self.s,
            "l": self.l,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "a": self.orbit.a,
            "Omega": self.omega,
            "omega": self.omega,
            "Eigenvalue": self.eigenvalue,
            "Type": self._type(),
            "rmin": rmin,
            "rmax": rmax,
            "Domain": self.domain,
            "SourceType": self.source_type,
            "Acceleration": self.acceleration,
            "Orbit": self.orbit,
            "RadialFunctions": {"In": self.radial_in, "Up": self.radial_up},
            "AngularFunction": self._angular_function(),
            "Amplitudes": self.amplitudes,
            "Fluxes": {"Energy": self._energy_flux_dict(), "AngularMomentum": self._angular_momentum_flux_dict()},
            "EnergyFlux": self._energy_flux_dict(),
            "AngularMomentumFlux": self._angular_momentum_flux_dict(),
        }
        return mapping.get(key, Missing("KeyAbsent", str(key)))

    def keys(self) -> list[object]:
        return [
            "s",
            "l",
            "m",
            "n",
            "k",
            "a",
            "Omega",
            "omega",
            "Eigenvalue",
            "Type",
            "rmin",
            "rmax",
            "Domain",
            "SourceType",
            "Acceleration",
            "Orbit",
            "RadialFunctions",
            "AngularFunction",
            "Amplitudes",
            "Fluxes",
            "EnergyFlux",
            "AngularMomentumFlux",
            ("ExtendedHomogeneous", "H"),
            ("ExtendedHomogeneous", "I"),
        ]

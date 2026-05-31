import math

from teukolsky.core import Missing
from teukolsky.geodesics import circular_orbit
from teukolsky.api import KerrGeoOrbit
from teukolsky.modes import solve_point_particle_mode
import pytest


def test_circular_flux_reference_case_is_close() -> None:
    orbit = circular_orbit(0.9, 10.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit)
    assert abs(mode.fluxes.energy_infinity - 0.0000222730005511805) < 1e-6
    assert abs(mode.fluxes.energy_horizon - (-5.983679213383364e-8)) < 1e-7
    assert abs(mode.fluxes.angular_momentum_infinity - 0.0007243798211752233) < 5e-5
    assert abs(mode.fluxes.angular_momentum_horizon - (-1.9460586231300613e-6)) < 3e-6


def test_mode_key_access_matches_mathematica_semantics() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    assert mode["Type"][0] == "PointParticleCircular"
    assert mode["rmin"] == orbit.p
    assert mode["rmax"] == orbit.p
    assert mode["SourceType"] == "Weyl"
    assert mode["Acceleration"] is None
    assert mode["Orbit"] == orbit
    assert mode["RadialFunctions"]["In"] == mode.radial_in
    assert callable(mode["AngularFunction"])
    assert mode["Amplitudes"] == mode.amplitudes
    assert mode["Fluxes"]["Energy"]["I"] == mode.fluxes.energy_infinity
    assert mode["Fluxes"]["Energy"]["H"] == mode.fluxes.energy_horizon
    assert mode["EnergyFlux"]["I"] == mode.fluxes.energy_infinity
    assert mode["AngularMomentumFlux"]["H"] == mode.fluxes.angular_momentum_horizon
    assert mode["NotAKey"] == Missing("KeyAbsent", "NotAKey")


def test_mode_keys_match_expected_semantics() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    assert mode.keys() == [
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


def test_mode_option_overrides_are_preserved() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0, domain=(6.0, 40.0), source_type="CustomSource")
    assert mode["Domain"] == (6.0, 40.0)
    assert mode["SourceType"] == "CustomSource"
    assert mode["RadialFunctions"]["In"]["Domain"] == (6.0, 40.0)
    assert mode["RadialFunctions"]["Up"]["Domain"] == (6.0, 40.0)
    assert mode["RadialFunctions"]["In"]["Method"] == ["NumericalIntegration", {"Domain": (6.0, 40.0)}]
    assert mode["Acceleration"] is None


def test_requesting_gpu_for_unsupported_orbit_kind_raises() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="only for eccentric-equatorial and generic"):
        solve_point_particle_mode(-2, 2, 2, orbit, accelerator="gpu")


def test_generic_mode_routes_to_dcu_backend(monkeypatch) -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    sentinel = object()

    def fake_dcu_solver(s, ell, m, orbit_arg, n, k):
        assert orbit_arg == orbit
        return sentinel

    monkeypatch.setattr("teukolsky.modes.point_particle._solve_generic_mode_dcu", fake_dcu_solver)
    result = solve_point_particle_mode(0, 2, 2, orbit, accelerator="gpu")
    assert result is sentinel


def test_eccentric_mode_routes_to_dcu_backend(monkeypatch) -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    sentinel = object()

    def fake_dcu_solver(s, ell, m, orbit_arg, n, k):
        assert orbit_arg == orbit
        return sentinel

    monkeypatch.setattr("teukolsky.modes.point_particle._solve_eccentric_equatorial_mode_dcu", fake_dcu_solver)
    result = solve_point_particle_mode(-2, 2, 2, orbit, accelerator="gpu")
    assert result is sentinel


def test_requesting_gpu_without_visible_device_raises(monkeypatch) -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)

    def fake_require_dcu(device_id):
        raise RuntimeError("GPU backend is not available in this session")

    monkeypatch.setattr("teukolsky.modes.point_particle.require_dcu", fake_require_dcu)
    with pytest.raises(RuntimeError, match="GPU backend is not available"):
        solve_point_particle_mode(0, 2, 2, orbit, accelerator="gpu")


def test_requesting_dcu_alias_routes_to_gpu_backend(monkeypatch) -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    sentinel = object()

    def fake_dcu_solver(s, ell, m, orbit_arg, n, k):
        assert orbit_arg == orbit
        return sentinel

    monkeypatch.setattr("teukolsky.modes.point_particle._solve_generic_mode_dcu", fake_dcu_solver)
    result = solve_point_particle_mode(0, 2, 2, orbit, accelerator="dcu")
    assert result is sentinel


def test_invalid_mode_indices_for_orbit_kind_raise() -> None:
    circular = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    spherical = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    eccentric = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)

    with pytest.raises(ValueError, match="n = k = 0"):
        solve_point_particle_mode(-2, 2, 2, circular, n=1, k=0)
    with pytest.raises(ValueError, match="n = 0"):
        solve_point_particle_mode(-2, 2, 2, spherical, n=1, k=0)
    with pytest.raises(ValueError, match="k = 0"):
        solve_point_particle_mode(-2, 2, 2, eccentric, n=0, k=1)


def test_domain_option_is_rejected_for_static_point_particle_modes() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="Domain option is not supported"):
        solve_point_particle_mode(0, 2, 0, orbit, n=0, k=0, domain=(6.0, 40.0))


def test_eccentric_mode_uses_internal_libration_domain_for_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0, domain=(9.0, 12.0))
    assert mode["Domain"] == (9.0, 12.0)
    assert mode["RadialFunctions"]["In"]["Domain"] == (9.0, 12.0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0


def test_generic_mode_uses_internal_libration_domain_for_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0, domain=(9.0, 12.0))
    assert mode["Domain"] == (9.0, 12.0)
    assert mode["RadialFunctions"]["Up"]["Domain"] == (9.0, 12.0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0


def test_mode_type_details_match_orbit_kind() -> None:
    circular = solve_point_particle_mode(-2, 2, 2, KerrGeoOrbit(0.9, 10.0, 0.0, 1.0), n=0, k=0)
    spherical = solve_point_particle_mode(0, 2, 2, KerrGeoOrbit(0.5, 8.0, 0.0, 0.7), n=0, k=0)
    eccentric = solve_point_particle_mode(-2, 2, 2, KerrGeoOrbit(0.5, 10.0, 0.2, 1.0), n=0, k=0)
    generic = solve_point_particle_mode(0, 2, 2, KerrGeoOrbit(0.5, 10.0, 0.2, 0.7), n=0, k=0)

    assert circular["Type"] == ("PointParticleCircular", {"Radius": 10.0})
    assert spherical["Type"] == ("PointParticleSpherical", {"Radius": 8.0, "Inclination": 0.7})
    assert eccentric["Type"] == ("PointParticleEccentric", {"Semi-latus Rectum": 10.0, "Eccentricity": 0.2})
    assert generic["Type"] == (
        "PointParticleGeneric",
        {"Semi-latus Rectum": 10.0, "Eccentricity": 0.2, "Inclination": 0.7},
    )


def test_mode_extended_homogeneous_accessors_work() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ext_h = mode["ExtendedHomogeneous:H"]
    ext_i = mode["ExtendedHomogeneous:I"]
    assert abs(ext_h(5.0) - mode.amplitudes["H"] * mode.radial_in(5.0)) < 1e-15
    assert abs(ext_i(20.0) - mode.amplitudes["I"] * mode.radial_up(20.0)) < 1e-15
    assert abs(ext_h.derivative(1, 5.0) - mode.amplitudes["H"] * mode.radial_in.derivative(1, 5.0)) < 1e-15


def test_mode_and_extended_homogeneous_accept_numeric_sequences() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ext_h = mode["ExtendedHomogeneous:H"]
    ext_i = mode["ExtendedHomogeneous:I"]
    assert ext_h([5.0, 6.0]) == [ext_h(5.0), ext_h(6.0)]
    assert ext_i([20.0, 21.0]) == [ext_i(20.0), ext_i(21.0)]
    assert mode([9.0, 9.5]) == [mode(9.0), mode(9.5)]
    assert mode.derivative(1, [9.0, 9.5]) == [mode.derivative(1, 9.0), mode.derivative(1, 9.5)]


def test_mode_piecewise_evaluation_matches_radial_branches() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    rmin = orbit.p / (1.0 + orbit.e)
    rmax = orbit.p / (1.0 - orbit.e)
    assert abs(mode(rmin - 1.0) - mode.amplitudes["H"] * mode.radial_in(rmin - 1.0)) < 1e-15
    assert abs(mode(rmax + 1.0) - mode.amplitudes["I"] * mode.radial_up(rmax + 1.0)) < 1e-15
    with pytest.raises(ValueError, match="libration region"):
        mode((rmin + rmax) / 2.0)


def test_mode_piecewise_evaluation_rejects_particle_radius_for_circular_orbit() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    with pytest.raises(ValueError, match="particle radius"):
        mode(orbit.p)


def test_api_shorthand_overloads_match_full_signature() -> None:
    orbit_c = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    orbit_s = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    orbit_e = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)

    from teukolsky.api import TeukolskyPointParticleMode

    mode_c_short = TeukolskyPointParticleMode(-2, 2, 2, orbit_c)
    mode_c_full = TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit_c)
    assert mode_c_short.amplitudes == mode_c_full.amplitudes

    mode_s_short = TeukolskyPointParticleMode(-2, 2, 2, 0, orbit_s)
    mode_s_full = TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit_s)
    assert mode_s_short.amplitudes == mode_s_full.amplitudes

    mode_e_short = TeukolskyPointParticleMode(-2, 2, 2, 0, orbit_e)
    mode_e_full = TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit_e)
    assert mode_e_short.amplitudes == mode_e_full.amplitudes

    mode_kw = TeukolskyPointParticleMode(-2, 2, 2, orbit_c, domain=(6.0, 40.0), source_type="CustomSource")
    assert mode_kw["Domain"] == (6.0, 40.0)
    assert mode_kw["SourceType"] == "CustomSource"


def test_s_plus_2_spherical_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)
    assert mode.fluxes.energy_infinity.real > 0.0
    assert mode.fluxes.energy_horizon.real < 0.0


def test_s_plus_2_eccentric_equatorial_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)
    assert mode.fluxes.energy_infinity.real > 0.0
    assert mode.fluxes.energy_horizon.real < 0.0


def test_s_plus_2_generic_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)
    assert mode.fluxes.energy_infinity.real > 0.0
    assert mode.fluxes.energy_horizon.real < 0.0


def test_spin_two_circular_mode_matches_reference_amplitudes() -> None:
    orbit = circular_orbit(0.5, 10.0)
    mode = solve_point_particle_mode(2, 2, 2, orbit)
    assert abs(mode.amplitudes["I"] - (-396.5185079046605 + 121.68497998360989j)) < 3e-4
    assert abs(mode.amplitudes["H"] - (-0.0013801882310749624 + 0.007335365634528333j)) < 1e-8
    assert mode.fluxes.energy_infinity.real > 0.0
    assert mode.fluxes.energy_horizon.real < 0.0


def test_unimplemented_spin_weight_raises() -> None:
    orbit = circular_orbit(0.9, 10.0)
    with pytest.raises(NotImplementedError):
        solve_point_particle_mode(3, 3, 3, orbit)


def test_scalar_circular_flux_reference_case_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    assert abs(mode.fluxes.energy_infinity - 3.067116916288e-06) < 5e-10
    assert abs(mode.fluxes.energy_horizon - (-2.953421604327055e-09)) < 5e-12
    assert abs(mode.fluxes.angular_momentum_infinity - 9.85243115131647e-05) < 2e-8
    assert abs(mode.fluxes.angular_momentum_horizon - (-9.487210240638452e-08)) < 2e-10


def test_eccentric_equatorial_flux_reference_case_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    assert abs(mode.fluxes.energy_infinity - 1.4078283101297553e-05) < 2e-8
    assert abs(mode.fluxes.energy_horizon - (-9.004375421774648e-09)) < 2e-11
    assert abs(mode.fluxes.angular_momentum_infinity - 4.7419159510590406e-04) < 5e-7
    assert abs(mode.fluxes.angular_momentum_horizon - (-3.032897629250106e-07)) < 5e-10


def test_eccentric_equatorial_radial_harmonic_flux_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=1, k=0)
    assert abs(mode.fluxes.energy_infinity - 8.630621820066304e-06) < 2e-8
    assert abs(mode.fluxes.energy_horizon - (-7.939614404924082e-09)) < 2e-11
    assert abs(mode.fluxes.angular_momentum_infinity - 2.1376547166376627e-04) < 5e-7
    assert abs(mode.fluxes.angular_momentum_horizon - (-1.9665042142745523e-07)) < 5e-10


def test_spherical_flux_reference_case_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    assert abs(mode.fluxes.energy_infinity - 4.060182005305403e-05) < 5e-8
    assert abs(mode.fluxes.energy_horizon - (-4.5146327829027344e-08)) < 1e-10
    assert abs(mode.fluxes.angular_momentum_infinity - 9.234498882784354e-04) < 1e-6
    assert abs(mode.fluxes.angular_momentum_horizon - (-1.0268104074009613e-06)) < 2e-9


def test_generic_flux_reference_case_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    assert abs(mode.fluxes.energy_infinity - 7.622599715599874e-06) < 5e-9
    assert abs(mode.fluxes.energy_horizon - (-4.846144568045695e-09)) < 1e-11
    assert abs(mode.fluxes.angular_momentum_infinity - 2.5332622023492145e-04) < 2e-7
    assert abs(mode.fluxes.angular_momentum_horizon - (-1.6105469681460238e-07)) < 5e-10


def test_scalar_eccentric_equatorial_flux_reference_case_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    assert abs(mode.fluxes.energy_infinity - 1.8370607787246964e-06) < 5e-10
    assert abs(mode.fluxes.energy_horizon - (-1.5615446308960489e-09)) < 5e-12
    assert abs(mode.fluxes.angular_momentum_infinity - 6.187677678464e-05) < 2e-8
    assert abs(mode.fluxes.angular_momentum_horizon - (-5.259670756907926e-08)) < 2e-10


def test_scalar_spherical_flux_reference_case_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    assert abs(mode.fluxes.energy_infinity - 4.777241323673375e-06) < 5e-9
    assert abs(mode.fluxes.energy_horizon - (-7.209909976139032e-09)) < 2e-11
    assert abs(mode.fluxes.angular_momentum_infinity - 1.0865382292864655e-04) < 2e-7
    assert abs(mode.fluxes.angular_momentum_horizon - (-1.6398256416247426e-07)) < 5e-10


def test_scalar_generic_flux_reference_case_is_close() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    assert abs(mode.fluxes.energy_infinity - 9.918316203377088e-07) < 5e-10
    assert abs(mode.fluxes.energy_horizon - (-7.859825241061865e-10)) < 5e-12
    assert abs(mode.fluxes.angular_momentum_infinity - 3.2962108055526604e-05) < 2e-8
    assert abs(mode.fluxes.angular_momentum_horizon - (-2.6121007193259648e-08)) < 2e-10


def test_maxwell_circular_modes_are_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
    mode_minus = solve_point_particle_mode(-1, 1, 1, orbit, n=0, k=0)
    mode_plus = solve_point_particle_mode(1, 1, 1, orbit, n=0, k=0)
    for mode in (mode_minus, mode_plus):
        assert abs(mode.amplitudes["I"]) > 0.0
        assert abs(mode.amplitudes["H"]) > 0.0
        assert mode.fluxes.energy_infinity.real > 0.0
    assert abs(mode_minus.fluxes.energy_infinity - mode_plus.fluxes.energy_infinity) < 1e-10
    assert abs(mode_minus.fluxes.energy_horizon - mode_plus.fluxes.energy_horizon) < 1e-10


def test_maxwell_eccentric_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(-1, 1, 1, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)


def test_maxwell_spherical_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    mode = solve_point_particle_mode(-1, 1, 1, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)


def test_maxwell_generic_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(-1, 1, 1, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)


def test_maxwell_plus_one_eccentric_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    mode = solve_point_particle_mode(1, 1, 1, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)


def test_maxwell_plus_one_spherical_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    mode = solve_point_particle_mode(1, 1, 1, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)


def test_maxwell_plus_one_generic_mode_is_finite() -> None:
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(1, 1, 1, orbit, n=0, k=0)
    assert abs(mode.amplitudes["I"]) > 0.0
    assert abs(mode.amplitudes["H"]) > 0.0
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)


# ---------------------------------------------------------------------------
# Internal cross-validation tests — no external reference values needed
# ---------------------------------------------------------------------------

def _compute_wronskian(rin, rup, s, r):
    """Δ^(s+1)(R_up' R_in - R_in' R_up) at radius r."""
    delta = r * r - 2.0 * r + rin.a * rin.a
    return delta ** (s + 1) * (rup.derivative(1, r) * rin(r) - rin.derivative(1, r) * rup(r))


def test_spin_one_eccentric_wronskian_consistency() -> None:
    """Wronskian Δ^(s+1)(R_up' R_in - R_in' R_up) is r-independent for s=±1 eccentric."""
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    r_peri = orbit.p / (1.0 + orbit.e)
    r_apo = orbit.p / (1.0 - orbit.e)
    for s in (-1, 1):
        mode = solve_point_particle_mode(s, 1, 1, orbit, n=0, k=0)
        w_peri = _compute_wronskian(mode.radial_in, mode.radial_up, s, r_peri)
        w_apo = _compute_wronskian(mode.radial_in, mode.radial_up, s, r_apo)
        assert abs(w_peri) > 0.0, f"s={s}: Wronskian at periastron is zero"
        rel_diff = abs(w_peri - w_apo) / abs(w_peri)
        assert rel_diff < 1e-4, f"s={s}: Wronskian relative diff {rel_diff} exceeds tolerance"


def test_spin_two_positive_eccentric_wronskian_consistency() -> None:
    """Wronskian is r-independent for s=+2 eccentric."""
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    r_peri = orbit.p / (1.0 + orbit.e)
    r_apo = orbit.p / (1.0 - orbit.e)
    mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
    w_peri = _compute_wronskian(mode.radial_in, mode.radial_up, 2, r_peri)
    w_apo = _compute_wronskian(mode.radial_in, mode.radial_up, 2, r_apo)
    assert abs(w_peri) > 0.0
    rel_diff = abs(w_peri - w_apo) / abs(w_peri)
    assert rel_diff < 1e-4, f"Wronskian relative diff {rel_diff} exceeds tolerance"


def test_maxwell_eccentric_flux_positivity() -> None:
    """Energy flux at infinity > 0 and at horizon < 0 for s=±1 eccentric."""
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
    for s in (-1, 1):
        mode = solve_point_particle_mode(s, 1, 1, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0, f"s={s}: energy_infinity should be positive"
        assert mode.fluxes.energy_horizon.real < 0.0, f"s={s}: energy_horizon should be negative"


def test_spin_two_positive_spherical_flux_positivity() -> None:
    """Energy flux at infinity > 0 and at horizon < 0 for s=+2 spherical."""
    orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
    assert mode.fluxes.energy_infinity.real > 0.0
    assert mode.fluxes.energy_horizon.real < 0.0


def test_spin_two_positive_generic_flux_positivity() -> None:
    """Energy flux at infinity > 0 and at horizon < 0 for s=+2 generic."""
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
    assert mode.fluxes.energy_infinity.real > 0.0
    assert mode.fluxes.energy_horizon.real < 0.0


def test_maxwell_circular_limit_eccentric() -> None:
    """Eccentric mode with e=0.001 approaches circular mode for s=-1 and s=+1."""
    orbit_circ = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
    orbit_ecc = KerrGeoOrbit(0.5, 10.0, 0.001, 1.0)
    for s in (-1, 1):
        mode_circ = solve_point_particle_mode(s, 1, 1, orbit_circ, n=0, k=0)
        mode_ecc = solve_point_particle_mode(s, 1, 1, orbit_ecc, n=0, k=0)
        rel_inf = abs(mode_ecc.fluxes.energy_infinity - mode_circ.fluxes.energy_infinity) / abs(mode_circ.fluxes.energy_infinity)
        rel_hor = abs(mode_ecc.fluxes.energy_horizon - mode_circ.fluxes.energy_horizon) / abs(mode_circ.fluxes.energy_horizon)
        assert rel_inf < 1e-2, f"s={s}: energy_infinity relative diff {rel_inf} exceeds 1%"
        assert rel_hor < 1e-2, f"s={s}: energy_horizon relative diff {rel_hor} exceeds 1%"


def test_spin_two_positive_circular_limit_eccentric() -> None:
    """Eccentric mode with e=0.001 approaches circular mode for s=+2."""
    orbit_circ = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
    orbit_ecc = KerrGeoOrbit(0.5, 10.0, 0.001, 1.0)
    mode_circ = solve_point_particle_mode(2, 2, 2, orbit_circ, n=0, k=0)
    mode_ecc = solve_point_particle_mode(2, 2, 2, orbit_ecc, n=0, k=0)
    rel_inf = abs(mode_ecc.fluxes.energy_infinity - mode_circ.fluxes.energy_infinity) / abs(mode_circ.fluxes.energy_infinity)
    rel_hor = abs(mode_ecc.fluxes.energy_horizon - mode_circ.fluxes.energy_horizon) / abs(mode_circ.fluxes.energy_horizon)
    assert rel_inf < 1e-2, f"energy_infinity relative diff {rel_inf} exceeds 1%"
    assert rel_hor < 1e-2, f"energy_horizon relative diff {rel_hor} exceeds 1%"


# ---------------------------------------------------------------------------
# Regression tests ported from TeukolskyRadial.wlt
# Reference values generated by the Mathematica BlackHolePerturbationToolkit.
# ---------------------------------------------------------------------------


# -- Schwarzschild (a = 0) circular, p = 10 ----------------------------------

def test_schwarzschild_circular_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.09686911178330269 + 0.03469108886623385j
    ref_H = 0.0008524908874743823 - 0.00021231389467651976j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-6
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-6


def test_schwarzschild_circular_sm1_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-1, 2, 2, orbit, n=0, k=0)
    ref_I = 0.001908019370208005 + 0.005523060666840923j
    ref_H = 0.001091169268358212 + 0.00002231589868836576j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-6
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-5


def test_schwarzschild_circular_sp1_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(1, 2, 2, orbit, n=0, k=0)
    ref_I = -1.431014527655975 - 4.142295500130651j
    ref_H = -0.00008558574365842871 - 0.0003699661781058245j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-6
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-6


def test_schwarzschild_circular_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.001120640791925417 + 0.0003057608384581628j
    ref_H = -0.001471024726331052 - 0.000386130529924696j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


# -- Schwarzschild (a = 0) spherical, p = 10, x = cos(π/4) -----------------

def test_schwarzschild_spherical_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.07057431983348402 + 0.02527431041686724j
    ref_H = 0.0006210851265193302 - 0.0001546820078366417j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


def test_schwarzschild_spherical_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.0008164466485943937 + 0.00022276309556933678j
    ref_H = -0.0010717200520150304 - 0.0002813167067882148j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-5


# -- Schwarzschild (a = 0) eccentric, p = 10, e = 0.1 -----------------------

def test_schwarzschild_eccentric_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.1, 1.0)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.09096671505325699 + 0.03235726586996844j
    ref_H = 0.0007806442119256981 - 0.000192448035402558j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


def test_schwarzschild_eccentric_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.1, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.00102739509088208 + 0.00027888266315215j
    ref_H = -0.00135207925401707 - 0.00035093350945876j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


# -- Schwarzschild (a = 0) generic, p = 10, e = 0.1, x = cos(π/4) ----------

def test_schwarzschild_generic_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.1, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.0662741086831826 + 0.02357399575989346j
    ref_H = 0.0005687409874454815 - 0.000140208668705548j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


def test_schwarzschild_generic_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.0, 10.0, 0.1, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.00074851217701405 + 0.00020318090971475j
    ref_H = -0.00098506192486100 - 0.00025567379818942j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-5


# -- Kerr (a = 0.1) circular, p = 10 ----------------------------------------

def test_kerr_circular_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.09622758489476638 + 0.03442250589801311j
    ref_H = 0.0008296952987487495 - 0.00004165341379611483j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-6
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-5


def test_kerr_circular_sm1_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-1, 2, 2, orbit, n=0, k=0)
    ref_I = 0.0018866657836778435 + 0.005466865521073645j
    ref_H = 0.001063460021661906 + 0.000019474011014653604j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-6
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-5


def test_kerr_circular_sp1_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(1, 2, 2, orbit, n=0, k=0)
    ref_I = -1.4179854870221155 - 4.108801906224362j
    ref_H = -2.4468453070036285e-6 - 0.0000731084398930582j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-6
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-5


def test_kerr_circular_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.0011042513597484405 + 0.0003010513879342753j
    ref_H = -0.0014748445457123322 - 0.00022895111577079285j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


# -- Kerr (a = 0.1) spherical, p = 10, x = cos(π/4) ------------------------

def test_kerr_spherical_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.07046852842293279 + 0.02525252290396318j
    ref_H = 0.0006046377748504626 - 0.00003075360976614049j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-4
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 1e-4


def test_kerr_spherical_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.0008134090552314843 + 0.0002220559776171971j
    ref_H = -0.0010818095010835465 - 0.00016877032213978151j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-4
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 2e-4


# -- Kerr (a = 0.1) eccentric, p = 10, e = 0.1 -----------------------------

def test_kerr_eccentric_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.1, 1.0)
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.09068996756763358 + 0.03221537507695608j
    ref_H = 0.0007625072296231175 - 0.00003624478217354642j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-6
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


def test_kerr_eccentric_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.1, 1.0)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.00101539073433068 + 0.00027535746450647j
    ref_H = -0.001359631602157010 - 0.000207192301964262j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-5
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 5e-5


# -- Kerr (a = 0.1) generic, p = 10, e = 0.1, x = cos(π/4) ----------------

def test_kerr_generic_s0_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.1, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
    ref_I = -0.0663183242758333 + 0.023601036212611705j
    ref_H = 0.0005549011071009785 - 0.000026750779539412936j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 1e-4
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 2e-4


def test_kerr_generic_sm2_amplitudes() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.1, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
    ref_I = -0.0007470184367075 + 0.0002028616424164j
    ref_H = -0.0009960463270432 - 0.0001525666277914j
    assert abs(mode.amplitudes["I"] - ref_I) / abs(ref_I) < 5e-4
    assert abs(mode.amplitudes["H"] - ref_H) / abs(ref_H) < 2e-4


# -- Higher-k spherical harmonics (small amplitudes, smoke tests) -----------

def test_kerr_spherical_s0_k2_is_finite() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=2)
    assert math.isfinite(abs(mode.amplitudes["I"]))
    assert math.isfinite(abs(mode.amplitudes["H"]))
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)
    assert mode.fluxes.energy_infinity.real > 0.0


def test_kerr_spherical_sm2_k2_is_finite() -> None:
    orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
    mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=2)
    assert math.isfinite(abs(mode.amplitudes["I"]))
    assert math.isfinite(abs(mode.amplitudes["H"]))
    assert math.isfinite(mode.fluxes.energy_infinity.real)
    assert math.isfinite(mode.fluxes.energy_horizon.real)
    assert mode.fluxes.energy_infinity.real > 0.0

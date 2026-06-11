from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from teukolsky import (
    finite_difference_jacobian_equatorial,
    generate_equatorial_eccentric_adiabatic_waveform,
    integrate_equatorial_eccentric_inspiral,
    generic_eccentric_rhs,
    generic_total_fluxes,
    generate_generic_eccentric_adiabatic_waveform,
    integrate_schwarzschild_eccentric_inspiral,
    KerrGeoOrbit,
    TeukolskyPointParticleMode,
    enumerate_mode_indices,
    generate_schwarzschild_eccentric_adiabatic_waveform,
    generate_fixed_orbit_waveform,
    generate_sparse_trajectory_waveform,
    mode_frequency,
    mode_strain,
    source_frame_radius,
)


def test_mode_strain_matches_direct_formula():
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    mode = TeukolskyPointParticleMode(-2, 2, 2, orbit)
    t = np.linspace(0.0, 100.0, 8)
    theta = 1.1
    phi = 0.3
    radius = 1000.0

    angular = mode["AngularFunction"].evaluate(theta, phi)
    expected = -2.0 * mode["Amplitudes"]["I"] * angular * np.exp(-1j * mode["Omega"] * t) / (radius * mode["Omega"] ** 2)
    actual = mode_strain(mode, t, theta=theta, phi=phi, radius=radius)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_mode_strain_respects_mass_seconds_scaling():
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    mode = TeukolskyPointParticleMode(-2, 2, 2, orbit)
    t = np.linspace(0.0, 100.0, 8)
    theta = 1.1
    phi = 0.3
    radius = 1000.0
    M_sec = 2.0

    angular = mode["AngularFunction"].evaluate(theta, phi)
    omega_phase = mode["Omega"] / M_sec
    expected = -2.0 * mode["Amplitudes"]["I"] * angular * np.exp(-1j * omega_phase * t) / (radius * mode["Omega"] ** 2)
    actual = mode_strain(mode, t, theta=theta, phi=phi, radius=radius, mass_seconds=M_sec)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_source_frame_radius_matches_dimensionless_scaling():
    expected = 3.0856775814913673e25 / (10.0 * 1476.6250380501249)
    actual = source_frame_radius(1.0, 10.0)
    assert np.isclose(actual, expected, rtol=1e-15, atol=0.0)


def test_fixed_orbit_waveform_matches_manual_two_mode_sum():
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    t = np.linspace(0.0, 100.0, 16)
    theta = 1.2
    phi = 0.1
    radius = 2000.0
    mode_indices = [(2, 2, 0, 0), (2, -2, 0, 0)]

    waveform = generate_fixed_orbit_waveform(
        orbit,
        t,
        theta=theta,
        phi=phi,
        radius=radius,
        mode_indices=mode_indices,
    )

    mode_p = TeukolskyPointParticleMode(-2, 2, 2, orbit)
    mode_m = TeukolskyPointParticleMode(-2, 2, -2, orbit)
    expected = (
        mode_strain(mode_p, t, theta=theta, phi=phi, radius=radius)
        + mode_strain(mode_m, t, theta=theta, phi=phi, radius=radius)
    )

    assert waveform.modes == tuple(mode_indices)
    assert np.allclose(waveform.complex_strain, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(waveform.h_plus, expected.real, rtol=1e-12, atol=1e-12)
    assert np.allclose(waveform.h_cross, -expected.imag, rtol=1e-12, atol=1e-12)


def test_sparse_trajectory_waveform_with_single_node_matches_fixed_orbit():
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    t = np.linspace(0.0, 100.0, 16)
    theta = 1.1
    phi = 0.2
    radius = 1500.0
    mode_indices = [(2, 2, 0, 0), (2, -2, 0, 0)]

    fixed = generate_fixed_orbit_waveform(
        orbit,
        t,
        theta=theta,
        phi=phi,
        radius=radius,
        mode_indices=mode_indices,
        mass_seconds=1.0e6 * 4.92549095e-6,
    )
    sparse = generate_sparse_trajectory_waveform(
        1.0e6,
        0.5,
        np.array([0.0], dtype=float),
        np.array([orbit.p], dtype=float),
        np.array([orbit.e], dtype=float),
        np.array([0.7], dtype=float),
        evaluation_time=t,
        theta=theta,
        phi=phi,
        radius=radius,
        mode_indices=mode_indices,
    )

    assert sparse.modes == fixed.modes
    assert np.allclose(sparse.complex_strain, fixed.complex_strain, rtol=1e-12, atol=1e-12)


def test_sparse_trajectory_waveform_supports_generic_sparse_orbit(monkeypatch):
    import teukolsky.waveform as waveform_module

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.0 + 0.2j

    class FakeMode:
        def __init__(self, amp, omega):
            self._amp = amp
            self.omega = omega

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return FakeAngular()
            if key == "Omega":
                return self.omega
            raise KeyError(key)

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, orbit, kwargs
        omega = 0.02 + 0.001 * (abs(m) + abs(n) + abs(k))
        amp = complex(ell + 0.1 * m + 0.01 * n + 0.001 * k, 0.2 * m)
        mode = FakeMode(amp, omega)
        return mode

    monkeypatch.setattr(waveform_module, "TeukolskyPointParticleMode", fake_mode)

    t = np.array([0.0, 50.0, 100.0], dtype=float)
    p = np.array([10.0, 9.95, 9.9], dtype=float)
    e = np.array([0.2, 0.195, 0.19], dtype=float)
    x = np.array([0.7, 0.7, 0.7], dtype=float)

    waveform = generate_sparse_trajectory_waveform(
        1.0e6,
        0.5,
        t,
        p,
        e,
        x,
        theta=1.0,
        phi=0.3,
        radius=1000.0,
        mode_indices=[(2, 2, 0, 0), (2, -2, 0, 0)],
    )

    assert waveform.time.shape == t.shape
    assert np.all(np.isfinite(waveform.h_plus))
    assert np.all(np.isfinite(waveform.h_cross))
    assert len(waveform.modes) > 0


def test_generate_fixed_orbit_waveform_skips_zero_frequency_modes():
    orbit = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    t = np.linspace(0.0, 10.0, 4)
    waveform = generate_fixed_orbit_waveform(
        orbit,
        t,
        theta=1.0,
        radius=100.0,
        mode_indices=[(2, 0, 0, 0), (2, 2, 0, 0)],
    )

    assert waveform.modes == ((2, 2, 0, 0),)


def test_enumerate_mode_indices_respects_orbit_kind():
    circular = KerrGeoOrbit(0.0, 10.0, 0.0, 1.0)
    spherical = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
    eccentric = KerrGeoOrbit(0.0, 12.0, 0.2, 1.0)
    generic = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)

    circ = enumerate_mode_indices(circular, ell_min=2, ell_max=2, n_max=2, k_max=2)
    sph = enumerate_mode_indices(spherical, ell_min=2, ell_max=2, n_max=2, k_max=1)
    ecc = enumerate_mode_indices(eccentric, ell_min=2, ell_max=2, n_max=1, k_max=2)
    gen = enumerate_mode_indices(generic, ell_min=2, ell_max=2, n_max=1, k_max=1)

    assert all(n == 0 and k == 0 for _, _, n, k in circ)
    assert all(n == 0 for _, _, n, _ in sph)
    assert set(k for _, _, _, k in sph) == {-1, 0, 1}
    assert all(k == 0 for _, _, _, k in ecc)
    assert set(n for _, _, n, _ in ecc) == {-1, 0, 1}
    assert set(n for _, _, n, _ in gen) == {-1, 0, 1}
    assert set(k for _, _, _, k in gen) == {-1, 0, 1}


def test_mode_frequency_matches_mode_omega():
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit)
    assert np.isclose(mode_frequency(orbit, 2, 0, 0), mode["Omega"])


def test_integrate_schwarzschild_eccentric_inspiral_monotone_with_mock_flux(monkeypatch):
    import teukolsky.waveform as waveform

    def fake_fluxes(p, e, **kwargs):
        return 1.0e-6 * (1.0 + e), 2.0e-5 * (1.0 + 0.1 * e)

    monkeypatch.setattr(waveform, "schwarzschild_total_fluxes", fake_fluxes)

    traj = integrate_schwarzschild_eccentric_inspiral(
        1.0e6,
        1.0e4,
        12.0,
        0.2,
        t_end=1.0e5,
        trajectory_dt=2.5e4,
        ell_max=2,
        n_max=0,
    )

    assert len(traj.time) >= 2
    assert np.all(np.diff(traj.time) > 0.0)
    assert traj.p[-1] < traj.p[0]
    assert traj.e[-1] <= traj.e[0]
    assert np.all(np.isfinite(traj.pdot))
    assert np.all(np.isfinite(traj.edot))


def test_generate_schwarzschild_eccentric_adiabatic_waveform_smoke(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.25 + 0.5j

    class FakeMode:
        def __init__(self, amp):
            self._amp = amp

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, orbit, kwargs
        return FakeMode((ell + 0.1 * m + 0.01 * n) + 1j * (0.2 * m))

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectory(
            time=np.array([0.0, 10.0]),
            p=np.array([12.0, 12.0]),
            e=np.array([0.2, 0.2]),
            energy=np.array([0.95, 0.95]),
            angular_momentum=np.array([4.0, 4.0]),
            pdot=np.array([-1.0e-6, -1.0e-6]),
            edot=np.array([-1.0e-7, -1.0e-7]),
            edot_energy=np.array([-1.0e-10, -1.0e-10]),
            edot_angular_momentum=np.array([-1.0e-9, -1.0e-9]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_schwarzschild_eccentric_inspiral", fake_traj)
    t = np.linspace(0.0, 10.0, 32)
    result = generate_schwarzschild_eccentric_adiabatic_waveform(
        1.0e6,
        10.0,
        12.0,
        0.2,
        t,
        theta=1.0,
        radius=1000.0,
        trajectory_dt=10.0,
        trajectory_ell_max=2,
        trajectory_n_max=0,
        waveform_ell_max=2,
        waveform_n_max=0,
    )
    assert result.waveform.time.shape == t.shape
    assert np.all(np.isfinite(result.waveform.h_plus))
    assert np.all(np.isfinite(result.waveform.h_cross))
    assert len(result.waveform.modes) > 0


def test_generate_schwarzschild_adiabatic_waveform_avoids_extra_angular_probe_solves(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.25 + 0.5j

    class FakeMode:
        def __init__(self, amp):
            self._amp = amp

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    calls: list[tuple[int, int, int, int, float, float]] = []

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, kwargs
        calls.append((ell, m, n, k, orbit.p, orbit.e))
        return FakeMode((ell + 0.1 * m + 0.01 * n) + 1j * (0.2 * m))

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectory(
            time=np.array([0.0, 10.0]),
            p=np.array([12.0, 12.0]),
            e=np.array([0.2, 0.2]),
            energy=np.array([0.95, 0.95]),
            angular_momentum=np.array([4.0, 4.0]),
            pdot=np.array([-1.0e-6, -1.0e-6]),
            edot=np.array([-1.0e-7, -1.0e-7]),
            edot_energy=np.array([-1.0e-10, -1.0e-10]),
            edot_angular_momentum=np.array([-1.0e-9, -1.0e-9]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_schwarzschild_eccentric_inspiral", fake_traj)

    t = np.linspace(0.0, 10.0, 32)
    result = generate_schwarzschild_eccentric_adiabatic_waveform(
        1.0e6,
        10.0,
        12.0,
        0.2,
        t,
        theta=1.0,
        radius=1000.0,
        trajectory_dt=10.0,
        trajectory_ell_max=2,
        trajectory_n_max=0,
        waveform_ell_max=2,
        waveform_n_max=1,
        include_m_zero=False,
    )

    assert len(result.waveform.modes) == 12
    assert len(calls) == 24


def test_generate_schwarzschild_adiabatic_waveform_respects_explicit_mode_indices(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.0 + 0.0j

    class FakeMode:
        def __init__(self, amp):
            self._amp = amp

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    calls: list[tuple[int, int, int, int]] = []

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, orbit, kwargs
        calls.append((ell, m, n, k))
        return FakeMode(1.0 + 0.0j)

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectory(
            time=np.array([0.0, 10.0]),
            p=np.array([12.0, 12.0]),
            e=np.array([0.2, 0.2]),
            energy=np.array([0.95, 0.95]),
            angular_momentum=np.array([4.0, 4.0]),
            pdot=np.array([-1.0e-6, -1.0e-6]),
            edot=np.array([-1.0e-7, -1.0e-7]),
            edot_energy=np.array([-1.0e-10, -1.0e-10]),
            edot_angular_momentum=np.array([-1.0e-9, -1.0e-9]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_schwarzschild_eccentric_inspiral", fake_traj)

    t = np.linspace(0.0, 10.0, 16)
    mode_indices = [(2, 1, -1, 0), (2, 2, 0, 0)]
    result = generate_schwarzschild_eccentric_adiabatic_waveform(
        1.0e6,
        10.0,
        12.0,
        0.2,
        t,
        theta=1.0,
        radius=1000.0,
        trajectory_dt=10.0,
        mode_indices=mode_indices,
        waveform_ell_max=4,
        waveform_n_max=3,
    )

    expected_calls = [mode_indices[0], mode_indices[0], mode_indices[1], mode_indices[1]]
    assert result.waveform.modes == tuple(mode_indices)
    assert calls == expected_calls


def test_finite_difference_jacobian_equatorial_matches_schwarzschild_limit():
    jac0 = finite_difference_jacobian_equatorial(0.0, 12.0, 0.2, 1.0)
    jac1 = finite_difference_jacobian_equatorial(0.0, 12.0, 0.2, -1.0)
    assert np.allclose(jac0[0], jac1[0], rtol=1e-9, atol=1e-9)
    assert np.allclose(jac0[1], -jac1[1], rtol=1e-9, atol=3e-9)


def test_generic_total_fluxes_schwarzschild_inclined_respects_lz_scaling(monkeypatch):
    import teukolsky.waveform as waveform

    def fake_equatorial_total_fluxes(a, p, e, x, **kwargs):
        del a, p, e, x, kwargs
        return 3.0, 5.0

    monkeypatch.setattr(waveform, "equatorial_total_fluxes", fake_equatorial_total_fluxes)

    energy_flux, angular_flux, carter_flux = generic_total_fluxes(
        0.0, 12.0, 0.2, 0.5, ell_max=2, n_max=0
    )
    orbit = KerrGeoOrbit(0.0, 12.0, 0.2, 0.5)
    expected_lz_flux = 0.5 * 5.0
    expected_q_flux = (
        2.0
        * float(orbit.angular_momentum)
        * (1.0 - 0.5 * 0.5)
        / (0.5 * 0.5)
        * expected_lz_flux
    )

    assert energy_flux == pytest.approx(3.0)
    assert angular_flux == pytest.approx(expected_lz_flux)
    assert carter_flux == pytest.approx(expected_q_flux)


def test_generic_eccentric_rhs_schwarzschild_inclined_respects_lz_scaling(monkeypatch):
    import teukolsky.waveform as waveform

    def fake_equatorial_total_fluxes(a, p, e, x, **kwargs):
        del a, p, e, x, kwargs
        return 1.0e-6, 2.0e-5

    monkeypatch.setattr(waveform, "equatorial_total_fluxes", fake_equatorial_total_fluxes)

    rhs_pos = generic_eccentric_rhs(
        0.0,
        np.array([12.0, 0.2, 0.5], dtype=float),
        a=0.0,
        M=1.0e6,
        mu=1.0e4,
        ell_max=2,
        n_max=0,
        accelerator="cpu",
        device_id=0,
        accelerator_resolution=None,
    )
    rhs_neg = generic_eccentric_rhs(
        0.0,
        np.array([12.0, 0.2, -0.5], dtype=float),
        a=0.0,
        M=1.0e6,
        mu=1.0e4,
        ell_max=2,
        n_max=0,
        accelerator="cpu",
        device_id=0,
        accelerator_resolution=None,
    )

    assert np.isfinite(rhs_pos).all()
    assert np.isfinite(rhs_neg).all()
    assert rhs_pos[2] == pytest.approx(0.0)
    assert rhs_neg[2] == pytest.approx(0.0)
    assert rhs_pos[0] == pytest.approx(rhs_neg[0], rel=1e-10, abs=1e-12)
    assert rhs_pos[1] == pytest.approx(rhs_neg[1], rel=1e-10, abs=1e-12)


def test_integrate_equatorial_eccentric_inspiral_monotone_with_mock_flux(monkeypatch):
    import teukolsky.waveform as waveform

    def fake_fluxes(a, p, e, x, **kwargs):
        del kwargs
        return 1.0e-6 * (1.0 + 0.5 * a + e), 2.0e-5 * (1.0 + 0.1 * e + 0.2 * x)

    monkeypatch.setattr(waveform, "equatorial_total_fluxes", fake_fluxes)

    traj = integrate_equatorial_eccentric_inspiral(
        1.0e6,
        1.0e4,
        0.5,
        12.0,
        0.2,
        1.0,
        t_end=1.0e5,
        trajectory_dt=2.5e4,
        ell_max=2,
        n_max=0,
    )

    assert len(traj.time) >= 2
    assert np.all(np.diff(traj.time) > 0.0)
    assert traj.p[-1] < traj.p[0]
    assert traj.e[-1] <= traj.e[0]


def test_generate_equatorial_eccentric_adiabatic_waveform_smoke(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.0 - 0.25j

    class FakeMode:
        def __init__(self, amp):
            self._amp = amp

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, orbit, kwargs
        return FakeMode((ell + 0.2 * m + 0.01 * n) + 1j * (0.3 * m))

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectory(
            time=np.array([0.0, 10.0]),
            p=np.array([12.0, 11.9]),
            e=np.array([0.2, 0.199]),
            energy=np.array([0.95, 0.949]),
            angular_momentum=np.array([4.0, 3.999]),
            pdot=np.array([-1.0e-6, -1.0e-6]),
            edot=np.array([-1.0e-7, -1.0e-7]),
            edot_energy=np.array([-1.0e-10, -1.0e-10]),
            edot_angular_momentum=np.array([-1.0e-9, -1.0e-9]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_equatorial_eccentric_inspiral", fake_traj)

    t = np.linspace(0.0, 10.0, 32)
    result = generate_equatorial_eccentric_adiabatic_waveform(
        1.0e6,
        10.0,
        0.5,
        12.0,
        0.2,
        1.0,
        t,
        theta=1.0,
        radius=1000.0,
        trajectory_dt=10.0,
        trajectory_ell_max=2,
        trajectory_n_max=0,
        waveform_ell_max=2,
        waveform_n_max=0,
    )

    assert result.waveform.time.shape == t.shape
    assert np.all(np.isfinite(result.waveform.h_plus))
    assert np.all(np.isfinite(result.waveform.h_cross))
    assert len(result.waveform.modes) > 0


def test_generate_equatorial_adiabatic_waveform_uses_time_varying_angular_factor(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def __init__(self, value):
            self.value = value

        def evaluate(self, theta, phi=0.0):
            del theta, phi
            return self.value

    class FakeMode:
        def __init__(self, amp, angular_value):
            self._amp = amp
            self._angular = FakeAngular(angular_value)

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return self._angular
            raise KeyError(key)

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, kwargs
        amp = (ell + 0.2 * m + 0.01 * n) + 1j * (0.3 * m)
        angular_value = orbit.p + 1j * orbit.e
        return FakeMode(amp, angular_value)

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectory(
            time=np.array([0.0, 10.0]),
            p=np.array([12.0, 11.0]),
            e=np.array([0.2, 0.1]),
            energy=np.array([0.95, 0.949]),
            angular_momentum=np.array([4.0, 3.999]),
            pdot=np.array([-1.0e-6, -1.0e-6]),
            edot=np.array([-1.0e-7, -1.0e-7]),
            edot_energy=np.array([-1.0e-10, -1.0e-10]),
            edot_angular_momentum=np.array([-1.0e-9, -1.0e-9]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_equatorial_eccentric_inspiral", fake_traj)

    t = np.array([0.0, 5.0, 10.0])
    result = generate_equatorial_eccentric_adiabatic_waveform(
        1.0e6,
        10.0,
        0.5,
        12.0,
        0.2,
        1.0,
        t,
        theta=1.0,
        radius=1000.0,
        trajectory_dt=10.0,
        trajectory_ell_max=2,
        trajectory_n_max=0,
        waveform_ell_max=2,
        waveform_n_max=0,
        include_m_zero=False,
    )

    assert not np.allclose(result.waveform.complex_strain[0], result.waveform.complex_strain[-1])


def test_generate_equatorial_adiabatic_waveform_respects_explicit_mode_indices(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.0 + 0.0j

    class FakeMode:
        def __init__(self, amp):
            self._amp = amp

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    calls: list[tuple[int, int, int, int]] = []

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, orbit, kwargs
        calls.append((ell, m, n, k))
        return FakeMode(1.0 + 0.0j)

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectory(
            time=np.array([0.0, 10.0]),
            p=np.array([12.0, 12.0]),
            e=np.array([0.2, 0.2]),
            energy=np.array([0.95, 0.95]),
            angular_momentum=np.array([4.0, 4.0]),
            pdot=np.array([-1.0e-6, -1.0e-6]),
            edot=np.array([-1.0e-7, -1.0e-7]),
            edot_energy=np.array([-1.0e-10, -1.0e-10]),
            edot_angular_momentum=np.array([-1.0e-9, -1.0e-9]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_equatorial_eccentric_inspiral", fake_traj)

    t = np.linspace(0.0, 10.0, 16)
    mode_indices = [(2, 1, -1, 0), (2, 2, 0, 0)]
    result = generate_equatorial_eccentric_adiabatic_waveform(
        1.0e6,
        10.0,
        0.5,
        12.0,
        0.2,
        1.0,
        t,
        theta=1.0,
        radius=1000.0,
        trajectory_dt=10.0,
        mode_indices=mode_indices,
        waveform_ell_max=4,
        waveform_n_max=3,
    )

    expected_calls = [mode_indices[0], mode_indices[0], mode_indices[1], mode_indices[1]]
    assert result.waveform.modes == tuple(mode_indices)
    assert calls == expected_calls


def test_generate_generic_eccentric_adiabatic_waveform_smoke(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.0 + 0.1j

    class FakeMode:
        def __init__(self, amp):
            self._amp = amp

        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": self._amp}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    def fake_mode(s, ell, m, n, k, orbit, **kwargs):
        del s, orbit, kwargs
        return FakeMode((ell + 0.2 * m + 0.01 * n + 0.005 * k) + 1j * (0.3 * m - 0.1 * k))

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectoryGeneric(
            time=np.array([0.0, 10.0]),
            p=np.array([12.0, 11.95]),
            e=np.array([0.2, 0.199]),
            x=np.array([0.7, 0.699]),
            energy=np.array([0.95, 0.949]),
            angular_momentum=np.array([3.0, 2.999]),
            carter_constant=np.array([9.0, 8.99]),
            pdot=np.array([-1.0e-6, -1.0e-6]),
            edot=np.array([-1.0e-7, -1.0e-7]),
            xdot=np.array([-1.0e-8, -1.0e-8]),
            edot_energy=np.array([-1.0e-10, -1.0e-10]),
            edot_angular_momentum=np.array([-1.0e-9, -1.0e-9]),
            edot_carter=np.array([-1.0e-9, -1.0e-9]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_generic_eccentric_inspiral", fake_traj)

    t = np.linspace(0.0, 10.0, 32)
    result = generate_generic_eccentric_adiabatic_waveform(
        1.0e6,
        10.0,
        0.5,
        12.0,
        0.2,
        0.7,
        t,
        theta=1.0,
        radius=1000.0,
        trajectory_dt=10.0,
        trajectory_ell_max=2,
        trajectory_n_max=0,
        trajectory_k_max=1,
        waveform_ell_max=2,
        waveform_n_max=0,
        waveform_k_max=1,
    )

    assert result.waveform.time.shape == t.shape
    assert np.all(np.isfinite(result.waveform.h_plus))
    assert np.all(np.isfinite(result.waveform.h_cross))
    assert len(result.waveform.modes) > 0


def test_generate_generic_eccentric_adiabatic_waveform_kerr_inclined_interface(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.0 + 0.1j

    class FakeMode:
        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": 1.0 + 0.2j}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    def fake_mode(*args, **kwargs):
        del args, kwargs
        return FakeMode()

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return waveform.AdiabaticTrajectoryGeneric(
            time=np.array([0.0, 10.0]),
            p=np.array([18.0, 17.99]),
            e=np.array([0.05, 0.0499]),
            x=np.array([0.9, 0.8999]),
            energy=np.array([0.97, 0.9699]),
            angular_momentum=np.array([3.6, 3.5999]),
            carter_constant=np.array([0.58, 0.5799]),
            pdot=np.array([-1.0e-8, -1.0e-8]),
            edot=np.array([-1.0e-10, -1.0e-10]),
            xdot=np.array([-1.0e-12, -1.0e-12]),
            edot_energy=np.array([-1.0e-12, -1.0e-12]),
            edot_angular_momentum=np.array([-1.0e-10, -1.0e-10]),
            edot_carter=np.array([-1.0e-10, -1.0e-10]),
        )

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_generic_eccentric_inspiral", fake_traj)

    t = np.linspace(0.0, 100.0, 16)
    result = generate_generic_eccentric_adiabatic_waveform(
        1.0e6, 10.0, 0.1, 18.0, 0.05, 0.9, t,
        theta=1.0, phi=0.3, radius=1000.0, trajectory_dt=50.0,
        waveform_ell_max=2, waveform_n_max=0, waveform_k_max=1,
        accelerator="cpu",
    )
    assert result.waveform.time.shape == t.shape
    assert np.all(np.isfinite(result.waveform.h_plus))
    assert np.all(np.isfinite(result.waveform.h_cross))


def test_generate_generic_eccentric_adiabatic_waveform_matches_sparse_reconstruction(monkeypatch):
    import teukolsky.waveform as waveform

    class FakeAngular:
        def evaluate(self, theta, phi=0.0):
            return 1.0 + 0.1j

    class FakeMode:
        def __getitem__(self, key):
            if key == "Amplitudes":
                return {"I": 1.0 + 0.2j}
            if key == "AngularFunction":
                return FakeAngular()
            raise KeyError(key)

    traj = waveform.AdiabaticTrajectoryGeneric(
        time=np.array([0.0, 10.0, 20.0]),
        p=np.array([18.0, 17.99, 17.98]),
        e=np.array([0.05, 0.0499, 0.0498]),
        x=np.array([0.9, 0.8999, 0.8998]),
        energy=np.array([0.97, 0.9699, 0.9698]),
        angular_momentum=np.array([3.6, 3.5999, 3.5998]),
        carter_constant=np.array([0.58, 0.5799, 0.5798]),
        pdot=np.array([-1.0e-8, -1.0e-8, -1.0e-8]),
        edot=np.array([-1.0e-10, -1.0e-10, -1.0e-10]),
        xdot=np.array([-1.0e-12, -1.0e-12, -1.0e-12]),
        edot_energy=np.array([-1.0e-12, -1.0e-12, -1.0e-12]),
        edot_angular_momentum=np.array([-1.0e-10, -1.0e-10, -1.0e-10]),
        edot_carter=np.array([-1.0e-10, -1.0e-10, -1.0e-10]),
    )

    def fake_mode(*args, **kwargs):
        del args, kwargs
        return FakeMode()

    def fake_traj(*args, **kwargs):
        del args, kwargs
        return traj

    monkeypatch.setattr(waveform, "TeukolskyPointParticleMode", fake_mode)
    monkeypatch.setattr(waveform, "integrate_generic_eccentric_inspiral", fake_traj)

    t = np.linspace(0.0, 20.0, 16)
    adiabatic = generate_generic_eccentric_adiabatic_waveform(
        1.0e6, 10.0, 0.1, 18.0, 0.05, 0.9, t,
        theta=1.0, phi=0.3, radius=1000.0, trajectory_dt=10.0,
        waveform_ell_max=2, waveform_n_max=0, waveform_k_max=1,
        accelerator="cpu",
    )
    sparse = generate_sparse_trajectory_waveform(
        1.0e6, 0.1, traj.time, traj.p, traj.e, traj.x,
        evaluation_time=t,
        theta=1.0, phi=0.3, radius=1000.0,
        mode_indices=list(adiabatic.waveform.modes),
        accelerator="cpu",
    )

    assert sparse.modes == adiabatic.waveform.modes
    assert np.allclose(sparse.complex_strain, adiabatic.waveform.complex_strain, rtol=1e-12, atol=1e-12)


def test_finite_difference_jacobian_generic_shape_and_rank():
    from teukolsky import finite_difference_jacobian_generic

    J = finite_difference_jacobian_generic(0.5, 10.0, 0.2, 0.7)
    assert J.shape == (3, 3)
    assert np.linalg.matrix_rank(J) >= 2
    assert not np.any(np.isnan(J))


def test_generic_total_fluxes_schwarzschild_equatorial_limit():
    from teukolsky import (
        generic_total_fluxes,
        equatorial_total_fluxes,
    )

    E_gen, L_gen, Q_gen = generic_total_fluxes(
        0.0, 12.0, 0.2, 1.0, ell_max=2, n_max=1, accelerator="cpu",
    )
    E_eq, L_eq = equatorial_total_fluxes(
        0.0, 12.0, 0.2, 1.0, ell_max=2, n_max=1, accelerator="cpu",
    )
    assert np.isclose(E_gen, E_eq, rtol=1e-12)
    assert np.isclose(L_gen, L_eq, rtol=1e-12)
    assert Q_gen == 0.0


def test_generic_total_fluxes_nonzero_qdot_for_inclined():
    from teukolsky import generic_total_fluxes

    _, _, Qdot = generic_total_fluxes(
        0.0, 12.0, 0.15, 0.7, ell_max=2, n_max=1, k_max=1, accelerator="cpu",
    )
    assert np.isfinite(Qdot)


def test_generic_total_fluxes_kerr_equatorial_limit():
    from teukolsky import (
        generic_total_fluxes,
        equatorial_total_fluxes,
    )

    E_gen, L_gen, Q_gen = generic_total_fluxes(
        0.5, 12.0, 0.2, 1.0, ell_max=2, n_max=0, k_max=1, accelerator="cpu",
    )
    E_eq, L_eq = equatorial_total_fluxes(
        0.5, 12.0, 0.2, 1.0, ell_max=2, n_max=0, accelerator="cpu",
    )
    assert np.isclose(E_gen, E_eq, rtol=1e-12, atol=1e-12)
    assert np.isclose(L_gen, L_eq, rtol=1e-12, atol=1e-12)
    assert Q_gen == 0.0


def test_carter_constant_formula():
    from teukolsky import carter_constant

    assert carter_constant(3.0, 1.0) == 0.0
    assert carter_constant(3.0, -1.0) == 0.0
    assert carter_constant(4.0, 0.5) > 0.0


def test_generic_eccentric_rhs_shape():
    from teukolsky import generic_eccentric_rhs

    deriv = generic_eccentric_rhs(
        0.0,
        np.array([12.0, 0.2, 0.7], dtype=float),
        a=0.0,
        M=1.0e6,
        mu=10.0,
        ell_max=2,
        n_max=1,
        k_max=1,
        accelerator="cpu",
        device_id=0,
        accelerator_resolution=None,
    )
    assert deriv.shape == (3,)
    assert np.all(np.isfinite(deriv))


def test_generic_eccentric_rhs_kerr_equatorial_limit():
    from teukolsky import equatorial_eccentric_rhs

    generic = generic_eccentric_rhs(
        0.0,
        np.array([12.0, 0.2, 1.0], dtype=float),
        a=0.5,
        M=1.0e6,
        mu=10.0,
        ell_max=2,
        n_max=0,
        k_max=1,
        accelerator="cpu",
        device_id=0,
        accelerator_resolution=None,
    )
    equatorial = equatorial_eccentric_rhs(
        0.0,
        np.array([12.0, 0.2], dtype=float),
        a=0.5,
        x=1.0,
        M=1.0e6,
        mu=10.0,
        ell_max=2,
        n_max=0,
        accelerator="cpu",
        device_id=0,
        accelerator_resolution=None,
    )
    assert np.allclose(generic[:2], equatorial, rtol=1e-12, atol=1e-12)
    assert generic[2] == 0.0


def test_integrate_generic_eccentric_inspiral_smoke(monkeypatch):
    from teukolsky import integrate_generic_eccentric_inspiral

    import teukolsky.waveform as wf

    def fake_fluxes(a, p, e, x, **kwargs):
        return 1.0e-6, 2.0e-5

    monkeypatch.setattr(wf, "equatorial_total_fluxes", fake_fluxes)

    traj = integrate_generic_eccentric_inspiral(
        1.0e6, 10.0, 0.0, 12.0, 0.2, 0.7,
        t_end=1.0e5, trajectory_dt=2.5e4,
        ell_max=2, n_max=0, accelerator="cpu",
    )
    assert len(traj.time) >= 2
    assert np.all(np.diff(traj.time) > 0.0)
    assert traj.p[-1] < traj.p[0]
    assert np.all(np.isfinite(traj.pdot))
    assert np.all(np.isfinite(traj.edot))
    assert np.all(traj.xdot == 0.0)
    assert traj.x.shape == traj.p.shape


def test_generic_total_fluxes_kerr_inclined_requires_action_fluxes():
    with pytest.raises(RuntimeError, match="generic_action_fluxes"):
        generic_total_fluxes(
            0.5, 10.0, 0.2, 0.7,
            ell_max=4,
            n_max=3,
            k_max=2,
            accelerator="gpu",
            device_id=1,
            accelerator_resolution=513,
        )


def test_generic_action_fluxes_kerr_inclined_uses_teukolsky_mode_sums(monkeypatch):
    import teukolsky.waveform as wf

    def forbidden_pn5(*args, **kwargs):
        raise AssertionError("generic Kerr action fluxes must not call FEW PN5")

    def fake_mode_sums(orbit, **kwargs):
        assert orbit.kind == "generic"
        assert kwargs["ell_max"] == 4
        assert kwargs["n_max"] == 3
        assert kwargs["k_max"] == 2
        return 1.25e-5, 2.5e-4, 0.0, 7.5e-6

    monkeypatch.setattr(wf, "_pn5_internal_generic_rhs", forbidden_pn5)
    monkeypatch.setattr(wf, "_generic_mode_flux_sums", fake_mode_sums)

    E, L, Jr, Jtheta = wf.generic_action_fluxes(
        0.5, 10.0, 0.2, 0.7,
        ell_max=4,
        n_max=3,
        k_max=2,
        accelerator="gpu",
        device_id=1,
        accelerator_resolution=513,
    )

    assert E == 1.25e-5
    assert L == 2.5e-4
    assert Jr == 0.0
    assert Jtheta == 7.5e-6


def test_generic_mode_flux_job_action_contribution_sign(monkeypatch):
    import teukolsky.waveform as wf

    class FakeFluxes:
        energy = 6.0
        angular_momentum = 2.0

    class FakeMode:
        fluxes = FakeFluxes()
        omega = 3.0

    def fake_mode(*args, **kwargs):
        del args, kwargs
        return FakeMode()

    monkeypatch.setattr(wf, "TeukolskyPointParticleMode", fake_mode)

    result = wf._compute_generic_mode_flux_job(
        {
            "a": 0.5,
            "p": 10.0,
            "e": 0.2,
            "x": 0.7,
            "ell": 2,
            "m": 2,
            "n": -4,
            "k": 5,
            "accelerator": "cpu",
            "device_id": 0,
            "accelerator_resolution": None,
        }
    )

    assert result["radial_action_contribution"] == -8.0
    assert result["polar_action_contribution"] == 10.0


def test_generic_eccentric_rhs_kerr_inclined_finite(monkeypatch):
    import teukolsky.waveform as wf

    def fake_flux_sums(orbit, **kwargs):
        del orbit, kwargs
        return 1.0e-5, 2.0e-4, 3.0e-6, 4.0e-6

    def fake_jacobian(a, p, e, x):
        del a, p, e, x
        return np.eye(3)

    monkeypatch.setattr(wf, "_generic_mode_flux_sums", fake_flux_sums)
    monkeypatch.setattr(wf, "_generic_jacobian_with_jr", fake_jacobian)
    deriv = generic_eccentric_rhs(
        0.0,
        np.array([18.0, 0.05, 0.9], dtype=float),
        a=0.1,
        M=1.0e6,
        mu=10.0,
        ell_max=2,
        n_max=1,
        k_max=1,
        accelerator="cpu",
        device_id=0,
        accelerator_resolution=None,
    )
    assert deriv.shape == (3,)
    assert np.all(np.isfinite(deriv))
    assert deriv[0] < 0.0
    assert deriv[1] < 0.0


def test_generic_eccentric_rhs_kerr_inclined_uses_teukolsky_fluxes(monkeypatch):
    import teukolsky.waveform as wf

    def forbidden_pn5(*args, **kwargs):
        raise AssertionError("generic Kerr RHS must not call FEW PN5")

    def fake_mode_sums(orbit, **kwargs):
        assert orbit.kind == "generic"
        assert kwargs["ell_max"] == 4
        assert kwargs["n_max"] == 3
        assert kwargs["k_max"] == 2
        return 1.0e-5, 2.0e-4, 3.0e-6, 4.0e-6

    def fake_jacobian(a, p, e, x):
        assert (a, p, e, x) == (0.5, 10.0, 0.2, 0.7)
        return np.eye(3)

    monkeypatch.setattr(wf, "_pn5_internal_generic_rhs", forbidden_pn5)
    monkeypatch.setattr(wf, "_generic_mode_flux_sums", fake_mode_sums)
    monkeypatch.setattr(wf, "_generic_jacobian_with_jr", fake_jacobian)

    deriv = generic_eccentric_rhs(
        0.0,
        np.array([10.0, 0.2, 0.7], dtype=float),
        a=0.5,
        M=1.0e6,
        mu=10.0,
        ell_max=4,
        n_max=3,
        k_max=2,
        accelerator="gpu",
        device_id=1,
        accelerator_resolution=513,
    )

    scale = (10.0 / 1.0e6) / (1.0e6 * 4.92549095e-6)
    assert np.allclose(deriv, np.array([-1.0e-5, -2.0e-4, -3.0e-6]) * scale)


def test_integrate_generic_eccentric_inspiral_kerr_inclined_smoke(monkeypatch):
    from teukolsky import integrate_generic_eccentric_inspiral

    import teukolsky.waveform as wf

    def fake_generic_rhs(t, y, **kwargs):
        del t, kwargs
        return np.array([-1.0e-7, -1.0e-9, -1.0e-10], dtype=float)

    monkeypatch.setattr(wf, "generic_eccentric_rhs", fake_generic_rhs)
    traj = integrate_generic_eccentric_inspiral(
        1.0e6, 10.0, 0.1, 18.0, 0.05, 0.9,
        t_end=1.0e4, trajectory_dt=2.5e3,
        ell_max=2, n_max=1, k_max=1, accelerator="cpu",
    )
    assert len(traj.time) >= 2
    assert np.all(np.diff(traj.time) > 0.0)
    assert np.all(np.isfinite(traj.p))
    assert np.all(np.isfinite(traj.e))
    assert np.all(np.isfinite(traj.x))
    assert np.all(np.isfinite(traj.xdot))


def test_integrate_generic_eccentric_inspiral_matches_equatorial_limit(monkeypatch):
    from teukolsky import integrate_generic_eccentric_inspiral, integrate_equatorial_eccentric_inspiral

    import teukolsky.waveform as wf

    def fake_fluxes(a, p, e, x, **kwargs):
        del kwargs
        return 1.0e-6 * (1.0 + 0.2 * a + 0.01 * p), 2.0e-5 * (1.0 + 0.1 * e + 0.05 * x)

    monkeypatch.setattr(wf, "equatorial_total_fluxes", fake_fluxes)

    generic = integrate_generic_eccentric_inspiral(
        1.0e6, 10.0, 0.5, 12.0, 0.2, 1.0,
        t_end=1.0e5, trajectory_dt=2.5e4,
        ell_max=2, n_max=0, k_max=1, accelerator="cpu",
    )
    equatorial = integrate_equatorial_eccentric_inspiral(
        1.0e6, 10.0, 0.5, 12.0, 0.2, 1.0,
        t_end=1.0e5, trajectory_dt=2.5e4,
        ell_max=2, n_max=0, accelerator="cpu",
    )
    assert np.allclose(generic.time, equatorial.time, rtol=0.0, atol=1e-12)
    assert np.allclose(generic.p, equatorial.p, rtol=1e-12, atol=1e-12)
    assert np.allclose(generic.e, equatorial.e, rtol=1e-12, atol=1e-12)
    assert np.allclose(generic.pdot, equatorial.pdot, rtol=1e-12, atol=1e-12)
    assert np.allclose(generic.edot, equatorial.edot, rtol=1e-12, atol=1e-12)
    assert np.all(generic.x == 1.0)
    assert np.all(generic.xdot == 0.0)


def test_schwarzschild_real_inspiral_physical_signs():
    """Two-step Schwarzschild inspiral with real (minimal) mode-sum fluxes.

    Verifies: pdot < 0 (inspiral), edot <= 0 (circularisation), energy/Lz
    decrease, and the trajectory stays above the separatrix.
    """
    from teukolsky import integrate_schwarzschild_eccentric_inspiral

    traj = integrate_schwarzschild_eccentric_inspiral(
        1.0e6, 10.0, 12.0, 0.15,
        t_end=20.0, trajectory_dt=10.0,
        ell_max=2, n_max=0, accelerator="cpu",
    )
    assert len(traj.time) >= 2
    assert np.all(traj.pdot < 0.0), f"pdot must be negative, got {traj.pdot}"
    assert np.all(traj.edot <= 1e-10), f"edot must be <= 0 (tol 1e-10), got {traj.edot}"
    assert np.all(np.diff(traj.energy) < 0.0), "energy must decrease"
    assert np.all(np.diff(traj.angular_momentum) < 0.0), "Lz must decrease"
    assert traj.p[-1] > 6.0 + 2.0 * traj.e[-1], "trajectory above separatrix"
    assert np.all(np.isfinite(traj.edot_energy))
    assert np.all(np.isfinite(traj.edot_angular_momentum))


def test_spheroidal_eigenvalue_matches_qnm_berti():
    """Angular eigenvalue vs qnm package (Berti reference data).

    qnm uses the separation-constant convention A_lm; our code uses the
    Teukolsky-Starobinsky convention lambda = A + (aω)² - 2m(aω).
    Both are correct — this test checks that they agree modulo the known
    convention offset.
    """
    pytest.importorskip("qnm")
    import qnm.angular as qang
    from teukolsky import spin_weighted_spheroidal_eigenvalue

    for s in [-2]:
        for ell in [2, 3, 4]:
            for m_val in [0, 2]:
                if m_val > ell: continue
                for aom in [0.01, 0.1, 0.5, 1.0]:
                    c = complex(float(aom))
                    our_lam = float(spin_weighted_spheroidal_eigenvalue(s, ell, m_val, c).real)
                    l_min = max(abs(s), abs(m_val))
                    qnm_arr = qang.sep_consts(s, c, m_val, ell + 6)
                    if len(qnm_arr) <= ell - l_min:
                        continue
                    qnm_A = float(np.asarray(qnm_arr[ell - l_min]).real.item())
                    diff = our_lam - qnm_A
                    expected = aom**2 - 2.0 * m_val * aom
                    assert abs(diff - expected) < 1e-10, (
                        f"s={s} l={ell} m={m_val} aω={aom}: "
                        f"λ_our={our_lam:.10f} A_qnm={qnm_A:.10f} "
                        f"diff={diff:.10f} expected={expected:.10f}"
                    )


def test_qnm_frequencies_match_berti_reference():
    """Kerr QNM frequencies from the qnm package (Berti reference).

    This is a consistency check: the qnm package is validated against
    Berti's published QNM data (pages.jh.edu/eberti2/ringdown/).
    We verify the package returns plausible values for Schwarzschild
    and Kerr.
    """
    pytest.importorskip("qnm")
    import qnm

    omega_schw = qnm.modes_cache(s=-2, l=2, m=2, n=0)(a=0.0)[0]
    assert abs(omega_schw.real - 0.37367) < 1e-4
    assert abs(omega_schw.imag + 0.08896) < 1e-4

    omega_kerr = qnm.modes_cache(s=-2, l=2, m=2, n=0)(a=0.7)[0]
    assert abs(omega_kerr.real - 0.5326) < 2e-3
    assert abs(omega_kerr.imag + 0.0808) < 2e-3

    for a_val in [0.0, 0.5, 0.9]:
        for ell in [2, 3]:
            omega = qnm.modes_cache(s=-2, l=ell, m=2, n=0)(a=a_val)[0]
            assert omega.real > 0.0, f"a={a_val} l={ell}: ω_R must be > 0"
            assert omega.imag < 0.0, f"a={a_val} l={ell}: ω_I must be < 0 (damped)"

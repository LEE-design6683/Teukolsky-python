from teukolsky.geodesics import circular_orbit, equatorial_eccentric_orbit, generic_orbit, spherical_orbit


def test_circular_orbit_has_expected_frequency() -> None:
    orbit = circular_orbit(0.9, 10.0)
    assert orbit.kind == "circular-equatorial"
    assert abs(orbit.omega_phi - (1.0 / (10.0 ** 1.5 + 0.9))) < 1e-12


def test_eccentric_equatorial_orbit_has_finite_frequencies() -> None:
    orbit = equatorial_eccentric_orbit(0.5, 10.0, 0.2)
    assert orbit.kind == "eccentric-equatorial"
    assert orbit.omega_r > 0.0
    assert orbit.omega_phi > orbit.omega_r
    assert orbit.radial_phase_function is not None
    assert orbit.radial_velocity_function is not None


def test_spherical_orbit_has_finite_frequencies() -> None:
    orbit = spherical_orbit(0.5, 8.0, 0.7)
    assert orbit.kind == "spherical"
    assert orbit.omega_theta > 0.0
    assert orbit.omega_phi > orbit.omega_theta
    assert orbit.theta_phase_function is not None
    assert orbit.theta_velocity_function is not None


def test_generic_orbit_has_finite_frequencies() -> None:
    orbit = generic_orbit(0.5, 10.0, 0.2, 0.7)
    assert orbit.kind == "generic"
    assert orbit.omega_r > 0.0
    assert orbit.omega_theta > 0.0
    assert orbit.omega_phi > orbit.omega_theta
    assert orbit.radial_phase_function is not None
    assert orbit.theta_phase_function is not None


def test_orbit_key_access_matches_expected_semantics() -> None:
    orbit = generic_orbit(0.5, 10.0, 0.2, 0.7)
    assert orbit["a"] == 0.5
    assert orbit["p"] == 10.0
    assert orbit["e"] == 0.2
    assert orbit["Inclination"] == 0.7
    assert orbit["Parametrization"] == "Mino"
    assert orbit["Type"] == ("Bound", "Generic")
    assert orbit["Frequencies"]["Omega_r"] == orbit.omega_r
    assert orbit["Frequencies"]["Upsilon_t"] == orbit.upsilon_t
    assert callable(orbit["TrajectoryDeltas"]["Delta_tr"])
    assert callable(orbit["TrajectoryDeltas"]["Delta_ttheta"])
    assert orbit["InitialPhases"] == {"qt0": 0.0, "qr0": 0.0, "qtheta0": 0.0, "qphi0": 0.0}
    assert orbit.keys() == [
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

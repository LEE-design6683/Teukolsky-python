from teukolsky import KerrGeoOrbit, TeukolskyPointParticleMode


def test_homepage_style_example_runs() -> None:
    orbit = KerrGeoOrbit(0.9, 10.0, 0.0, 1.0)
    mode = TeukolskyPointParticleMode(-2, 2, 2, 0, 0, orbit)
    assert abs(mode.fluxes.energy_infinity - 0.0000222730005511805) < 1e-6

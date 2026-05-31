from teukolsky.angular import spin_weighted_spheroidal_eigenvalue


def test_eigenvalue_matches_reference_case() -> None:
    value = spin_weighted_spheroidal_eigenvalue(2, 2, 2, 0.05)
    assert abs(value - (-0.33267928615316333)) < 1e-10

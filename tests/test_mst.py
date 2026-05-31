import math

from teukolsky import RenormalizedAngularMomentum
from teukolsky.angular.eigen import spin_weighted_spheroidal_eigenvalue
from teukolsky.mst import renormalized_angular_momentum


def test_zero_frequency_returns_ell() -> None:
    assert renormalized_angular_momentum(-2, 2, 0, 0.0, 0.0, 4.0) == 2
    assert renormalized_angular_momentum(-2, 2, 0, 0.0, 0.0, 4.0, method="Monodromy") == 2
    assert renormalized_angular_momentum(-2, 2, 0, 0.0, 0.0, 4.0, method=("Monodromy", {"nmax": 25})) == 2
    assert renormalized_angular_momentum(0, 2, 0, 0.0, 0.0, 4.0) == 2


def test_ell_below_abs_s_returns_zero() -> None:
    assert renormalized_angular_momentum(-2, 1, 0, 0.0, 0.0, 0.0) == 0.0
    assert renormalized_angular_momentum(-2, 1, 0, 0.5, 0.1, 0.0) == 0.0


def test_series_method_nonzero_frequency_s_minus_2() -> None:
    lam = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.5 * 0.1)
    nu = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method="series")
    assert math.isfinite(abs(nu))
    assert abs(nu - 2.0) < 0.05


def test_eigenvalue_can_be_omitted() -> None:
    lam = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.5 * 0.1)
    explicit = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method="monodromy")
    inferred = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, method="monodromy")
    assert abs(explicit - inferred) < 1e-12


def test_monodromy_method_s_minus_2() -> None:
    lam = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.5 * 0.1)
    nu = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method="monodromy")
    assert math.isfinite(abs(nu))
    assert abs(nu - 2.0) < 0.05


def test_series_and_monodromy_agree_all_spin_weights() -> None:
    a, omega = 0.5, 0.1
    for s in [-2, -1, 0, 1, 2]:
        ell, m = 2, 2
        lam = spin_weighted_spheroidal_eigenvalue(s, ell, m, a * omega)
        nu_s = renormalized_angular_momentum(s, ell, m, a, omega, lam, method="series")
        nu_m = renormalized_angular_momentum(s, ell, m, a, omega, lam, method="monodromy")
        assert abs(nu_s - nu_m) < 5e-4, f"s={s}: series={nu_s}, monodromy={nu_m}, diff={abs(nu_s-nu_m)}"
        assert abs(nu_s - ell) < 0.05, f"s={s}: nu={nu_s} too far from ell={ell}"
        assert abs(nu_m - ell) < 0.05, f"s={s}: nu={nu_m} too far from ell={ell}"


def test_findroot_and_monodromy_agree_all_spin_weights() -> None:
    """Verify the new confluent-Heun monodromy and continued-fraction findroot agree.

    The two methods are mathematically independent (monodromy uses confluent Heun
    recurrence; findroot uses continued-fraction root-finding), so they serve as
    cross-checks for each other.  Tolerance is set to 2e-6 to allow for differing
    numerical-error accumulation paths at double precision.
    """
    a, omega = 0.5, 0.1
    for s in [-2, -1, 0, 1, 2]:
        ell, m = 2, 2
        lam = spin_weighted_spheroidal_eigenvalue(s, ell, m, a * omega)
        nu_f = renormalized_angular_momentum(s, ell, m, a, omega, lam, method="findroot")
        nu_m = renormalized_angular_momentum(s, ell, m, a, omega, lam, method="monodromy")
        assert abs(nu_f - nu_m) < 2e-6, (
            f"s={s}: findroot={nu_f}, monodromy={nu_m}, diff={abs(nu_f - nu_m)}"
        )
        assert abs(nu_m - ell) < 0.05, (
            f"s={s}: monodromy nu={nu_m} too far from ell={ell}"
        )


def test_confluent_heun_monodromy_all_spin_weights() -> None:
    """Direct test of _confluent_heun_monodromy with correct eigenvalues.

    a=0.5, omega=0.1, l=2, m=2 for all spin weights s=-2,-1,0,1,2.
    """
    from teukolsky.mst.renormalized import _confluent_heun_monodromy

    a, omega = 0.5, 0.1
    ell, m_val = 2, 2
    for s in [-2, -1, 0, 1, 2]:
        lam = spin_weighted_spheroidal_eigenvalue(s, ell, m_val, a * omega)
        nu = _confluent_heun_monodromy(s, ell, m_val, a, omega, lam)
        assert math.isfinite(abs(nu)), f"s={s}: nu={nu} is not finite"
        assert abs(nu - ell) < 0.05, f"s={s}: nu={nu} too far from ell={ell}"


def test_confluent_heun_monodromy_explicit_nmax() -> None:
    """Test _confluent_heun_monodromy with explicit nmax parameter."""
    from teukolsky.mst.renormalized import _confluent_heun_monodromy

    s, ell, m_val, a, omega = -2, 2, 2, 0.5, 0.1
    lam = spin_weighted_spheroidal_eigenvalue(s, ell, m_val, a * omega)

    nu_auto = _confluent_heun_monodromy(s, ell, m_val, a, omega, lam)
    nu_n50 = _confluent_heun_monodromy(s, ell, m_val, a, omega, lam, nmax=50)

    assert math.isfinite(abs(nu_auto)), f"auto nmax gave non-finite nu={nu_auto}"
    assert math.isfinite(abs(nu_n50)), f"nmax=50 gave non-finite nu={nu_n50}"
    assert abs(nu_auto - nu_n50) < 1e-8, (
        f"auto={nu_auto}, nmax=50={nu_n50}, diff={abs(nu_auto - nu_n50)}"
    )


def test_public_method_tuple_forms_are_supported() -> None:
    lam = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.5 * 0.1)
    nu_default = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method="monodromy")
    nu_tuple = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method=("Monodromy", {"nmax": 50}))
    assert abs(nu_default - nu_tuple) < 1e-8


def test_findroot_initial_guess_option_is_supported() -> None:
    lam = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.5 * 0.1)
    nu_guess = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method=("FindRoot", {"InitialGuess": 2.0}))
    nu_plain = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method="findroot")
    assert abs(nu_guess - nu_plain) < 5e-6


def test_public_renormalized_angular_momentum_alias_matches_function() -> None:
    lam = spin_weighted_spheroidal_eigenvalue(-2, 2, 2, 0.5 * 0.1)
    nu_alias = RenormalizedAngularMomentum(-2, 2, 2, 0.5, 0.1, lam, method=("Monodromy", {"nmax": 25}))
    nu_func = renormalized_angular_momentum(-2, 2, 2, 0.5, 0.1, lam, method=("Monodromy", {"nmax": 25}))
    assert abs(nu_alias - nu_func) < 1e-12

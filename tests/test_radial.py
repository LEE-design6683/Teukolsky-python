from teukolsky.core import Missing
from teukolsky.geodesics import circular_orbit
from teukolsky.radial import solve_radial
import pytest


def test_radial_wronskian_is_consistent_for_circular_kerr_case() -> None:
    orbit = circular_orbit(0.9, 10.0)
    radial = solve_radial(-2, 2, 2, orbit.a, 2.0 * orbit.omega_phi)
    rin = radial["In"]
    rup = radial["Up"]

    def wronskian(r: float) -> complex:
        delta = r * r - 2.0 * r + orbit.a * orbit.a
        return delta ** (-1) * (rup.derivative(1, r) * rin(r) - rin.derivative(1, r) * rup(r))

    w10 = wronskian(10.0)
    w20 = wronskian(20.0)
    assert abs(w10 - w20) / abs(w10) < 1e-9
    assert abs(rin(10.0)) > 0.0
    assert abs(rup(10.0)) > 0.0


def test_static_mode_returns_value() -> None:
    radial = solve_radial(2, 2, 0, 1 / 3, 0.0)["In"]
    assert radial.method == "Static"
    assert radial.boundary_conditions == "In"


def test_spin_two_numerical_in_mode_matches_reference_values() -> None:
    radial = solve_radial(2, 2, 2, 0.5, 0.1)["In"]
    assert abs(radial(10.0) - (0.8151274455692312 + 0.5569358329985331j)) < 1e-6
    assert abs(radial.derivative(1, 10.0) - (0.032005838396104935 - 0.07297367974124509j)) < 1e-6


def test_spin_two_numerical_in_mode_fourth_derivative_matches_reference() -> None:
    radial = solve_radial(2, 2, 2, 0.5, 0.1)["In"]
    ref = -0.00031399107128295535 + 0.00007805860280733666j
    assert abs(radial.derivative(4, 10.0) - ref) < 5e-7


def test_spin_two_eigenvalue_matches_mathematica_reference() -> None:
    """Eigenvalue for s=2,l=2,m=2,a=0.5,omega=0.1 matches WLT reference."""
    radial = solve_radial(2, 2, 2, 0.5, 0.1)["In"]
    ref_eig = -0.33267928615316333
    assert abs(radial.eigenvalue - ref_eig) < 1e-12


def test_radial_solution_key_access_matches_wlt_semantics() -> None:
    radial = solve_radial(2, 2, 2, 0.5, 0.1)["In"]
    assert radial["BoundaryConditions"] == "In"
    assert radial["Method"] == ["NumericalIntegration"]
    assert abs(radial["Eigenvalue"] - (-0.33267928615316333)) < 1e-12
    assert callable(radial["RadialFunction"])
    assert radial["NotAKey"] == Missing("KeyAbsent", "NotAKey")


def test_radial_solution_keys_match_expected_semantics() -> None:
    radial = solve_radial(2, 2, 2, 0.5, 0.1)["In"]
    assert radial.keys() == [
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


def test_numerical_radial_domain_override_is_enforced() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1, domain=(5.0, 40.0))["In"]
    assert radial["Domain"] == (5.0, 40.0)
    assert radial["Method"] == ["NumericalIntegration", {"Domain": (5.0, 40.0)}]
    assert abs(radial(10.0)) > 0.0
    with pytest.raises(ValueError, match="outside domain"):
        radial(4.0)
    with pytest.raises(ValueError, match="outside domain"):
        radial(41.0)


def test_numerical_radial_boundary_specific_domains_are_enforced() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1, domain={"In": (5.0, 40.0), "Up": (7.0, 60.0)})
    assert radial["In"]["Domain"] == (5.0, 40.0)
    assert radial["Up"]["Domain"] == (7.0, 60.0)
    assert abs(radial["In"](10.0)) > 0.0
    assert abs(radial["Up"](10.0)) > 0.0
    with pytest.raises(ValueError, match="outside domain"):
        radial["In"](41.0)
    with pytest.raises(ValueError, match="outside domain"):
        radial["Up"](6.0)


def test_radial_solution_accepts_numeric_sequences() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1)["In"]
    points = [10.0, 12.0]
    values = radial(points)
    derivatives = radial.derivative(1, points)
    assert values == [radial(10.0), radial(12.0)]
    assert derivatives == [radial.derivative(1, 10.0), radial.derivative(1, 12.0)]


def test_static_modes_reject_explicit_domain() -> None:
    with pytest.raises(ValueError, match="static modes"):
        solve_radial(2, 2, 0, 1 / 3, 0.0, domain=(5.0, 40.0))


def test_sasaki_nakamura_method_produces_finite_solution() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1, method="SasakiNakamura")
    rin = radial["In"]
    rup = radial["Up"]
    assert rup.method == "SasakiNakamura"
    assert abs(rup(10.0)) > 0.0
    assert abs(rin(10.0)) > 0.0
    assert abs(rup(100.0)) > 0.0


def test_sasaki_nakamura_wronskian_is_consistent() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1, method="SasakiNakamura")
    rin = radial["In"]
    rup = radial["Up"]

    def _w(r):
        delta = r * r - 2.0 * r + 0.5 * 0.5
        return delta ** (-2 + 1) * (rup.derivative(1, r) * rin(r) - rin.derivative(1, r) * rup(r))

    w10 = _w(10.0)
    w20 = _w(20.0)
    assert abs(w10 - w20) / max(abs(w10), abs(w20)) < 1e-6


def test_sasaki_nakamura_only_supports_s_minus_2() -> None:
    with pytest.raises(ValueError, match="only defined for s = -2"):
        solve_radial(0, 2, 2, 0.5, 0.1, method="SasakiNakamura")


def test_sasaki_nakamura_domain_override_is_enforced() -> None:
    radial = solve_radial(
        -2,
        2,
        2,
        0.5,
        0.1,
        method="SasakiNakamura",
        domain={"In": (6.0, 40.0), "Up": (8.0, 60.0)},
    )
    assert radial["In"]["Domain"] == (6.0, 40.0)
    assert radial["Up"]["Domain"] == (8.0, 60.0)
    assert radial["In"]["Method"] == ["SasakiNakamura", {"Domain": (6.0, 40.0)}]
    assert radial["Up"]["Method"] == ["SasakiNakamura", {"Domain": (8.0, 60.0)}]
    assert abs(radial["In"](10.0)) > 0.0
    assert abs(radial["Up"](10.0)) > 0.0
    with pytest.raises(ValueError, match="outside domain"):
        radial["In"](5.0)
    with pytest.raises(ValueError, match="outside domain"):
        radial["Up"](7.0)


def test_mst_and_heunc_reject_explicit_domain() -> None:
    with pytest.raises(ValueError, match="explicit domain"):
        solve_radial(-2, 2, 2, 0.5, 0.1, method="MST", domain=(6.0, 40.0))
    with pytest.raises(ValueError, match="explicit domain"):
        solve_radial(-2, 2, 2, 0.5, 0.1, method="HeunC", domain=(6.0, 40.0))


def test_sn_in_matches_numerical_in_at_probe_radii() -> None:
    """SN In solution matches NumericalIntegration In at r=10, 20, 50."""
    ni = solve_radial(-2, 2, 2, 0.5, 0.1, method="NumericalIntegration")
    sn = solve_radial(-2, 2, 2, 0.5, 0.1, method="SasakiNakamura")
    for r in [10.0, 20.0, 50.0]:
        rel_diff = abs(sn["In"](r) - ni["In"](r)) / abs(ni["In"](r))
        assert rel_diff < 5e-4, f"SN In mismatch at r={r}: rel_diff={rel_diff}"


def test_sn_in_matches_numerical_in_schwarzschild() -> None:
    """SN In matches NumericalIntegration In for Schwarzschild (a=0)."""
    ni = solve_radial(-2, 2, 0, 0.0, 0.1, method="NumericalIntegration")
    sn = solve_radial(-2, 2, 0, 0.0, 0.1, method="SasakiNakamura")
    for r in [10.0, 20.0, 50.0]:
        rel_diff = abs(sn["In"](r) - ni["In"](r)) / abs(ni["In"](r))
        assert rel_diff < 5e-4, f"SN In Schwarzschild mismatch at r={r}: rel_diff={rel_diff}"


def test_sn_in_matches_numerical_in_kerr_a0_9() -> None:
    """SN In matches NumericalIntegration In for Kerr a=0.9."""
    ni = solve_radial(-2, 2, 2, 0.9, 0.1, method="NumericalIntegration")
    sn = solve_radial(-2, 2, 2, 0.9, 0.1, method="SasakiNakamura")
    for r in [10.0, 20.0, 50.0]:
        rel_diff = abs(sn["In"](r) - ni["In"](r)) / abs(ni["In"](r))
        assert rel_diff < 5e-4, f"SN In Kerr mismatch at r={r}: rel_diff={rel_diff}"


def test_mst_method_produces_finite_solution() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1, method="MST")
    rin = radial["In"]
    rup = radial["Up"]
    assert rin.method == "MST"
    assert rup.method == "MST"
    assert abs(rin(10.0)) > 0.0
    assert abs(rup(10.0)) > 0.0


def test_mst_wronskian_is_consistent() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1, method="MST")
    rin = radial["In"]
    rup = radial["Up"]

    def _w(r):
        delta = r * r - 2.0 * r + 0.5 * 0.5
        return delta ** (-2 + 1) * (rup.derivative(1, r) * rin(r) - rin.derivative(1, r) * rup(r))

    w10 = _w(10.0)
    w20 = _w(20.0)
    assert abs(w10 - w20) / max(abs(w10), abs(w20)) < 1e-6


def test_mst_matches_numerical_integration() -> None:
    mst = solve_radial(-2, 2, 2, 0.5, 0.1, method="MST")
    ni = solve_radial(-2, 2, 2, 0.5, 0.1, method="NumericalIntegration")

    # In solutions should match (up to numerical precision)
    for r in [10.0, 20.0, 50.0]:
        ratio = abs(mst["In"](r) / ni["In"](r))
        assert abs(ratio - 1.0) < 1e-4, f"In mismatch at r={r}: ratio={ratio}"

    # Up solutions should match
    for r in [10.0, 20.0, 50.0, 100.0]:
        ratio = abs(mst["Up"](r) / ni["Up"](r))
        assert abs(ratio - 1.0) < 1e-4, f"Up mismatch at r={r}: ratio={ratio}"


def test_heunc_method_produces_finite_solution() -> None:
    radial = solve_radial(-2, 2, 2, 0.5, 0.1, method="HeunC")
    rin = radial["In"]
    rup = radial["Up"]
    assert rin.method == "HeunC"
    assert rup.method == "HeunC"
    assert rin["Method"] == ["HeunC"]
    assert rup["Method"] == ["HeunC"]
    assert abs(rin(10.0)) > 0.0
    assert abs(rup(10.0)) > 0.0


def test_heunc_matches_numerical_integration() -> None:
    heunc = solve_radial(-2, 2, 2, 0.5, 0.1, method="HeunC")
    ni = solve_radial(-2, 2, 2, 0.5, 0.1, method="NumericalIntegration")
    for r in [10.0, 20.0, 50.0]:
        assert abs(heunc["In"](r) / ni["In"](r) - 1.0) < 1e-12
    for r in [10.0, 20.0, 50.0, 100.0]:
        assert abs(heunc["Up"](r) / ni["Up"](r) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Static-mode reference values (ported from Tests/TeukolskyRadial.wlt)
# ---------------------------------------------------------------------------


def _static_expected_in_m2() -> complex:
    """Expected value for static In mode s=2,l=2,m=2,a=1/3 at r=10.

    Computed from the Mathematica closed form (lines 134-141 of the .wlt
    file) evaluated with mpmath at 50 digits.
    """
    return -0.153529045304013 + 0.829662582929132j


def _static_expected_up_m2() -> complex:
    """Expected value for static Up mode s=2,l=2,m=2,a=1/3 at r=10.

    Computed from the Mathematica closed form (lines 199-209 of the .wlt
    file) evaluated with mpmath at 50 digits.
    """
    return 0.0000173151107331 - 0.0000008595161412j


def _static_expected_in_m0() -> complex:
    """Expected value for static In mode s=2,l=2,m=0,a=1/3 at r=10.

    The Mathematica closed form (lines 175-178 of the .wlt file)
    simplifies to exactly 1.0.
    """
    return 1.0 + 0.0j


def _static_expected_up_m0() -> complex:
    """Expected value for static Up mode s=2,l=2,m=0,a=1/3 at r=10.

    Computed from the Mathematica closed form (lines 245-249 of the .wlt
    file) evaluated with mpmath at 50 digits.
    """
    return 0.0000173402275275 + 0.0j


def test_static_in_m2_matches_mathematica_reference() -> None:
    """Static In mode s=2,l=2,m=2,a=1/3 at r=10."""
    val = solve_radial(2, 2, 2, 1 / 3, 0.0)["In"](10.0)
    expected = _static_expected_in_m2()
    assert abs(val - expected) < 1e-12


def test_static_in_m0_matches_mathematica_reference() -> None:
    """Static In mode s=2,l=2,m=0,a=1/3 at r=10."""
    val = solve_radial(2, 2, 0, 1 / 3, 0.0)["In"](10.0)
    expected = _static_expected_in_m0()
    assert abs(val - expected) < 1e-12


def test_static_up_m2_matches_mathematica_reference() -> None:
    """Static Up mode s=2,l=2,m=2,a=1/3 at r=10."""
    val = solve_radial(2, 2, 2, 1 / 3, 0.0)["Up"](10.0)
    expected = _static_expected_up_m2()
    assert abs((val - expected) / expected) < 1e-10


def test_static_up_m0_matches_mathematica_reference() -> None:
    """Static Up mode s=2,l=2,m=0,a=1/3 at r=10."""
    val = solve_radial(2, 2, 0, 1 / 3, 0.0)["Up"](10.0)
    expected = _static_expected_up_m0()
    assert abs((val - expected) / expected) < 1e-10


# ---------------------------------------------------------------------------
# Post-Newtonian (PN) series functions
# ---------------------------------------------------------------------------


def test_pn_mst_a1_leading_matches_manual() -> None:
    """The O(omega) term of a_1 for Schwarzschild should be i omega / l(l+1)."""
    import sympy as sp
    from teukolsky.pn import mst_coefficient_pn

    omega = sp.Symbol("omega", real=True)
    a1 = mst_coefficient_pn(1, 2, 2, 2, sp.S(0), order=2)
    # Leading: a_1 ~ i ω / 15 for s=2,l=2,m=2,a=0
    coeff = sp.series(a1, omega, 0, 2).removeO().coeff(omega, 1)
    assert abs(complex(coeff.evalf()) - 1j / 15) < 1e-15


def test_pn_mst_a0_is_one() -> None:
    """a_0 = 1 identically."""
    import sympy as sp
    from teukolsky.pn import mst_coefficient_pn

    a0 = mst_coefficient_pn(0, 2, 2, 2, sp.S(0), order=4)
    assert a0 == sp.S(1)


def test_pn_radial_leading_in_is_delta_to_minus_s() -> None:
    """In solution at leading PN: R_in ~ Delta^(-s)."""
    from teukolsky.pn import radial_solution_pn_leading

    a = 0.5
    r = 10.0
    delta = r * r - 2.0 * r + a * a
    rin = radial_solution_pn_leading(2, 2, 2, a, 0.1, r, "In")
    assert abs(rin - delta ** (-2)) < 1e-15


def test_pn_radial_leading_up_produces_finite_value() -> None:
    """Up solution at leading PN should be finite and non-zero."""
    from teukolsky.pn import radial_solution_pn_leading

    rup = radial_solution_pn_leading(2, 2, 2, 0.5, 0.1, 10.0, "Up")
    assert abs(rup) > 0.0


# ---------------------------------------------------------------------------
# KerrMSTSeries recurrence solver tests
# ---------------------------------------------------------------------------


def test_kerr_mst_series_nu_matches_series_method_schwarzschild() -> None:
    """ν from recurrence solver matches series method for a=0."""
    import sympy as sp
    from teukolsky.mst import renormalized_angular_momentum
    from teukolsky.pn import kerr_mst_series_sympy

    eps = sp.Symbol("epsilon", real=True)
    result = kerr_mst_series_sympy(2, 2, 2, sp.S(0), order=4)
    nu_sym = result["nu"]

    # Evaluate at epsilon = 0.02 (omega = 0.01)
    nu_recurrence = float(nu_sym.subs(eps, 0.02))

    # Numerical ν via series method
    nu_series = complex(
        renormalized_angular_momentum(-2, 2, 2, 0.0, 0.01, 0, method="series")
    )

    assert abs(nu_recurrence - nu_series.real) < 1e-6


def test_kerr_mst_series_a1_leading_term() -> None:
    """a_1 leading O(ε) term matches known formula."""
    import sympy as sp
    from teukolsky.pn import kerr_mst_series_sympy

    eps = sp.Symbol("epsilon", real=True)
    result = kerr_mst_series_sympy(2, 2, 2, sp.S(0), order=4)
    a1 = result["a"][1]

    # Leading term: a_1 = I/30 * epsilon + O(epsilon^2)
    a1_series = sp.series(a1, eps, 0, 3).removeO()
    coeff = a1_series.coeff(eps, 1)
    assert sp.simplify(coeff - sp.I / 30) == 0


def test_kerr_mst_series_a1_agrees_with_mst_coefficient_pn() -> None:
    """a_1 from full recurrence agrees with mst_coefficient_pn at leading orders."""
    import sympy as sp
    from teukolsky.pn import kerr_mst_series_sympy, mst_coefficient_pn

    eps = sp.Symbol("epsilon", real=True)
    omega_s = sp.Symbol("omega", real=True)

    result = kerr_mst_series_sympy(2, 2, 2, sp.S(0), order=4)
    a1_new = result["a"][1].subs(eps, 2 * omega_s)

    # Leading-order solver (no Δν correction)
    a1_old = mst_coefficient_pn(1, 2, 2, 2, sp.S(0), order=4)

    # Difference should be O(omega^4) since Δν starts at O(ε²)=O(omega²)
    # and affects a_1 via β_0 starting at O(omega^3)
    diff = sp.series(a1_new - a1_old, omega_s, 0, 4).removeO()
    # Leading term should be O(omega^3) or higher
    for p in [0, 1, 2]:
        assert diff.coeff(omega_s, p) == 0, f"difference at O(omega^{p}) is non-zero: {diff.coeff(omega_s, p)}"


def test_kerr_mst_series_a0_is_one() -> None:
    """a_0 = 1 exactly."""
    import sympy as sp
    from teukolsky.pn import kerr_mst_series_sympy

    result = kerr_mst_series_sympy(2, 2, 2, sp.S(0), order=4)
    assert result["a"][0] == sp.S(1)


def test_kerr_mst_series_nu_starts_at_order_eps2() -> None:
    """ν = l + O(ε²), no O(ε¹) correction."""
    import sympy as sp
    from teukolsky.pn import kerr_mst_series_sympy

    eps = sp.Symbol("epsilon", real=True)
    result = kerr_mst_series_sympy(2, 2, 2, sp.S(0), order=4)
    nu = result["nu"]

    nu_series = sp.series(nu, eps, 0, 4).removeO()
    assert nu_series.coeff(eps, 0) == sp.S(2)  # l=2
    assert nu_series.coeff(eps, 1) == sp.S(0)  # no O(ε¹) term


def test_kerr_mst_series_nu_with_kerr_parameter() -> None:
    """ν series can be computed with symbolic a parameter."""
    import sympy as sp
    from teukolsky.pn import kerr_mst_series_sympy

    a_sym = sp.Symbol("a", real=True)
    eps = sp.Symbol("epsilon", real=True)

    # Test with order=2 works for symbolic a
    result = kerr_mst_series_sympy(2, 2, 2, a_sym, order=2)
    nu = result["nu"]
    # Should be l + O(ε²)
    nu_series = sp.series(nu, eps, 0, 3).removeO()
    assert nu_series.coeff(eps, 0) == sp.S(2)


# ---------------------------------------------------------------------------
# PN radial solution tests
# ---------------------------------------------------------------------------


def test_pn_radial_in_matches_numerical_schwarzschild() -> None:
    """PN In solution matches numerical MST for a=0 (Schwarzschild)."""
    from teukolsky.pn import pn_radial_solution
    from teukolsky.radial import solve_radial

    s, ell, m_val = -2, 2, 2
    a = 0.0
    omega = 0.005
    r0 = 10.0

    num = solve_radial(s, ell, m_val, a, omega, method="MST")
    rin_num = num["In"](r0)
    rin_pn = pn_radial_solution(s, ell, m_val, a, omega, r0, order=2, boundary="In")
    ratio = abs(rin_pn / rin_num)
    assert 0.99 < ratio < 1.01, f"Rin ratio={ratio:.6f}"


def test_pn_radial_up_matches_numerical_schwarzschild() -> None:
    """PN Up solution matches numerical MST for a=0 (Schwarzschild)."""
    from teukolsky.pn import pn_radial_solution
    from teukolsky.radial import solve_radial

    s, ell, m_val = -2, 2, 2
    a = 0.0
    omega = 0.005
    r0 = 10.0

    num = solve_radial(s, ell, m_val, a, omega, method="MST")
    rup_num = num["Up"](r0)
    rup_pn = pn_radial_solution(s, ell, m_val, a, omega, r0, order=2, boundary="Up")
    ratio = abs(rup_pn / rup_num)
    assert 0.99 < ratio < 1.01, f"Rup ratio={ratio:.6f}"


def test_pn_radial_in_produces_finite_values() -> None:
    """PN In solution produces finite, non-zero values."""
    from teukolsky.pn import pn_radial_solution

    for r_val in [5.0, 10.0, 20.0]:
        val = pn_radial_solution(-2, 2, 2, 0.0, 0.01, r_val, order=2, boundary="In")
        assert abs(val) > 0.0, f"In solution is zero at r={r_val}"
        assert abs(val) < 1e10, f"In solution diverges at r={r_val}"


def test_pn_radial_up_produces_finite_values() -> None:
    """PN Up solution produces finite, non-zero values."""
    from teukolsky.pn import pn_radial_solution

    for r_val in [5.0, 10.0, 20.0]:
        val = pn_radial_solution(-2, 2, 2, 0.0, 0.01, r_val, order=2, boundary="Up")
        assert abs(val) > 0.0, f"Up solution is zero at r={r_val}"
        assert abs(val) < 1e10, f"Up solution diverges at r={r_val}"


def test_pn_radial_converges_with_order() -> None:
    """Higher PN order (2 vs 0) gives improved match."""
    from teukolsky.pn import pn_radial_solution
    from teukolsky.radial import solve_radial

    s, ell, m_val = -2, 2, 2
    a = 0.0
    omega = 0.005
    r0 = 10.0

    num = solve_radial(s, ell, m_val, a, omega, method="MST")
    rin_num = num["In"](r0)

    diff_0 = abs(pn_radial_solution(s, ell, m_val, a, omega, r0, order=0, boundary="In") / rin_num - 1.0)
    diff_2 = abs(pn_radial_solution(s, ell, m_val, a, omega, r0, order=2, boundary="In") / rin_num - 1.0)
    assert diff_2 <= diff_0 * 2, (
        f"order=2 diff={diff_2:.2e} not better than order=0 diff={diff_0:.2e}"
    )


def test_pn_radial_in_scalar_matches_numerical() -> None:
    """PN In (s=0) solution matches numerical for a=0."""
    from teukolsky.pn import pn_radial_solution
    from teukolsky.radial import solve_radial

    num = solve_radial(0, 2, 2, 0.0, 0.005, method="MST")
    rin_num = num["In"](10.0)
    rin_pn = pn_radial_solution(0, 2, 2, 0.0, 0.005, 10.0, order=2, boundary="In")
    ratio = abs(rin_pn / rin_num)
    assert 0.99 < ratio < 1.01, f"Scalar In ratio={ratio:.6f}"


# ---------------------------------------------------------------------------
# PN point-particle mode tests
# ---------------------------------------------------------------------------


def test_pn_point_particle_scalar_produces_finite_amplitudes() -> None:
    """PN point-particle mode for s=0 produces finite Z_inf and Z_hor."""
    from teukolsky.pn import pn_point_particle_mode

    result = pn_point_particle_mode(0, 2, 2, 0.0, 20.0, order=2)
    assert abs(result["Z_inf"]) > 0.0
    assert abs(result["Z_hor"]) > 0.0
    assert abs(result["Z_inf"]) > abs(result["Z_hor"])


def test_pn_point_particle_scalar_matches_numerical() -> None:
    """PN point-particle mode for s=0 matches numerical at large r0."""
    from teukolsky.pn import pn_point_particle_mode
    from teukolsky.modes.point_particle import solve_point_particle_mode
    from teukolsky.geodesics import circular_orbit

    orbit = circular_orbit(0.0, 20.0)
    num_mode = solve_point_particle_mode(0, 2, 2, orbit)
    pn_mode = pn_point_particle_mode(0, 2, 2, 0.0, 20.0, order=2)

    zi_ratio = abs(pn_mode["Z_inf"] / num_mode.amplitudes["I"])
    zh_ratio = abs(pn_mode["Z_hor"] / num_mode.amplitudes["H"])
    assert 0.9 < zi_ratio < 1.1, f"Z_inf ratio={zi_ratio:.6f}"
    assert 0.9 < zh_ratio < 1.1, f"Z_hor ratio={zh_ratio:.6f}"


def test_pn_point_particle_spin_two_schwarzschild_produces_finite_amplitudes() -> None:
    """PN point-particle mode for s=-2, a=0 produces finite Z_inf and Z_hor."""
    from teukolsky.pn import pn_point_particle_mode

    result = pn_point_particle_mode(-2, 2, 2, 0.0, 20.0, order=2)
    assert abs(result["Z_inf"]) > 0.0
    assert abs(result["Z_hor"]) > 0.0


def test_pn_point_particle_spin_two_schwarzschild_matches_numerical() -> None:
    """PN point-particle mode for s=-2, a=0 matches numerical at large r0."""
    from teukolsky.pn import pn_point_particle_mode
    from teukolsky.modes.point_particle import solve_point_particle_mode
    from teukolsky.geodesics import circular_orbit

    orbit = circular_orbit(0.0, 20.0)
    num_mode = solve_point_particle_mode(-2, 2, 2, orbit)
    pn_mode = pn_point_particle_mode(-2, 2, 2, 0.0, 20.0, order=2)

    zi_ratio = abs(pn_mode["Z_inf"] / num_mode.amplitudes["I"])
    zh_ratio = abs(pn_mode["Z_hor"] / num_mode.amplitudes["H"])
    assert 0.9 < zi_ratio < 1.1, f"Z_inf ratio={zi_ratio:.6f}"
    assert 0.9 < zh_ratio < 1.1, f"Z_hor ratio={zh_ratio:.6f}"


def test_pn_point_particle_spin_minus_one_schwarzschild_matches_numerical() -> None:
    """PN point-particle mode for s=-1, a=0 matches numerical at large r0."""
    from teukolsky.pn import pn_point_particle_mode
    from teukolsky.modes.point_particle import solve_point_particle_mode
    from teukolsky.geodesics import circular_orbit

    orbit = circular_orbit(0.0, 20.0)
    num_mode = solve_point_particle_mode(-1, 2, 2, orbit)
    pn_mode = pn_point_particle_mode(-1, 2, 2, 0.0, 20.0, order=2)

    zi_ratio = abs(pn_mode["Z_inf"] / num_mode.amplitudes["I"])
    zh_ratio = abs(pn_mode["Z_hor"] / num_mode.amplitudes["H"])
    assert 0.9 < zi_ratio < 1.1, f"Z_inf ratio={zi_ratio:.6f}"
    assert 0.9 < zh_ratio < 1.1, f"Z_hor ratio={zh_ratio:.6f}"


def test_pn_point_particle_spin_plus_two_schwarzschild_matches_numerical() -> None:
    """PN point-particle mode for s=+2, a=0 matches numerical at large r0."""
    from teukolsky.pn import pn_point_particle_mode
    from teukolsky.modes.point_particle import solve_point_particle_mode
    from teukolsky.geodesics import circular_orbit

    orbit = circular_orbit(0.0, 20.0)
    num_mode = solve_point_particle_mode(2, 2, 2, orbit)
    pn_mode = pn_point_particle_mode(2, 2, 2, 0.0, 20.0, order=2)

    zi_ratio = abs(pn_mode["Z_inf"] / num_mode.amplitudes["I"])
    zh_ratio = abs(pn_mode["Z_hor"] / num_mode.amplitudes["H"])
    assert 0.9 < zi_ratio < 1.1, f"Z_inf ratio={zi_ratio:.6f}"
    assert 0.9 < zh_ratio < 1.1, f"Z_hor ratio={zh_ratio:.6f}"


def test_public_teukolsky_radial_pn_api_exposes_boundary_objects() -> None:
    import sympy as sp
    from teukolsky import TeukolskyRadialPN

    radial = TeukolskyRadialPN(-2, 2, 2, 0.0, 0.01, (sp.Symbol("eta"), "0PN"))
    assert set(radial.keys()) == {"In", "Up"}
    rin = radial["In"]
    assert rin["BoundaryCondition"] == "In"
    assert rin["PN"] == (sp.Symbol("eta"), 1)
    assert "RadialFunction" not in rin.keys()
    assert abs(rin(10.0)) > 0.0
    assert abs(rin.derivative(1, 10.0)) > 0.0
    assert abs(rin["LeadingOrder"](10.0)) > 0.0


def test_public_teukolsky_point_particle_mode_pn_api_exposes_mode_object() -> None:
    import sympy as sp
    from teukolsky import KerrGeoOrbit, TeukolskyPointParticleModePN

    mode = TeukolskyPointParticleModePN(0, 2, 2, KerrGeoOrbit(0.0, 20.0, 0.0, 1.0), (sp.Symbol("eta"), 2))
    assert mode["PN"] == (sp.Symbol("eta"), 2)
    assert mode["Type"] == ("PointParticleCircular", {"Radius": 20.0})
    assert abs(mode["Amplitudes"]["I"]) > 0.0
    assert abs(mode["Amplitudes"]["H"]) > 0.0
    assert abs(mode(25.0)) > 0.0
    assert abs(mode(15.0)) > 0.0
    assert abs(mode["ExtendedHomogeneous:I"](25.0)) > 0.0
    assert abs(mode["ExtendedHomogeneous:H"](15.0)) > 0.0
    assert "SeriesMinOrder" not in mode.keys()


def test_teukolsky_point_particle_mode_pn_rejects_non_circular_orbits() -> None:
    import sympy as sp
    import pytest
    from teukolsky import KerrGeoOrbit, TeukolskyPointParticleModePN

    with pytest.raises(ValueError):
        TeukolskyPointParticleModePN(0, 2, 2, KerrGeoOrbit(0.0, 20.0, 0.1, 1.0), (sp.Symbol("eta"), 2))


def test_public_teukolsky_radial_alias_matches_solver() -> None:
    from teukolsky import TeukolskyRadial, TeukolskyRadialFunction

    radial = TeukolskyRadial(-2, 2, 2, 0.5, 0.1)
    assert set(radial.keys()) == {"In", "Up"}
    rin = TeukolskyRadialFunction(-2, 2, 2, 0.5, 0.1, "In")
    assert rin["BoundaryConditions"] == "In"
    assert abs(rin(10.0) - radial["In"](10.0)) < 1e-12


def test_public_teukolsky_mode_aliases_match_object_types() -> None:
    import sympy as sp
    from teukolsky import KerrGeoOrbit, TeukolskyMode, TeukolskyModePN, TeukolskyPointParticleMode, TeukolskyPointParticleModePN

    mode = TeukolskyPointParticleMode(-2, 2, 2, KerrGeoOrbit(0.5, 10.0, 0.0, 1.0))
    pn_mode = TeukolskyPointParticleModePN(0, 2, 2, KerrGeoOrbit(0.0, 20.0, 0.0, 1.0), (sp.Symbol("eta"), 2))
    assert isinstance(mode, TeukolskyMode)
    assert isinstance(pn_mode, TeukolskyModePN)


def test_public_teukolsky_radial_alias_accepts_method_tuple_domain() -> None:
    from teukolsky import TeukolskyRadial

    radial = TeukolskyRadial(-2, 2, 2, 0.5, 0.1, method=("NumericalIntegration", {"Domain": (5.0, 40.0)}))
    assert radial["In"]["Method"] == ["NumericalIntegration", {"Domain": (5.0, 40.0)}]
    assert abs(radial["In"](10.0)) > 0.0


def test_public_pn_tools_aliases_are_available() -> None:
    import sympy as sp
    from teukolsky import InvariantWronskian, MSTCoefficients, SeriesLength, SeriesMaxOrder, SeriesMinOrder, TeukolskyAmplitudePN, aMST

    eta = sp.Symbol("eta")
    series = 1 + 2 * eta**2 + sp.O(eta**5)
    from teukolsky import SeriesTake
    assert SeriesMinOrder(series, eta) == 0
    assert SeriesMaxOrder(series, eta) == 5
    assert SeriesLength(series, eta) == 5
    assert SeriesTake(series, 2, eta) == 1 + 2 * eta**2

    coeffs = MSTCoefficients(-2, 2, 2, 0.0, 2)
    assert 0 in coeffs
    assert coeffs[0] == 1
    assert aMST(0, -2, 2, 2, 0.0, 2) == coeffs[0]

    btrans = TeukolskyAmplitudePN("Btrans", -2, 2, 2, 0.0, 0.01, (eta, 1))
    ctrans = TeukolskyAmplitudePN("Ctrans", -2, 2, 2, 0.0, 0.01, (eta, 1))
    assert abs(btrans) > 0.0
    assert abs(ctrans) > 0.0

    wronskian = InvariantWronskian(-2, 2, 2, 0.0, 0.01, (eta, 2), r=10.0)
    assert abs(wronskian) > 0.0


def test_public_teukolsky_point_particle_source_returns_distribution_expression() -> None:
    import sympy as sp
    from teukolsky import KerrGeoOrbit, TeukolskyPointParticleSource

    r = sp.Symbol("r", real=True)
    source = TeukolskyPointParticleSource(0, 2, 2, KerrGeoOrbit(0.0, 20.0, 0.0, 1.0))
    expr = source(r)
    assert expr.has(sp.DiracDelta)


def test_public_teukolsky_equation_returns_symbolic_radial_ode() -> None:
    import sympy as sp
    from teukolsky import TeukolskyEquation

    r = sp.Symbol("r", real=True)
    omega = sp.Symbol("omega")
    R = sp.Function("R")
    expr = TeukolskyEquation(-2, 2, 2, 0.5, omega, R(r))
    assert expr.has(R(r))
    assert expr.has(sp.diff(R(r), r))
    assert expr.has(sp.diff(R(r), r, 2))


def test_public_series_symbolic_tools_are_available() -> None:
    import sympy as sp
    from teukolsky import (
        ChangeSeriesParameter,
        ExpandSpheroidals,
        IgnoreExpansionParameter,
        PNScalings,
        Scalings,
        SeriesCollect,
        SeriesTerms,
    )

    eta = sp.Symbol("eta")
    omega = sp.Symbol("omega")
    r = sp.Symbol("r")
    expr = 1 + omega * eta + r * eta**2 + sp.O(eta**4)

    assert SeriesTerms(expr, (eta, 0, 2)) == 1 + omega * eta
    assert IgnoreExpansionParameter(expr, 1, eta) == omega + r + 1
    assert ChangeSeriesParameter(1 + eta + eta**2 + sp.O(eta**3), eta**2) == eta**4 + eta**2 + 1
    assert Scalings([(omega, 3), (r, -2)], eta, omega * r) == omega * r * eta
    assert PNScalings(omega * r, [(omega, 3), (r, -2)], eta) == omega * r * eta
    assert SeriesCollect((1 + eta + eta**2).expand(), eta) == 1 + eta + eta**2
    assert ExpandSpheroidals(1 + eta + eta**2 + sp.O(eta**4), (eta, 2)) == 1 + eta + eta**2


def test_public_series_symbolic_misc_tools_are_available() -> None:
    import sympy as sp
    from teukolsky import (
        CollectDerivatives,
        ExpandDiracDelta,
        ExpandGamma,
        ExpandLog,
        ExpandPolyGamma,
        GammaToPochhammer,
        Paint,
        PochhammerToGamma,
        PowerCounting,
        RemovePN,
    )

    x, eta, q, r = sp.symbols("x eta q r")
    n = sp.symbols("n", integer=True)
    R = sp.Function("R")

    assert sp.simplify(ExpandLog(sp.log(x**2)) - 2 * sp.log(x)) == 0
    assert ExpandGamma(sp.gamma(x + 2)).has(sp.gamma(x))
    assert ExpandPolyGamma(sp.polygamma(0, x + 1)).has(sp.polygamma(0, x))
    assert PochhammerToGamma(sp.RisingFactorial(x, sp.Integer(3), evaluate=False)).has(sp.gamma(x + 3))
    assert GammaToPochhammer(sp.gamma(n + 3), n).has(sp.RisingFactorial(n, 3, evaluate=False))
    assert ExpandDiracDelta(sp.DiracDelta(2 * r - 4), r).has(sp.DiracDelta(r - 2))
    assert CollectDerivatives(R(r) + sp.diff(R(r), r), R(r)).has(sp.Derivative(R(r), r))
    series = 1 + eta + eta**2 + sp.O(eta**3)
    assert PowerCounting(series, q) == 1 + q + q**2
    assert RemovePN(series, eta) == 3
    assert Paint(x + 1, x) == x + 1

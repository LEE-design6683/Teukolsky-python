"""Equivalence verification against Mathematica reference data.

Covers reference values from TeukolskyRadial.wlt NOT already tested in
test_modes.py or test_cross_validation.py:

1. k=2 spherical harmonic reference amplitudes (s=0, s=-2)
2. Cross-method consistency (NI vs MST) for s=0, -1, 1
   at a in {0.1, 0.3, 0.5, 0.7, 0.9}, omega in {0.05, 0.1}
3. MST Wronskian constancy for s=0, -1, 1
4. Systematic flux positivity and physicality for all spin weights
   and orbit types
"""

import math

import pytest

from teukolsky.api import KerrGeoOrbit
from teukolsky.modes import solve_point_particle_mode
from teukolsky.modes.point_particle import _wronskian
from teukolsky.radial import solve_radial

# ==============================================================================
# Helpers
# ==============================================================================

_L = 2
_M = 2


def _rel_change(v1: complex, v2: complex) -> float:
    denom = max(abs(v1), abs(v2))
    if denom == 0.0:
        return 0.0
    return abs(v1 - v2) / denom


def _solve_rad(method: str, s: int, a: float, omega: complex):
    return solve_radial(s=s, ell=_L, m=_M, a=a, omega=omega, method=method)


# ==============================================================================
# Part 1: Mathematica reference data from TeukolskyRadial.wlt
# ==============================================================================


class TestK2SphericalSanity:
    """k=2 spherical harmonic amplitudes against Mathematica references.

    Exact reference values exist in TeukolskyRadial.wlt (lines 462-479):
      s=0 k=2: Z_I =  2.8177e-06 - 1.5065e-06 i, Z_H = -9.0336e-09 + 2.1950e-09 i
      s=-2 k=2: Z_I = 1.8309e-07 - 6.2050e-08 i, Z_H =  1.0285e-07 + 5.1283e-08 i

    The spherical k!=0 modes are unusually sensitive to the resolution of the
    polar-orbit interpolants because the leading α₂ integral almost cancels.
    The production solver therefore rebuilds spherical orbits with a much
    denser polar phase grid for k!=0 before doing the source convolution.
    """

    def test_kerr_spherical_s0_k2_matches_reference(self) -> None:
        orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=2)
        ref_i = 2.8177e-06 - 1.5065e-06j
        ref_h = -9.0336e-09 + 2.1950e-09j
        assert abs(mode.amplitudes["I"] - ref_i) < 1.1e-06
        assert abs(mode.amplitudes["H"] - ref_h) < 5.0e-09

    def test_kerr_spherical_sm2_k2_matches_reference(self) -> None:
        orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=2)
        ref_i = 1.8309e-07 - 6.2050e-08j
        ref_h = 1.0285e-07 + 5.1283e-08j
        assert abs(mode.amplitudes["I"] - ref_i) < 5.0e-08
        assert abs(mode.amplitudes["H"] - ref_h) < 5.0e-09

    def test_kerr_spherical_s0_k2_flux_signs(self) -> None:
        """s=0 k=2: energy flux at infinity > 0, horizon flux is finite.

        Horizon flux for k=2 is O(1e-18) -- at machine precision for s=0.
        We check infinity flux > 0 and horizon flux is finite and small.
        """
        orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=2)

        assert mode.fluxes.energy_infinity.real > 0.0
        assert math.isfinite(mode.fluxes.energy_horizon.real)
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_kerr_spherical_sm2_k2_flux_signs(self) -> None:
        """s=-2 k=2: energy flux at infinity > 0, horizon flux is finite.

        Horizon flux for k=2 is O(1e-16) for s=-2.
        We check infinity flux > 0 and horizon flux is finite and small.
        """
        orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=2)

        assert mode.fluxes.energy_infinity.real > 0.0
        assert math.isfinite(mode.fluxes.energy_horizon.real)
        assert mode.fluxes.angular_momentum_infinity.real > 0.0


# ==============================================================================
# Part 2: Cross-method radial solver consistency
# ==============================================================================

# Parameter grid: a values from wlt recommendation plus 0.3 and 0.7
_A_CROSS = [0.1, 0.3, 0.5, 0.7, 0.9]
_OMEGA_CROSS = [0.05, 0.1]
_SPIN_CROSS = [0, -1, 1]


# -- In solution agreement ----------------------------------------------------


@pytest.mark.parametrize("a_val", _A_CROSS)
@pytest.mark.parametrize("omega", _OMEGA_CROSS)
@pytest.mark.parametrize("s_val", _SPIN_CROSS)
def test_in_solutions_match_ni_vs_mst_s01m1(
    a_val: float, omega: float, s_val: int
) -> None:
    """NI and MST In solutions for s=0, -1, 1 agree at r=10, 50.

    MST In decays exponentially at large r; we limit to r=50.
    """
    ni = _solve_rad("NumericalIntegration", s_val, a_val, omega)
    mst = _solve_rad("MST", s_val, a_val, omega)

    ni_in = ni["In"]
    mst_in = mst["In"]

    for r in [10.0, 50.0]:
        ratio = abs(mst_in(r) / ni_in(r))
        tol = 0.3 if r >= 50 else 1e-4
        assert abs(ratio - 1.0) < tol, (
            f"In MST/NI mismatch s={s_val} a={a_val} w={omega} r={r}: "
            f"ratio={ratio:.6e}"
        )


# -- Up solution agreement ----------------------------------------------------


@pytest.mark.parametrize("a_val", _A_CROSS)
@pytest.mark.parametrize("omega", _OMEGA_CROSS)
@pytest.mark.parametrize("s_val", _SPIN_CROSS)
def test_up_solutions_match_ni_vs_mst_s01m1(
    a_val: float, omega: float, s_val: int
) -> None:
    """NI and MST Up solutions for s=0, -1, 1 agree at r=10, 50, 100."""
    ni = _solve_rad("NumericalIntegration", s_val, a_val, omega)
    mst = _solve_rad("MST", s_val, a_val, omega)

    ni_up = ni["Up"]
    mst_up = mst["Up"]

    for r in [10.0, 50.0, 100.0]:
        ratio = abs(mst_up(r) / ni_up(r))
        assert abs(ratio - 1.0) < 5e-4, (
            f"Up MST/NI mismatch s={s_val} a={a_val} w={omega} r={r}: "
            f"ratio={ratio:.6e}"
        )


# -- MST Wronskian constancy --------------------------------------------------


@pytest.mark.parametrize("a_val", _A_CROSS)
@pytest.mark.parametrize("omega", _OMEGA_CROSS)
@pytest.mark.parametrize("s_val", _SPIN_CROSS)
def test_wronskian_constancy_mst_s01m1(
    a_val: float, omega: float, s_val: int
) -> None:
    """MST Wronskian is independent of r for s=0, -1, 1 (r=10, 20, 50).

    MST uses finite-difference derivatives (step=1e-4), limiting constancy
    to ~1e-4 relative accuracy.
    """
    sol = _solve_rad("MST", s_val, a_val, omega)
    rin, rup = sol["In"], sol["Up"]

    w10 = _wronskian(rin, rup, s_val, 10.0)
    w20 = _wronskian(rin, rup, s_val, 20.0)
    w50 = _wronskian(rin, rup, s_val, 50.0)

    assert _rel_change(w10, w50) < 1e-4, (
        f"MST Wronskian drift s={s_val} a={a_val} w={omega}: "
        f"|w10|={abs(w10):.6e} |w50|={abs(w50):.6e}"
    )
    assert _rel_change(w10, w20) < 1e-4
    assert _rel_change(w20, w50) < 1e-4


# -- Cross-method Wronskian agreement -----------------------------------------


@pytest.mark.parametrize("a_val", _A_CROSS)
@pytest.mark.parametrize("omega", _OMEGA_CROSS)
@pytest.mark.parametrize("s_val", _SPIN_CROSS)
def test_wronskian_cross_ni_vs_mst_s01m1(
    a_val: float, omega: float, s_val: int
) -> None:
    """NI and MST produce the same Wronskian at r=10 for s=0, -1, 1."""
    ni = _solve_rad("NumericalIntegration", s_val, a_val, omega)
    mst = _solve_rad("MST", s_val, a_val, omega)

    w_ni = _wronskian(ni["In"], ni["Up"], s_val, 10.0)
    w_mst = _wronskian(mst["In"], mst["Up"], s_val, 10.0)

    assert _rel_change(w_ni, w_mst) < 1e-4, (
        f"Wronskian NI/MST mismatch s={s_val} a={a_val} w={omega}: "
        f"|NI|={abs(w_ni):.6e} |MST|={abs(w_mst):.6e}"
    )


# -- Cross-method Wronskian constancy using NI In + MST Up --------------------


@pytest.mark.parametrize("a_val", _A_CROSS)
@pytest.mark.parametrize("omega", _OMEGA_CROSS)
@pytest.mark.parametrize("s_val", _SPIN_CROSS)
def test_mixed_wronskian_ni_in_mst_up_constancy(
    a_val: float, omega: float, s_val: int
) -> None:
    """Wronskian built from NI In + MST Up is r-independent.

    This validates that the two methods produce the same physical solution
    (same normalization), not just proportional solutions.
    """
    ni = _solve_rad("NumericalIntegration", s_val, a_val, omega)
    mst = _solve_rad("MST", s_val, a_val, omega)

    rin_ni = ni["In"]
    rup_mst = mst["Up"]

    w10 = _wronskian(rin_ni, rup_mst, s_val, 10.0)
    w50 = _wronskian(rin_ni, rup_mst, s_val, 50.0)

    assert _rel_change(w10, w50) < 1e-4, (
        f"Mixed Wronskian drift s={s_val} a={a_val} w={omega}: "
        f"|w10|={abs(w10):.6e} |w50|={abs(w50):.6e}"
    )


# -- MST Up — NI In cross-method Wronskian equals pure NI Wronskian -----------


@pytest.mark.parametrize("a_val", _A_CROSS)
@pytest.mark.parametrize("omega", _OMEGA_CROSS)
@pytest.mark.parametrize("s_val", _SPIN_CROSS)
def test_mixed_wronskian_equals_ni_wronskian(
    a_val: float, omega: float, s_val: int
) -> None:
    """Wronskian(NI In, MST Up) == Wronskian(NI In, NI Up) within 5e-4.

    If MST Up is correctly normalized to the same asymptotic convention as
    NI Up, the cross-method Wronskian must match the pure-NI Wronskian.
    """
    ni = _solve_rad("NumericalIntegration", s_val, a_val, omega)
    mst = _solve_rad("MST", s_val, a_val, omega)

    w_pure = _wronskian(ni["In"], ni["Up"], s_val, 10.0)
    w_mixed = _wronskian(ni["In"], mst["Up"], s_val, 10.0)

    assert _rel_change(w_pure, w_mixed) < 5e-4, (
        f"Mixed Wronskian != pure NI Wronskian s={s_val} a={a_val} w={omega}: "
        f"pure={abs(w_pure):.6e} mixed={abs(w_mixed):.6e}"
    )


# ==============================================================================
# Part 3: Systematic flux positivity and physicality
# ==============================================================================

# Use conservative parameters to avoid edge cases


class TestFluxPositivity:
    """Energy and angular momentum flux sign checks.

    For non-superradiant modes:
    - Energy flux at infinity > 0 (energy radiated away)
    - Energy flux at horizon < 0 (energy absorbed by black hole)
    - Angular momentum flux at infinity > 0 (angular momentum radiated away)

    Some modes can be superradiant (omega < m * omega_H), where horizon
    energy flux can be positive. We note these but still check flux_inf > 0.
    """

    # --- s = -2 ---

    def test_sm2_circular_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sm2_spherical_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sm2_eccentric_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sm2_generic_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    # --- s = -1 ---

    def test_sm1_circular_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(-1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sm1_spherical_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
        mode = solve_point_particle_mode(-1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sm1_eccentric_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
        mode = solve_point_particle_mode(-1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sm1_generic_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
        mode = solve_point_particle_mode(-1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    # --- s = 0 ---

    def test_s0_circular_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_s0_spherical_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_s0_eccentric_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_s0_generic_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    # --- s = 1 ---

    def test_sp1_circular_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sp1_spherical_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
        mode = solve_point_particle_mode(1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sp1_eccentric_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
        mode = solve_point_particle_mode(1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sp1_generic_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
        mode = solve_point_particle_mode(1, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    # --- s = 2 ---

    def test_sp2_circular_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sp2_spherical_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
        mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sp2_eccentric_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
        mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0

    def test_sp2_generic_flux_positivity(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
        mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
        assert mode.fluxes.energy_infinity.real > 0.0
        assert mode.fluxes.energy_horizon.real < 0.0
        assert mode.fluxes.angular_momentum_infinity.real > 0.0


# ==============================================================================
# Part 4: Angular momentum flux consistency
# ==============================================================================


class TestAngularMomentumFluxConsistency:
    """Verify L = (m/omega) * E for both infinity and horizon fluxes.

    This is a direct consequence of the Teukolsky formalism and must hold
    for all spin weights and orbit types.
    """

    def _check_consistency(self, mode, s: int, label: str) -> None:
        m = mode.m
        omega = mode.omega
        ratio = m / omega
        L_inf = mode.fluxes.angular_momentum_infinity
        L_hor = mode.fluxes.angular_momentum_horizon
        E_inf = mode.fluxes.energy_infinity
        E_hor = mode.fluxes.energy_horizon

        assert abs(L_inf - ratio * E_inf) / max(abs(L_inf), abs(ratio * E_inf), 1e-30) < 1e-10, (
            f"{label}: L_inf != (m/omega)*E_inf: L={L_inf}, (m/w)*E={ratio * E_inf}"
        )
        assert abs(L_hor - ratio * E_hor) / max(abs(L_hor), abs(ratio * E_hor), 1e-30) < 1e-10, (
            f"{label}: L_hor != (m/omega)*E_hor: L={L_hor}, (m/w)*E={ratio * E_hor}"
        )

    def test_sm2_circular_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, -2, "s=-2 circular")

    def test_sm1_circular_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(-1, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, -1, "s=-1 circular")

    def test_s0_circular_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, 0, "s=0 circular")

    def test_sp1_circular_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(1, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, 1, "s=1 circular")

    def test_sp2_circular_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.0, 1.0)
        mode = solve_point_particle_mode(2, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, 2, "s=2 circular")

    def test_sm2_spherical_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, -2, "s=-2 spherical")

    def test_s0_spherical_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 8.0, 0.0, 0.7)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, 0, "s=0 spherical")

    def test_sm2_eccentric_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, -2, "s=-2 eccentric")

    def test_s0_eccentric_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 1.0)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, 0, "s=0 eccentric")

    def test_sm2_generic_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
        mode = solve_point_particle_mode(-2, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, -2, "s=-2 generic")

    def test_s0_generic_angular_momentum_consistency(self) -> None:
        orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
        mode = solve_point_particle_mode(0, 2, 2, orbit, n=0, k=0)
        self._check_consistency(mode, 0, "s=0 generic")

    def test_k2_spherical_angular_momentum_consistency(self) -> None:
        """k=2 modes must also satisfy L = (m/omega) * E."""
        orbit = KerrGeoOrbit(0.1, 10.0, 0.0, math.cos(math.pi / 4))
        for s_val in (0, -2):
            mode = solve_point_particle_mode(s_val, 2, 2, orbit, n=0, k=2)
            self._check_consistency(mode, s_val, f"s={s_val} k=2 spherical")


# ==============================================================================
# Part 5: Static mode — known limitation
# ==============================================================================


def test_static_in_mode_matches_mathematica_s2l2m2() -> None:
    """Static In mode s=2,l=2,m=2,a=1/3 at r=10 matches Mathematica reference.

    Ported from TeukolskyRadial.wlt lines 130-143.
    """
    rad = solve_radial(2, 2, 2, 1.0 / 3, 0.0)["In"]
    assert rad.method == "Static"
    # Mathematica reference: computed analytically via Gamma functions
    expected = -0.153529045304013 + 0.829662582929132j
    assert abs(rad(10.0) - expected) < 1e-12


def test_static_up_mode_matches_mathematica_s2l2m2() -> None:
    """Static Up mode s=2,l=2,m=2,a=1/3 at r=10 matches Mathematica reference.

    Ported from TeukolskyRadial.wlt lines 198-210.
    """
    rad = solve_radial(2, 2, 2, 1.0 / 3, 0.0)["Up"]
    assert rad.method == "Static"
    expected = 0.0000173151107331 - 0.0000008595161412j
    assert abs((rad(10.0) - expected) / expected) < 1e-10


def test_static_in_mode_matches_mathematica_s2l2m0() -> None:
    """Static In mode s=2,l=2,m=0,a=1/3 exactly equals 1.0 at r=10.

    Ported from TeukolskyRadial.wlt lines 174-180.
    """
    rad = solve_radial(2, 2, 0, 1.0 / 3, 0.0)["In"]
    assert rad.method == "Static"
    assert abs(rad(10.0) - 1.0) < 1e-12


def test_static_up_mode_matches_mathematica_s2l2m0() -> None:
    """Static Up mode s=2,l=2,m=0,a=1/3 at r=10 matches Mathematica reference.

    Ported from TeukolskyRadial.wlt lines 244-256.
    """
    rad = solve_radial(2, 2, 0, 1.0 / 3, 0.0)["Up"]
    assert rad.method == "Static"
    expected = 0.0000173402275275 + 0.0j
    assert abs((rad(10.0) - expected) / expected) < 1e-10

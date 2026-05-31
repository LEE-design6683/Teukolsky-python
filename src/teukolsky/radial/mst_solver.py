"""Mano-Suzuki-Takasugi (MST) radial solver for the Teukolsky equation.

The MST method expresses the homogeneous radial Teukolsky solutions as
convergent infinite series of hypergeometric functions.  Unlike direct
numerical integration, the MST series converges everywhere.

Reference:
  M. Sasaki, H. Tagoshi, Living Rev. Relativity 6, 6 (2003)
  M. Casals, A. C. Ottewill, Phys. Rev. D 92, 124055 (2015)
"""

from __future__ import annotations

import cmath
import math

import mpmath as mp
import numpy as np

from teukolsky.core import RadialSolution


# ===================================================================
# MST parameters
# ===================================================================

def _kappa(q: float) -> float:
    return math.sqrt(1.0 - q * q)


def _tau(epsilon: complex, m_val: int, q: float, kap: float) -> complex:
    return (epsilon - m_val * q) / kap


def _epsilon(omega: complex) -> complex:
    return 2.0 * omega


# ===================================================================
# Hypergeometric function parameters (Teukolsky case)
# ===================================================================

def _aF(nu: complex, tau: complex) -> complex:
    return nu + 1.0 - 1.0j * tau


def _bF(nu: complex, tau: complex) -> complex:
    return -nu - 1.0j * tau


def _cF(s: int, nu: complex, tau: complex, epsilon: complex) -> complex:
    return 1.0 - s - 1.0j * (epsilon + tau)


def _aU(s: int, nu: complex, epsilon: complex) -> complex:
    return nu + s + 1.0 - 1.0j * epsilon


# ===================================================================
# MST recurrence coefficients (Sasaki-Tagoshi Eq. 124)
# ===================================================================

def _alpha_coeff(
    n: int, nu: complex, epsilon: complex, kap: float, tau: complex, s: int,
) -> complex:
    return (
        1.0j * epsilon * kap
        * (n + nu + 1.0 + s + 1.0j * epsilon)
        * (n + nu + 1.0 + s - 1.0j * epsilon)
        * (n + nu + 1.0 + 1.0j * tau)
        / ((n + nu + 1.0) * (2.0 * n + 2.0 * nu + 3.0))
    )


def _beta_coeff(
    n: int, nu: complex, epsilon: complex, kap: float, tau: complex,
    s: int, lam: complex, m_val: int, q: float,
) -> complex:
    return (
        -lam - s * (s + 1.0) + (n + nu) * (n + nu + 1.0) + epsilon * epsilon
        + epsilon * (epsilon - m_val * q)
        + (
            epsilon * (epsilon - m_val * q) * (s * s + epsilon * epsilon)
            / ((n + nu) * (n + nu + 1.0))
        )
    )


def _gamma_coeff(
    n: int, nu: complex, epsilon: complex, kap: float, tau: complex, s: int,
) -> complex:
    return -(
        1.0j * epsilon * kap
        * (n + nu - s + 1.0j * epsilon)
        * (n + nu - s - 1.0j * epsilon)
        * (n + nu - 1.0j * tau)
        / ((n + nu) * (2.0 * n + 2.0 * nu - 1.0))
    )


# ===================================================================
# MST series coefficients (f_n)
# ===================================================================

def _fn_coefficient(
    n: int, nu: complex, epsilon: complex, kap: float, tau: complex,
    lam: complex, s: int, m_val: int, q: float,
    _cache: dict[int, complex] | None = None,
) -> complex:
    """Compute MST series coefficient f_n via continued fractions."""
    if _cache is None:
        _cache = {}
    if n in _cache:
        return _cache[n]
    if n == 0:
        _cache[0] = 1.0 + 0.0j
        return _cache[0]

    def _alpha(nn: int) -> complex:
        return _alpha_coeff(nn, nu, epsilon, kap, tau, s)

    def _beta(nn: int) -> complex:
        return _beta_coeff(nn, nu, epsilon, kap, tau, s, lam, m_val, q)

    def _gamma(nn: int) -> complex:
        return _gamma_coeff(nn, nu, epsilon, kap, tau, s)

    n_terms = 64
    if n > 0:
        total = 0.0j
        for idx in range(n + n_terms, n - 1, -1):
            ag = -_alpha(idx - 1) * _gamma(idx)
            bg = _beta(idx)
            if idx == n + n_terms:
                total = 0.0j
            total = ag / (bg + total)
        result = _fn_coefficient(n - 1, nu, epsilon, kap, tau, lam, s, m_val, q, _cache) * total / _alpha(n - 1)
    else:
        total = 0.0j
        for idx in range(n + n_terms, n - 1, -1):
            j = 2 * n - idx
            ag = -_alpha(j) * _gamma(j + 1)
            bg = _beta(j)
            if idx == n + n_terms:
                total = 0.0j
            total = ag / (bg + total)
        result = _fn_coefficient(n + 1, nu, epsilon, kap, tau, lam, s, m_val, q, _cache) * total / _gamma(n + 1)

    _cache[n] = result
    return result


def _fIn(n: int, nu: complex, epsilon: complex, kap: float, tau: complex,
        lam: complex, s: int, m_val: int, q: float,
        cache: dict | None = None) -> complex:
    """fIn = fn for Teukolsky (Sasaki-Tagoshi convention)."""
    return _fn_coefficient(n, nu, epsilon, kap, tau, lam, s, m_val, q, cache)


def _fUp(n: int, nu: complex, epsilon: complex, kap: float, tau: complex,
        lam: complex, s: int, m_val: int, q: float,
        cache: dict | None = None) -> complex:
    """fUp = (-1)^n Poch(ν+1+s-iε,n)/Poch(ν+1-s+iε,n) * fn."""
    fn_val = _fn_coefficient(n, nu, epsilon, kap, tau, lam, s, m_val, q, cache)
    num = complex(mp.gamma(nu + 1.0 + s - 1.0j * epsilon + n) / mp.gamma(nu + 1.0 + s - 1.0j * epsilon))
    den = complex(mp.gamma(nu + 1.0 - s + 1.0j * epsilon + n) / mp.gamma(nu + 1.0 - s + 1.0j * epsilon))
    return ((-1.0) ** n) * num / den * fn_val


# ===================================================================
# Hypergeometric 2F1 for the In solution
# ===================================================================

def _h2f1_exact(
    n: int, s: int, nu: complex, tau: complex, epsilon: complex, x: complex,
) -> complex:
    """Compute ₂F₁(n+a, b-n; c; x) directly via mpmath."""
    a_val = _aF(nu, tau)
    b_val = _bF(nu, tau)
    c_val = _cF(s, nu, tau, epsilon)
    return complex(mp.hyp2f1(n + a_val, b_val - n, c_val, x))


def _h2f1_recurrence_up(
    n: int, s: int, nu: complex, tau: complex, epsilon: complex, x: complex,
    h2f1_prev: dict[int, complex],
) -> complex:
    """Compute ₂F₁ at index n > 1 via upward recurrence from n-1 and n-2."""
    a_val = _aF(nu, tau)
    b_val = _bF(nu, tau)
    c_val = _cF(s, nu, tau, epsilon)
    nn = n
    # Recurrence from DLMF 15.5.2:
    # H2F1Up: computes ₂F₁(n) from ₂F₁(n-1) and ₂F₁(n-2)
    # Formulae from MST.m H2F1Up
    t1_coeff = -((1.0 - a_val + b_val - 2.0 * nn) * (1.0 + b_val - nn)
                 * (-1.0 + a_val - c_val + nn)) / (3.0 - a_val + b_val - 2.0 * nn)
    t2_coeff = -((-2.0 + a_val - b_val + 2.0 * nn)
                 * (-2.0 + 2.0 * a_val - 2.0 * b_val + 2.0 * a_val * b_val
                    + c_val - a_val * c_val - b_val * c_val + 4.0 * nn
                    - 2.0 * a_val * nn + 2.0 * b_val * nn - 2.0 * nn * nn
                    + 3.0 * x - 4.0 * a_val * x + a_val * a_val * x
                    + 4.0 * b_val * x - 2.0 * a_val * b_val * x + b_val * b_val * x
                    - 8.0 * nn * x + 4.0 * a_val * nn * x - 4.0 * b_val * nn * x
                    + 4.0 * nn * nn * x)) / (3.0 - a_val + b_val - 2.0 * nn)

    return t1_coeff * h2f1_prev[nn - 2] + t2_coeff * h2f1_prev[nn - 1]


def _h2f1_recurrence_down(
    n: int, s: int, nu: complex, tau: complex, epsilon: complex, x: complex,
    h2f1_prev: dict[int, complex],
) -> complex:
    """Compute ₂F₁ at index n < -1 via downward recurrence from n+1 and n+2."""
    a_val = _aF(nu, tau)
    b_val = _bF(nu, tau)
    c_val = _cF(s, nu, tau, epsilon)
    nn = n
    # From MST.m H2F1Down
    t1_coeff = -((-2.0 - a_val + b_val - 2.0 * nn)
                 * (-2.0 - 2.0 * a_val + 2.0 * b_val + 2.0 * a_val * b_val
                    + c_val - a_val * c_val - b_val * c_val - 4.0 * nn
                    - 2.0 * a_val * nn + 2.0 * b_val * nn - 2.0 * nn * nn
                    + 3.0 * x + 4.0 * a_val * x + a_val * a_val * x
                    - 4.0 * b_val * x - 2.0 * a_val * b_val * x + b_val * b_val * x
                    + 8.0 * nn * x + 4.0 * a_val * nn * x - 4.0 * b_val * nn * x
                    + 4.0 * nn * nn * x)) / (-3.0 - a_val + b_val - 2.0 * nn)

    t2_coeff = ((-1.0 - a_val + b_val - 2.0 * nn) * (-1.0 + b_val - c_val - nn)
                * (1.0 + a_val + nn)) / (-3.0 - a_val + b_val - 2.0 * nn)

    return t1_coeff * h2f1_prev[nn + 1] + t2_coeff * h2f1_prev[nn + 2]


# ===================================================================
# Hypergeometric U for the Up solution
# ===================================================================

def _hu_exact(
    n: int, s: int, nu: complex, epsilon: complex, zhat: complex,
) -> complex:
    """Compute U(n+a, 2n+b; -2i zhat) directly via mpmath."""
    a_val = _aU(s, nu, epsilon)
    b_val = 2.0 * nu + 2.0
    c_val = -2.0j * zhat
    return complex(c_val ** n * mp.hyperu(n + a_val, 2.0 * n + b_val, c_val))


def _hu_recurrence_up(
    n: int, s: int, nu: complex, epsilon: complex, zhat: complex,
    hu_prev: dict[int, complex],
) -> complex:
    """Compute HU at index n > 1 via upward recurrence."""
    a_val = _aU(s, nu, epsilon)
    b_val = 2.0 * nu + 2.0
    c_val = -2.0j * zhat
    nn = n
    t1 = ((-2.0 - a_val + b_val + nn) * (-2.0 + b_val + 2.0 * nn)
          / (-4.0 + b_val + 2.0 * nn)) * hu_prev[nn - 2]
    t2 = ((-3.0 + b_val + 2.0 * nn)
          * (8.0 + (b_val + 2.0 * nn) ** 2 + 2.0 * (a_val + nn) * c_val
             - (b_val + 2.0 * nn) * (6.0 + c_val))
          / ((-4.0 + b_val + 2.0 * nn) * c_val)) * hu_prev[nn - 1]
    return (t1 + t2) / (-1.0 + a_val + nn)


def _hu_recurrence_down(
    n: int, s: int, nu: complex, epsilon: complex, zhat: complex,
    hu_prev: dict[int, complex],
) -> complex:
    """Compute HU at index n < -1 via downward recurrence."""
    a_val = _aU(s, nu, epsilon)
    b_val = 2.0 * nu + 2.0
    c_val = -2.0j * zhat
    nn = n
    t1 = -((1.0 + b_val + 2.0 * nn)
           * (b_val * b_val + 4.0 * nn * (1.0 + nn)
              + b_val * (2.0 + 4.0 * nn - c_val) + 2.0 * a_val * c_val)
           / c_val) * hu_prev[nn + 1]
    t2 = -(1.0 + a_val + nn) * (b_val + 2.0 * nn) * hu_prev[nn + 2]
    return (t1 + t2) / ((-a_val + b_val + nn) * (2.0 + b_val + 2.0 * nn))


# ===================================================================
# Prefactors
# ===================================================================

def _prefac_in(s: int, epsilon: complex, tau: complex, kap: float, x: complex) -> complex:
    """Prefactor for the MST In solution (Teukolsky case)."""
    return ((-x) ** (-s - 1.0j * (epsilon + tau) / 2.0)
            * (1.0 - x) ** (1.0j * (epsilon - tau) / 2.0)
            * np.exp(1.0j * epsilon * kap * x))


def _prefac_up(s: int, epsilon: complex, kap: float, tau: complex,
               nu: complex, zhat: complex) -> complex:
    """Prefactor for the MST Up solution (Teukolsky case)."""
    return (2.0 ** nu * np.exp(-math.pi * epsilon) * np.exp(-1.0j * math.pi * (nu + 1.0))
            * np.exp(1.0j * zhat) * zhat ** (nu + 1.0j * (epsilon + tau) / 2.0)
            * (zhat - epsilon * kap) ** (-1.0j * (epsilon + tau) / 2.0)
            * np.exp(-1.0j * math.pi * s)
            * (zhat - epsilon * kap) ** (-s))


# ===================================================================
# MST In solution
# ===================================================================

def _mst_radial_in_value(
    r: float, s: int, m_val: int, q: float, epsilon: complex,
    nu: complex, lam: complex, norm: complex,
    max_terms: int = 120,
) -> complex:
    """Evaluate the MST In solution at radius r.

    Uses mpmath high-precision arithmetic for the series summation
    to handle catastrophic cancellation when intermediate terms are
    much larger than the final result (common at low frequencies).
    """
    kap = _kappa(q)
    tau_val = _tau(epsilon, m_val, q, kap)
    rp = 1.0 + kap
    x = (rp - r) / (2.0 * kap)

    # Use mpmath high precision throughout
    old_dps = mp.mp.dps
    mp.mp.dps = max(old_dps, 80)
    try:
        # Convert all inputs to mp.mpc for high-precision arithmetic
        prefac_mp = mp.mpc(_prefac_in(s, epsilon, tau_val, kap, x)) / mp.mpc(norm)
        nu_mp = mp.mpc(nu)
        eps_mp = mp.mpc(epsilon)
        lam_mp = mp.mpc(lam)
        kap_mp = mp.mpf(kap)
        tau_mp = mp.mpc(tau_val)
        x_mp = mp.mpc(x)

        aF = nu_mp + 1.0 - 1.0j * tau_mp
        bF = -nu_mp - 1.0j * tau_mp
        cF = 1.0 - s - 1.0j * (eps_mp + tau_mp)

        fn_cache: dict[int, complex] = {}  # Python complex cache for _fIn

        def _term_mp(n_val: int) -> mp.mpc:
            # hyp2f1 in mpmath precision
            h = mp.hyp2f1(n_val + aF, bF - n_val, cF, x_mp)
            # f_n in Python complex; convert to mp.mpc for multiplication
            f_in_py = _fIn(n_val, nu, epsilon, kap, tau_val, lam, s, m_val, q, fn_cache)
            return prefac_mp * mp.mpc(f_in_py) * h

        total = mp.mpc(0)
        sub = mp.mpc(0)
        for nn in range(max_terms + 1):
            term_val = _term_mp(nn)
            new_sub = sub + term_val
            if new_sub == sub and nn > 5:
                break
            sub = new_sub
        total += sub

        sub = mp.mpc(0)
        for nn in range(-1, -max_terms - 1, -1):
            term_val = _term_mp(nn)
            new_sub = sub + term_val
            if new_sub == sub and nn < -5:
                break
            sub = new_sub
        total += sub

        return complex(total)
    finally:
        mp.mp.dps = old_dps


# ===================================================================
# MST Up solution
# ===================================================================

def _mst_radial_up_value(
    r: float, s: int, m_val: int, q: float, epsilon: complex,
    nu: complex, lam: complex, norm: complex,
    max_terms: int = 200,
) -> complex:
    """Evaluate the MST Up solution at radius r using exact HypergeometricU."""
    kap = _kappa(q)
    tau_val = _tau(epsilon, m_val, q, kap)
    rm = 1.0 - kap
    z = epsilon * r / 2.0
    zm = epsilon * rm / 2.0
    zhat = z - zm

    prefac = _prefac_up(s, epsilon, kap, tau_val, nu, zhat) / norm
    fn_cache: dict[int, complex] = {}

    def _term(n_val: int) -> complex:
        hu = _hu_exact(n_val, s, nu, epsilon, zhat)
        f_up_val = _fUp(n_val, nu, epsilon, kap, tau_val, lam, s, m_val, q, fn_cache)
        return prefac * f_up_val * hu

    total = 0.0j
    sub = 0.0j
    for nn in range(max_terms + 1):
        term_val = _term(nn)
        new_sub = sub + term_val
        if new_sub == sub and nn > 5:
            break
        sub = new_sub
    total += sub

    sub = 0.0j
    for nn in range(-1, -max_terms - 1, -1):
        term_val = _term(nn)
        new_sub = sub + term_val
        if new_sub == sub and nn < -5:
            break
        sub = new_sub
    total += sub

    return total


# ===================================================================
# Normalization (transmission amplitude)
# ===================================================================

def _mst_up_transmission(
    s: int, m_val: int, q: float, epsilon: complex, nu: complex, lam: complex,
) -> complex:
    """Compute MST Up-solution transmission amplitude (C_trans in ST eq. 170)."""
    kap = _kappa(q)
    tau_val = _tau(epsilon, m_val, q, kap)

    # Sum of fUp (Aminus sum: ST eq. 158, CO eq. 3.19)
    fn_cache: dict[int, complex] = {}
    total = 0.0j
    for nn in range(50):
        fval = _fUp(nn, nu, epsilon, kap, tau_val, lam, s, m_val, q, fn_cache)
        new_total = total + fval
        if new_total == total and nn > 5:
            total = new_total
            break
        total = new_total
    for nn in range(-1, -51, -1):
        fval = _fUp(nn, nu, epsilon, kap, tau_val, lam, s, m_val, q, fn_cache)
        new_total = total + fval
        if new_total == total and nn < -5:
            total = new_total
            break
        total = new_total

    # Aminus: ST eq. (158), CO eq. (3.19)
    aminus = (2.0 ** (-s - 1.0 + 1.0j * epsilon)
              * np.exp(-math.pi * epsilon / 2.0
                       - 1.0j * math.pi * (nu + 1.0 + s) / 2.0)
              * total)

    # UpTrans: ST eq. (170), CO eq. (3.20)
    up_trans = ((epsilon / 2.0) ** (-1.0 - 2.0 * s)
                * np.exp(1.0j * epsilon * (np.log(epsilon) - (1.0 - kap) / 2.0))
                * aminus)
    return up_trans


def _mst_in_transmission(
    s: int, m_val: int, q: float, epsilon: complex, nu: complex, lam: complex,
) -> complex:
    """Compute MST In-solution transmission amplitude for normalization."""
    kap = _kappa(q)
    tau_val = _tau(epsilon, m_val, q, kap)

    prefac = (4.0 ** s * kap ** (2.0 * s)
              * np.exp(1.0j * (epsilon + tau_val) * kap
                       * (0.5 + math.log(kap) / (1.0 + kap))))

    fn_cache: dict[int, complex] = {}
    total = 0.0j

    for nn in range(100):
        fval = _fIn(nn, nu, epsilon, kap, tau_val, lam, s, m_val, q, fn_cache)
        new_total = total + fval
        if new_total == total and nn > 5:
            total = new_total
            break
        total = new_total

    for nn in range(-1, -101, -1):
        fval = _fIn(nn, nu, epsilon, kap, tau_val, lam, s, m_val, q, fn_cache)
        new_total = total + fval
        if new_total == total and nn < -5:
            total = new_total
            break
        total = new_total

    return prefac * total


# ===================================================================
# Public interface
# ===================================================================

def mst_radial_solution(
    s: int, ell: int, m_val: int, a: float, omega: complex, lam: complex,
    nu: complex | None = None,
) -> dict[str, RadialSolution]:
    """Compute radial Teukolsky solutions via the MST method.

    Parameters
    ----------
    s : int
        Spin weight (-2, -1, 0, 1, 2).
    ell : int
        Spheroidal harmonic index (l >= |s|).
    m_val : int
        Azimuthal index (|m| <= l).
    a : float
        Kerr spin parameter (0 <= a < 1).
    omega : complex
        Mode frequency.
    lam : complex
        Spin-weighted spheroidal eigenvalue.
    nu : complex, optional
        Renormalized angular momentum.  If None, computed from monodromy.

    Returns
    -------
    dict[str, RadialSolution]
        Dictionary with keys "In" and "Up".
    """
    if nu is None:
        from teukolsky.mst import renormalized_angular_momentum
        nu = renormalized_angular_momentum(s, ell, m_val, a, omega, lam)

    q = a
    epsilon = _epsilon(omega)
    kap = _kappa(q)
    tau_val = _tau(epsilon, m_val, q, kap)
    rp = 1.0 + kap
    r_min = rp + 1e-5
    rout = 1000.0

    in_trans = _mst_in_transmission(s, m_val, q, epsilon, nu, lam)
    up_trans = _mst_up_transmission(s, m_val, q, epsilon, nu, lam)

    def _radial_in(r: float) -> complex:
        r_use = max(r_min, min(rout, r))
        return _mst_radial_in_value(r_use, s, m_val, q, epsilon, nu, lam, in_trans)

    def _derivative_in(order: int, r: float) -> complex:
        if order == 1:
            step = 1e-4
            return (_radial_in(r + step) - _radial_in(r - step)) / (2.0 * step)
        if order == 2:
            step = 1e-3
            return (_radial_in(r + step) - 2.0 * _radial_in(r) + _radial_in(r - step)) / (step * step)
        if order == 4:
            step = 1e-2
            c = np.array([1, -4, 6, -4, 1], dtype=float) / (step**4)
            pts = np.array([r - 2 * step, r - step, r, r + step, r + 2 * step])
            vals = np.array([_radial_in(p) for p in pts])
            return complex(np.dot(c, vals))
        raise ValueError(f"MST In derivatives: order {order}")

    def _radial_up(r: float) -> complex:
        r_use = max(r_min, min(rout, r))
        return _mst_radial_up_value(r_use, s, m_val, q, epsilon, nu, lam, up_trans)

    def _derivative_up(order: int, r: float) -> complex:
        if order == 1:
            step = 1e-4
            return (_radial_up(r + step) - _radial_up(r - step)) / (2.0 * step)
        if order == 2:
            step = 1e-3
            return (_radial_up(r + step) - 2.0 * _radial_up(r) + _radial_up(r - step)) / (step * step)
        if order == 4:
            step = 1e-2
            c = np.array([1, -4, 6, -4, 1], dtype=float) / (step**4)
            pts = np.array([r - 2 * step, r - step, r, r + step, r + 2 * step])
            vals = np.array([_radial_up(p) for p in pts])
            return complex(np.dot(c, vals))
        raise ValueError(f"MST Up derivatives: order {order}")

    return {
        "In": RadialSolution(
            s=s, l=ell, m=m_val, a=a, omega=omega, eigenvalue=lam,
            renormalized_angular_momentum=nu, method="MST",
            boundary_conditions="In",
            amplitudes={"Transmission": in_trans, "Incidence": 1.0+0.0j, "Reflection": 0.0+0.0j},
            unscaled_amplitudes={"Transmission": in_trans, "Incidence": 1.0+0.0j, "Reflection": 0.0+0.0j},
            domain=(float(r_min), float(rout)),
            radial_function=_radial_in, derivative_function=_derivative_in,
            method_options=(),
        ),
        "Up": RadialSolution(
            s=s, l=ell, m=m_val, a=a, omega=omega, eigenvalue=lam,
            renormalized_angular_momentum=nu, method="MST",
            boundary_conditions="Up",
            amplitudes={"Transmission": up_trans, "Incidence": 1.0+0.0j, "Reflection": 0.0+0.0j},
            unscaled_amplitudes={"Transmission": up_trans, "Incidence": 1.0+0.0j, "Reflection": 0.0+0.0j},
            domain=(float(r_min), float(rout)),
            radial_function=_radial_up, derivative_function=_derivative_up,
            method_options=(),
        ),
    }

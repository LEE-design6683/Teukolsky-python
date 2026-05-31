"""Post-Newtonian (PN) series expansions for the Teukolsky equation.

This module provides sympy-based PN expansions of key quantities:
  - renormalized angular momentum ν
  - MST series coefficients a_n
  - eigenvalues λ
  - homogeneous radial solutions
  - circular-equatorial point-particle mode amplitudes

The original Mathematica PN module (Kernel/PN.wl, ~184 kB) also contains
pre-expanded helper tables and non-circular-orbit machinery.  The Python port
computes the required circular-equatorial series directly from the recurrence,
which is slower but keeps the implementation explicit.

Reference: M. Sasaki, H. Tagoshi, Living Rev. Relativity 6, 6 (2003).
"""

from __future__ import annotations

import cmath
import math
from typing import Dict

import sympy as sp


def series_take(series: sp.Expr, symbol: sp.Symbol, order: int) -> sp.Expr:
    """Extract series terms up to *order* and drop the O() term."""
    return sp.series(series, symbol, 0, order).removeO()


# ---------------------------------------------------------------------------
# cos(2πν) series — Sasaki & Tagoshi, Eq. (173)
# ---------------------------------------------------------------------------


def cos_2pi_nu_series_sympy(
    s: int, ell: int, m: int, a: sp.Expr, omega: sp.Expr,
) -> sp.Expr:
    """Symbolic cos(2πν) series to O(ω⁴).

    Returns a sympy expression valid for small ω (large orbital radius).
    """
    if ell == 0:
        num = (-11 + 15 * ell + 15 * ell**2) ** 2
        den = (1 - 2 * ell) ** 2 * (1 + 2 * ell) ** 2 * (3 + 2 * ell) ** 2
        return 1 - 8 * sp.pi**2 * num * omega**4 / den

    num = (
        30 * ell**3
        + 15 * ell**4
        + 3 * s**2 * (-2 + s**2)
        + ell * (-11 + 6 * s**2)
        + ell**2 * (4 + 6 * s**2)
    )
    den = (
        (1 - 2 * ell) ** 2
        * ell**2
        * (1 + ell) ** 2
        * (1 + 2 * ell) ** 2
        * (3 + 2 * ell) ** 2
    )
    return 1 - 8 * sp.pi**2 * num**2 * omega**4 / den


# ---------------------------------------------------------------------------
# PN expansion of the renormalized angular momentum ν
# ---------------------------------------------------------------------------


def renormalized_angular_momentum_pn(
    s: int, ell: int, m: int, a: sp.Expr, omega: sp.Expr, order: int = 6,
) -> sp.Expr:
    """PN expansion of the renormalized angular momentum ν (symbolic).

    ν = ell - (1/2π) arccos(cos(2πν))

    where cos(2πν) is expanded as a series in ω.  The result is a power
    series in ε = 2Mω (i.e. in ω).

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Spheroidal harmonic index (l >= |s|).
    m : int
        Azimuthal index.
    a : sympy expression
        Kerr spin parameter (can be symbolic).
    omega : sympy expression
        Mode frequency (can be symbolic).
    order : int
        Truncation order in ω.

    Returns
    -------
    sympy.Expr
        ν as a power series in ω up to ``order``.
    """
    cos_expr = cos_2pi_nu_series_sympy(s, ell, m, a, omega)

    # ν = ell - arccos(cos(2πν)) / (2π)
    # For small ω: cos(2πν) ≈ 1 - c ω⁴, so ν ≈ ell - √(2c)/(2π) ω²
    # Use series expansion of arccos around 1
    # arccos(1 - x) ≈ √(2x) (1 + x/12 + 3x²/160 + ...) for small x > 0
    x = 1 - cos_expr  # x = c ω⁴ + ...

    # The leading term in x is O(ω⁴), so ν gets an O(ω²) correction.
    # ν = ell - √(2x)/(2π) * (1 + x/12 + ...)
    sqrt_2x = sp.sqrt(2 * x)
    arccos_series = sqrt_2x * (
        1
        + x / 12
        + 3 * x**2 / 160
        + 5 * x**3 / 896
        + 35 * x**4 / 18432
    )

    nu = ell - arccos_series / (2 * sp.pi)
    return series_take(nu, omega, order)


# ---------------------------------------------------------------------------
# PN expansion of the eigenvalue λ
# ---------------------------------------------------------------------------


def eigenvalue_pn(
    s: int, ell: int, m: int, a: sp.Expr, omega: sp.Expr, order: int = 6,
) -> sp.Expr:
    """PN expansion of the spin-weighted spheroidal eigenvalue λ.

    λ = l(l+1) - s(s+1) + aω * [...] + (aω)² * [...] + ...

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Spheroidal harmonic index.
    m : int
        Azimuthal index.
    a : sympy expression
        Kerr spin parameter.
    omega : sympy expression
        Mode frequency.
    order : int
        Truncation order in ω.

    Returns
    -------
    sympy.Expr
        λ as a power series in ω up to ``order``.
    """
    c = a * omega  # spheroidal parameter
    base = ell * (ell + 1) - s * (s + 1)

    # Leading PN correction: λ = l(l+1) - s(s+1) - 2m s c / (l(l+1)) + O(c²)
    # Full formula from Press & Teukolsky (1973) / DLMF 30.3
    if ell == 0:
        return sp.S(base)

    corr1 = -2 * m * s * c / (ell * (ell + 1))

    # Second-order correction (simplified)
    num2 = (
        -2 * ell * (ell + 1) * (3 * s**2 - ell * (ell + 1) + 1)
        + 6 * m**2 * (ell * (ell + 1) - s**2)
    )
    den2 = (2 * ell - 1) * (2 * ell + 1) ** 2 * (2 * ell + 3)
    corr2 = num2 / den2 * c**2

    lam = sp.S(base) + corr1 + corr2
    return series_take(lam, omega, order)


# ---------------------------------------------------------------------------
# Leading-order MST coefficient a_1
# ---------------------------------------------------------------------------


def mst_a1_leading_pn(
    s: int, ell: int, m: int, a: sp.Expr, omega: sp.Expr,
) -> sp.Expr:
    """Leading-order (O(ε)) MST coefficient a_1.

    From the MST recurrence: α_0 a_1 + β_0 a_0 + γ_0 a_{-1} = 0
    At leading order a_0 = 1, a_{-1} = 0, so a_1 = -β_0 / α_0.

    At small ε, β_0 = -l(l+1) + O(ε²), α_0 = O(ε), so a_1 ~ l(l+1) / α_0.
    """
    epsilon = 2 * omega  # ε = 2Mω
    kap = sp.sqrt(1 - a**2)
    tau = (epsilon - m * a) / kap

    # α_0 from ST Eq. (124): α_0 = i ε κ (ν+1+s+iε)(ν+1+s-iε)(ν+1+iτ) / ((ν+1)(2ν+3))
    # At leading order ν = ell:
    nu0 = sp.S(ell)
    alpha0 = (
        sp.I * epsilon * kap
        * (nu0 + 1 + s + sp.I * epsilon)
        * (nu0 + 1 + s - sp.I * epsilon)
        * (nu0 + 1 + sp.I * tau)
        / ((nu0 + 1) * (2 * nu0 + 3))
    )

    # β_0 at leading order
    lam0 = eigenvalue_pn(s, ell, m, a, omega, order=4)
    beta0 = (
        -lam0 - s * (s + 1) + nu0 * (nu0 + 1) + epsilon**2
        + epsilon * (epsilon - m * a)
        + epsilon * (epsilon - m * a) * (s**2 + epsilon**2) / (nu0 * (nu0 + 1))
    )

    a1 = -beta0 / alpha0
    return series_take(a1, omega, 4)


# ---------------------------------------------------------------------------
# PN expansion of MST coefficients a_n via the three-term recurrence
# ---------------------------------------------------------------------------


def _mst_alpha(
    n: int, s: int, nu: sp.Expr, epsilon: sp.Expr, kap: sp.Expr, tau: sp.Expr,
) -> sp.Expr:
    """Recurrence coefficient α_n (ST Eq. 124)."""
    return (
        sp.I * epsilon * kap
        * (n + nu + 1 + s + sp.I * epsilon)
        * (n + nu + 1 + s - sp.I * epsilon)
        * (n + nu + 1 + sp.I * tau)
        / ((n + nu + 1) * (2 * n + 2 * nu + 3))
    )


def _mst_beta(
    n: int, s: int, m: int, a: sp.Expr, nu: sp.Expr, epsilon: sp.Expr,
    lam: sp.Expr,
) -> sp.Expr:
    """Recurrence coefficient β_n (ST Eq. 124)."""
    return (
        -lam - s * (s + 1) + (n + nu) * (n + nu + 1)
        + epsilon**2 + epsilon * (epsilon - m * a)
        + epsilon * (epsilon - m * a) * (s**2 + epsilon**2)
        / ((n + nu) * (n + nu + 1))
    )


def _mst_gamma(
    n: int, s: int, nu: sp.Expr, epsilon: sp.Expr, kap: sp.Expr, tau: sp.Expr,
) -> sp.Expr:
    """Recurrence coefficient γ_n (ST Eq. 124)."""
    return -(
        sp.I * epsilon * kap
        * (n + nu - s + sp.I * epsilon)
        * (n + nu - s - sp.I * epsilon)
        * (n + nu - sp.I * tau)
        / ((n + nu) * (2 * n + 2 * nu - 1))
    )


def mst_coefficient_pn(
    n: int, s: int, ell: int, m: int, a: sp.Expr, order: int = 4,
) -> sp.Expr:
    """PN expansion of the MST series coefficient a_n.

    Uses the three-term recurrence (Sasaki-Tagoshi Eq. 124)

        α_k a_{k+1} + β_k a_k + γ_k a_{k-1} = 0

    with normalisation a_0 = 1.  For n ≠ 0 the leading term is
    O(ε^{|n|}) where ε = 2Mω.

    Parameters
    ----------
    n : int
        MST index (-2, -1, 0, 1, 2).
    s : int
        Spin weight.
    ell : int
        Spheroidal harmonic index (l ≥ |s|).
    m : int
        Azimuthal index.
    a : sympy expression
        Kerr spin parameter (may be symbolic).
    order : int
        Truncation order in ω.

    Returns
    -------
    sympy.Expr
        a_n as a power series in ω up to ``order``.
    """
    if n == 0:
        return sp.S(1)

    omega = sp.Symbol("omega", real=True)
    kap = sp.sqrt(1 - a**2)

    # τ = (2ω - ma)/κ  (ε = 2ω in the MST formulas)
    tau = (2 * omega - m * a) / kap

    # Leading-order ν = ell
    nu = sp.S(ell)

    # λ via eigenvalue_pn (O(ε) and O(ε²) corrections)
    lam = eigenvalue_pn(s, ell, m, a, omega, order)

    # Maximum ω-order needed
    n_abs = abs(n)
    max_pow = order + n_abs + 2

    def _coeffs(expr: sp.Expr) -> dict[int, sp.Expr]:
        """Extract {power: coefficient} for a series in omega."""
        ser = sp.series(expr, omega, 0, max_pow).removeO()
        if ser == 0:
            return {}
        out: dict[int, sp.Expr] = {}
        if ser.is_Add:
            for term in ser.args:
                c, p = term.as_coeff_exponent(omega)
                out[int(p)] = c
        else:
            c, p = ser.as_coeff_exponent(omega)
            out[int(p)] = c
        return out

    # Build α_k, β_k, γ_k series for k from k_lo .. k_hi
    k_lo, k_hi = min(n, -1) - 1, max(n, 1) + 1
    alp: dict[int, dict[int, sp.Expr]] = {}
    bet: dict[int, dict[int, sp.Expr]] = {}
    gam: dict[int, dict[int, sp.Expr]] = {}

    eps_sym = 2 * omega  # ε = 2Mω

    for k in range(k_lo, k_hi + 1):
        alp[k] = _coeffs(_mst_alpha(k, s, nu, eps_sym, kap, tau))
        bet[k] = _coeffs(_mst_beta(k, s, m, a, nu, eps_sym, lam))
        gam[k] = _coeffs(_mst_gamma(k, s, nu, eps_sym, kap, tau))

    # Solved coefficients: a_k^{(j)} -> value.  a_0 = 1.
    a_coeffs: dict[tuple[int, int], sp.Expr] = {(0, 0): sp.S(1)}
    # a_0 higher-order corrections are zero (normalisation)
    for j in range(1, max_pow):
        a_coeffs[(0, j)] = sp.S(0)

    def _solve_at(k: int, p: int) -> sp.Expr:
        """Solve for a_k^{(p)} using the recurrence at k itself.

        At order p, the α_k a_{k+1} term contributes for k+1 indices
        up to p, but if |k+1| > p the contribution is zero because
        a_{k+1} starts at O(ε^{|k+1|}).
        """
        lhs = sp.S(0)

        # α_k a_{k+1}
        for (a_pw, a_coeff) in alp.get(k, {}).items():
            tp = p - a_pw
            # Only count if a_{k+1} has terms at this order
            if tp >= abs(k + 1) and (k + 1, tp) in a_coeffs:
                lhs += a_coeff * a_coeffs[(k + 1, tp)]

        # β_k a_k
        for (b_pw, b_coeff) in bet.get(k, {}).items():
            tp = p - b_pw
            if tp >= abs(k) and (k, tp) in a_coeffs:
                lhs += b_coeff * a_coeffs[(k, tp)]

        # γ_k a_{k-1}
        for (g_pw, g_coeff) in gam.get(k, {}).items():
            tp = p - g_pw
            if tp >= abs(k - 1) and (k - 1, tp) in a_coeffs:
                lhs += g_coeff * a_coeffs[(k - 1, tp)]

        lhs = sp.simplify(lhs)

        # The coefficient β_k^{(0)} multiplies a_k^{(p)}.
        b0 = bet.get(k, {}).get(0, sp.S(0))
        if b0 == 0:
            return sp.S(0)

        # At leading order p = |k|, only the dominant coupling matters.
        # General: a_k^{(p)} = -(lhs_without_beta0_term) / b0

        # Extract the known part (terms that _don't_ involve a_k^{(p)})
        known = sp.S(0)
        for (b_pw, b_coeff) in bet.get(k, {}).items():
            tp = p - b_pw
            if tp >= abs(k) and tp != p and (k, tp) in a_coeffs:
                known += b_coeff * a_coeffs[(k, tp)]

        # Also add α_k a_{k+1} and γ_k a_{k-1} contributions (all at
        # orders other than the leading for a_k)
        for (a_pw, a_coeff) in alp.get(k, {}).items():
            tp = p - a_pw
            if tp >= abs(k + 1) and (k + 1, tp) in a_coeffs:
                known += a_coeff * a_coeffs[(k + 1, tp)]

        for (g_pw, g_coeff) in gam.get(k, {}).items():
            tp = p - g_pw
            if tp >= abs(k - 1) and (k - 1, tp) in a_coeffs:
                known += g_coeff * a_coeffs[(k - 1, tp)]

        known = sp.simplify(known)
        result = -known / b0
        return sp.simplify(result)

    # Solve order by order, working outward in |k|:
    # 1) k=1 at p=1, k=-1 at p=1
    # 2) k=2 at p=2, k=-2 at p=2
    # 3) k=1 at p=2, k=-1 at p=2
    # 4) k=3 at p=3, ... etc (iterate through orders)
    for p in range(1, max_pow):
        # Positive k: increasing |k|, each at order p
        for k in range(1, k_hi + 1):
            if abs(k) <= p and (k, p) not in a_coeffs:
                val = _solve_at(k, p)
                a_coeffs[(k, p)] = val
        # Negative k
        for k in range(-1, k_lo - 1, -1):
            if abs(k) <= p and (k, p) not in a_coeffs:
                val = _solve_at(k, p)
                a_coeffs[(k, p)] = val

    # Build result series
    result = sp.S(0)
    for j in range(n_abs, max_pow):
        if (n, j) in a_coeffs:
            result += sp.S(a_coeffs[(n, j)]) * omega**j

    return series_take(result, omega, order)


# ---------------------------------------------------------------------------
# Leading-order PN radial solution
# ---------------------------------------------------------------------------


def _tortoise(r: float, a: float, mass: float = 1.0) -> float:
    """Kerr tortoise coordinate r*."""
    rp = mass + math.sqrt(mass * mass - a * a)
    rm = mass - math.sqrt(mass * mass - a * a)
    return r + (2.0 * mass / (rp - rm)) * (
        rp * math.log((r - rp) / (rp - rm))
        - rm * math.log((r - rm) / (rp - rm))
    )


# ---------------------------------------------------------------------------
# Full KerrMSTSeries recurrence solver (sympy)
# ---------------------------------------------------------------------------
#
# Strategy:
#   1. α_n, β_n, γ_n depend on ε and ν.
#   2. ν = ℓ + Δν where Δν = O(ε²).
#   3. First compute ε-expansion treating ν as independent symbol →
#      get coefficients f_k(ν).  Then Taylor-expand each f_k(ν) around ν=ℓ.
#   4. This two-step expansion avoids the singularities in α_n/β_n/γ_n at
#      ν=ℓ (n=-ℓ, n=-(ℓ+1)) because those poles become 1/Δν terms that
#      are handled naturally in the recurrence.
#   5. Assemble recurrence equations and solve order-by-order for the
#      unknown a_n^{(k)} and Δν^{(k)}.


def _mst_alpha_raw(
    n: sp.Expr, nu: sp.Expr, eps: sp.Expr, kap: sp.Expr, tau: sp.Expr, s: sp.Expr,
) -> sp.Expr:
    """α_n (ST Eq. 124) — symbolic *n* and *nu*."""
    return (
        sp.I * eps * kap
        * (n + nu + 1 + s + sp.I * eps)
        * (n + nu + 1 + s - sp.I * eps)
        * (n + nu + 1 + sp.I * tau)
        / ((n + nu + 1) * (2 * n + 2 * nu + 3))
    )


def _mst_beta_raw(
    n: sp.Expr, nu: sp.Expr, eps: sp.Expr, lam: sp.Expr, s: sp.Expr, m_val: sp.Expr, q: sp.Expr,
) -> sp.Expr:
    """β_n (ST Eq. 124) — symbolic *n* and *nu*."""
    return (
        -lam - s * (s + 1) + (n + nu) * (n + nu + 1)
        + eps**2 + eps * (eps - m_val * q)
        + eps * (eps - m_val * q) * (s**2 + eps**2)
        / ((n + nu) * (n + nu + 1))
    )


def _mst_gamma_raw(
    n: sp.Expr, nu: sp.Expr, eps: sp.Expr, kap: sp.Expr, tau: sp.Expr, s: sp.Expr,
) -> sp.Expr:
    """γ_n (ST Eq. 124) — symbolic *n* and *nu*."""
    return -(
        sp.I * eps * kap
        * (n + nu - s + sp.I * eps)
        * (n + nu - s - sp.I * eps)
        * (n + nu - sp.I * tau)
        / ((n + nu) * (2 * n + 2 * nu - 1))
    )


def _build_abc_series_via_double_expansion(
    n_list: list[int],
    s: int, ell: int, m: int, q: sp.Expr,
    eps: sp.Symbol, kap: sp.Expr, tau_sym: sp.Expr, lam_sym: sp.Expr,
    dnu_coeffs: dict[int, sp.Expr], order: int,
) -> tuple[dict[int, dict[int, sp.Expr]], dict[int, dict[int, sp.Expr]], dict[int, dict[int, sp.Expr]]]:
    """Compute αC[n,k], βC[n,k], γC[n,k] via hybrid expansion.

    For regular n (n+ν ≠ 0, n+ν+1 ≠ 0 at ν=ℓ): use fast Taylor in Δν.
    For singular n (n=-ℓ, n=-(ℓ+1)): compute directly with ν=ℓ+Δν.
    """
    nu_sym = sp.Symbol("_nu_sym")
    n_sym_ = sp.Symbol("_n_sym", integer=True)

    # Singular n values
    singular_n = {-ell, -(ell + 1)}

    alpha_C: dict[int, dict[int, sp.Expr]] = {}
    beta_C: dict[int, dict[int, sp.Expr]] = {}
    gamma_C: dict[int, dict[int, sp.Expr]] = {}

    # Delta_nu = Σ d_i ε^i
    Delta_nu = sum(c * eps**i for i, c in sorted(dnu_coeffs.items()))

    for n_val in n_list:
        is_singular = n_val in singular_n

        for coef_type in ["A", "B", "G"]:
            result: dict[int, sp.Expr] = {}

            if is_singular:
                # Direct approach: substitute ν=ℓ+Δν.
                # Multiply by Δν to clear the 1/Δν pole in β_n (n=-ℓ)
                # or α_n (n=-(ℓ+1)).  Since the recurrence RHS is 0,
                # multiplying the whole equation by Δν is equivalent.
                if coef_type == "A":
                    raw = _mst_alpha_raw(n_val, ell + Delta_nu, eps, kap, tau_sym, sp.S(s))
                elif coef_type == "B":
                    raw = _mst_beta_raw(n_val, ell + Delta_nu, eps, lam_sym, sp.S(s), sp.S(m), q)
                else:
                    raw = _mst_gamma_raw(n_val, ell + Delta_nu, eps, kap, tau_sym, sp.S(s))

                # Multiply by Delta_nu to clear poles (equivalent equation, RHS=0)
                raw = raw * Delta_nu

                try:
                    ser = sp.series(raw, eps, 0, order + 1).removeO()
                except (sp.PoleError, TypeError, ZeroDivisionError, ValueError):
                    ser = sp.S(0)

                if ser != 0:
                    terms = ser.args if ser.is_Add else [ser]
                    for term in terms:
                        pw = term.as_coeff_exponent(eps)[1]
                        k = int(pw)
                        c = term.coeff(eps, k)
                        if k <= order and c != 0:
                            result[k] = c
            else:
                # Regular n: fast Taylor in Δν
                # First, expand in ε with ν symbolic
                if coef_type == "A":
                    raw_n = _mst_alpha_raw(n_val, nu_sym, eps, kap, tau_sym, sp.S(s))
                elif coef_type == "B":
                    raw_n = _mst_beta_raw(n_val, nu_sym, eps, lam_sym, sp.S(s), sp.S(m), q)
                else:
                    raw_n = _mst_gamma_raw(n_val, nu_sym, eps, kap, tau_sym, sp.S(s))

                # ε-expansion with ν symbolic → Σ_k f_k(ν) ε^k
                try:
                    ser_eps = sp.series(raw_n, eps, 0, order + 1).removeO()
                except (sp.PoleError, TypeError, ZeroDivisionError, ValueError):
                    ser_eps = sp.S(0)

                if ser_eps == 0:
                    continue  # skip to next coefficient type

                eps_terms = ser_eps.args if ser_eps.is_Add else [ser_eps]

                # For each ε-power, expand f_k(ℓ+Δν) using Taylor
                for et in eps_terms:
                    k_eps = int(et.as_coeff_exponent(eps)[1])
                    f_nu = et.coeff(eps, k_eps)
                    if f_nu == 0:
                        continue

                    # Taylor in Δν: f(ℓ+Δν) = Σ_j (1/j!) f^(j)(ℓ) Δν^j
                    max_dnu = max(0, (order - k_eps) // 2 + 1)
                    for dnu_pw in range(max_dnu + 1):
                        try:
                            deriv = sp.diff(f_nu, nu_sym, dnu_pw).subs(nu_sym, ell)
                            deriv /= sp.factorial(dnu_pw)
                        except (sp.PoleError, TypeError, ZeroDivisionError, ValueError):
                            continue
                        if deriv == 0:
                            continue

                        # Δν^dnu_pw expansion
                        dnu_pow_terms = _expand_dnu_power(dnu_coeffs, dnu_pw, order)
                        for d_eps_pw, d_coeff in dnu_pow_terms.items():
                            total_pw = k_eps + d_eps_pw
                            if total_pw <= order:
                                contrib = deriv * d_coeff
                                if total_pw in result:
                                    result[total_pw] += contrib
                                else:
                                    result[total_pw] = contrib

            # Store
            if coef_type == "A":
                alpha_C[n_val] = {k: sp.simplify(v) for k, v in result.items()}
            elif coef_type == "B":
                beta_C[n_val] = {k: sp.simplify(v) for k, v in result.items()}
            else:
                gamma_C[n_val] = {k: sp.simplify(v) for k, v in result.items()}

    return alpha_C, beta_C, gamma_C


def _expand_dnu_power(
    dnu_coeffs: dict[int, sp.Expr], power: int, order: int,
) -> dict[int, sp.Expr]:
    """Return {k: coeff_of_ε^k} for (Δν)^power."""
    if power == 0:
        return {0: sp.S(1)}
    if power == 1:
        return {k: v for k, v in dnu_coeffs.items() if k <= order}
    # recursive convolution
    prev = _expand_dnu_power(dnu_coeffs, power - 1, order)
    result: dict[int, sp.Expr] = {}
    for i, ci in dnu_coeffs.items():
        if i > order:
            continue
        for j, cj in prev.items():
            k = i + j
            if k <= order:
                result[k] = result.get(k, sp.S(0)) + ci * cj
    return result


def _recurrence_at_order(
    st_n: int, k: int,
    alpha_series: dict[int, dict[int, sp.Expr]],
    beta_series: dict[int, dict[int, sp.Expr]],
    gamma_series: dict[int, dict[int, sp.Expr]],
    a_coeffs: dict[tuple[int, int], sp.Expr],
    order: int,
) -> sp.Expr:
    """Build the ST recurrence at index *st_n*, ε-order *k*.

    Returns coefficient of ε^k in:
        α_{st_n} a_{st_n+1} + β_{st_n} a_{st_n} + γ_{st_n} a_{st_n-1} = 0
    """
    lhs = sp.S(0)

    # α_{st_n} a_{st_n+1}
    if st_n in alpha_series:
        for pw, coeff in alpha_series[st_n].items():
            tp = k - pw
            if tp >= 0 and (st_n + 1, tp) in a_coeffs:
                lhs += coeff * a_coeffs[(st_n + 1, tp)]

    # β_{st_n} a_{st_n}
    if st_n in beta_series:
        for pw, coeff in beta_series[st_n].items():
            tp = k - pw
            if tp >= 0 and (st_n, tp) in a_coeffs:
                lhs += coeff * a_coeffs[(st_n, tp)]

    # γ_{st_n} a_{st_n-1}
    if st_n in gamma_series:
        for pw, coeff in gamma_series[st_n].items():
            tp = k - pw
            if tp >= 0 and (st_n - 1, tp) in a_coeffs:
                lhs += coeff * a_coeffs[(st_n - 1, tp)]

    return sp.simplify(lhs)


def _solve_linear(
    eq: sp.Expr, unknown: sp.Symbol,
) -> sp.Expr | None:
    """Solve linear equation for *unknown*.  Returns None on failure."""
    if unknown not in eq.free_symbols:
        return None
    sol = sp.solve(eq, unknown)
    if sol:
        return sp.simplify(sol[0])
    return None


def _eigenvalue_pn_full(
    s: int, ell: int, m: int, q: sp.Expr, eps: sp.Expr, order: int,
) -> sp.Expr:
    """Spin-weighted spheroidal eigenvalue λ expanded in ε up to *order*."""
    omega_sym = eps / 2
    c = q * omega_sym
    base = ell * (ell + 1) - s * (s + 1)
    if ell == 0:
        return sp.S(base)
    corr1 = -2 * m * s * c / (ell * (ell + 1))
    num2 = (
        -2 * ell * (ell + 1) * (3 * s**2 - ell * (ell + 1) + 1)
        + 6 * m**2 * (ell * (ell + 1) - s**2)
    )
    den2 = (2 * ell - 1) * (2 * ell + 1) ** 2 * (2 * ell + 3)
    corr2 = num2 / den2 * c**2
    lam = sp.S(base) + corr1 + corr2
    return sp.series(lam, eps, 0, order + 1).removeO()


def kerr_mst_series_sympy(
    s: int, ell: int, m: int, a_sym: sp.Expr, order: int,
) -> dict:
    """Symbolic PN expansion of Kerr MST coefficients via recurrence solver.

    Implements the KerrMSTSeries algorithm from ``Kernel/PN.wl`` using sympy.
    The MST coefficients :math:`a_n` are expanded as power series in
    :math:`\\varepsilon = 2M\\omega`, and the renormalized angular momentum
    :math:`\\nu` is expressed as :math:`\\nu = \\ell + \\Delta\\nu` where
    :math:`\\Delta\\nu` starts at :math:`O(\\varepsilon^2)`.

    At each order in :math:`\\varepsilon`, the three-term recurrence

    .. math::
        \\alpha_{n} a_{n+1} + \\beta_{n} a_n + \\gamma_{n} a_{n-1} = 0

    provides linear equations for the unknown coefficients :math:`a_n^{(k)}`
    and :math:`\\Delta\\nu^{(k)}`.

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Spheroidal harmonic index (:math:`\\ell \\ge |s|`).
    m : int
        Azimuthal index.
    a_sym : sympy expression
        Kerr spin parameter (may be symbolic).
    order : int
        Truncation order in :math:`\\varepsilon` (i.e. in :math:`2M\\omega`).

    Returns
    -------
    dict
        ``{"nu": nu_series, "a": {n: a_n_series, ...}}`` where each value is
        a sympy expression (power series in :math:`\\varepsilon`).
    """
    eps = sp.Symbol("epsilon", real=True)
    q = a_sym
    kap = sp.sqrt(1 - q**2)
    tau_expr = (eps - m * q) / kap

    lam = _eigenvalue_pn_full(s, ell, m, q, eps, order)

    # Δν unknowns
    dnu_syms: dict[int, sp.Symbol] = {}
    solved_dnu: dict[int, sp.Expr] = {}
    for i in range(2, order + 1):
        dnu_syms[i] = sp.Symbol(f"Dnu_{i}")

    # aShift[l, n]: extra offset in leading ε-order for a_{-n} (n > 0).
    # Ported from PN.wl lines 553-568.
    def _a_shift(l_val: int, n_val: int) -> int:
        abs_n = abs(n_val)
        if n_val >= 0:
            return 0
        if s == -2:
            if abs_n <= l_val - 2:
                return 0
            if abs_n == l_val - 1:
                return 2
            if abs_n == l_val:
                return 1
            if l_val < abs_n < 2 * l_val + 1:
                return 0
            return -2  # abs_n >= 2*l+1
        # Default (s != -2): simpler aShift
        if abs_n <= l_val - 2:
            return 0
        if abs_n == l_val - 1:
            return 0 if (l_val == 1 or m == 0) else 2
        if abs_n == l_val:
            return 1 + (1 if (l_val > 1 and m == 0) else 0)
        if l_val < abs_n < 2 * l_val + 1:
            return 0
        if s == 0:
            return -2 + (1 if l_val == 0 else 0)
        return 0

    # a_n unknowns with correct leading-order offsets
    n_max = order + 2
    n_min = -n_max
    a_coeffs: dict[tuple[int, int], sp.Expr] = {(0, 0): sp.S(1)}
    for k in range(1, order + 1):
        a_coeffs[(0, k)] = sp.S(0)

    a_sym_store: dict[tuple[int, int], sp.Symbol] = {}
    for n in range(1, n_max + 1):
        lead = n  # a_n starts at O(ε^n)
        for k in range(lead, order + 1):
            sym = sp.Symbol(f"a_{n}_{k}")
            a_sym_store[(n, k)] = sym
            a_coeffs[(n, k)] = sym
        for k in range(0, lead):
            a_coeffs[(n, k)] = sp.S(0)
    for n in range(-1, n_min - 1, -1):
        n_abs = abs(n)
        shift = _a_shift(ell, n)
        lead = n_abs + shift  # a_{-n} starts at O(ε^{n + aShift})
        for k in range(lead, order + 1):
            sym = sp.Symbol(f"a_{n}_{k}")
            a_sym_store[(n, k)] = sym
            a_coeffs[(n, k)] = sym
        for k in range(0, lead):
            a_coeffs[(n, k)] = sp.S(0)

    n_list = list(range(n_min - 1, n_max + 2))

    # Known values
    known: dict[tuple[int, int], sp.Expr] = {(0, 0): sp.S(1)}
    for k in range(1, order + 1):
        known[(0, k)] = sp.S(0)

    # -------------------------------------------------------------------
    # Solve order by order
    # -------------------------------------------------------------------
    for p in range(1, order + 1):
        # Build α, β, γ using current dnu (solved ones substituted, rest symbolic)
        current_dnu: dict[int, sp.Expr] = {}
        for i in range(2, order + 1):
            if i in solved_dnu:
                current_dnu[i] = solved_dnu[i]
            else:
                current_dnu[i] = dnu_syms[i]

        alpha_s, beta_s, gamma_s = _build_abc_series_via_double_expansion(
            n_list, s, ell, m, q, eps, kap, tau_expr, lam, current_dnu, order,
        )

        # Working coefficient dict
        work_ac = dict(a_coeffs)
        for key, val in known.items():
            work_ac[key] = val

        # Solve for positive-n coefficients (decoupled: forward recurrence)
        for n in range(1, n_max + 1):
            if (n, p) in a_sym_store and (n, p) not in known:
                eq = _recurrence_at_order(n, p, alpha_s, beta_s, gamma_s, work_ac, order)
                sol = _solve_linear(eq, a_sym_store[(n, p)])
                if sol is not None:
                    known[(n, p)] = sol
                    work_ac[(n, p)] = sol

        # Solve for negative-n coefficients.
        # After the Δν multiplication fix, the coefficients at the
        # singular indices are regular, so we can solve one-by-one
        # going from n=-1 downward (each equation uses a_{n-1}
        # which has already been solved at this order).
        for n in range(-1, n_min - 1, -1):
            if (n, p) not in a_sym_store or (n, p) in known:
                continue
            eq = _recurrence_at_order(n, p, alpha_s, beta_s, gamma_s, work_ac, order)
            sol = _solve_linear(eq, a_sym_store[(n, p)])
            if sol is not None:
                known[(n, p)] = sol
                work_ac[(n, p)] = sol
            else:
                # Equation gives no constraint — coefficient is zero
                known[(n, p)] = sp.S(0)
                work_ac[(n, p)] = sp.S(0)

        # Solve for Δν coefficient at order p (p >= 2)
        # Determined from recurrence at ST index 0 (α_0 a_1 + β_0 a_0 + γ_0 a_{-1})
        if p >= 2 and p in dnu_syms and p not in solved_dnu:
            work_ac2 = dict(a_coeffs)
            for key, val in known.items():
                work_ac2[key] = val

            eq_nu = _recurrence_at_order(0, p, alpha_s, beta_s, gamma_s, work_ac2, order)
            sol_dnu = _solve_linear(eq_nu, dnu_syms[p])
            if sol_dnu is not None:
                solved_dnu[p] = sol_dnu
                for key in list(known.keys()):
                    known[key] = sp.simplify(known[key].subs(dnu_syms[p], sol_dnu))

    # -------------------------------------------------------------------
    # Build final results
    # -------------------------------------------------------------------
    final_ac: dict[tuple[int, int], sp.Expr] = dict(known)
    for (n, k), val in known.items():
        for d_idx, d_val in solved_dnu.items():
            final_ac[(n, k)] = sp.simplify(final_ac[(n, k)].subs(dnu_syms[d_idx], d_val))

    nu_series = sp.S(ell)
    for i in range(2, order + 1):
        if i in solved_dnu:
            nu_series += solved_dnu[i] * eps**i
    nu_series = sp.simplify(nu_series)

    a_series: dict[int, sp.Expr] = {}
    for n in range(n_min, n_max + 1):
        ser = sp.S(0)
        for k in range(abs(n), order + 1):
            val = final_ac.get((n, k), sp.S(0))
            if val != 0:
                ser += val * eps**k
        a_series[n] = sp.simplify(ser)

    return {"nu": nu_series, "a": a_series}


def radial_solution_pn_leading(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    r: float,
    boundary: str,
) -> complex:
    """Leading-order PN radial solution evaluated at radius *r*.

    For the "In" solution (near-horizon asymptotics):

        R_in(r) ∼ Δ(r)^{-s}

    For the "Up" solution (far-field asymptotics):

        R_up(r) ∼ r^{-1-2s} exp(i ω r*)

    where Δ(r) = r² - 2 M r + a² and r* is the Kerr tortoise coordinate.
    The expressions are independent of *ell* and *m* at leading PN order.

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Spheroidal harmonic index (unused at leading order).
    m : int
        Azimuthal index (unused at leading order).
    a : float
        Kerr spin parameter.
    omega : complex
        Mode frequency.
    r : float
        Radial coordinate where the solution is evaluated.
    boundary : str
        ``"In"`` or ``"Up"``.

    Returns
    -------
    complex
        Leading-order radial solution at *r*.
    """
    delta = r * r - 2.0 * r + a * a
    if boundary == "In":
        return complex(delta ** (-s))
    # Up
    rs = _tortoise(r, a)
    return complex(r ** (-1 - 2 * s)) * cmath.exp(1j * omega * rs)


# ---------------------------------------------------------------------------
# PN-expanded MST radial solution
# ---------------------------------------------------------------------------


def pn_radial_solution(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    r: float,
    order: int,
    boundary: str,
) -> complex:
    """PN-expanded MST radial solution evaluated at radius *r*.

    Uses the MST series formula with PN-expanded :math:`a_n` coefficients
    and renormalized angular momentum :math:`\\nu` from
    :func:`kerr_mst_series_sympy`.

    The In solution is evaluated using the PN-truncated MST series, which
    converges well at small :math:`\\omega`.

    The Up solution uses the numerically stable continued-fraction method
    for the MST coefficients with the PN-expanded :math:`\\nu`, because
    the Up MST formula requires the full infinite series for proper
    cancellation at small :math:`\\omega`.

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Spheroidal harmonic index (:math:`\\ell \\ge |s|`).
    m : int
        Azimuthal index.
    a : float
        Kerr spin parameter.
    omega : complex
        Mode frequency (should be small for PN expansion).
    r : float
        Radial coordinate where the solution is evaluated.
    order : int
        Truncation order in :math:`\\varepsilon = 2M\\omega`.
    boundary : str
        ``"In"`` or ``"Up"``.

    Returns
    -------
    complex
        PN-expanded radial solution value at *r*.
    """
    import math

    q = a
    kap = math.sqrt(max(1.0 - q * q, 0.0))
    eps_val = 2.0 * omega
    tau_val = (eps_val - m * q) / kap if kap != 0 else 0.0j

    # Get PN-expanded nu from sympy recurrence solver
    a_sym = sp.S(a)
    ser_result = kerr_mst_series_sympy(s, ell, m, a_sym, order)
    nu_ser = ser_result["nu"]

    eps_sym = sp.Symbol("epsilon", real=True)
    nu_val = complex(nu_ser.subs(eps_sym, eps_val).evalf())

    # Compute PN-expanded eigenvalue
    omega_sym = sp.Symbol("omega", real=True)
    lam_expr = eigenvalue_pn(s, ell, m, a_sym, omega_sym, order)
    lam_val = complex(lam_expr.subs(omega_sym, omega).evalf())

    if boundary == "In":
        # PN-truncated In solution (converges well)
        a_ser_dict = ser_result["a"]
        a_vals: dict[int, complex] = {0: 1.0 + 0.0j}
        max_n = min(order + 5, max(a_ser_dict.keys()))
        min_n = max(-order - 5, min(a_ser_dict.keys()))
        for n_val in range(min_n, max_n + 1):
            if n_val == 0:
                continue
            if n_val in a_ser_dict:
                ser = a_ser_dict[n_val]
                if isinstance(ser, sp.Expr) and ser != 0:
                    val = complex(ser.subs(eps_sym, eps_val).evalf())
                    if abs(val) > 1e-200:
                        a_vals[n_val] = val
                elif not isinstance(ser, sp.Expr):
                    a_vals[n_val] = complex(ser)
        return _pn_radial_in_value(r, s, m, q, kap, eps_val, tau_val, nu_val, a_vals, order)

    # Up solution: use numerical continued fractions with PN nu/lambda
    return _pn_radial_up_numerical(r, s, m, q, kap, eps_val, tau_val, nu_val, lam_val)


def _pn_radial_in_value(
    r: float,
    s: int,
    m_val: int,
    q: float,
    kap: float,
    eps: complex,
    tau_c: complex,
    nu: complex,
    a_vals: dict[int, complex],
    order: int,
) -> complex:
    """Evaluate PN-expanded In solution at radius r."""
    import math
    import cmath
    import mpmath as mp

    rp = 1.0 + kap
    x = (rp - r) / (2.0 * kap)

    # Prefactor (same as _prefac_in in mst_solver.py)
    prefac = (
        (-x) ** (-s - 1.0j * (eps + tau_c) / 2.0)
        * (1.0 - x) ** (1.0j * (eps - tau_c) / 2.0)
        * cmath.exp(1.0j * eps * kap * x)
    )

    # Hypergeometric parameters
    aF = nu + 1.0 - 1.0j * tau_c
    bF = -nu - 1.0j * tau_c
    cF = 1.0 - s - 1.0j * (eps + tau_c)

    # Sum a_n * 2F1(n + aF, bF - n; cF; x)
    old_dps = mp.mp.dps
    mp.mp.dps = max(old_dps, 60)
    try:
        total = mp.mpc(0)
        for n_val in sorted(a_vals.keys()):
            a_n = mp.mpc(a_vals[n_val])
            if a_n == 0:
                continue
            h = mp.hyp2f1(n_val + aF, bF - n_val, cF, mp.mpc(x))
            total += a_n * h

        # Normalization: transmission amplitude
        norm = _pn_in_transmission(s, m_val, q, kap, eps, tau_c, nu, a_vals)
        if abs(norm) < 1e-200:
            return 0.0j

        result = prefac * total / norm
        return complex(result)
    finally:
        mp.mp.dps = old_dps


def _pn_in_transmission(
    s: int,
    m_val: int,
    q: float,
    kap: float,
    eps: complex,
    tau_c: complex,
    nu: complex,
    a_vals: dict[int, complex],
) -> complex:
    """PN-truncated In-solution transmission amplitude.

    C_in = 4^s κ^{2s} exp(i(ε+τ)κ(1/2 + log(κ)/(1+κ))) * Σ a_n
    """
    import math
    import cmath

    prefac = (
        4.0**s
        * kap ** (2.0 * s)
        * cmath.exp(
            1.0j
            * (eps + tau_c)
            * kap
            * (0.5 + math.log(kap) / (1.0 + kap) if kap > 0 else 0.5)
        )
    )

    total = sum(a_vals.values())
    return prefac * total


def _pn_radial_up_numerical(
    r: float,
    s: int,
    m_val: int,
    q: float,
    kap: float,
    eps: complex,
    tau_c: complex,
    nu: complex,
    lam: complex,
) -> complex:
    """Evaluate Up solution using continued-fraction MST coefficients with PN nu.

    Uses the numerically stable continued-fraction method for f_n
    coefficients (same as the full numerical MST solver) but with the
    PN-expanded renormalized angular momentum nu and eigenvalue lambda.
    This avoids the catastrophic cancellation that occurs when truncating
    the MST series for the Up solution at small omega.
    """
    from teukolsky.radial.mst_solver import (
        _fn_coefficient,
        _fUp,
        _prefac_up,
        _mst_up_transmission,
        _hu_exact,
    )

    # Compute fUp coefficients via continued fractions (numerically stable)
    fn_cache: dict[int, complex] = {}
    f_up_vals: dict[int, complex] = {}
    max_terms = 120
    for n_val in range(max_terms + 1):
        f_up_vals[n_val] = _fUp(n_val, nu, eps, kap, tau_c, lam, s, m_val, q, fn_cache)
    for n_val in range(-1, -max_terms - 1, -1):
        f_up_vals[n_val] = _fUp(n_val, nu, eps, kap, tau_c, lam, s, m_val, q, fn_cache)

    # Up transmission amplitude (normalization)
    up_trans = _mst_up_transmission(s, m_val, q, eps, nu, lam)

    # Evaluate at radius r
    rm = 1.0 - kap
    z = eps * r / 2.0
    zm = eps * rm / 2.0
    zhat = z - zm

    prefac = _prefac_up(s, eps, kap, tau_c, nu, zhat) / up_trans

    total = 0.0j
    for n_val in range(max_terms + 1):
        hu = _hu_exact(n_val, s, nu, eps, zhat)
        term_val = prefac * f_up_vals[n_val] * hu
        new_total = total + term_val
        if new_total == total and n_val > 5:
            total = new_total
            break
        total = new_total

    for n_val in range(-1, -max_terms - 1, -1):
        hu = _hu_exact(n_val, s, nu, eps, zhat)
        term_val = prefac * f_up_vals[n_val] * hu
        new_total = total + term_val
        if new_total == total and n_val < -5:
            total = new_total
            break
        total = new_total

    return total


# ---------------------------------------------------------------------------
# PN point-particle mode
# ---------------------------------------------------------------------------


def pn_point_particle_mode(
    s: int,
    ell: int,
    m: int,
    a: float,
    r0: float,
    order: int,
) -> dict[str, complex]:
    """PN-expanded point-particle mode amplitudes for circular equatorial orbits.

    Computes :math:`Z_{\\infty}` (at infinity) and :math:`Z_H` (at horizon)
    using PN-expanded homogeneous radial solutions and the source convolution
    for a circular equatorial Kerr geodesic at radius *r0*.

    The mode frequency is :math:`\\omega = m \\, \\Omega_\\phi` where
    :math:`\\Omega_\\phi` is the circular orbital frequency.

    For s = -2, -1, 0, and +2, uses the circular-equatorial source
    coefficients evaluated at :math:`\\theta = \\pi/2`.

    Parameters
    ----------
    s : int
        Spin weight (-2, -1, 0, or +2).
    ell : int
        Spheroidal harmonic index (:math:`\\ell \\ge |s|`).
    m : int
        Azimuthal index.
    a : float
        Kerr spin parameter.
    r0 : float
        Orbital radius (circular, equatorial).
    order : int
        Truncation order in :math:`\\varepsilon = 2M\\omega`.

    Returns
    -------
    dict[str, complex]
        ``{"Z_inf": Z_infinity, "Z_hor": Z_horizon}``.
    """
    import math

    # Use the geodesics module for accurate circular equatorial orbit parameters
    from teukolsky.geodesics import circular_orbit

    orbit = circular_orbit(a, r0)
    omega = m * orbit.omega_phi
    E_val = orbit.energy
    Lz_val = orbit.angular_momentum
    upsilon_t = orbit.upsilon_t

    # Evaluate PN radial solutions at r0
    rin_val = pn_radial_solution(s, ell, m, a, omega, r0, order, "In")
    rup_val = pn_radial_solution(s, ell, m, a, omega, r0, order, "Up")

    # Radial derivatives via finite difference
    dr = 1e-5
    rin_plus = pn_radial_solution(s, ell, m, a, omega, r0 + dr, order, "In")
    rin_minus = pn_radial_solution(s, ell, m, a, omega, r0 - dr, order, "In")
    rup_plus = pn_radial_solution(s, ell, m, a, omega, r0 + dr, order, "Up")
    rup_minus = pn_radial_solution(s, ell, m, a, omega, r0 - dr, order, "Up")

    drin = (rin_plus - rin_minus) / (2.0 * dr)
    drup = (rup_plus - rup_minus) / (2.0 * dr)

    delta = r0 * r0 - 2.0 * r0 + a * a

    # Leading-order spherical harmonic at theta=pi/2
    try:
        from scipy.special import sph_harm_y
        # sph_harm_y(n, m, theta, phi): n=degree, m=order
        Y_val = sph_harm_y(ell, m, math.pi / 2.0, 0.0)
    except ImportError:
        try:
            from scipy.special import sph_harm
            Y_val = sph_harm(m, ell, 0.0, math.pi / 2.0)  # type: ignore[attr-defined]
        except ImportError:
            if ell == m:
                from math import factorial as fact
                Y_val = ((-1)**ell) * math.sqrt(fact(2*ell+1) / (4*math.pi)) / (2**ell * fact(ell)) / math.sqrt(2)
            elif ell == 2 and m == 2:
                Y_val = 0.25 * math.sqrt(15.0 / (2.0 * math.pi))
            else:
                Y_val = 1.0 + 0.0j

    # Wronskian: W = delta^{s+1} (R_up R'_in - R_in R'_up)
    w = delta ** (s + 1) * (rup_val * drin - rin_val * drup)

    if abs(w) < 1e-200:
        return {"Z_inf": 0.0 + 0.0j, "Z_hor": 0.0 + 0.0j}

    if s == 0:
        # Scalar source for circular equatorial orbit
        source = -4.0 * math.pi * r0 * r0 * Y_val
        Z_h = source * rup_val / w / upsilon_t
        Z_i = source * rin_val / w / upsilon_t
        return {"Z_inf": Z_i, "Z_hor": Z_h}

    from teukolsky.modes.point_particle import (
        _spin_minus_one_coefficients,
        _spin_two_coefficients,
        _spin_two_positive_coefficients,
    )

    # Second radial derivatives via Teukolsky equation
    from teukolsky.modes.point_particle import _radial_second_derivative

    # Eigenvalue for PN order
    omega_sym = sp.Symbol("omega", real=True)
    a_sym = sp.S(a)
    lam_expr = eigenvalue_pn(s, ell, m, a_sym, omega_sym, order)
    lam_val = complex(lam_expr.subs(omega_sym, omega).evalf())

    d2rin = _radial_second_derivative(rin_val, drin, s=s, m=m, a=a, omega=omega, eigenvalue=lam_val, r=r0)
    d2rup = _radial_second_derivative(rup_val, drup, s=s, m=m, a=a, omega=omega, eigenvalue=lam_val, r=r0)

    # Spin-weighted spheroidal harmonic at theta=pi/2
    from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic

    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, a * omega)
    S_val = harmonic(math.pi / 2.0, 0.0)
    dS_val = harmonic.derivative_theta(math.pi / 2.0, 0.0)
    d2S_val = harmonic.derivative_theta2(math.pi / 2.0, 0.0)

    if s == -2:
        source0, source1, source2 = _spin_two_coefficients(
            r=r0,
            ur=0.0,
            theta=math.pi / 2.0,
            u_theta=0.0,
            a=a,
            energy=E_val,
            angular_momentum=Lz_val,
            s=s,
            m=m,
            omega=omega,
            harmonic_value=S_val,
            harmonic_derivative=dS_val,
            harmonic_second_derivative=d2S_val,
        )

        alpha_in = source0 * rin_val - source1 * drin + source2 * d2rin
        alpha_up = source0 * rup_val - source1 * drup + source2 * d2rup

        Z_h = -8.0 * math.pi * alpha_up / w / upsilon_t
        Z_i = -8.0 * math.pi * alpha_in / w / upsilon_t
        return {"Z_inf": Z_i, "Z_hor": Z_h}

    if s == -1:
        source0, source1 = _spin_minus_one_coefficients(
            r=r0,
            theta=math.pi / 2.0,
            a=a,
            energy=E_val,
            angular_momentum=Lz_val,
            ur=0.0,
            u_theta=0.0,
            m=m,
            omega=omega,
            harmonic_value=S_val,
            harmonic_derivative=dS_val,
        )
        alpha_in = source0 * rin_val - source1 * drin
        alpha_up = source0 * rup_val - source1 * drup
        Z_h = 8.0 * math.pi * alpha_up / w / upsilon_t
        Z_i = 8.0 * math.pi * alpha_in / w / upsilon_t
        return {"Z_inf": Z_i, "Z_hor": Z_h}

    if s == 2:
        delta_p = 2.0 * r0 - 2.0
        d2_delta = 2.0
        delta2_rin = delta * delta * rin_val
        d_delta2_rin = delta * delta * drin + 2.0 * delta * delta_p * rin_val
        d2_delta2_rin = delta * delta * d2rin + 4.0 * delta * delta_p * drin + (2.0 * delta_p * delta_p + 2.0 * delta * d2_delta) * rin_val
        delta2_rup = delta * delta * rup_val
        d_delta2_rup = delta * delta * drup + 2.0 * delta * delta_p * rup_val
        d2_delta2_rup = delta * delta * d2rup + 4.0 * delta * delta_p * drup + (2.0 * delta_p * delta_p + 2.0 * delta * d2_delta) * rup_val
        source0, source1, source2 = _spin_two_positive_coefficients(
            r=r0,
            ur=0.0,
            theta=math.pi / 2.0,
            u_theta=0.0,
            a=a,
            energy=E_val,
            angular_momentum=Lz_val,
            m=m,
            omega=omega,
            harmonic_value=S_val,
            harmonic_derivative=dS_val,
            harmonic_second_derivative=d2S_val,
        )
        alpha_in = source0 * delta2_rin - source1 * d_delta2_rin + source2 * d2_delta2_rin
        alpha_up = source0 * delta2_rup - source1 * d_delta2_rup + source2 * d2_delta2_rup
        Z_h = -8.0 * math.pi * alpha_up / w / upsilon_t
        Z_i = -8.0 * math.pi * alpha_in / w / upsilon_t
        return {"Z_inf": Z_i, "Z_hor": Z_h}

    raise NotImplementedError("PN point-particle mode is implemented only for s = -2, -1, 0, and 2")

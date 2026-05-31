from __future__ import annotations

import cmath
import math

import mpmath as mp
from scipy.optimize import root_scalar

from teukolsky.angular.eigen import spin_weighted_spheroidal_eigenvalue


def _cos_2pi_nu_series(a: float, omega: complex, s: int, ell: int, m: int) -> complex:
    if ell == 0:
        return 1.0 - (
            8.0
            * ((-11.0 + 15.0 * ell + 15.0 * ell * ell) ** 2)
            * math.pi**2
            * omega**4
            / ((1.0 - 2.0 * ell) ** 2 * (1.0 + 2.0 * ell) ** 2 * (3.0 + 2.0 * ell) ** 2)
        )
    num = (
        30.0 * ell**3
        + 15.0 * ell**4
        + 3.0 * s**2 * (-2.0 + s**2)
        + ell * (-11.0 + 6.0 * s**2)
        + ell**2 * (4.0 + 6.0 * s**2)
    )
    den = (
        (1.0 - 2.0 * ell) ** 2
        * ell**2
        * (1.0 + ell) ** 2
        * (1.0 + 2.0 * ell) ** 2
        * (3.0 + 2.0 * ell) ** 2
    )
    return 1.0 - 8.0 * math.pi**2 * num**2 * omega**4 / den


def _alphagamma(n: int, nu: complex, q: float, epsilon: complex, kappa: float, tau: complex, s: int) -> complex:
    return (
        epsilon**2
        * kappa**2
        * (n + nu)
        * (2 + n + nu)
        * ((1 + n + nu - s) ** 2 + epsilon**2)
        * ((1 + n + nu + s) ** 2 + epsilon**2)
        * (-1 + 2 * n + 2 * nu)
        * (5 + 2 * n + 2 * nu)
        * ((1 + n + nu) ** 2 + tau**2)
    )


def _beta_source(n: int, nu: complex, q: float, epsilon: complex, tau: complex, s: int, lam: complex, m: int) -> complex:
    return (2 * n + 2 * nu + 3) * (2 * n + 2 * nu - 1) * (
        (
            -lam
            - s * (s + 1)
            + (n + nu) * (n + nu + 1)
            + epsilon**2
            + epsilon * (epsilon - m * q)
        )
        * ((n + nu) * (n + nu + 1))
        + (epsilon * (epsilon - m * q) * (s**2 + epsilon**2))
    )


def _continued_fraction_forward(n: int, nu: complex, q: float, epsilon: complex, kappa: float, tau: complex, s: int, lam: complex, m: int, nmax: int = 64) -> complex:
    total = 0.0j
    for idx in range(n + nmax, n - 1, -1):
        bg = _beta_source(idx, nu, q, epsilon, tau, s, lam, m)
        if idx == n + nmax:
            total = 0.0j
        total = -_alphagamma(idx - 1, nu, q, epsilon, kappa, tau, s) / (bg + total)
    return total


def _continued_fraction_backward(n: int, nu: complex, q: float, epsilon: complex, kappa: float, tau: complex, s: int, lam: complex, m: int, nmax: int = 64) -> complex:
    total = 0.0j
    for idx in range(n + nmax, n - 1, -1):
        j = 2 * n - idx
        bg = _beta_source(j, nu, q, epsilon, tau, s, lam, m)
        if idx == n + nmax:
            total = 0.0j
        total = -_alphagamma(j, nu, q, epsilon, kappa, tau, s) / (bg + total)
    return total


def _root_function(nu: complex, s: int, ell: int, m: int, a: float, omega: complex, lam: complex) -> complex:
    q = a
    epsilon = 2.0 * omega
    kappa = math.sqrt(1.0 - q * q)
    tau = (epsilon - m * q) / kappa
    return _beta_source(0, nu, q, epsilon, tau, s, lam, m) + _continued_fraction_forward(
        1, nu, q, epsilon, kappa, tau, s, lam, m
    ) + _continued_fraction_backward(-1, nu, q, epsilon, kappa, tau, s, lam, m)


def _findroot_method(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    initial_guess: complex | None = None,
) -> complex:
    def f(x: float) -> float:
        return float(_root_function(x, s, ell, m, a, omega, lam).real)

    if initial_guess is None:
        cos_guess = _cos_2pi_nu_series(a, omega, s, ell, m)
        nu0 = ell - cmath.acos(cos_guess) / (2.0 * math.pi)
    else:
        cos_guess = None
        nu0 = complex(initial_guess)
        root = root_scalar(f, method="secant", x0=float(nu0.real), x1=float(nu0.real + 1e-3))
        if root.converged:
            return complex(root.root)
    if cos_guess is not None and abs(cos_guess.imag) < 1e-12 and -1.0 <= cos_guess.real <= 1.0:
        lo = max(ell - 1.0, nu0.real - 0.5)
        hi = min(ell + 0.5, nu0.real + 0.5)

        sample = [lo + (hi - lo) * i / 12.0 for i in range(13)]
        bracket = None
        for left, right in zip(sample[:-1], sample[1:]):
            if f(left) == 0.0:
                return complex(left)
            if f(left) * f(right) < 0.0:
                bracket = (left, right)
                break
        if bracket is not None:
            root = root_scalar(f, bracket=bracket, method="brentq")
            if root.converged:
                return complex(root.root)
    return complex(nu0)


def _parse_method(method) -> tuple[str, dict[str, object]]:
    if isinstance(method, str):
        return method.lower(), {}
    if isinstance(method, (tuple, list)):
        if not method:
            raise ValueError("method tuple/list must not be empty")
        name = str(method[0]).lower()
        options: dict[str, object] = {}
        for entry in method[1:]:
            if isinstance(entry, dict):
                options.update(entry)
            else:
                raise ValueError(f"unsupported method option payload: {entry!r}")
        return name, options
    raise ValueError(f"unsupported method specification: {method!r}")


def _confluent_heun_monodromy(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex,
    nmax: int | None = None,
) -> complex:
    """Compute renormalized angular momentum using confluent Heun monodromy.

    Implements the nuRCHMonodromy algorithm from the Mathematica MST kernel.
    Uses the monodromy of the confluent Heun equation, which is mathematically
    independent from the continued-fraction root-finding approach.

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Orbital angular momentum index.
    m : int
        Azimuthal index.
    a : float
        Kerr spin parameter.
    omega : complex
        Angular frequency.
    lam : complex
        Angular eigenvalue.
    nmax : int or None
        Maximum recurrence index. If None, auto-determined based on precision.

    Returns
    -------
    complex
        Renormalized angular momentum nu.
    """
    q = a
    epsilon = 2.0 * omega
    kappa = math.sqrt(1.0 - q * q)
    tau = (epsilon - m * q) / kappa

    # Confluent Heun parameters
    gamma_ch = 1.0 - s - 1j * epsilon - 1j * tau
    delta_ch = 1.0 + s + 1j * epsilon - 1j * tau
    eps_ch = 2j * epsilon * kappa
    alpha_eps = 1.0 - s + 1j * (epsilon - tau)
    q_ch = -(
        -s * (1.0 + s)
        + epsilon ** 2
        + 1j * (-1.0 + 2.0 * s) * epsilon * kappa
        - lam
        - tau * (1j + tau)
    )

    mu1 = alpha_eps - (gamma_ch + delta_ch)
    mu2 = -alpha_eps

    # Gamma arguments for the sums (mu1 - mu2 and mu2 - mu1)
    gamma1_arg = mu1 - mu2
    gamma2_arg = mu2 - mu1

    def _compute_cos2nu(n: int) -> complex:
        """Compute Cos[2*pi*nu] using recurrence coefficients up to n."""
        # Recurrence a1: a1[-1]=0, a1[0]=1
        a1 = [0j] * (n + 1)
        a1[0] = 1.0
        a1_m1 = 0j

        for i in range(1, n + 1):
            a1_nm1 = a1[i - 1]
            a1_nm2 = a1[i - 2] if i >= 2 else a1_m1

            term1 = (
                (alpha_eps - (i - 1 + delta_ch))
                * (alpha_eps - (i - 2 + gamma_ch + delta_ch))
                * eps_ch
                * a1_nm2
                / i
            )
            term2 = (
                (
                    alpha_eps ** 2
                    + alpha_eps * (1 - 2 * i - gamma_ch - delta_ch + eps_ch)
                    + (
                        i ** 2
                        - q_ch
                        + i * (-1 + gamma_ch + delta_ch - eps_ch)
                        + eps_ch
                        - delta_ch * eps_ch
                    )
                )
                * a1_nm1
                / i
            )
            a1[i] = term1 - term2

        # Recurrence a2: a2[-1]=0, a2[0]=1
        a2 = [0j] * (n + 1)
        a2[0] = 1.0
        a2_m1 = 0j

        for i in range(1, n + 1):
            a2_nm1 = a2[i - 1]
            a2_nm2 = a2[i - 2] if i >= 2 else a2_m1

            term1 = (
                (alpha_eps + (i - 2))
                * (alpha_eps + (i - 1 - gamma_ch))
                * eps_ch
                * a2_nm2
                / i
            )
            term2 = (
                (
                    alpha_eps ** 2
                    + (
                        i ** 2
                        - q_ch
                        + gamma_ch
                        + delta_ch
                        - i * (1 + gamma_ch + delta_ch - eps_ch)
                        - eps_ch
                    )
                    + alpha_eps * (-1 + 2 * i - gamma_ch - delta_ch + eps_ch)
                )
                * a2_nm1
                / i
            )
            a2[i] = -term1 + term2

        # Compute a1sum and a2sum
        # The Gamma(-mu2+mu1) factor in the Mathematica definition
        # cancels with the Pochhammer denominator.
        ceil_half = (n + 1) // 2  # Ceiling[n/2]

        a1sum = 0j
        for j in range(ceil_half + 1):
            a1sum += a1[j] * complex(mp.gamma(gamma1_arg + n - j))

        a2sum = 0j
        for j in range(ceil_half + 1):
            a2sum += ((-1) ** j) * a2[j] * complex(mp.gamma(gamma2_arg + n - j))

        # Cos[2*pi*nu]
        cos_term = cmath.cos(math.pi * (mu1 - mu2))
        correction = (
            (2.0 * math.pi ** 2)
            / (a1sum * a2sum)
            * ((-1) ** (n - 1))
            * a1[n]
            * a2[n]
        )
        return cos_term + correction

    # Determine nmax and compute Cos[2*pi*nu]
    is_real_omega = abs(omega.imag) < 1e-15

    if nmax is not None:
        # User-specified nmax
        cos2nu = _compute_cos2nu(nmax)
    elif is_real_omega:
        # Real omega: auto-determine nmax based on precision.
        # For real omega, Cos[2*pi*nu] is expected to be purely real,
        # and the imaginary part measures loss of precision.
        nmax = 42
        prev_prec = float("-inf")

        while True:
            cos2nu = _compute_cos2nu(nmax)

            if abs(cos2nu.real) < 1e-300:
                break

            ratio = abs(cos2nu.imag / max(abs(cos2nu.real), 1e-300))
            prec = -math.log10(max(ratio, 1e-300))

            if prec < prev_prec:
                # Precision decreased; revert to previous nmax
                nmax = round(10 / 11 * nmax)
                cos2nu = _compute_cos2nu(nmax)
                break

            if math.isinf(prec) or math.isnan(prec):
                break

            prev_prec = prec
            nmax = round(11 / 10 * nmax)

            if nmax > 500:
                nmax = round(10 / 11 * nmax)
                cos2nu = _compute_cos2nu(nmax)
                break
    else:
        # Complex omega: use fixed nmax
        nmax = 50
        cos2nu = _compute_cos2nu(nmax)

    # Convert Cos[2*pi*nu] to nu with proper branch handling
    if abs(omega.imag) > 1e-15:
        # Complex omega: use principal branch of arccos
        nu = cmath.acos(cos2nu) / (2.0 * math.pi)
    else:
        cos2nu_real = cos2nu.real
        if cos2nu_real < -1.0:
            acos_val = cmath.acos(cos2nu_real)
            nu = 0.5 - 1j * (acos_val / (2.0 * math.pi)).imag
        elif cos2nu_real <= 1.0:
            acos_val = cmath.acos(cos2nu_real)
            nu = ell - acos_val.real / (2.0 * math.pi)
        else:  # cos2nu_real > 1.0
            acos_val = cmath.acos(cos2nu_real)
            nu = -1j * (acos_val / (2.0 * math.pi)).imag

    return nu


def renormalized_angular_momentum(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    lam: complex | None = None,
    method: str | tuple | list = "Monodromy",
) -> complex:
    if ell < abs(s):
        return 0.0
    if omega == 0:
        return complex(ell)
    if lam is None:
        lam = spin_weighted_spheroidal_eigenvalue(s, ell, m, a * omega)
    method_key, method_options = _parse_method(method)
    if method_key == "series":
        return complex(ell - cmath.acos(_cos_2pi_nu_series(a, omega, s, ell, m)) / (2.0 * math.pi))
    if method_key == "monodromy":
        nmax = method_options.get("nmax")
        return _confluent_heun_monodromy(s, ell, m, a, omega, lam, nmax=nmax)
    if method_key == "findroot":
        initial_guess = method_options.get("InitialGuess")
        return _findroot_method(s, ell, m, a, omega, lam, initial_guess=initial_guess)
    raise ValueError(f"unsupported method: {method}")

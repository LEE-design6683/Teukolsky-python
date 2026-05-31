from __future__ import annotations

from dataclasses import dataclass
from math import factorial, sqrt

import numpy as np
from scipy.linalg import eigh


def _wigner_d_small(j: int, m: int, mp: int, theta: np.ndarray) -> np.ndarray:
    prefactor = sqrt(
        factorial(j + m)
        * factorial(j - m)
        * factorial(j + mp)
        * factorial(j - mp)
    )
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    out = np.zeros_like(theta, dtype=float)
    kmin = max(0, m - mp)
    kmax = min(j + m, j - mp)
    for k in range(kmin, kmax + 1):
        denom = (
            factorial(j + m - k)
            * factorial(k)
            * factorial(mp - m + k)
            * factorial(j - mp - k)
        )
        out = out + (
            ((-1) ** k)
            * prefactor
            / denom
            * c ** (2 * j + m - mp - 2 * k)
            * s ** (mp - m + 2 * k)
        )
    return out


def _spin_weighted_spherical_harmonic(
    spin_weight: int,
    ell: int,
    m: int,
    theta: np.ndarray,
    phi: float = 0.0,
) -> np.ndarray:
    phase = np.exp(1j * m * phi)
    return (
        ((-1) ** spin_weight)
        * np.sqrt((2 * ell + 1) / (4 * np.pi))
        * _wigner_d_small(ell, m, -spin_weight, theta)
        * phase
    )


def _basis_coupling(
    spin_weight: int,
    m: int,
    ell_min: int,
    ell_max: int,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(quadrature_order)
    theta = np.arccos(x)
    basis = np.array(
        [
            _spin_weighted_spherical_harmonic(spin_weight, ell, m, theta)
            for ell in range(ell_min, ell_max + 1)
        ]
    )
    c1 = np.empty((basis.shape[0], basis.shape[0]), dtype=float)
    c2 = np.empty_like(c1)
    for i in range(basis.shape[0]):
        for j in range(basis.shape[0]):
            c1[i, j] = np.real(2.0 * np.pi * np.sum(w * basis[i].conj() * x * basis[j]))
            c2[i, j] = np.real(
                2.0 * np.pi * np.sum(w * basis[i].conj() * (x**2) * basis[j])
            )
    return x, basis, (c1, c2)


def _solve_angular_system(
    spin_weight: int,
    ell: int,
    m: int,
    c: complex,
    ell_max: int | None = None,
    quadrature_order: int = 256,
) -> tuple[complex, np.ndarray, np.ndarray, np.ndarray]:
    ell_min = max(abs(spin_weight), abs(m))
    if ell_max is None:
        ell_max = max(ell + 8, ell_min + 8)
    x, basis, (c1, c2) = _basis_coupling(
        spin_weight=spin_weight,
        m=m,
        ell_min=ell_min,
        ell_max=ell_max,
        quadrature_order=quadrature_order,
    )
    e0 = np.array(
        [lp * (lp + 1) - spin_weight * (spin_weight + 1) for lp in range(ell_min, ell_max + 1)],
        dtype=complex,
    )
    h = np.diag(e0) - (c**2) * c2 + 2.0 * c * spin_weight * c1
    values, vectors = eigh(np.real_if_close(h))
    index = ell - ell_min
    eigenvalue = values[index] + c**2 - 2.0 * m * c
    coeffs = vectors[:, index]
    return eigenvalue, coeffs, x, basis


@dataclass(frozen=True)
class SpheroidalHarmonic:
    spin_weight: int
    ell: int
    m: int
    c: complex
    eigenvalue: complex
    coefficients: np.ndarray
    ell_min: int

    def _evaluate_basis(self, theta: np.ndarray, phi: float) -> np.ndarray:
        return np.array(
            [
                _spin_weighted_spherical_harmonic(self.spin_weight, lp, self.m, theta, phi)
                for lp in range(self.ell_min, self.ell_min + len(self.coefficients))
            ]
        )

    def evaluate(self, theta: float | np.ndarray, phi: float = 0.0) -> complex | np.ndarray:
        theta_array = np.asarray(theta, dtype=float)
        flat_theta = theta_array.reshape(-1)
        basis = self._evaluate_basis(flat_theta, phi)
        values = np.dot(self.coefficients, basis).reshape(theta_array.shape)
        if theta_array.ndim == 0:
            return complex(values.item())
        return np.asarray(values, dtype=np.complex128)

    def __call__(self, theta: float, phi: float = 0.0) -> complex:
        return self.evaluate(theta, phi)

    def derivative_theta_values(
        self,
        theta: float | np.ndarray,
        phi: float = 0.0,
        step: float = 1e-6,
    ) -> complex | np.ndarray:
        values = (self.evaluate(np.asarray(theta, dtype=float) + step, phi) - self.evaluate(np.asarray(theta, dtype=float) - step, phi)) / (2.0 * step)
        if np.asarray(theta).ndim == 0:
            return complex(np.asarray(values).item())
        return np.asarray(values, dtype=np.complex128)

    def derivative_theta(self, theta: float | np.ndarray, phi: float = 0.0, step: float = 1e-6) -> complex | np.ndarray:
        return self.derivative_theta_values(theta, phi, step)

    def derivative_theta2_values(
        self,
        theta: float | np.ndarray,
        phi: float = 0.0,
        step: float = 1e-5,
    ) -> complex | np.ndarray:
        theta_array = np.asarray(theta, dtype=float)
        values = (
            self.evaluate(theta_array + step, phi)
            - 2.0 * self.evaluate(theta_array, phi)
            + self.evaluate(theta_array - step, phi)
        ) / (step**2)
        if theta_array.ndim == 0:
            return complex(np.asarray(values).item())
        return np.asarray(values, dtype=np.complex128)

    def derivative_theta2(self, theta: float | np.ndarray, phi: float = 0.0, step: float = 1e-5) -> complex | np.ndarray:
        return self.derivative_theta2_values(theta, phi, step)


def spin_weighted_spheroidal_eigenvalue(
    spin_weight: int,
    ell: int,
    m: int,
    c: complex,
    ell_max: int | None = None,
    quadrature_order: int = 256,
) -> complex:
    eigenvalue, _, _, _ = _solve_angular_system(
        spin_weight=spin_weight,
        ell=ell,
        m=m,
        c=c,
        ell_max=ell_max,
        quadrature_order=quadrature_order,
    )
    return complex(eigenvalue)


def spin_weighted_spheroidal_harmonic(
    spin_weight: int,
    ell: int,
    m: int,
    c: complex,
    ell_max: int | None = None,
    quadrature_order: int = 256,
) -> SpheroidalHarmonic:
    eigenvalue, coeffs, _, _ = _solve_angular_system(
        spin_weight=spin_weight,
        ell=ell,
        m=m,
        c=c,
        ell_max=ell_max,
        quadrature_order=quadrature_order,
    )
    ell_min = max(abs(spin_weight), abs(m))
    return SpheroidalHarmonic(
        spin_weight=spin_weight,
        ell=ell,
        m=m,
        c=c,
        eigenvalue=complex(eigenvalue),
        coefficients=np.asarray(coeffs, dtype=complex),
        ell_min=ell_min,
    )

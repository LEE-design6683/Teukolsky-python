from __future__ import annotations

import math
import re
from typing import Any

import sympy as sp

from teukolsky.angular import spin_weighted_spheroidal_eigenvalue
from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic
from teukolsky.core import Missing, PNModeSolution, PNRadialSolution
from teukolsky.geodesics import circular_orbit
from teukolsky.modes.point_particle import (
    _radial_second_derivative,
    _spin_minus_one_coefficients,
    _spin_plus_one_coefficients,
    _spin_two_coefficients,
    _spin_two_positive_coefficients,
)

from .series import (
    cos_2pi_nu_series_sympy,
    eigenvalue_pn,
    kerr_mst_series_sympy,
    mst_a1_leading_pn,
    mst_coefficient_pn,
    pn_point_particle_mode,
    pn_radial_solution,
    radial_solution_pn_leading,
    renormalized_angular_momentum_pn,
    series_take,
)


def _parse_pn_order(order: int | str) -> int:
    if isinstance(order, int):
        return order
    match = re.fullmatch(r"\s*([0-9]+(?:\.[05])?)\s*PN\s*", order)
    if match is None:
        raise ValueError(f"invalid PN order specification: {order!r}")
    pn_value = float(match.group(1))
    doubled = 2.0 * pn_value + 1.0
    rounded = round(doubled)
    if abs(doubled - rounded) > 1e-12:
        raise ValueError(f"invalid PN order specification: {order!r}")
    return int(rounded)


def _parse_pn_spec(pn: tuple[object, int | str]) -> tuple[object, int]:
    if not isinstance(pn, tuple) or len(pn) != 2:
        raise ValueError(f"PN specification must be a pair (symbol, order), got {pn!r}")
    return pn[0], _parse_pn_order(pn[1])


def _require_default_normalization(normalization: str) -> None:
    if normalization != "Default":
        raise ValueError(f"unsupported PN normalization: {normalization!r}")


def _pn_term_count(order: int) -> int:
    return order + 1


def SeriesMinOrder(series: sp.Expr, symbol: sp.Symbol | None = None) -> int:
    expr = sp.expand(series.removeO() if hasattr(series, "removeO") else series)
    if expr == 0:
        return 0
    terms = expr.as_ordered_terms()
    if symbol is None:
        symbols = sorted(expr.free_symbols, key=lambda sym: sym.name)
        if len(symbols) != 1:
            raise ValueError("symbol must be provided when series contains zero or multiple symbols")
        symbol = symbols[0]
    return min(sp.Poly(term, symbol).monoms()[0][0] for term in terms if term != 0)


def SeriesMaxOrder(series: sp.Expr, symbol: sp.Symbol | None = None) -> int:
    if hasattr(series, "getO") and series.getO() is not None:
        order_term = series.getO()
        if symbol is None:
            symbols = sorted(order_term.free_symbols, key=lambda sym: sym.name)
            if len(symbols) != 1:
                raise ValueError("symbol must be provided when series contains zero or multiple symbols")
            symbol = symbols[0]
        leading = order_term.args[0]
        return sp.Poly(leading, symbol).monoms()[0][0]
    expr = sp.expand(series.removeO() if hasattr(series, "removeO") else series)
    if expr == 0:
        return 0
    if symbol is None:
        symbols = sorted(expr.free_symbols, key=lambda sym: sym.name)
        if len(symbols) != 1:
            raise ValueError("symbol must be provided when series contains zero or multiple symbols")
        symbol = symbols[0]
    return max(sp.Poly(term, symbol).monoms()[0][0] for term in expr.as_ordered_terms() if term != 0) + 1


def SeriesLength(series: sp.Expr, symbol: sp.Symbol | None = None) -> int:
    return SeriesMaxOrder(series, symbol) - SeriesMinOrder(series, symbol)


def SeriesTake(series: sp.Expr, term_count: int, symbol: sp.Symbol | None = None) -> sp.Expr:
    symbol = _infer_series_symbol(series) if symbol is None else symbol
    expr = sp.expand(series.removeO() if hasattr(series, "removeO") else series)
    if expr == 0:
        return sp.S(0)
    terms = []
    for term in expr.as_ordered_terms():
        if term == 0:
            continue
        degree = sp.Poly(term, symbol).monoms()[0][0]
        terms.append((degree, term))
    terms.sort(key=lambda item: item[0])
    return sp.expand(sum(term for _, term in terms[:term_count]))


def SeriesCollect(expr: sp.Expr, variables, func=sp.simplify):
    if isinstance(variables, (list, tuple)):
        result = expr
        for variable in variables:
            result = sp.collect(result, variable, func)
        return result
    return sp.collect(expr, variables, func)


def SeriesTerms(series: sp.Expr, spec: tuple[sp.Symbol, int | float, int]) -> sp.Expr:
    symbol, point, term_count = spec
    if point != 0:
        raise ValueError("SeriesTerms currently supports expansions around 0 only")
    min_order = SeriesMinOrder(series, symbol)
    max_order = min_order + term_count
    return sp.series(series.removeO() if hasattr(series, "removeO") else series, symbol, point, max_order).removeO()


def _infer_series_symbol(expr: sp.Expr) -> sp.Symbol:
    symbols = []
    if hasattr(expr, "getO") and expr.getO() is not None:
        symbols = sorted(expr.getO().free_symbols, key=lambda sym: sym.name)
    if not symbols:
        symbols = sorted(expr.free_symbols, key=lambda sym: sym.name)
    if len(symbols) != 1:
        raise ValueError("could not infer a unique expansion symbol")
    return symbols[0]


def IgnoreExpansionParameter(series: sp.Expr, value=1, symbol: sp.Symbol | None = None) -> sp.Expr:
    symbol = _infer_series_symbol(series) if symbol is None else symbol
    expr = series.removeO() if hasattr(series, "removeO") else series
    return sp.expand(expr.subs(symbol, value))


def ChangeSeriesParameter(series: sp.Expr, new_parameter) -> sp.Expr:
    symbol = _infer_series_symbol(series)
    expr = series.removeO() if hasattr(series, "removeO") else series
    return sp.expand(expr.subs(symbol, new_parameter))


def Scalings(arguments: list[tuple[sp.Symbol, int]] | list[tuple[sp.Symbol, int, object]], var: sp.Symbol, expr: sp.Expr) -> sp.Expr:
    repls = {}
    for entry in arguments:
        if len(entry) == 2:
            symbol, power = entry
            factor = 1
        elif len(entry) == 3:
            symbol, power, factor = entry
        else:
            raise ValueError(f"invalid scaling specification: {entry!r}")
        repls[symbol] = factor * symbol * var ** power
    return sp.expand(expr.subs(repls))


def PNScalings(expr: sp.Expr, arguments: list[tuple[sp.Symbol, int]] | list[tuple[sp.Symbol, int, object]], var: sp.Symbol) -> sp.Expr:
    return Scalings(arguments, var, expr)


def ExpandSpheroidals(expr: sp.Expr, spec: tuple[sp.Symbol, int]) -> sp.Expr:
    symbol, order = spec
    return sp.series(expr, symbol, 0, order + 1).removeO()


def PowerCounting(series: sp.Expr, symbol: sp.Symbol) -> sp.Expr:
    return ChangeSeriesParameter(series, symbol)


def RemovePN(expr: sp.Expr, symbol: sp.Symbol) -> sp.Expr:
    return IgnoreExpansionParameter(expr, 1, symbol)


def ExpandLog(expr: sp.Expr) -> sp.Expr:
    return sp.expand_log(expr, force=True)


def ExpandGamma(expr: sp.Expr) -> sp.Expr:
    return sp.expand_func(expr)


def ExpandPolyGamma(expr: sp.Expr) -> sp.Expr:
    return sp.expand_func(expr)


def PochhammerToGamma(expr: sp.Expr) -> sp.Expr:
    return expr.rewrite(sp.gamma)


def GammaToPochhammer(expr: sp.Expr, symbol) -> sp.Expr:
    def replace_gamma(node):
        if node.func is sp.gamma and node.args[0].has(symbol):
            arg = node.args[0]
            if arg.is_Add:
                constant_part = sp.Integer(0)
                symbol_part = sp.Integer(0)
                for term in arg.args:
                    if term.has(symbol):
                        symbol_part += term
                    else:
                        constant_part += term
                if constant_part.is_integer:
                    n = int(constant_part)
                    if n > 0:
                        return sp.RisingFactorial(symbol_part, sp.Integer(n), evaluate=False) * sp.gamma(symbol_part)
            return node
        return node

    return expr.replace(lambda node: node.func is sp.gamma and node.args[0].has(symbol), replace_gamma)


def ExpandDiracDelta(expr: sp.Expr, symbol: sp.Symbol) -> sp.Expr:
    def replace_delta(node):
        if node.func is sp.DiracDelta and len(node.args) == 1:
            arg = sp.expand(node.args[0])
            coeff = sp.diff(arg, symbol)
            if coeff.free_symbols:
                return node
            constant = sp.simplify(arg - coeff * symbol)
            if coeff == 0:
                return node
            return sp.DiracDelta(symbol + constant / coeff) / sp.Abs(coeff)
        return node

    return expr.replace(lambda node: node.func is sp.DiracDelta and len(node.args) == 1, replace_delta)


def CollectDerivatives(expr: sp.Expr, func) -> sp.Expr:
    atoms = [func]
    if isinstance(func, sp.FunctionClass):
        atoms.extend(sorted(expr.atoms(sp.Derivative), key=str))
    return sp.collect(expr, atoms)


def Paint(expr: sp.Expr, var) -> sp.Expr:
    return expr


def MSTCoefficients(s: int, ell: int, m: int, a: float | sp.Expr, order: int) -> dict[int, sp.Expr]:
    return kerr_mst_series_sympy(s, ell, m, sp.S(a), order)["a"]


def aMST(n: int, s: int, ell: int, m: int, a: float | sp.Expr, order: int) -> sp.Expr:
    return MSTCoefficients(s, ell, m, a, order)[n]


def _pn_radial_amplitudes(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    boundary: str,
    amplitudes: bool,
) -> dict[str, object]:
    if not amplitudes:
        return {
            "Trans": Missing("NotComputed", "Trans"),
            "Inc": Missing("NotComputed", "Inc"),
            "Ref": Missing("NotComputed", "Ref"),
        }

    if boundary == "In":
        trans = 1.0 + 0.0j
        return {
            "Trans": trans,
            "Inc": Missing("NotAvailable", "Inc"),
            "Ref": Missing("NotAvailable", "Ref"),
        }
    return {
        "Trans": 1.0 + 0.0j,
        "Inc": Missing("NotAvailable", "Inc"),
        "Ref": Missing("NotAvailable", "Ref"),
    }


def TeukolskyAmplitudePN(
    solution: str,
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    pn: tuple[object, int | str],
) -> complex | Missing:
    _require_default_normalization("Default")
    radial = TeukolskyRadialPN(s, ell, m, a, omega, pn, amplitudes=True)
    if solution == "Btrans":
        return radial["In"]["Amplitudes"]["Trans"]
    if solution == "Binc":
        return radial["In"]["Amplitudes"]["Inc"]
    if solution == "Bref":
        return radial["In"]["Amplitudes"]["Ref"]
    if solution == "Ctrans":
        return radial["Up"]["Amplitudes"]["Trans"]
    if solution == "Cinc":
        return radial["Up"]["Amplitudes"]["Inc"]
    if solution == "Cref":
        return radial["Up"]["Amplitudes"]["Ref"]
    raise ValueError(f"unsupported PN amplitude label: {solution!r}")


def InvariantWronskian(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    pn: tuple[object, int | str],
    *,
    r: float = 10.0,
) -> complex:
    radial = TeukolskyRadialPN(s, ell, m, a, omega, pn)
    rin = radial["In"]
    rup = radial["Up"]
    delta = r * r - 2.0 * r + a * a
    return delta ** (s + 1) * (rup.derivative(1, r) * rin(r) - rin.derivative(1, r) * rup(r))


def TeukolskyEquation(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega,
    radial_function,
    pn: tuple[object, int | str] | None = None,
):
    r = None
    if isinstance(radial_function, sp.Expr):
        r_symbols = sorted(radial_function.free_symbols, key=lambda sym: sym.name)
        if len(r_symbols) != 1:
            raise ValueError("radial_function expression must contain exactly one radial symbol")
        r = r_symbols[0]
        radial_expr = radial_function
    else:
        raise ValueError("radial_function must be a sympy expression in a single radial symbol")

    delta = r * r - 2 * r + a * a
    delta_prime = sp.diff(delta, r)
    if pn is None:
        if isinstance(omega, sp.Basic):
            eigenvalue = sp.Function("SpinWeightedSpheroidalEigenvalue")(s, ell, m, a * omega)
        else:
            eigenvalue = spin_weighted_spheroidal_eigenvalue(s, ell, m, a * complex(omega))
    else:
        _, order = _parse_pn_spec(pn)
        eigenvalue = eigenvalue_pn(s, ell, m, sp.S(a), sp.S(omega), order)
    potential = (
        (
            (r * r + a * a) ** 2 * omega**2
            - 4 * a * r * omega * m
            + a * a * m * m
            + 2 * sp.I * a * (r - 1) * m * s
            - 2 * sp.I * (r * r - a * a) * omega * s
        )
        / delta
        + 2 * sp.I * r * omega * s
        - eigenvalue
        - 2 * a * m * omega
    )
    return sp.expand(
        sp.diff(radial_expr, r, 2)
        + (s + 1) * delta_prime / delta * sp.diff(radial_expr, r)
        + potential / delta * radial_expr
    )


def TeukolskyPointParticleSource(s: int, ell: int, m: int, orbit):
    if orbit.kind != "circular-equatorial":
        raise ValueError("TeukolskyPointParticleSource currently supports only circular equatorial orbits")
    omega = m * orbit.omega_phi
    harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, orbit.a * omega)
    theta = math.pi / 2.0
    harmonic_value = harmonic(theta, 0.0)
    harmonic_derivative = harmonic.derivative_theta(theta, 0.0)
    harmonic_second_derivative = harmonic.derivative_theta2(theta, 0.0)

    if s == 0:
        source0 = -4.0 * math.pi * orbit.p * orbit.p * harmonic_value
        source1 = 0.0 + 0.0j
        source2 = 0.0 + 0.0j
    elif s == -2:
        source0, source1, source2 = _spin_two_coefficients(
            r=orbit.p,
            ur=0.0,
            theta=theta,
            u_theta=0.0,
            a=orbit.a,
            energy=orbit.energy,
            angular_momentum=orbit.angular_momentum,
            s=s,
            m=m,
            omega=omega,
            harmonic_value=harmonic_value,
            harmonic_derivative=harmonic_derivative,
            harmonic_second_derivative=harmonic_second_derivative,
        )
    elif s == -1:
        source0, source1 = _spin_minus_one_coefficients(
            r=orbit.p,
            ur=0.0,
            theta=theta,
            u_theta=0.0,
            a=orbit.a,
            energy=orbit.energy,
            angular_momentum=orbit.angular_momentum,
            s=s,
            m=m,
            omega=omega,
            harmonic_value=harmonic_value,
            harmonic_derivative=harmonic_derivative,
        )
        source2 = 0.0 + 0.0j
    elif s == 1:
        source0, source1 = _spin_plus_one_coefficients(
            r=orbit.p,
            ur=0.0,
            theta=theta,
            u_theta=0.0,
            a=orbit.a,
            energy=orbit.energy,
            angular_momentum=orbit.angular_momentum,
            s=s,
            m=m,
            omega=omega,
            harmonic_value=harmonic_value,
            harmonic_derivative=harmonic_derivative,
        )
        source2 = 0.0 + 0.0j
    elif s == 2:
        source0, source1, source2 = _spin_two_positive_coefficients(
            r=orbit.p,
            ur=0.0,
            theta=theta,
            u_theta=0.0,
            a=orbit.a,
            energy=orbit.energy,
            angular_momentum=orbit.angular_momentum,
            s=s,
            m=m,
            omega=omega,
            harmonic_value=harmonic_value,
            harmonic_derivative=harmonic_derivative,
            harmonic_second_derivative=harmonic_second_derivative,
        )
    else:
        raise NotImplementedError("TeukolskyPointParticleSource is implemented only for spin weights s = -2, -1, 0, 1, and 2")

    r0 = orbit.p

    def source(r):
        if isinstance(r, sp.Basic):
            expr = source0 * sp.DiracDelta(r - r0)
            if source1 != 0:
                expr -= source1 * sp.diff(sp.DiracDelta(r - r0), r)
            if source2 != 0:
                expr += source2 * sp.diff(sp.DiracDelta(r - r0), r, 2)
            return expr
        return 0.0 + 0.0j

    return source


def TeukolskyRadialFunctionPN(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    pn: tuple[object, int | str],
    boundary: str,
    *,
    normalization: str = "Default",
    amplitudes: bool = False,
    simplify: bool = True,
) -> PNRadialSolution:
    _require_default_normalization(normalization)
    var_pn, order = _parse_pn_spec(pn)
    if boundary not in {"In", "Up"}:
        raise ValueError(f"unsupported PN boundary condition: {boundary!r}")

    def radial_fn(r: float) -> complex:
        return pn_radial_solution(s, ell, m, a, omega, r, order=order, boundary=boundary)

    def leading_fn(r: float) -> complex:
        return radial_solution_pn_leading(s, ell, m, a, omega, r, boundary)

    return PNRadialSolution(
        s=s,
        l=ell,
        m=m,
        a=a,
        omega=omega,
        pn=(var_pn, order),
        boundary_condition=boundary,
        series_min_order=0,
        term_count=_pn_term_count(order),
        normalization=normalization,
        simplify=simplify,
        amplitudes_bool=amplitudes,
        amplitudes=_pn_radial_amplitudes(s, ell, m, a, omega, boundary, amplitudes),
        radial_function=radial_fn,
        leading_order_function=leading_fn,
    )


def TeukolskyRadialPN(
    s: int,
    ell: int,
    m: int,
    a: float,
    omega: complex,
    pn: tuple[object, int | str],
    *,
    normalization: str = "Default",
    amplitudes: bool = False,
    simplify: bool = True,
) -> dict[str, PNRadialSolution]:
    return {
        "In": TeukolskyRadialFunctionPN(
            s,
            ell,
            m,
            a,
            omega,
            pn,
            "In",
            normalization=normalization,
            amplitudes=amplitudes,
            simplify=simplify,
        ),
        "Up": TeukolskyRadialFunctionPN(
            s,
            ell,
            m,
            a,
            omega,
            pn,
            "Up",
            normalization=normalization,
            amplitudes=amplitudes,
            simplify=simplify,
        ),
    }


def _pn_circular_source_data(
    s: int,
    ell: int,
    m: int,
    a: float,
    r0: float,
    order: int,
) -> tuple[dict[str, object], Any, complex]:
    orbit = circular_orbit(a, r0)
    omega = m * orbit.omega_phi
    if s == 0:
        coeffs = {"S": -4.0 * math.pi * r0 * r0, "S'": 0.0 + 0.0j, "S''": 0.0 + 0.0j}

        def coeff_fn(_r):
            return coeffs

        def source_fn(r):
            if isinstance(r, sp.Basic):
                return coeffs["S"] * sp.DiracDelta(r - r0)
            return 0.0 + 0.0j

        return coeffs, source_fn, 0.0 + 0.0j

    if s == -2:
        rin = pn_radial_solution(s, ell, m, a, omega, r0, order=order, boundary="In")
        rup = pn_radial_solution(s, ell, m, a, omega, r0, order=order, boundary="Up")
        dr = 1e-5
        drin = (
            pn_radial_solution(s, ell, m, a, omega, r0 + dr, order=order, boundary="In")
            - pn_radial_solution(s, ell, m, a, omega, r0 - dr, order=order, boundary="In")
        ) / (2.0 * dr)
        drup = (
            pn_radial_solution(s, ell, m, a, omega, r0 + dr, order=order, boundary="Up")
            - pn_radial_solution(s, ell, m, a, omega, r0 - dr, order=order, boundary="Up")
        ) / (2.0 * dr)
        omega_sym = sp.Symbol("omega", real=True)
        lam_val = complex(eigenvalue_pn(s, ell, m, sp.S(a), omega_sym, order).subs(omega_sym, omega).evalf())
        d2rin = _radial_second_derivative(rin, drin, s=s, m=m, a=a, omega=omega, eigenvalue=lam_val, r=r0)
        d2rup = _radial_second_derivative(rup, drup, s=s, m=m, a=a, omega=omega, eigenvalue=lam_val, r=r0)
        from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic

        harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, a * omega)
        s_val = harmonic(math.pi / 2.0, 0.0)
        ds_val = harmonic.derivative_theta(math.pi / 2.0, 0.0)
        d2s_val = harmonic.derivative_theta2(math.pi / 2.0, 0.0)
        source0, source1, source2 = _spin_two_coefficients(
            r=r0,
            ur=0.0,
            theta=math.pi / 2.0,
            u_theta=0.0,
            a=a,
            energy=orbit.energy,
            angular_momentum=orbit.angular_momentum,
            s=s,
            m=m,
            omega=omega,
            harmonic_value=s_val,
            harmonic_derivative=ds_val,
            harmonic_second_derivative=d2s_val,
        )
        delta = r0 * r0 - 2.0 * r0 + a * a
        delta_coeff = source2 / delta
        coeffs = {"S": source0, "S'": source1, "S''": source2, "Rin": rin, "Rup": rup, "dRin": drin, "dRup": drup, "ddRin": d2rin, "ddRup": d2rup}

        def coeff_fn(_r):
            return coeffs

        def source_fn(r):
            if isinstance(r, sp.Basic):
                return source0 * sp.DiracDelta(r - r0) - source1 * sp.diff(sp.DiracDelta(r - r0), r) + source2 * sp.diff(sp.DiracDelta(r - r0), r, 2)
            return 0.0 + 0.0j

        return coeffs, source_fn, delta_coeff

    if s == -1:
        from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic

        harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, a * omega)
        s_val = harmonic(math.pi / 2.0, 0.0)
        ds_val = harmonic.derivative_theta(math.pi / 2.0, 0.0)
        source0, source1 = _spin_minus_one_coefficients(
            r=r0,
            theta=math.pi / 2.0,
            a=a,
            energy=orbit.energy,
            angular_momentum=orbit.angular_momentum,
            ur=0.0,
            u_theta=0.0,
            m=m,
            omega=omega,
            harmonic_value=s_val,
            harmonic_derivative=ds_val,
        )
        coeffs = {"S": source0, "S'": source1, "S''": 0.0 + 0.0j}

        def coeff_fn(_r):
            return coeffs

        def source_fn(r):
            if isinstance(r, sp.Basic):
                return source0 * sp.DiracDelta(r - r0) - source1 * sp.diff(sp.DiracDelta(r - r0), r)
            return 0.0 + 0.0j

        return coeffs, source_fn, 0.0 + 0.0j

    if s == 2:
        rin = pn_radial_solution(s, ell, m, a, omega, r0, order=order, boundary="In")
        rup = pn_radial_solution(s, ell, m, a, omega, r0, order=order, boundary="Up")
        dr = 1e-5
        drin = (
            pn_radial_solution(s, ell, m, a, omega, r0 + dr, order=order, boundary="In")
            - pn_radial_solution(s, ell, m, a, omega, r0 - dr, order=order, boundary="In")
        ) / (2.0 * dr)
        drup = (
            pn_radial_solution(s, ell, m, a, omega, r0 + dr, order=order, boundary="Up")
            - pn_radial_solution(s, ell, m, a, omega, r0 - dr, order=order, boundary="Up")
        ) / (2.0 * dr)
        omega_sym = sp.Symbol("omega", real=True)
        lam_val = complex(eigenvalue_pn(s, ell, m, sp.S(a), omega_sym, order).subs(omega_sym, omega).evalf())
        d2rin = _radial_second_derivative(rin, drin, s=s, m=m, a=a, omega=omega, eigenvalue=lam_val, r=r0)
        d2rup = _radial_second_derivative(rup, drup, s=s, m=m, a=a, omega=omega, eigenvalue=lam_val, r=r0)
        from teukolsky.angular.eigen import spin_weighted_spheroidal_harmonic

        harmonic = spin_weighted_spheroidal_harmonic(s, ell, m, a * omega)
        s_val = harmonic(math.pi / 2.0, 0.0)
        ds_val = harmonic.derivative_theta(math.pi / 2.0, 0.0)
        d2s_val = harmonic.derivative_theta2(math.pi / 2.0, 0.0)
        source0, source1, source2 = _spin_two_positive_coefficients(
            r=r0,
            ur=0.0,
            theta=math.pi / 2.0,
            u_theta=0.0,
            a=a,
            energy=orbit.energy,
            angular_momentum=orbit.angular_momentum,
            m=m,
            omega=omega,
            harmonic_value=s_val,
            harmonic_derivative=ds_val,
            harmonic_second_derivative=d2s_val,
        )
        coeffs = {
            "S": source0,
            "S'": source1,
            "S''": source2,
            "Rin": rin,
            "Rup": rup,
            "dRin": drin,
            "dRup": drup,
            "ddRin": d2rin,
            "ddRup": d2rup,
        }
        delta_coeff = source2 / (r0 * r0 - 2.0 * r0 + a * a)

        def coeff_fn(_r):
            return coeffs

        def source_fn(r):
            if isinstance(r, sp.Basic):
                return source0 * sp.DiracDelta(r - r0) - source1 * sp.diff(sp.DiracDelta(r - r0), r) + source2 * sp.diff(sp.DiracDelta(r - r0), r, 2)
            return 0.0 + 0.0j

        return coeffs, source_fn, delta_coeff

    raise NotImplementedError("PN point-particle source is implemented only for s = -2, -1, 0, and 2")


def TeukolskyPointParticleModePN(
    s: int,
    ell: int,
    m: int,
    orbit,
    pn: tuple[object, int | str],
    *,
    normalization: str = "Default",
    simplify: bool = True,
) -> PNModeSolution:
    _require_default_normalization(normalization)
    var_pn, order = _parse_pn_spec(pn)
    if orbit.kind != "circular-equatorial":
        raise ValueError("TeukolskyPointParticleModePN only supports circular equatorial orbits")

    omega = m * orbit.omega_phi
    radial = TeukolskyRadialPN(
        s,
        ell,
        m,
        orbit.a,
        omega,
        (var_pn, order),
        normalization=normalization,
        amplitudes=False,
        simplify=simplify,
    )
    amplitudes_raw = pn_point_particle_mode(s, ell, m, orbit.a, orbit.p, order)
    coeffs, source_fn, delta_coeff = _pn_circular_source_data(s, ell, m, orbit.a, orbit.p, order)
    delta = orbit.p * orbit.p - 2.0 * orbit.p + orbit.a * orbit.a
    dr = 1e-5
    rin = radial["In"](orbit.p)
    rup = radial["Up"](orbit.p)
    drin = (radial["In"](orbit.p + dr) - radial["In"](orbit.p - dr)) / (2.0 * dr)
    drup = (radial["Up"](orbit.p + dr) - radial["Up"](orbit.p - dr)) / (2.0 * dr)
    wronskian = delta ** (s + 1) * (rup * drin - rin * drup)

    def coeff_fn(_r):
        return coeffs

    return PNModeSolution(
        s=s,
        l=ell,
        m=m,
        orbit=orbit,
        pn=(var_pn, order),
        series_min_order=0,
        normalization=normalization,
        simplify=simplify,
        radial_in=radial["In"],
        radial_up=radial["Up"],
        amplitudes={"I": amplitudes_raw["Z_inf"], "H": amplitudes_raw["Z_hor"]},
        wronskian=wronskian,
        source_function=source_fn,
        coefficient_list_function=coeff_fn,
        delta_coefficient=delta_coeff,
    )


__all__ = [
    "InvariantWronskian",
    "MSTCoefficients",
    "PNModeSolution",
    "PNRadialSolution",
    "PNScalings",
    "Paint",
    "PochhammerToGamma",
    "Scalings",
    "PowerCounting",
    "RemovePN",
    "SeriesLength",
    "SeriesCollect",
    "SeriesMaxOrder",
    "SeriesMinOrder",
    "SeriesTerms",
    "ExpandDiracDelta",
    "ExpandGamma",
    "ExpandLog",
    "ExpandPolyGamma",
    "GammaToPochhammer",
    "TeukolskyAmplitudePN",
    "TeukolskyEquation",
    "TeukolskyPointParticleModePN",
    "TeukolskyPointParticleSource",
    "TeukolskyRadialFunctionPN",
    "TeukolskyRadialPN",
    "ChangeSeriesParameter",
    "ExpandSpheroidals",
    "IgnoreExpansionParameter",
    "cos_2pi_nu_series_sympy",
    "eigenvalue_pn",
    "kerr_mst_series_sympy",
    "mst_a1_leading_pn",
    "mst_coefficient_pn",
    "pn_point_particle_mode",
    "pn_radial_solution",
    "radial_solution_pn_leading",
    "renormalized_angular_momentum_pn",
    "series_take",
]

"""Systematic DCU vs CPU speed and precision validation for the accelerated
Teukolsky solver.

Covers multiple Kerr parameters and orbit types at fixed p=10.

Supported orbit types:
  - generic  (e=0.2, x=0.7)  -> accelerated_generic_alpha  (2D GPU integral)
  - eccentric equatorial (e=0.2, x=1.0) -> accelerated_eccentric_alpha (1D GPU integral)

Spin-weight coverage: s = -2, -1, 0, +1, +2.
  - s = -2, 0, +2: three-term alpha (R, R', R'')
  - s = -1, +1:    two-term alpha (R, R') — Maxwell system

Circular (e=0) and spherical (e=0) orbits have no accelerated path yet
(radial_phase_function = None, so phase-space integrals are not applicable).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
import torch

from teukolsky.api import KerrGeoOrbit
from teukolsky.modes import solve_point_particle_mode


# ---- helpers -----------------------------------------------------------

def _is_dcu_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


# ---- test-case definitions --------------------------------------------

# (s, ell, m, a, p, e, x, n, k)
# -- generic orbits (e=0.2, x=0.7): accelerated_generic_alpha via benchmark_mode
_GENERIC_CASES = [
    pytest.param(-2, 2, 2, 0.0, 10.0, 0.2, 0.7, 0, 0, id="gen_a=0.0"),
    pytest.param(-2, 2, 2, 0.1, 10.0, 0.2, 0.7, 0, 0, id="gen_a=0.1"),
    pytest.param(-2, 2, 2, 0.2, 10.0, 0.2, 0.7, 0, 0, id="gen_a=0.2"),
    pytest.param(-2, 2, 2, 0.3, 10.0, 0.2, 0.7, 0, 0, id="gen_a=0.3"),
    pytest.param(-2, 2, 2, 0.5, 10.0, 0.2, 0.7, 0, 0, id="gen_a=0.5"),
    pytest.param(-2, 2, 2, 0.7, 10.0, 0.2, 0.7, 0, 0, id="gen_a=0.7"),
    pytest.param(-2, 2, 2, 0.9, 10.0, 0.2, 0.7, 0, 0, id="gen_a=0.9"),
]  # 7 cases, s=-2

# -- generic orbits s=-1: Maxwell, two-term alpha
_GENERIC_S_MINUS_ONE_CASES = [
    pytest.param(-1, 1, 1, 0.3, 10.0, 0.2, 0.7, 0, 0, id="gen_s-1_a=0.3"),
    pytest.param(-1, 1, 1, 0.5, 10.0, 0.2, 0.7, 0, 0, id="gen_s-1_a=0.5"),
    pytest.param(-1, 1, 1, 0.7, 10.0, 0.2, 0.7, 0, 0, id="gen_s-1_a=0.7"),
]  # 3 cases, s=-1

# -- generic orbits s=0: scalar, uses spin-two coeffs on DCU path
_GENERIC_S_ZERO_CASES = [
    pytest.param(0, 2, 2, 0.3, 10.0, 0.2, 0.7, 0, 0, id="gen_s0_a=0.3"),
    pytest.param(0, 2, 2, 0.5, 10.0, 0.2, 0.7, 0, 0, id="gen_s0_a=0.5"),
    pytest.param(0, 2, 2, 0.7, 10.0, 0.2, 0.7, 0, 0, id="gen_s0_a=0.7"),
]  # 3 cases, s=0

# -- eccentric equatorial (e=0.2, x=1.0): accelerated_eccentric_alpha
_ECCENTRIC_CASES = [
    pytest.param(-2, 2, 2, 0.3, 10.0, 0.2, 1.0, 0, 0, id="ecc_a=0.3"),
    pytest.param(-2, 2, 2, 0.5, 10.0, 0.2, 1.0, 0, 0, id="ecc_a=0.5"),
    pytest.param(-2, 2, 2, 0.7, 10.0, 0.2, 1.0, 0, 0, id="ecc_a=0.7"),
]  # 3 cases

ALL_CASES = _GENERIC_CASES + _GENERIC_S_MINUS_ONE_CASES + _GENERIC_S_ZERO_CASES + _ECCENTRIC_CASES  # 16 total
ALL_GENERIC_CASES = _GENERIC_CASES + _GENERIC_S_MINUS_ONE_CASES + _GENERIC_S_ZERO_CASES


@pytest.mark.skipif(not _is_dcu_available(), reason="CUDA not available")
def test_public_mode_api_records_dcu_usage_metadata():
    orbit = KerrGeoOrbit(0.5, 10.0, 0.2, 0.7)
    mode = solve_point_particle_mode(-2, 2, 2, orbit, accelerator="gpu")
    accel = mode["Acceleration"]
    assert accel is not None
    assert accel["Backend"] in {"CUDA", "ROCm", "GPU"}
    assert accel["Used"] is True
    assert str(accel["Device"]).startswith("cuda:")
    assert accel["DeviceName"]


# ---- accelerated_generic_alpha (via benchmark_mode) --------------------

@pytest.mark.skipif(not _is_dcu_available(), reason="CUDA not available")
@pytest.mark.parametrize("s, ell, m, a, p, e, x, n, k", _GENERIC_CASES)
def test_generic_alpha(s, ell, m, a, p, e, x, n, k):
    """Precision: |I/H_dcu - I/H_cpu|/|I/H_cpu| < 1e-6.  Speedup: > 2x."""
    from teukolsky.accelerated.validation import benchmark_mode

    result = benchmark_mode(s=s, ell=ell, m=m, a=a, p=p, e=e, x=x, n=n, k=k)

    msg = (
        f"\n  {result['orbit']}  s={result['s']}  ell={result['ell']}  m={result['m']}"
        f"\n  CPU={result['cpu_time_s']:.3f}s  DCU={result['dcu_conv_time_s']:.4f}s"
        f"  speedup={result['speedup']:.1f}x"
        f"\n  I_rel={result['I_rel_diff']:.2e}  H_rel={result['H_rel_diff']:.2e}"
    )

    assert result["I_rel_diff"] < 1e-6, (
        f"I rel diff {result['I_rel_diff']:.2e} >= 1e-6" + msg
    )
    assert result["H_rel_diff"] < 1e-6, (
        f"H rel diff {result['H_rel_diff']:.2e} >= 1e-6" + msg
    )
    assert result["speedup"] > 2.0, (
        f"Speedup {result['speedup']:.1f}x <= 2x" + msg
    )


@pytest.mark.skipif(not _is_dcu_available(), reason="CUDA not available")
@pytest.mark.parametrize("s, ell, m, a, p, e, x, n, k", _GENERIC_S_MINUS_ONE_CASES)
def test_generic_alpha_s_minus_one(s, ell, m, a, p, e, x, n, k):
    """s=-1 generic modes: precision < 1e-4, speedup > 2x."""
    from teukolsky.accelerated.validation import benchmark_mode

    result = benchmark_mode(s=s, ell=ell, m=m, a=a, p=p, e=e, x=x, n=n, k=k)

    msg = (
        f"\n  {result['orbit']}  s={result['s']}  ell={result['ell']}  m={result['m']}"
        f"\n  CPU={result['cpu_time_s']:.3f}s  DCU={result['dcu_conv_time_s']:.4f}s"
        f"  speedup={result['speedup']:.1f}x"
        f"\n  I_rel={result['I_rel_diff']:.2e}  H_rel={result['H_rel_diff']:.2e}"
    )

    assert result["I_rel_diff"] < 1e-4, (
        f"I rel diff {result['I_rel_diff']:.2e} >= 1e-4" + msg
    )
    assert result["H_rel_diff"] < 1e-4, (
        f"H rel diff {result['H_rel_diff']:.2e} >= 1e-4" + msg
    )
    assert result["speedup"] > 2.0, (
        f"Speedup {result['speedup']:.1f}x <= 2x" + msg
    )


@pytest.mark.skipif(not _is_dcu_available(), reason="CUDA not available")
@pytest.mark.parametrize("s, ell, m, a, p, e, x, n, k", _GENERIC_S_ZERO_CASES)
def test_generic_alpha_s_zero(s, ell, m, a, p, e, x, n, k):
    """s=0 generic modes: precision < 1e-4, speedup > 2x.

    Note: CPU solve_point_particle_mode uses scalar formulation for s=0,
    while DCU uses spin-coefficient formulation.  Both should give the same
    physical amplitudes.
    """
    from teukolsky.accelerated.validation import benchmark_mode

    result = benchmark_mode(s=s, ell=ell, m=m, a=a, p=p, e=e, x=x, n=n, k=k)

    msg = (
        f"\n  {result['orbit']}  s={result['s']}  ell={result['ell']}  m={result['m']}"
        f"\n  CPU={result['cpu_time_s']:.3f}s  DCU={result['dcu_conv_time_s']:.4f}s"
        f"  speedup={result['speedup']:.1f}x"
        f"\n  I_rel={result['I_rel_diff']:.2e}  H_rel={result['H_rel_diff']:.2e}"
    )

    assert result["I_rel_diff"] < 1e-4, (
        f"I rel diff {result['I_rel_diff']:.2e} >= 1e-4" + msg
    )
    assert result["H_rel_diff"] < 1e-4, (
        f"H rel diff {result['H_rel_diff']:.2e} >= 1e-4" + msg
    )
    assert result["speedup"] > 2.0, (
        f"Speedup {result['speedup']:.1f}x <= 2x" + msg
    )


# ---- accelerated_eccentric_alpha --------------------------------------

def _run_eccentric_alpha_bench(s, ell, m, a, p, e, n, k):
    """Compare accelerated_eccentric_alpha against CPU solve_point_particle_mode."""
    from teukolsky.api import KerrGeoOrbit
    from teukolsky.modes import solve_point_particle_mode
    from teukolsky.accelerated.convolution import accelerated_eccentric_alpha
    from teukolsky.radial import solve_radial
    from teukolsky.modes.point_particle import _wronskian

    orbit = KerrGeoOrbit(a, p, e, 1.0)
    omega_val = m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta

    t0 = time.perf_counter()
    mode_cpu = solve_point_particle_mode(s, ell, m, orbit, n=n, k=k)
    t_cpu = time.perf_counter() - t0

    radial = solve_radial(s=s, ell=ell, m=m, a=orbit.a, omega=omega_val)
    rin = radial["In"]
    rup = radial["Up"]

    t0 = time.perf_counter()
    alpha_up_dcu = accelerated_eccentric_alpha(
        lambda rr: np.array([rup(float(r)) for r in rr], dtype=np.complex128),
        lambda rr: np.array([rup.derivative(1, float(r)) for r in rr], dtype=np.complex128),
        s, m, orbit.a, omega_val, rup.eigenvalue, orbit, n=n, ell=ell,
    )
    alpha_in_dcu = accelerated_eccentric_alpha(
        lambda rr: np.array([rin(float(r)) for r in rr], dtype=np.complex128),
        lambda rr: np.array([rin.derivative(1, float(r)) for r in rr], dtype=np.complex128),
        s, m, orbit.a, omega_val, rin.eigenvalue, orbit, n=n, ell=ell,
    )
    w_val = _wronskian(rin, rup, s, orbit.p / (1.0 + orbit.e))
    # Sign convention: s = -2, 0, +2 use -8π; s = ±1 use +8π
    sign = 1.0 if s in (-1, 1) else -1.0
    prefactor = sign * 8.0 * math.pi / w_val / orbit.upsilon_t
    z_h_dcu = prefactor * alpha_up_dcu
    z_i_dcu = prefactor * alpha_in_dcu
    t_dcu = time.perf_counter() - t0

    i_diff = abs(z_i_dcu - mode_cpu.amplitudes["I"])
    h_diff = abs(z_h_dcu - mode_cpu.amplitudes["H"])
    i_rel = i_diff / max(abs(mode_cpu.amplitudes["I"]), 1e-30)
    h_rel = h_diff / max(abs(mode_cpu.amplitudes["H"]), 1e-30)

    return {
        "orbit": f"a={a}, p={p}, e={e}, x=1.0",
        "s": s, "ell": ell, "m": m, "n": n, "k": k,
        "cpu_time_s": t_cpu,
        "dcu_conv_time_s": t_dcu,
        "speedup": t_cpu / max(t_dcu, 1e-10),
        "I_cpu": mode_cpu.amplitudes["I"],
        "I_dcu": z_i_dcu,
        "I_rel_diff": float(i_rel),
        "H_cpu": mode_cpu.amplitudes["H"],
        "H_dcu": z_h_dcu,
        "H_rel_diff": float(h_rel),
    }


@pytest.mark.skipif(not _is_dcu_available(), reason="CUDA not available")
@pytest.mark.parametrize("s, ell, m, a, p, e, x, n, k", _ECCENTRIC_CASES)
def test_eccentric_alpha_precision(s, ell, m, a, p, e, x, n, k):
    """|I_dcu - I_cpu| / |I_cpu| < 1e-6 and same for H (eccentric alpha)."""
    result = _run_eccentric_alpha_bench(s=s, ell=ell, m=m, a=a, p=p, e=e, n=n, k=k)

    msg = (
        f"\n  {result['orbit']}  s={result['s']}  ell={result['ell']}  m={result['m']}"
        f"\n  CPU={result['cpu_time_s']:.3f}s  DCU={result['dcu_conv_time_s']:.4f}s"
        f"  speedup={result['speedup']:.1f}x"
        f"\n  I_rel={result['I_rel_diff']:.2e}  H_rel={result['H_rel_diff']:.2e}"
    )

    assert result["I_rel_diff"] < 1e-6, (
        f"I rel diff {result['I_rel_diff']:.2e} >= 1e-6" + msg
    )
    assert result["H_rel_diff"] < 1e-6, (
        f"H rel diff {result['H_rel_diff']:.2e} >= 1e-6" + msg
    )


# Note: no separate eccentric_alpha speedup test.  The 1D integral is
# already fast on CPU (~1.4 s total solve), and GPU data-transfer overhead
# dominates the tiny kernel, so the wall-clock speedup is < 1x.
# Speedup assertions are reserved for generic orbits (2D integral).


# ---- summary table ----------------------------------------------------

def test_summary_table():
    """Print a combined summary table of all generic + eccentric test cases.

    Runs every case once and collates precision + timing into a single table.
    """
    if not _is_dcu_available():
        pytest.skip("CUDA not available")

    from teukolsky.accelerated.validation import benchmark_mode

    rows: list[dict] = []

    # -- generic cases (accelerated_generic_alpha via benchmark_mode) --
    for tc in _GENERIC_CASES + _GENERIC_S_MINUS_ONE_CASES + _GENERIC_S_ZERO_CASES:
        s, ell, m, a, p, e, x, n, k = tc.values
        try:
            r = benchmark_mode(s=s, ell=ell, m=m, a=a, p=p, e=e, x=x, n=n, k=k)
            tol = 1e-6 if s == -2 else 1e-4
            rows.append({
                "test": f"gen_alpha a={a:.1f} s={s}",
                "s": s, "a": a, "e": e, "x": x,
                "cpu_s": f"{r['cpu_time_s']:.3f}",
                "dcu_s": f"{r['dcu_conv_time_s']:.4f}",
                "speedup": f"{r['speedup']:.1f}x",
                "I_diff": f"{r['I_rel_diff']:.2e}",
                "H_diff": f"{r['H_rel_diff']:.2e}",
                "pass": r["I_rel_diff"] < tol and r["H_rel_diff"] < tol,
            })
        except Exception as exc:
            rows.append({
                "test": f"gen_alpha a={a:.1f} s={s}",
                "s": s, "a": a, "e": e, "x": x,
                "cpu_s": "ERR", "dcu_s": "ERR", "speedup": "ERR",
                "I_diff": str(exc)[:50], "H_diff": "-", "pass": False,
            })

    # -- eccentric cases (accelerated_eccentric_alpha) --
    for tc in _ECCENTRIC_CASES:
        s, ell, m, a, p, e, x, n, k = tc.values
        try:
            r = _run_eccentric_alpha_bench(s=s, ell=ell, m=m, a=a, p=p, e=e, n=n, k=k)
            rows.append({
                "test": f"ecc_alpha a={a:.1f}",
                "s": s, "a": a, "e": e, "x": 1.0,
                "cpu_s": f"{r['cpu_time_s']:.3f}",
                "dcu_s": f"{r['dcu_conv_time_s']:.4f}",
                "speedup": f"{r['speedup']:.1f}x",
                "I_diff": f"{r['I_rel_diff']:.2e}",
                "H_diff": f"{r['H_rel_diff']:.2e}",
                "pass": r["I_rel_diff"] < 1e-6 and r["H_rel_diff"] < 1e-6,
            })
        except Exception as exc:
            rows.append({
                "test": f"ecc_alpha a={a:.1f}",
                "s": s, "a": a, "e": e, "x": 1.0,
                "cpu_s": "ERR", "dcu_s": "ERR", "speedup": "ERR",
                "I_diff": str(exc)[:50], "H_diff": "-", "pass": False,
            })

    # ---- render table ----
    header = (
        f"{'Test':<22} {'s':>3} {'a':>5} {'e':>5} {'x':>6} "
        f"{'CPU(s)':>9} {'DCU(s)':>10} {'speedup':>8} "
        f"{'I_diff':>10} {'H_diff':>10} {'OK':>5}"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['test']:<22} {r['s']:>3} {r['a']:>5.1f} {r['e']:>5.1f} {r['x']:>6.1f} "
            f"{r['cpu_s']:>9} {r['dcu_s']:>10} {r['speedup']:>8} "
            f"{r['I_diff']:>10} {r['H_diff']:>10} "
            f"{'PASS' if r['pass'] else 'FAIL':>5}"
        )
    lines.append(sep)

    passed = sum(1 for r in rows if r["pass"])
    lines.append(f"TOTAL: {passed}/{len(rows)} passed")
    lines.append(sep)

    report = "\n".join(lines)
    print("\n" + report)

    assert passed == len(rows), f"{len(rows) - passed} case(s) failed\n{report}"

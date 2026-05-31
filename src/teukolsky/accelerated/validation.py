"""Speed and precision validation for GPU-accelerated Teukolsky computations.

Compares CPU (NumericalIntegration) results with GPU-accelerated versions
to verify that:
  1. Numerical precision is maintained (relative difference < 1e-6)
  2. GPU acceleration provides meaningful speedup (> 3x for generic modes)
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch


def dcu_execution_report(device_id: int = 0) -> dict[str, Any]:
    from teukolsky.accelerated.backend import require_dcu

    status = require_dcu(device_id)
    torch.cuda.reset_peak_memory_stats(device_id)
    before = int(torch.cuda.memory_allocated(device_id))
    x = torch.arange(1024, device=f"cuda:{device_id}", dtype=torch.float64)
    y = (x * x).sum()
    torch.cuda.synchronize(device_id)
    after = int(torch.cuda.memory_allocated(device_id))
    peak = int(torch.cuda.max_memory_allocated(device_id))
    return {
        "backend": status["backend"],
        "device": status["device"],
        "device_name": status["device_name"],
        "device_id": device_id,
        "torch_version": status["torch_version"],
        "memory_before_bytes": before,
        "memory_after_bytes": after,
        "peak_memory_bytes": peak,
        "sample_result": float(y.cpu()),
    }


def benchmark_mode(
    s: int, ell: int, m: int, a: float, p: float, e: float, x: float,
    n: int = 0, k: int = 0,
) -> dict[str, Any]:
    """Benchmark CPU vs DCU for a single point-particle mode.

    Returns dict with timing and precision metrics.
    """
    from teukolsky.api import KerrGeoOrbit
    from teukolsky.modes import solve_point_particle_mode

    orbit = KerrGeoOrbit(a, p, e, x)

    # CPU benchmark
    # For s=0 generic/eccentric orbits, the scalar path in
    # solve_point_particle_mode uses a different formulation.
    # Use _solve_generic_mode / _solve_eccentric_equatorial_mode directly
    # to match the spin-coefficient formulation used by the DCU path.
    t0 = time.perf_counter()
    if s == 0 and e > 0:
        if x < 1.0 or k != 0:
            from teukolsky.modes.point_particle import _solve_generic_mode
            mode_cpu = _solve_generic_mode(s, ell, m, orbit, n, k)
        else:
            from teukolsky.modes.point_particle import _solve_eccentric_equatorial_mode
            mode_cpu = _solve_eccentric_equatorial_mode(s, ell, m, orbit, n, k)
    else:
        mode_cpu = solve_point_particle_mode(s, ell, m, orbit, n=n, k=k)
    t_cpu = time.perf_counter() - t0

    # GPU benchmark through the public solve_point_particle_mode API
    from teukolsky.accelerated.backend import require_dcu
    device_status = require_dcu(0)
    torch.cuda.reset_peak_memory_stats(0)
    mem_before = torch.cuda.memory_allocated(0)
    t0 = time.perf_counter()
    mode_dcu = solve_point_particle_mode(
        s, ell, m, orbit, n=n, k=k, accelerator="gpu", device_id=0
    )
    torch.cuda.synchronize(0)
    t_dcu = time.perf_counter() - t0
    mem_after = torch.cuda.memory_allocated(0)
    peak_mem = torch.cuda.max_memory_allocated(0)

    # Compare amplitudes
    z_i_dcu = mode_dcu.amplitudes["I"]
    z_h_dcu = mode_dcu.amplitudes["H"]
    i_diff = abs(z_i_dcu - mode_cpu.amplitudes["I"])
    h_diff = abs(z_h_dcu - mode_cpu.amplitudes["H"])
    i_rel = i_diff / max(abs(mode_cpu.amplitudes["I"]), 1e-30)
    h_rel = h_diff / max(abs(mode_cpu.amplitudes["H"]), 1e-30)

    return {
        "orbit": f"a={a}, p={p}, e={e}, x={x:.4f}",
        "s": s, "ell": ell, "m": m, "n": n, "k": k,
        "cpu_time_s": t_cpu,
        "dcu_conv_time_s": t_dcu,
        "speedup": t_cpu / max(t_dcu, 1e-10),
        "dcu_backend": device_status["backend"],
        "dcu_device": device_status["device"],
        "dcu_device_name": device_status["device_name"],
        "dcu_used": bool(mode_dcu["Acceleration"] and mode_dcu["Acceleration"]["Used"]),
        "dcu_memory_before_bytes": int(mem_before),
        "dcu_memory_after_bytes": int(mem_after),
        "dcu_peak_memory_bytes": int(peak_mem),
        "I_cpu": mode_cpu.amplitudes["I"],
        "I_dcu": z_i_dcu,
        "I_rel_diff": float(i_rel),
        "H_cpu": mode_cpu.amplitudes["H"],
        "H_dcu": z_h_dcu,
        "H_rel_diff": float(h_rel),
        "dcu_matches": bool(i_rel < 1e-6 and h_rel < 1e-6),
    }


def validate_precision(
    test_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run precision validation across a set of test cases.

    Parameters
    ----------
    test_cases : list of dict, optional
        Each dict has keys: s, ell, m, a, p, e, x, n, k.
        If None, uses a default set.

    Returns
    -------
    dict with summary statistics.
    """
    if test_cases is None:
        test_cases = [
            {"s": -2, "ell": 2, "m": 2, "a": 0.5, "p": 10.0, "e": 0.0, "x": 1.0, "n": 0, "k": 0},
            {"s": -2, "ell": 2, "m": 2, "a": 0.5, "p": 10.0, "e": 0.2, "x": 1.0, "n": 0, "k": 0},
            {"s": -2, "ell": 2, "m": 2, "a": 0.5, "p": 8.0, "e": 0.0, "x": 0.7, "n": 0, "k": 0},
            {"s": -2, "ell": 2, "m": 2, "a": 0.5, "p": 10.0, "e": 0.2, "x": 0.7, "n": 0, "k": 0},
            {"s": 0, "ell": 2, "m": 2, "a": 0.5, "p": 10.0, "e": 0.2, "x": 0.7, "n": 0, "k": 0},
        ]

    results = []
    for tc in test_cases:
        print(f"Testing s={tc['s']}, a={tc['a']}, p={tc['p']}, e={tc['e']}, x={tc['x']:.4f} ...")
        result = benchmark_mode(**tc)
        results.append(result)
        status = "✓" if result["dcu_matches"] else "✗"
        print(f"  {status} CPU={result['cpu_time_s']:.2f}s, DCU={result['dcu_conv_time_s']:.3f}s, "
              f"speedup={result['speedup']:.1f}x, I_diff={result['I_rel_diff']:.2e}, H_diff={result['H_rel_diff']:.2e}")

    passed = sum(1 for r in results if r["dcu_matches"])
    max_i_diff = max(r["I_rel_diff"] for r in results)
    max_h_diff = max(r["H_rel_diff"] for r in results)
    avg_speedup = np.mean([r["speedup"] for r in results])

    return {
        "total": len(results),
        "passed": passed,
        "max_I_rel_diff": max_i_diff,
        "max_H_rel_diff": max_h_diff,
        "avg_speedup": avg_speedup,
        "results": results,
    }


def benchmark_batch_modes(s: int, ell: int, m: int, orbit,
                          nk_list: list[tuple[int, int]],
                          n_workers: int | None = None):
    """Benchmark batch vs sequential radial solves for multiple (n,k) harmonics.

    Parameters
    ----------
    s : int
        Spin weight.
    ell : int
        Angular mode number.
    m : int
        Azimuthal mode number.
    orbit : Orbit
        Kerr geodesic orbit.
    nk_list : list of tuple[int, int]
        List of (n, k) harmonic pairs to solve.
    n_workers : int or None
        Number of parallel workers. Defaults to min(len(nk_list), cpu_count()).

    Returns
    -------
    dict with keys:
        n_modes, seq_time_s, batch_time_s, speedup,
        in_match (bool), up_match (bool), max_in_rel_diff, max_up_rel_diff
    """
    import time

    from teukolsky.radial.solver import solve_radial
    from teukolsky.angular.eigen import spin_weighted_spheroidal_eigenvalue
    from teukolsky.mst import renormalized_angular_momentum
    from teukolsky.accelerated.radial_dcu import batch_solve_point_particle_modes

    a = orbit.a
    n_list = [nk[0] for nk in nk_list]
    k_list = [nk[1] for nk in nk_list]
    omega_list = [
        m * orbit.omega_phi + n * orbit.omega_r + k * orbit.omega_theta
        for n, k in nk_list
    ]
    lam_list = [
        spin_weighted_spheroidal_eigenvalue(s, ell, m, a * omega)
        for omega in omega_list
    ]
    nu_list = [
        renormalized_angular_momentum(s=s, ell=ell, m=m, a=a, omega=omega,
                                      lam=lam)
        for omega, lam in zip(omega_list, lam_list)
    ]

    # Sequential timing
    t0 = time.perf_counter()
    seq_results = []
    for omega, lam, nu in zip(omega_list, lam_list, nu_list):
        seq_results.append(
            solve_radial(
                s=s, ell=ell, m=m, a=a, omega=omega,
                method="NumericalIntegration",
                boundary_conditions=("In", "Up"),
                eigenvalue=lam,
                renormalized_angular_momentum_value=nu,
            )
        )
    t_seq = time.perf_counter() - t0

    # Batch timing
    t0 = time.perf_counter()
    batch_results = batch_solve_point_particle_modes(
        s=s, ell=ell, m=m, orbit=orbit,
        n_list=n_list, k_list=k_list,
        n_workers=n_workers,
    )
    t_batch = time.perf_counter() - t0

    # Precision check: compare radial_function at a test radius
    r_test = 10.0
    in_diffs = []
    up_diffs = []
    for seq_res, batch_res in zip(seq_results, batch_results):
        for bdry in ("In", "Up"):
            seq_val = seq_res[bdry].radial_function(r_test)
            batch_val = batch_res[bdry].radial_function(r_test)
            abs_diff = abs(seq_val - batch_val)
            rel_diff = abs_diff / max(abs(seq_val), 1e-30)
            if bdry == "In":
                in_diffs.append(float(rel_diff))
            else:
                up_diffs.append(float(rel_diff))

    max_in_rel_diff = max(in_diffs) if in_diffs else 0.0
    max_up_rel_diff = max(up_diffs) if up_diffs else 0.0
    tolerance = 1e-12

    return {
        "n_modes": len(nk_list),
        "nk_list": nk_list,
        "seq_time_s": t_seq,
        "batch_time_s": t_batch,
        "speedup": t_seq / max(t_batch, 1e-10),
        "in_match": bool(max_in_rel_diff < tolerance),
        "up_match": bool(max_up_rel_diff < tolerance),
        "max_in_rel_diff": max_in_rel_diff,
        "max_up_rel_diff": max_up_rel_diff,
    }

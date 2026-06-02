from __future__ import annotations

import os
import numpy as np
import pytest


few = pytest.importorskip("few.waveform")
few_flux = pytest.importorskip("few.trajectory.ode.flux")
few_constants = pytest.importorskip("few.utils.constants")

from few.waveform import FastKerrEccentricEquatorialFlux, FastSchwarzschildEccentricFlux
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import KerrEccEqFlux
from few.utils.constants import MTSUN_SI

from teukolsky import (
    equatorial_eccentric_rhs,
    generate_equatorial_eccentric_adiabatic_waveform,
    generate_sparse_trajectory_waveform,
    generate_schwarzschild_eccentric_adiabatic_waveform,
    source_frame_radius,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("TEUKOLSKY_RUN_FEW_TESTS") != "1",
    reason="set TEUKOLSKY_RUN_FEW_TESTS=1 to run FEW waveform comparisons",
)


def test_schwarzschild_short_segment_matches_few_source_frame():
    M = 1.0e6
    mu = 10.0
    p0 = 10.0
    e0 = 0.2
    theta = 1.0
    phi = 0.3
    dt = 10.0
    T_years = 1.0e-6
    T_seconds = T_years * 365.25 * 24.0 * 3600.0
    time = np.arange(0.0, T_seconds + 0.5 * dt, dt, dtype=float)
    time = time[time <= T_seconds]

    mode_indices_teukolsky = [
        (2, 1, -1, 0),
        (2, 1, 0, 0),
        (2, 1, 1, 0),
        (2, 2, -1, 0),
        (2, 2, 0, 0),
        (2, 2, 1, 0),
    ]
    # FEW uses (l, m, k, n) ordering; convert from Teukolsky (l, m, n, k)
    mode_indices_few = [(l, m, k, n) for l, m, n, k in mode_indices_teukolsky]

    waveform = generate_schwarzschild_eccentric_adiabatic_waveform(
        M,
        mu,
        p0,
        e0,
        time,
        theta=theta,
        phi=phi,
        radius=source_frame_radius(1.0, mu),
        trajectory_dt=10.0,
        mode_indices=mode_indices_teukolsky,
        trajectory_ell_max=2,
        trajectory_n_max=1,
        accelerator="cpu",
    )
    h_python = waveform.waveform.complex_strain

    few_waveform = FastSchwarzschildEccentricFlux(force_backend="cpu")
    h_few = np.asarray(
        few_waveform(
            M,
            mu,
            p0,
            e0,
            theta,
            phi,
            T=T_years,
            dt=dt,
            dist=1.0,
            mode_selection=mode_indices_few,
            include_minus_mkn=False,
        ),
        dtype=np.complex128,
    )

    numerator = np.vdot(h_few, h_python)
    denominator = np.sqrt(np.vdot(h_few, h_few).real * np.vdot(h_python, h_python).real)
    overlap = abs(numerator / denominator)
    mean_ratio = float(np.mean(np.abs(h_python) / np.maximum(np.abs(h_few), 1e-300)))
    max_relative_error = float(np.max(np.abs(h_python - h_few) / np.maximum(np.abs(h_few), 1e-300)))

    assert h_python.shape == h_few.shape
    assert overlap > 0.99999
    assert 0.99 < mean_ratio < 1.02
    assert max_relative_error < 5.0e-3


def test_equatorial_rhs_matches_few_flux_table_after_time_rescaling():
    M = 1.0e6
    mu = 10.0
    a = 0.5
    p = 10.0
    e = 0.2
    x = 1.0

    few_rhs = KerrEccEqFlux()
    few_rhs.add_fixed_parameters(M, mu, a, additional_args=[])
    few_internal = np.asarray(
        few_rhs(np.array([p, e, x, 0.0, 0.0, 0.0], dtype=float))[:2],
        dtype=float,
    )
    time_scale = (mu / M) / (M * MTSUN_SI)
    few_physical = few_internal * time_scale

    ours = equatorial_eccentric_rhs(
        0.0,
        np.array([p, e], dtype=float),
        a=a,
        x=x,
        M=M,
        mu=mu,
        ell_max=5,
        n_max=5,
        accelerator="cpu",
        device_id=0,
        accelerator_resolution=None,
    )

    relative_error = np.abs((few_physical - ours) / np.maximum(np.abs(few_physical), 1e-300))

    assert np.all(np.isfinite(ours))
    assert np.all(relative_error < 5.0e-3)


def test_kerr_equatorial_short_segment_matches_few_source_frame():
    """Kerr eccentric-equatorial source-frame waveform against FEW.

    Minimal mode set (lmax=2, nmax=0) and shortest plausible time window
    to keep runtime acceptable.  The FEW waveform generator uses
    pre-computed interpolation tables; this test only checks that the
    two pipelines agree to within a few percent on a handful of samples.
    """
    M = 1.0e6
    mu = 10.0
    a = 0.5
    p0 = 10.0
    e0 = 0.2
    x0 = 1.0
    theta = 1.0
    phi = 0.3
    dt = 10.0
    T_years = 1.0e-6
    T_seconds = T_years * 365.25 * 24.0 * 3600.0
    time = np.arange(0.0, T_seconds + 0.5 * dt, dt, dtype=float)
    time = time[time <= T_seconds]

    mode_indices_teukolsky = [
        (2, 1, 0, 0),
        (2, 2, 0, 0),
    ]
    mode_indices_few = [(l, m, k, n) for l, m, n, k in mode_indices_teukolsky]

    waveform = generate_equatorial_eccentric_adiabatic_waveform(
        M, mu, a, p0, e0, x0,
        time,
        theta=theta, phi=phi,
        radius=source_frame_radius(1.0, mu),
        trajectory_dt=10.0,
        mode_indices=mode_indices_teukolsky,
        trajectory_ell_max=2,
        trajectory_n_max=0,
        accelerator="cpu",
    )
    h_python = waveform.waveform.complex_strain

    few_waveform = FastKerrEccentricEquatorialFlux(
        force_backend="cpu", frame="source", return_list=False, lmax=2, nmax=0,
    )
    h_few = np.asarray(
        few_waveform(
            M, mu, a, p0, e0, x0,
            theta, phi,
            T=T_years, dt=dt, dist=1.0,
            mode_selection=mode_indices_few,
            include_minus_mkn=False,
        ),
        dtype=np.complex128,
    )

    numerator = np.vdot(h_few, h_python)
    denominator_sq = np.vdot(h_few, h_few).real * np.vdot(h_python, h_python).real
    denominator = np.sqrt(max(denominator_sq, 0.0))
    if denominator < 1e-300:
        pytest.skip("zero-amplitude segment, skipping overlap check")
    overlap = abs(numerator / denominator)
    mean_ratio = float(np.mean(np.abs(h_python) / np.maximum(np.abs(h_few), 1e-300)))
    max_rel_err = float(np.max(np.abs(h_python - h_few) / np.maximum(np.abs(h_few), 1e-300)))

    assert h_python.shape == h_few.shape
    assert overlap > 0.9999
    assert 0.95 < mean_ratio < 1.05
    assert max_rel_err < 1.0e-2


def test_schwarzschild_hour_scale_sparse_trajectory_matches_few_source_frame():
    M = 1.0e6
    mu = 10.0
    p0 = 10.0
    e0 = 0.2
    a = 0.0
    x0 = 1.0
    theta = 1.0
    phi = 0.3
    dt = 10.0
    T_years = 2.0e-4
    T_seconds = T_years * 365.25 * 24.0 * 3600.0
    dense_time = np.arange(0.0, T_seconds + 0.5 * dt, dt, dtype=float)
    dense_time = dense_time[dense_time <= T_seconds]
    mode_indices_teukolsky = [(2, 2, 0, 0)]
    mode_indices_few = [(l, m, k, n) for l, m, n, k in mode_indices_teukolsky]

    inspiral = EMRIInspiral(func="SchwarzEccFlux", force_backend="cpu")
    sparse = inspiral(M, mu, a, p0, e0, x0, T=T_years, dt=900.0)
    sparse_time, sparse_p, sparse_e, sparse_x = [np.asarray(value, dtype=float) for value in sparse[:4]]

    waveform = generate_sparse_trajectory_waveform(
        M,
        a,
        sparse_time,
        sparse_p,
        sparse_e,
        sparse_x,
        evaluation_time=dense_time,
        theta=theta,
        phi=phi,
        radius=source_frame_radius(1.0, mu),
        mode_indices=mode_indices_teukolsky,
        accelerator="cpu",
    )
    h_python = waveform.complex_strain

    few_waveform = FastSchwarzschildEccentricFlux(force_backend="cpu")
    h_few = np.asarray(
        few_waveform(
            M,
            mu,
            p0,
            e0,
            theta,
            phi,
            T=T_years,
            dt=dt,
            dist=1.0,
            mode_selection=mode_indices_few,
            include_minus_mkn=False,
        ),
        dtype=np.complex128,
    )

    numerator = np.vdot(h_few, h_python)
    denominator = np.sqrt(np.vdot(h_few, h_few).real * np.vdot(h_python, h_python).real)
    overlap = abs(numerator / denominator)
    mean_ratio = float(np.mean(np.abs(h_python) / np.maximum(np.abs(h_few), 1e-300)))
    max_relative_error = float(np.max(np.abs(h_python - h_few) / np.maximum(np.abs(h_few), 1e-300)))

    assert h_python.shape == h_few.shape
    assert overlap > 0.99999
    assert 0.99 < mean_ratio < 1.02
    assert max_relative_error < 5.0e-3


def test_kerr_equatorial_hour_scale_sparse_trajectory_matches_few_source_frame():
    M = 1.0e6
    mu = 10.0
    a = 0.5
    p0 = 10.0
    e0 = 0.2
    x0 = 1.0
    theta = 1.0
    phi = 0.3
    dt = 10.0
    T_years = 2.0e-4
    T_seconds = T_years * 365.25 * 24.0 * 3600.0
    dense_time = np.arange(0.0, T_seconds + 0.5 * dt, dt, dtype=float)
    dense_time = dense_time[dense_time <= T_seconds]
    mode_indices_teukolsky = [(2, 2, 0, 0)]
    mode_indices_few = [(l, m, k, n) for l, m, n, k in mode_indices_teukolsky]

    inspiral = EMRIInspiral(func="KerrEccEqFlux", force_backend="cpu")
    sparse = inspiral(M, mu, a, p0, e0, x0, T=T_years, dt=900.0)
    sparse_time, sparse_p, sparse_e, sparse_x = [np.asarray(value, dtype=float) for value in sparse[:4]]

    waveform = generate_sparse_trajectory_waveform(
        M,
        a,
        sparse_time,
        sparse_p,
        sparse_e,
        sparse_x,
        evaluation_time=dense_time,
        theta=theta,
        phi=phi,
        radius=source_frame_radius(1.0, mu),
        mode_indices=mode_indices_teukolsky,
        accelerator="cpu",
    )
    h_python = waveform.complex_strain

    few_waveform = FastKerrEccentricEquatorialFlux(
        force_backend="cpu", frame="source", return_list=False, lmax=2, nmax=0,
    )
    h_few = np.asarray(
        few_waveform(
            M,
            mu,
            a,
            p0,
            e0,
            x0,
            theta,
            phi,
            T=T_years,
            dt=dt,
            dist=1.0,
            mode_selection=mode_indices_few,
            include_minus_mkn=False,
        ),
        dtype=np.complex128,
    )

    numerator = np.vdot(h_few, h_python)
    denominator = np.sqrt(np.vdot(h_few, h_few).real * np.vdot(h_python, h_python).real)
    overlap = abs(numerator / denominator)
    mean_ratio = float(np.mean(np.abs(h_python) / np.maximum(np.abs(h_few), 1e-300)))
    max_relative_error = float(np.max(np.abs(h_python - h_few) / np.maximum(np.abs(h_few), 1e-300)))

    assert h_python.shape == h_few.shape
    assert overlap > 0.99999
    assert 0.99 < mean_ratio < 1.02
    assert max_relative_error < 5.0e-3

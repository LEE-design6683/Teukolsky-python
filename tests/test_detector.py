from __future__ import annotations

import numpy as np

from teukolsky import (
    optimal_snr,
    project_signal_to_taiji,
    project_waveform_to_taiji,
    taiji_antenna_pattern,
    taiji_detector_snr,
    taiji_frequency_response,
    taiji_orbital_phase,
    taiji_psd,
    taiji_response_snr,
)


def test_taiji_psd_positive():
    freq = np.logspace(-4, 0, 100)
    psd = taiji_psd(freq)
    assert np.all(np.isfinite(psd))
    assert np.all(psd > 0.0)


def test_taiji_psd_shape():
    freq = np.array([0.001, 0.01])
    psd = taiji_psd(freq)
    assert psd.shape == freq.shape


def test_taiji_psd_no_galactic_is_smaller():
    freq = np.logspace(-4, -1, 50)
    psd_inst = taiji_psd(freq, include_galactic=False)
    psd_full = taiji_psd(freq, include_galactic=True)
    assert np.all(psd_inst <= psd_full)


def test_optimal_snr_zero_for_zero_signal():
    t = np.linspace(0.0, 1000.0, 100)
    signal = np.zeros_like(t)
    snr = optimal_snr(t, signal)
    assert snr == 0.0


def test_optimal_snr_finite():
    t = np.linspace(0.0, 1000.0, 200)
    signal = 1e-22 * np.sin(2.0 * np.pi * 0.002 * t)
    snr = optimal_snr(t, signal)
    assert np.isfinite(snr)
    assert snr > 0.0


def test_taiji_orbital_phase_shape():
    t = np.array([0.0, 10.0, 20.0])
    phase = taiji_orbital_phase(t)
    assert phase.shape == t.shape
    assert np.all(np.diff(phase) > 0.0)


def test_taiji_antenna_pattern_shape_and_finite():
    t = np.linspace(0.0, 100.0, 16)
    f_plus, f_cross = taiji_antenna_pattern(t, theta=1.0, phi=0.2, psi=0.3, channel="X")
    assert f_plus.shape == t.shape
    assert f_cross.shape == t.shape
    assert np.all(np.isfinite(f_plus))
    assert np.all(np.isfinite(f_cross))


def test_taiji_frequency_response_matches_long_wavelength_limit():
    t = np.array([0.0], dtype=float)
    f_plus_lw, f_cross_lw = taiji_antenna_pattern(t, theta=1.0, phi=0.2, psi=0.3, channel="X")
    f_plus_hf, f_cross_hf = taiji_frequency_response(
        np.array([1.0e-6], dtype=float),
        theta=1.0,
        phi=0.2,
        psi=0.3,
        channel="X",
        reference_time=0.0,
    )
    assert np.allclose(f_plus_hf.real, f_plus_lw, rtol=1e-6, atol=1e-9)
    assert np.allclose(f_cross_hf.real, f_cross_lw, rtol=1e-6, atol=1e-9)
    assert np.all(np.abs(f_plus_hf.imag) < 1e-4)
    assert np.all(np.abs(f_cross_hf.imag) < 1e-4)


def test_project_signal_to_taiji_zero_signal():
    t = np.linspace(0.0, 100.0, 32)
    signal = project_signal_to_taiji(
        t,
        np.zeros_like(t),
        np.zeros_like(t),
        theta=1.1,
        phi=0.2,
        psi=0.4,
        channel="A",
    )
    assert np.all(signal == 0.0)


def test_project_signal_to_taiji_tdi_combinations():
    t = np.linspace(0.0, 1000.0, 64)
    h_plus = 1e-22 * np.sin(2.0 * np.pi * 0.002 * t)
    h_cross = 0.5e-22 * np.cos(2.0 * np.pi * 0.002 * t)
    x = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="X")
    y = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="Y")
    z = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="Z")
    a = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="A")
    e = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="E")
    tdi_t = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="T")
    assert np.allclose(a, (2.0 * x - y - z) / 3.0)
    assert np.allclose(e, (z - y) / np.sqrt(3.0))
    assert np.allclose(tdi_t, (x + y + z) / 3.0)


def test_project_signal_to_taiji_finite_arm_low_frequency_matches_long_wavelength():
    t = np.linspace(0.0, 10000.0, 2048)
    freq = 1.0e-4
    h_plus = 1e-22 * np.sin(2.0 * np.pi * freq * t)
    h_cross = 0.5e-22 * np.cos(2.0 * np.pi * freq * t)
    lw = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="X")
    hf = project_signal_to_taiji(
        t,
        h_plus,
        h_cross,
        theta=1.0,
        phi=0.3,
        channel="X",
        finite_arm=True,
        reference_time=0.0,
    )
    assert np.linalg.norm(hf - lw) / np.linalg.norm(lw) < 5.0e-3


def test_project_signal_to_taiji_finite_arm_high_frequency_is_suppressed_and_keeps_tdi_relations():
    t = np.linspace(0.0, 2000.0, 4096)
    freq = 0.2
    h_plus = 1e-22 * np.sin(2.0 * np.pi * freq * t)
    h_cross = 0.5e-22 * np.cos(2.0 * np.pi * freq * t)
    x_lw = project_signal_to_taiji(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="X")
    x_hf = project_signal_to_taiji(
        t,
        h_plus,
        h_cross,
        theta=1.0,
        phi=0.3,
        channel="X",
        finite_arm=True,
        reference_time=0.0,
    )
    y_hf = project_signal_to_taiji(
        t,
        h_plus,
        h_cross,
        theta=1.0,
        phi=0.3,
        channel="Y",
        finite_arm=True,
        reference_time=0.0,
    )
    z_hf = project_signal_to_taiji(
        t,
        h_plus,
        h_cross,
        theta=1.0,
        phi=0.3,
        channel="Z",
        finite_arm=True,
        reference_time=0.0,
    )
    a_hf = project_signal_to_taiji(
        t,
        h_plus,
        h_cross,
        theta=1.0,
        phi=0.3,
        channel="A",
        finite_arm=True,
        reference_time=0.0,
    )
    e_hf = project_signal_to_taiji(
        t,
        h_plus,
        h_cross,
        theta=1.0,
        phi=0.3,
        channel="E",
        finite_arm=True,
        reference_time=0.0,
    )
    t_hf = project_signal_to_taiji(
        t,
        h_plus,
        h_cross,
        theta=1.0,
        phi=0.3,
        channel="T",
        finite_arm=True,
        reference_time=0.0,
    )
    assert np.linalg.norm(x_hf) < np.linalg.norm(x_lw)
    assert np.allclose(a_hf, (2.0 * x_hf - y_hf - z_hf) / 3.0)
    assert np.allclose(e_hf, (z_hf - y_hf) / np.sqrt(3.0))
    assert np.allclose(t_hf, (x_hf + y_hf + z_hf) / 3.0)


def test_taiji_detector_snr_finite():
    t = np.linspace(0.0, 1000.0, 200)
    h_plus = 1e-22 * np.sin(2.0 * np.pi * 0.002 * t)
    h_cross = 0.5e-22 * np.cos(2.0 * np.pi * 0.002 * t)
    snr = taiji_detector_snr(t, h_plus, h_cross, theta=1.0, phi=0.3, channel="A")
    assert np.isfinite(snr)
    assert snr > 0.0


def test_project_waveform_to_taiji_channels_consistent():
    class FakeWaveform:
        def __init__(self):
            self.time = np.linspace(0.0, 1000.0, 200)
            self.h_plus = 1e-22 * np.sin(2.0 * np.pi * 0.002 * self.time)
            self.h_cross = 0.5e-22 * np.cos(2.0 * np.pi * 0.002 * self.time)
            self.theta = 1.0
            self.phi = 0.3

    wf = FakeWaveform()
    response = project_waveform_to_taiji(wf, psi=0.2)
    assert response.time.shape == wf.time.shape
    assert np.allclose(response.a, (2.0 * response.x - response.y - response.z) / 3.0)
    assert np.allclose(response.e, (response.z - response.y) / np.sqrt(3.0))
    assert np.allclose(response.t, (response.x + response.y + response.z) / 3.0)


def test_project_waveform_to_taiji_finite_arm_records_metadata():
    class FakeWaveform:
        def __init__(self):
            self.time = np.linspace(0.0, 1000.0, 512)
            self.h_plus = 1e-22 * np.sin(2.0 * np.pi * 0.05 * self.time)
            self.h_cross = 0.5e-22 * np.cos(2.0 * np.pi * 0.05 * self.time)
            self.theta = 1.0
            self.phi = 0.3

    response = project_waveform_to_taiji(FakeWaveform(), psi=0.2, finite_arm=True, reference_time=10.0)
    assert response.finite_arm is True
    assert response.reference_time == 10.0
    assert response.arm_length > 0.0


def test_taiji_response_snr_finite():
    class FakeWaveform:
        def __init__(self):
            self.time = np.linspace(0.0, 1000.0, 200)
            self.h_plus = 1e-22 * np.sin(2.0 * np.pi * 0.002 * self.time)
            self.h_cross = 0.5e-22 * np.cos(2.0 * np.pi * 0.002 * self.time)
            self.theta = 1.0
            self.phi = 0.3

    response = project_waveform_to_taiji(FakeWaveform(), psi=0.2)
    snr = taiji_response_snr(response, channel="A")
    assert np.isfinite(snr)
    assert snr > 0.0

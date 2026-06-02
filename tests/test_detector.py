from __future__ import annotations

import numpy as np

from teukolsky import optimal_snr, taiji_psd


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

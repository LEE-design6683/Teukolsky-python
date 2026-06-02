from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# Taiji sensitivity from v4 sampling pipeline
# S_n(f) = ASD(f)^2 / 2  (one TDI channel, no galactic foreground)
_TAIJI_FILE = Path(
    "/public/home/licm/M1/用张师兄的数据计算/sensitivity_Taiji_X2.0(1).txt"
)

_taiji_interp: interp1d | None = None


def _load_taiji():
    global _taiji_interp
    if _taiji_interp is not None:
        return _taiji_interp
    if not _TAIJI_FILE.exists():
        raise FileNotFoundError(f"Taiji sensitivity file not found: {_TAIJI_FILE}")
    data = np.loadtxt(str(_TAIJI_FILE), comments="#")
    freq = np.asarray(data[:, 0], dtype=float)
    asd = np.asarray(data[:, 1], dtype=float)
    psd = asd ** 2 / 2.0
    _taiji_interp = interp1d(
        freq, psd, kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    return _taiji_interp


def _taiji_gal_noise(f: np.ndarray) -> np.ndarray:
    A = 9.69230889e-44
    a = -1.69927109
    alpha = 3.05655225
    f0 = 1.56737297e-03
    conversion = 3.877
    f_arr = np.asarray(f, dtype=float)
    lnSc = np.log(A) + a * np.log(f_arr) - (f_arr / f0) ** alpha
    return conversion * np.exp(lnSc)


def taiji_psd(
    frequency: np.ndarray,
    *,
    include_galactic: bool = False,
) -> np.ndarray:
    """Taiji one-channel noise PSD from the v4 sampling pipeline."""
    interp = _load_taiji()
    f_arr = np.asarray(frequency, dtype=float)
    psd = np.asarray(interp(f_arr), dtype=float)
    if include_galactic:
        psd = psd + _taiji_gal_noise(f_arr)
    return psd


def optimal_snr(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    psd_func: callable = taiji_psd,
) -> float:
    """Optimal SNR against a one-sided PSD."""
    dt = float(np.median(np.diff(time)))
    if dt <= 0.0:
        raise ValueError("time step must be > 0")
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=dt)
    signal_fft = np.fft.rfft(signal) * dt
    psd = psd_func(freq[1:])
    inner = 4.0 * np.sum(np.abs(signal_fft[1:]) ** 2 / np.maximum(psd, 1e-60)) / n
    return float(np.sqrt(max(inner, 0.0)))

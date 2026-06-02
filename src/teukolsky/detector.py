from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from scipy.interpolate import interp1d

# Taiji sensitivity from v4 sampling pipeline
# S_n(f) = ASD(f)^2 / 2  (one TDI channel, no galactic foreground)
_TAIJI_FILE = Path(
    "/public/home/licm/M1/用张师兄的数据计算/sensitivity_Taiji_X2.0(1).txt"
)

_taiji_interp: interp1d | None = None
_SIDEREAL_YEAR_SI = 365.256363004 * 86400.0
_CHANNEL_PHASE = {
    "X": 0.0,
    "Y": 2.0 * math.pi / 3.0,
    "Z": 4.0 * math.pi / 3.0,
}


@dataclass(frozen=True)
class TaijiResponse:
    time: np.ndarray
    h_plus: np.ndarray
    h_cross: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    a: np.ndarray
    e: np.ndarray
    t: np.ndarray
    theta: float
    phi: float
    psi: float
    initial_phase: float


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


def taiji_orbital_phase(
    time: np.ndarray,
    *,
    initial_phase: float = 0.0,
) -> np.ndarray:
    time_arr = np.asarray(time, dtype=float)
    return initial_phase + 2.0 * math.pi * time_arr / _SIDEREAL_YEAR_SI


def taiji_antenna_pattern(
    time: np.ndarray,
    *,
    theta: float,
    phi: float,
    psi: float = 0.0,
    channel: str = "X",
    initial_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    channel_name = str(channel).upper()
    if channel_name not in _CHANNEL_PHASE:
        raise ValueError("channel must be one of 'X', 'Y', 'Z'")
    time_arr = np.asarray(time, dtype=float)
    phase = taiji_orbital_phase(time_arr, initial_phase=initial_phase) + _CHANNEL_PHASE[channel_name]
    phi_det = float(phi) - phase
    cos_theta = math.cos(float(theta))
    prefactor = math.sqrt(3.0) / 2.0
    cos_2phi = np.cos(2.0 * phi_det)
    sin_2phi = np.sin(2.0 * phi_det)
    cos_2psi = math.cos(2.0 * float(psi))
    sin_2psi = math.sin(2.0 * float(psi))
    f_plus = prefactor * (0.5 * (1.0 + cos_theta * cos_theta) * cos_2phi * cos_2psi - cos_theta * sin_2phi * sin_2psi)
    f_cross = prefactor * (0.5 * (1.0 + cos_theta * cos_theta) * cos_2phi * sin_2psi + cos_theta * sin_2phi * cos_2psi)
    return np.asarray(f_plus, dtype=float), np.asarray(f_cross, dtype=float)


def project_signal_to_taiji(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    theta: float,
    phi: float,
    psi: float = 0.0,
    channel: str = "A",
    initial_phase: float = 0.0,
) -> np.ndarray:
    time_arr = np.asarray(time, dtype=float)
    h_plus_arr = np.asarray(h_plus, dtype=float)
    h_cross_arr = np.asarray(h_cross, dtype=float)
    if time_arr.shape != h_plus_arr.shape or time_arr.shape != h_cross_arr.shape:
        raise ValueError("time, h_plus, and h_cross must have the same shape")

    def project_single(name: str) -> np.ndarray:
        f_plus, f_cross = taiji_antenna_pattern(
            time_arr,
            theta=theta,
            phi=phi,
            psi=psi,
            channel=name,
            initial_phase=initial_phase,
        )
        return f_plus * h_plus_arr + f_cross * h_cross_arr

    channel_name = str(channel).upper()
    if channel_name in _CHANNEL_PHASE:
        return project_single(channel_name)

    x = project_single("X")
    y = project_single("Y")
    z = project_single("Z")
    if channel_name == "A":
        return (2.0 * x - y - z) / 3.0
    if channel_name == "E":
        return (z - y) / math.sqrt(3.0)
    if channel_name == "T":
        return (x + y + z) / 3.0
    raise ValueError("channel must be one of 'X', 'Y', 'Z', 'A', 'E', 'T'")


def project_waveform_to_taiji(
    waveform,
    *,
    psi: float = 0.0,
    initial_phase: float = 0.0,
) -> TaijiResponse:
    x = project_signal_to_taiji(
        waveform.time,
        waveform.h_plus,
        waveform.h_cross,
        theta=waveform.theta,
        phi=waveform.phi,
        psi=psi,
        channel="X",
        initial_phase=initial_phase,
    )
    y = project_signal_to_taiji(
        waveform.time,
        waveform.h_plus,
        waveform.h_cross,
        theta=waveform.theta,
        phi=waveform.phi,
        psi=psi,
        channel="Y",
        initial_phase=initial_phase,
    )
    z = project_signal_to_taiji(
        waveform.time,
        waveform.h_plus,
        waveform.h_cross,
        theta=waveform.theta,
        phi=waveform.phi,
        psi=psi,
        channel="Z",
        initial_phase=initial_phase,
    )
    a = (2.0 * x - y - z) / 3.0
    e = (z - y) / math.sqrt(3.0)
    t = (x + y + z) / 3.0
    return TaijiResponse(
        time=np.asarray(waveform.time, dtype=float),
        h_plus=np.asarray(waveform.h_plus, dtype=float),
        h_cross=np.asarray(waveform.h_cross, dtype=float),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        a=np.asarray(a, dtype=float),
        e=np.asarray(e, dtype=float),
        t=np.asarray(t, dtype=float),
        theta=float(waveform.theta),
        phi=float(waveform.phi),
        psi=float(psi),
        initial_phase=float(initial_phase),
    )


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


def taiji_detector_snr(
    time: np.ndarray,
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    theta: float,
    phi: float,
    psi: float = 0.0,
    channel: str = "A",
    include_galactic: bool = False,
    initial_phase: float = 0.0,
) -> float:
    signal = project_signal_to_taiji(
        time,
        h_plus,
        h_cross,
        theta=theta,
        phi=phi,
        psi=psi,
        channel=channel,
        initial_phase=initial_phase,
    )
    return optimal_snr(
        np.asarray(time, dtype=float),
        signal,
        psd_func=lambda f: taiji_psd(f, include_galactic=include_galactic),
    )


def taiji_response_snr(
    response: TaijiResponse,
    *,
    channel: str = "A",
    include_galactic: bool = False,
) -> float:
    channel_name = str(channel).upper()
    mapping = {
        "X": response.x,
        "Y": response.y,
        "Z": response.z,
        "A": response.a,
        "E": response.e,
        "T": response.t,
    }
    if channel_name not in mapping:
        raise ValueError("channel must be one of 'X', 'Y', 'Z', 'A', 'E', 'T'")
    return optimal_snr(
        response.time,
        mapping[channel_name],
        psd_func=lambda f: taiji_psd(f, include_galactic=include_galactic),
    )

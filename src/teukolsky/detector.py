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
_C_SI = 299792458.0
_TAIJI_ARM_LENGTH_SI = 3.0e9
_REFERENCE_ROTATION = math.pi / 4.0
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
    finite_arm: bool
    arm_length: float
    reference_time: float | None


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


def _source_direction(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ],
        dtype=float,
    )


def _polarization_tensors(theta: float, phi: float, psi: float) -> tuple[np.ndarray, np.ndarray]:
    # The wave propagates from the source toward the detector, so the
    # propagation direction is the negative of the sky-position unit vector.
    e_theta = np.array(
        [
            -math.cos(theta) * math.cos(phi),
            -math.cos(theta) * math.sin(phi),
            math.sin(theta),
        ],
        dtype=float,
    )
    e_phi = np.array(
        [
            -math.sin(phi),
            math.cos(phi),
            0.0,
        ],
        dtype=float,
    )
    plus_0 = np.outer(e_theta, e_theta) - np.outer(e_phi, e_phi)
    cross_0 = np.outer(e_theta, e_phi) + np.outer(e_phi, e_theta)
    cos_2psi = math.cos(2.0 * psi)
    sin_2psi = math.sin(2.0 * psi)
    plus = cos_2psi * plus_0 - sin_2psi * cross_0
    cross = sin_2psi * plus_0 + cos_2psi * cross_0
    return plus, cross


def _channel_arm_vectors(
    time: np.ndarray,
    *,
    channel: str,
    initial_phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    phase = taiji_orbital_phase(time, initial_phase=initial_phase)
    alpha = phase + _CHANNEL_PHASE[channel] + _REFERENCE_ROTATION
    u = np.stack(
        [
            np.cos(alpha - math.pi / 6.0),
            np.sin(alpha - math.pi / 6.0),
            np.zeros_like(alpha),
        ],
        axis=-1,
    )
    v = np.stack(
        [
            np.cos(alpha + math.pi / 6.0),
            np.sin(alpha + math.pi / 6.0),
            np.zeros_like(alpha),
        ],
        axis=-1,
    )
    return np.asarray(u, dtype=float), np.asarray(v, dtype=float)


def _long_wavelength_response_tensor(
    time: np.ndarray,
    *,
    channel: str,
    initial_phase: float = 0.0,
) -> np.ndarray:
    u, v = _channel_arm_vectors(time, channel=channel, initial_phase=initial_phase)
    return 0.5 * (
        np.einsum("...i,...j->...ij", u, u)
        - np.einsum("...i,...j->...ij", v, v)
    )


def _arm_transfer_function(
    frequency: np.ndarray,
    mu: float,
    *,
    arm_length: float,
) -> np.ndarray:
    f_arr = np.asarray(frequency, dtype=float)
    f_star = _C_SI / (2.0 * math.pi * arm_length)
    omega = f_arr / f_star
    arg_minus = 0.5 * omega * (1.0 - mu)
    arg_plus = 0.5 * omega * (1.0 + mu)
    sinc_minus = np.sinc(arg_minus / math.pi)
    sinc_plus = np.sinc(arg_plus / math.pi)
    phase_minus = np.exp(-0.5j * omega * (3.0 + mu))
    phase_plus = np.exp(-0.5j * omega * (1.0 + mu))
    return 0.5 * (sinc_minus * phase_minus + sinc_plus * phase_plus)


def taiji_frequency_response(
    frequency: np.ndarray,
    *,
    theta: float,
    phi: float,
    psi: float = 0.0,
    channel: str = "X",
    reference_time: float = 0.0,
    initial_phase: float = 0.0,
    arm_length: float = _TAIJI_ARM_LENGTH_SI,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen-constellation equal-arm Michelson response in the frequency domain."""
    channel_name = str(channel).upper()
    if channel_name in {"A", "E", "T"}:
        fxp, fxc = taiji_frequency_response(
            frequency,
            theta=theta,
            phi=phi,
            psi=psi,
            channel="X",
            reference_time=reference_time,
            initial_phase=initial_phase,
            arm_length=arm_length,
        )
        fyp, fyc = taiji_frequency_response(
            frequency,
            theta=theta,
            phi=phi,
            psi=psi,
            channel="Y",
            reference_time=reference_time,
            initial_phase=initial_phase,
            arm_length=arm_length,
        )
        fzp, fzc = taiji_frequency_response(
            frequency,
            theta=theta,
            phi=phi,
            psi=psi,
            channel="Z",
            reference_time=reference_time,
            initial_phase=initial_phase,
            arm_length=arm_length,
        )
        if channel_name == "A":
            return (2.0 * fxp - fyp - fzp) / 3.0, (2.0 * fxc - fyc - fzc) / 3.0
        if channel_name == "E":
            return (fzp - fyp) / math.sqrt(3.0), (fzc - fyc) / math.sqrt(3.0)
        return (fxp + fyp + fzp) / 3.0, (fxc + fyc + fzc) / 3.0

    if channel_name not in _CHANNEL_PHASE:
        raise ValueError("channel must be one of 'X', 'Y', 'Z', 'A', 'E', 'T'")

    plus, cross = _polarization_tensors(float(theta), float(phi), float(psi))
    propagation = -_source_direction(float(theta), float(phi))
    u, v = _channel_arm_vectors(
        np.array([reference_time], dtype=float),
        channel=channel_name,
        initial_phase=initial_phase,
    )
    u0 = u[0]
    v0 = v[0]
    transfer_u = _arm_transfer_function(
        frequency,
        float(np.dot(propagation, u0)),
        arm_length=arm_length,
    )
    transfer_v = _arm_transfer_function(
        frequency,
        float(np.dot(propagation, v0)),
        arm_length=arm_length,
    )
    detector = 0.5 * (
        np.einsum("i,j,...->...ij", u0, u0, transfer_u)
        - np.einsum("i,j,...->...ij", v0, v0, transfer_v)
    )
    f_plus = np.einsum("...ij,ij->...", detector, plus)
    f_cross = np.einsum("...ij,ij->...", detector, cross)
    return np.asarray(f_plus, dtype=np.complex128), np.asarray(f_cross, dtype=np.complex128)


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
    plus, cross = _polarization_tensors(float(theta), float(phi), float(psi))
    detector = _long_wavelength_response_tensor(
        np.asarray(time, dtype=float),
        channel=channel_name,
        initial_phase=initial_phase,
    )
    f_plus = np.einsum("...ij,ij->...", detector, plus)
    f_cross = np.einsum("...ij,ij->...", detector, cross)
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
    finite_arm: bool = False,
    arm_length: float = _TAIJI_ARM_LENGTH_SI,
    reference_time: float | None = None,
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

    def project_single_finite_arm(name: str) -> np.ndarray:
        dt = float(np.median(np.diff(time_arr)))
        if dt <= 0.0:
            raise ValueError("time step must be > 0")
        freq = np.fft.rfftfreq(time_arr.size, d=dt)
        ref = (
            float(time_arr[0] + 0.5 * (time_arr[-1] - time_arr[0]))
            if reference_time is None
            else float(reference_time)
        )
        f_plus, f_cross = taiji_frequency_response(
            freq,
            theta=theta,
            phi=phi,
            psi=psi,
            channel=name,
            reference_time=ref,
            initial_phase=initial_phase,
            arm_length=arm_length,
        )
        h_plus_fft = np.fft.rfft(h_plus_arr)
        h_cross_fft = np.fft.rfft(h_cross_arr)
        signal_fft = f_plus * h_plus_fft + f_cross * h_cross_fft
        return np.asarray(np.fft.irfft(signal_fft, n=time_arr.size), dtype=float)

    channel_name = str(channel).upper()
    projector = project_single_finite_arm if finite_arm else project_single
    if channel_name in _CHANNEL_PHASE:
        return projector(channel_name)

    x = projector("X")
    y = projector("Y")
    z = projector("Z")
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
    finite_arm: bool = False,
    arm_length: float = _TAIJI_ARM_LENGTH_SI,
    reference_time: float | None = None,
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
        finite_arm=finite_arm,
        arm_length=arm_length,
        reference_time=reference_time,
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
        finite_arm=finite_arm,
        arm_length=arm_length,
        reference_time=reference_time,
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
        finite_arm=finite_arm,
        arm_length=arm_length,
        reference_time=reference_time,
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
        finite_arm=bool(finite_arm),
        arm_length=float(arm_length),
        reference_time=None if reference_time is None else float(reference_time),
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
    finite_arm: bool = False,
    arm_length: float = _TAIJI_ARM_LENGTH_SI,
    reference_time: float | None = None,
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
        finite_arm=finite_arm,
        arm_length=arm_length,
        reference_time=reference_time,
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

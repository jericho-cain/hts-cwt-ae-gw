"""
Line-of-sight acceleration (LOSA) wrappers for synthetic waveforms.

Phase 0: constant acceleration only.
Implements kinematic Romer delay via time shift:
  h_obs(t) = h_iso(t + Delta t(t))
with Delta t(t) = 0.5 * (a_los/c) * t^2  (optionally + (v0/c) t).
"""

from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


def apply_losa_constant_accel(
    h: np.ndarray,
    sample_rate: float,
    a_los: float,
    v0_los: float = 0.0,
    t0: float = 0.0,
    c: float = C_LIGHT,
) -> np.ndarray:
    """
    Apply constant line-of-sight acceleration to a time series by resampling.

    Parameters
    ----------
    h : np.ndarray
        Input waveform, shape (T,)
    sample_rate : float
        Sampling rate in Hz
    a_los : float
        Constant LOS acceleration in m/s^2 (signed)
    v0_los : float
        Optional initial LOS velocity (m/s)
    t0 : float
        Reference time (seconds) at which Delta t(t0)=0
    c : float
        Speed of light

    Returns
    -------
    np.ndarray
        Time-shifted waveform, shape (T,)
    """
    h = np.asarray(h, dtype=np.float64)
    T = h.shape[0]
    t = np.arange(T, dtype=np.float64) / sample_rate

    # Time shift Delta t(t) = (v0/c)(t-t0) + 0.5(a/c)(t-t0)^2
    dt = (v0_los / c) * (t - t0) + 0.5 * (a_los / c) * (t - t0) ** 2

    # Sample original signal at shifted times
    t_shifted = t + dt

    # Interpolate: numpy interp is linear but stable; use this for Phase 0.
    # (Upgrade to cubic later if needed.)
    h_shifted = np.interp(t_shifted, t, h, left=0.0, right=0.0)
    return h_shifted.astype(np.float32)


def epsilon_from_accel(a_los: float, duration_s: float, c: float = C_LIGHT) -> float:
    """epsilon approx max|v|/c for constant acceleration over duration."""
    vmax = abs(a_los) * duration_s
    return vmax / c


def accel_from_delta_phi(
    delta_phi_rad: float,
    duration_s: float,
    f_star_hz: float = 100.0,
    c: float = C_LIGHT,
) -> float:
    """
    Acceleration needed for target cumulative phase distortion.

    delta t_max = delta_phi / (2 pi f_star)
    a = 2 c delta_t_max / T^2 = c delta_phi / (pi f_star T^2)
    """
    delta_t_max = delta_phi_rad / (2 * np.pi * f_star_hz)
    return float(c * delta_phi_rad / (np.pi * f_star_hz * duration_s**2))

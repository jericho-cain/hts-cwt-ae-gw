"""
Minimal synthetic signal generator for smoke tests.

Generates chirp-like signals (sum of sinusoids with envelope) for baseline
experiments. Supports optional LOSA (line-of-sight acceleration) via
generate_chirp_with_losa.
"""

import numpy as np

from .losa import apply_losa_constant_accel


def generate_isolated_chirp(
    T: int = 4096,
    sample_rate: float = 1024.0,
    f_start: float = 30.0,
    f_end: float = 200.0,
    amplitude: float = 1e-20,
    t_peak: float = 0.5,
    sigma: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a simple chirp-like signal (sum of sinusoids with Gaussian envelope).

    Parameters
    ----------
    T : int
        Number of time samples
    sample_rate : float
        Sampling rate in Hz
    f_start : float
        Initial frequency (Hz)
    f_end : float
        Final frequency (Hz)
    amplitude : float
        Signal amplitude
    t_peak : float
        Peak time as fraction of segment (0–1)
    sigma : float
        Gaussian envelope width
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Time series of shape (T,)
    """
    if seed is not None:
        np.random.seed(seed)

    amplitude = float(amplitude)
    t = np.arange(T, dtype=np.float64) / sample_rate
    t_norm = t / t[-1]

    # Linear chirp frequency
    f = f_start + (f_end - f_start) * t_norm

    # Phase accumulation
    phase = np.cumsum(2 * np.pi * f / sample_rate)
    phase = np.concatenate([[0], phase[:-1]])

    # Gaussian envelope centered at t_peak fraction
    peak_idx = int(t_peak * T)
    env = np.exp(-0.5 * ((np.arange(T) - peak_idx) / (sigma * T)) ** 2)

    signal = amplitude * env * np.sin(phase)
    return signal.astype(np.float32)


def generate_chirp_with_losa(
    T: int = 4096,
    sample_rate: float = 1024.0,
    a_los: float = 0.0,
    v0_los: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """
    Generate isolated chirp with optional LOSA applied.

    Parameters
    ----------
    T : int
        Number of time samples
    sample_rate : float
        Sampling rate in Hz
    a_los : float
        Constant LOS acceleration (m/s^2). If 0, returns isolated chirp only.
    v0_los : float
        Initial LOS velocity (m/s). If both a_los and v0_los are 0, no LOSA.
    **kwargs
        Passed to generate_isolated_chirp (f_start, f_end, amplitude, etc.)

    Returns
    -------
    np.ndarray
        Time series of shape (T,)
    """
    h = generate_isolated_chirp(T=T, sample_rate=sample_rate, **kwargs)
    if a_los != 0.0 or v0_los != 0.0:
        h = apply_losa_constant_accel(
            h, sample_rate=sample_rate, a_los=a_los, v0_los=v0_los
        )
    return h

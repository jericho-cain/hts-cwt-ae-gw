"""
Phase 0 dataset builders for LOSA experiments.

Simple batch generation for sanity checks and training.
"""

from __future__ import annotations

import numpy as np

from .isolated_generator import generate_isolated_chirp
from .losa import apply_losa_constant_accel
from .noise_models import gaussian_noise


def make_phase0_batch(
    n: int,
    T: int,
    sample_rate: float,
    snr: float,
    a_los: float = 0.0,
    noise_sigma: float = 1e-21,
    seed: int | None = None,
    f_start_range: tuple[float, float] = (10.0, 15.0),
    f_end_range: tuple[float, float] = (55.0, 70.0),
    t_peak_range: tuple[float, float] = (0.4, 0.7),
    sigma_range: tuple[float, float] = (0.06, 0.14),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) where:
      x is noisy strain (n, T)
      y is clean strain (n, T) for diagnostics / denoising AE later.

    Chirp band (f_start_range, f_end_range) should lie below Nyquist of the
    representation. For downsample_factor=8 at 1024 Hz, use ~10-70 Hz.

    Use narrow ranges (e.g. (12,12), (0.55,0.55)) for tight distribution
    to test identifiability: LOSA becomes true deformation, not absorbed by
    chirp-parameter degeneracy.
    """
    rng = np.random.default_rng(seed)
    xs = []
    ys = []

    for i in range(n):
        f0 = rng.uniform(f_start_range[0], f_start_range[1])
        f1 = rng.uniform(f_end_range[0], f_end_range[1])
        t_peak = rng.uniform(t_peak_range[0], t_peak_range[1])
        sigma_env = rng.uniform(sigma_range[0], sigma_range[1])

        h = generate_isolated_chirp(
            T=T,
            sample_rate=sample_rate,
            f_start=float(f0),
            f_end=float(f1),
            t_peak=float(t_peak),
            sigma=float(sigma_env),
        )

        if a_los != 0.0:
            h = apply_losa_constant_accel(h, sample_rate=sample_rate, a_los=a_los)

        nse = gaussian_noise(
            T=T, sigma=noise_sigma, seed=(seed + i) if seed is not None else None
        )
        h_norm = np.sqrt(np.mean(h**2)) + 1e-30
        scale = (snr * noise_sigma) / h_norm
        h_scaled = (h * scale).astype(np.float32)

        x = (h_scaled + nse).astype(np.float32)

        xs.append(x)
        ys.append(h_scaled)

    return np.stack(xs, axis=0), np.stack(ys, axis=0)

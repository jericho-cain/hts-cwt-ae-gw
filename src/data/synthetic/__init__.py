"""Synthetic data generators for smoke tests and baseline experiments."""

from .isolated_generator import generate_isolated_chirp, generate_chirp_with_losa
from .noise_models import gaussian_noise
from .losa import apply_losa_constant_accel, accel_from_delta_phi, epsilon_from_accel

__all__ = [
    "generate_isolated_chirp",
    "generate_chirp_with_losa",
    "gaussian_noise",
    "apply_losa_constant_accel",
    "accel_from_delta_phi",
    "epsilon_from_accel",
]

"""Synthetic data generators for smoke tests and baseline experiments."""

from .isolated_generator import generate_isolated_chirp
from .noise_models import gaussian_noise

__all__ = ['generate_isolated_chirp', 'gaussian_noise']

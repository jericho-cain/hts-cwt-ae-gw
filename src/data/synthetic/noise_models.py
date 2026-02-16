"""Noise models for synthetic data generation."""

import numpy as np


def gaussian_noise(
    T: int,
    sigma: float = 1e-21,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate Gaussian white noise.

    Parameters
    ----------
    T : int
        Number of samples
    sigma : float
        Standard deviation
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Noise time series of shape (T,)
    """
    if seed is not None:
        np.random.seed(seed)
    return (np.random.randn(T) * sigma).astype(np.float32)

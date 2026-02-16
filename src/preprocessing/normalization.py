"""Normalization utilities for time series and scalograms."""

import numpy as np
from typing import Optional


def normalize_log(scalogram: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Log-transform and normalize scalogram to zero mean, unit variance.

    Parameters
    ----------
    scalogram : np.ndarray
        Raw magnitude scalogram
    eps : float
        Small constant for numerical stability

    Returns
    -------
    np.ndarray
        Normalized scalogram
    """
    log_s = np.log10(scalogram + eps)
    norm = (log_s - np.mean(log_s)) / (np.std(log_s) + eps)
    return norm.astype(np.float32)

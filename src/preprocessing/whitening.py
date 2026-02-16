"""Generic whitening utilities (zero mean, unit variance)."""

import numpy as np
from typing import Optional, Tuple


def whiten(
    x: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    eps: float = 1e-10,
) -> Tuple[np.ndarray, float, float]:
    """
    Whiten time series: zero mean, unit variance.

    Parameters
    ----------
    x : np.ndarray
        Input time series
    mean : float, optional
        Precomputed mean (if None, use per-array mean)
    std : float, optional
        Precomputed std (if None, use per-array std)
    eps : float
        Small constant to avoid division by zero

    Returns
    -------
    Tuple[np.ndarray, float, float]
        Whitened array, mean used, std used
    """
    m = float(np.mean(x)) if mean is None else mean
    s = float(np.std(x)) if std is None else std
    whitened = (x - m) / (s + eps)
    return whitened.astype(np.float32), m, s

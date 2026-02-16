"""
Preprocessing module for CWT-based time-frequency analysis.

Provides Continuous Wavelet Transform (CWT) utilities for generic time series,
including whitening, normalization, and scalogram computation.
"""

from .cwt import (
    cwt_clean,
    peak_time_from_cwt,
    fixed_preprocess_with_cwt,
    CWTPreprocessor,
    compute_global_normalization_stats,
)
from .whitening import whiten
from .normalization import normalize_log

__all__ = [
    'cwt_clean',
    'peak_time_from_cwt',
    'fixed_preprocess_with_cwt',
    'CWTPreprocessor',
    'compute_global_normalization_stats',
    'whiten',
    'normalize_log',
]

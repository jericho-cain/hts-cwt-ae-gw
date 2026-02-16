"""
Λ tracking + recommended Δφ for observability pipeline.

Λ = Δφ × SNR collapses centroid-based detectability onto a single control parameter.
"""

from __future__ import annotations

# From observability_dfc_lambda_fit.py sigmoid fit
LAMBDA_08 = 18.9690
LAMBDA_095 = 26.9520


def lambda_from(dphi: float, snr: float) -> float:
    """Λ = Δφ × SNR."""
    return dphi * snr


def recommended_dphi(snr: float, target: str = "0.8") -> float:
    """
    Δφ to achieve target AUROC at given SNR.
    Uses fitted thresholds: Λ₀.8 = 18.969, Λ₀.95 = 26.952.
    """
    if target == "0.8":
        return LAMBDA_08 / snr
    if target == "0.95":
        return LAMBDA_095 / snr
    raise ValueError(f"Unknown target: {target!r}. Use '0.8' or '0.95'.")

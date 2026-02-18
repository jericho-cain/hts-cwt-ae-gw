#!/usr/bin/env python3
"""
Fit scaling law AUROC(Λ) with Λ = Δφ × SNR.
Sigmoid: AUROC = 0.5 + 0.5 / (1 + exp(-k(Λ - Λ0)))
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "experiments/outputs_corrected/observability_dfc_snr_sweep.json"
OUT_PATH = ROOT / "experiments/outputs_corrected/observability_dfc_lambda_fit.png"


def sigmoid(Lambda: np.ndarray, k: float, L0: float) -> np.ndarray:
    """AUROC(Λ) = 0.5 + 0.5 / (1 + exp(-k(Λ - Λ0)))"""
    return 0.5 + 0.5 / (1 + np.exp(-k * (Lambda - L0)))


def main():
    with open(IN_PATH) as f:
        data = json.load(f)

    results = data["results"]

    # Build flat list: (Λ, AUROC) - iterate over auroc keys for robustness (handles auto_dphi)
    Lambda_list = []
    auroc_list = []
    for r in results:
        snr = r["snr"]
        lam_dict = r.get("lambda", {})
        for key in r["auroc"]:
            dphi = float(key)
            auroc = r["auroc"][key]
            lam = lam_dict.get(key, dphi * snr)
            Lambda_list.append(lam)
            auroc_list.append(auroc)

    Lambda_arr = np.array(Lambda_list)
    auroc_arr = np.array(auroc_list)

    # Fit
    k_init = 0.1
    L0_init = float(np.median(Lambda_arr))
    popt, pcov = curve_fit(sigmoid, Lambda_arr, auroc_arr, p0=[k_init, L0_init])
    k_fit, L0_fit = popt[0], popt[1]

    # Solve for Λ₀.8 and Λ₀.95 analytically
    # From AUROC = 0.5 + 0.5/(1+exp(-k(Λ-Λ0))):
    # 2*y - 1 = 1/(1+exp(-k(Λ-Λ0)))
    # 1 + exp(-k(Λ-Λ0)) = 1/(2*y-1)
    # exp(-k(Λ-Λ0)) = 1/(2*y-1) - 1 = (2-2y)/(2y-1)
    # -k(Λ-Λ0) = ln((2-2y)/(2y-1))
    # Λ = Λ0 - ln((2-2y)/(2y-1)) / k
    # For y=0.8: Λ = Λ0 - ln(0.4/0.6) / k = Λ0 - ln(2/3) / k
    # For y=0.95: Λ = Λ0 - ln(0.1/0.9) / k = Λ0 - ln(1/9) / k

    def lambda_at_auroc(y: float, k: float, L0: float) -> float:
        """Λ such that sigmoid(Λ) = y. Requires 0.5 < y < 1."""
        return L0 - np.log((2 - 2 * y) / (2 * y - 1)) / k

    L_08 = lambda_at_auroc(0.8, k_fit, L0_fit)
    L_095 = lambda_at_auroc(0.95, k_fit, L0_fit)

    # R²
    auroc_pred = sigmoid(Lambda_arr, k_fit, L0_fit)
    ss_res = np.sum((auroc_arr - auroc_pred) ** 2)
    ss_tot = np.sum((auroc_arr - np.mean(auroc_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(Lambda_arr, auroc_arr, color="steelblue", s=60, zorder=3, label="data")
    L_plot = np.linspace(Lambda_arr.min() - 5, Lambda_arr.max() + 20, 200)
    ax.plot(L_plot, sigmoid(L_plot, k_fit, L0_fit), "r-", linewidth=2, label="fit")
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(L_08, color="orange", linestyle=":", alpha=0.8, label=f"Λ₀.₈ = {L_08:.1f}")
    ax.axvline(L_095, color="green", linestyle=":", alpha=0.8, label=f"Λ₀.₉₅ = {L_095:.1f}")
    ax.set_xlabel("Λ = Δφ × SNR")
    ax.set_ylabel("AUROC(S_fc)")
    ax.set_title("AUROC(Λ) with sigmoid fit")
    ax.set_ylim(0.48, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"Saved {OUT_PATH}")

    # Summary
    print()
    print("Summary:")
    print(f"  k = {k_fit:.4f}")
    print(f"  Λ0 = {L0_fit:.4f}")
    print(f"  Λ₀.8 = {L_08:.4f}")
    print(f"  Λ₀.95 = {L_095:.4f}")
    print(f"  R² = {r_squared:.4f}")


if __name__ == "__main__":
    main()

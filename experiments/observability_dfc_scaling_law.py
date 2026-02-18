#!/usr/bin/env python3
"""
Plot AUROC(S_fc) vs Λ = Δφ × SNR for all (Δφ, SNR) grid points.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "experiments/outputs_corrected/observability_dfc_snr_sweep.json"
OUT_PATH = ROOT / "experiments/outputs_corrected/observability_dfc_scaling_law.png"


def main():
    with open(IN_PATH) as f:
        data = json.load(f)

    results = data["results"]

    # Flatten to (SNR, Δφ, AUROC) - iterate over auroc keys for robustness (handles auto_dphi)
    points = []
    for r in results:
        snr = r["snr"]
        for key in r["auroc"]:
            dphi = float(key)
            auroc = r["auroc"][key]
            points.append((snr, dphi, auroc))

    snr_arr = np.array([p[0] for p in points])
    dphi_arr = np.array([p[1] for p in points])
    auroc_arr = np.array([p[2] for p in points])

    Lambda1 = dphi_arr * snr_arr

    # Single plot: Λ₁ = Δφ × SNR
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    order = np.argsort(Lambda1)
    L_sort = Lambda1[order]
    A_sort = auroc_arr[order]
    ax.plot(L_sort, A_sort, "o-", color="steelblue", linewidth=2, markersize=10)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Λ = Δφ × SNR")
    ax.set_ylabel("AUROC(S_fc)")
    ax.set_title("AUROC(S_fc) vs Λ = Δφ × SNR")
    ax.set_ylim(0.48, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

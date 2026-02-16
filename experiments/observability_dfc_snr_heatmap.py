#!/usr/bin/env python3
"""
Heatmap of AUROC(S_fc) over (SNR, Δφ) from the SNR sweep results.
Δφ on x-axis, SNR on y-axis. Interpolated smoothly; labels at measured grid points only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "experiments/outputs_corrected/observability_dfc_snr_sweep.json"
OUT_PATH = ROOT / "experiments/outputs_corrected/observability_dfc_snr_heatmap.png"


def main():
    with open(IN_PATH) as f:
        data = json.load(f)

    snr_vals = np.array(data["snr_values"])
    dphi_raw = data["delta_phi_values"]
    results = data["results"]

    # Handle both regular grid (flat dphi list) and auto_dphi (list of lists)
    if all(isinstance(x, (int, float)) for x in dphi_raw):
        dphi_vals = np.array(dphi_raw, dtype=float)
        Z = np.zeros((len(snr_vals), len(dphi_vals)))
        for i, r in enumerate(results):
            for j, dphi in enumerate(dphi_vals):
                key = str(dphi) if str(dphi) in r["auroc"] else str(int(dphi)) if dphi == int(dphi) else f"{dphi:.1f}"
                Z[i, j] = r["auroc"].get(key, r["auroc"].get(str(dphi)))
        snr_dense = np.linspace(snr_vals.min(), snr_vals.max(), 80)
        dphi_dense = np.linspace(dphi_vals.min(), dphi_vals.max(), 60)
        spline = RectBivariateSpline(snr_vals, dphi_vals, Z, kx=1, ky=1)
        Z_dense = spline(snr_dense, dphi_dense)
        extent = [dphi_vals.min(), dphi_vals.max(), snr_vals.min(), snr_vals.max()]
    else:
        # auto_dphi: irregular grid - collect (snr, dphi, auroc) and use griddata
        from scipy.interpolate import griddata
        points_xy = []
        values = []
        for r in results:
            snr = r["snr"]
            for key in r["auroc"]:
                dphi = float(key)
                points_xy.append((snr, dphi))
                values.append(r["auroc"][key])
        points_xy = np.array(points_xy)
        values = np.array(values)
        snr_min, snr_max = snr_vals.min(), snr_vals.max()
        dphi_min = min(p[1] for p in points_xy)
        dphi_max = max(p[1] for p in points_xy)
        snr_dense = np.linspace(snr_min, snr_max, 80)
        dphi_dense = np.linspace(dphi_min, dphi_max, 60)
        sgrid, dgrid = np.meshgrid(snr_dense, dphi_dense, indexing="ij")
        Z_dense = griddata(points_xy, values, (sgrid, dgrid), method="linear", fill_value=0.5)
        extent = [dphi_min, dphi_max, snr_min, snr_max]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(
        Z_dense,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax.set_xlabel("Δφ (rad)")
    ax.set_ylabel("SNR")
    if all(isinstance(x, (int, float)) for x in dphi_raw):
        ax.set_xticks(dphi_vals)
    ax.set_yticks(snr_vals)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AUROC(S_fc)")
    ax.set_title("AUROC(S_fc) vs Δφ and SNR (interpolated)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()

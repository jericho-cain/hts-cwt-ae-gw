#!/usr/bin/env python3
"""
Diagnostic plot: Δf_c(t) vs time for Δφ = {0, 0.3, 1, 3}.

Confirms that the time-local centroid shift behaves physically (smooth, grows
toward late times, scales with Δφ).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import accel_from_delta_phi, apply_losa_constant_accel
from utils.io import load_yaml

from experiments.observability_phase_metric import (
    DEFAULT_CONFIG,
    complex_cwt_pipeline,
    joint_gate,
    compute_freq_centroid,
)

DELTA_PHI_PLOT = [0.0, 0.3, 1.0, 3.0]


def main():
    cfg = load_yaml(DEFAULT_CONFIG)
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    T = int(data_cfg.get("T", 4096))
    fs = float(data_cfg.get("sample_rate", 1024))
    snr = float(data_cfg.get("snr", 5.0))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    duration = T / fs
    p0 = cfg.get("phase0_losa", {})
    f_star = float(p0.get("f_star_hz", 40.0))
    syn = cfg.get("synthetic", {})
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    fmin = float(cwt_cfg.get("fmin", 20.0))
    fmax = float(cwt_cfg.get("fmax", 512.0))
    n_scales = int(cwt_cfg.get("target_height", 8))
    downsample_factor = 4
    f_start = float(syn.get("chirp_f_start", 12.0))
    f_end = float(syn.get("chirp_f_end", 65.0))
    t_peak = float(syn.get("chirp_t_peak", 0.55))
    sigma = float(syn.get("chirp_sigma", 0.10))
    amplitude = float(syn.get("signal_amplitude", 1e-20))

    h_iso = generate_isolated_chirp(
        T=T, sample_rate=fs, f_start=f_start, f_end=f_end,
        t_peak=t_peak, sigma=sigma, amplitude=amplitude, seed=seed,
    ).astype(np.float64)
    h_norm = np.sqrt(np.mean(h_iso**2)) + 1e-30
    scale = (snr * 1e-21) / h_norm
    h_iso = (h_iso * scale).astype(np.float32)

    C_iso, _, freqs_pywt = complex_cwt_pipeline(
        h_iso, int(fs), downsample_factor, fmin, fmax, n_scales
    )
    n_time = C_iso.shape[1]
    fs_down = fs / downsample_factor
    t_sec = np.arange(n_time, dtype=np.float64) / fs_down

    P_iso = np.abs(C_iso) ** 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(DELTA_PHI_PLOT)))

    for i, delta_phi in enumerate(DELTA_PHI_PLOT):
        if delta_phi == 0:
            h_losa = h_iso.copy()
        else:
            a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)
            h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=fs, a_los=a_los)
            h_losa = h_losa.astype(np.float32)

        C_losa, _, _ = complex_cwt_pipeline(
            h_losa, int(fs), downsample_factor, fmin, fmax, n_scales
        )
        P_losa = np.abs(C_losa) ** 2
        gate = joint_gate(P_iso, P_losa)

        f_c_iso = compute_freq_centroid(P_iso, freqs_pywt, gate)
        f_c_losa = compute_freq_centroid(P_losa, freqs_pywt, gate)
        delta_f_c = f_c_losa - f_c_iso

        ax.plot(t_sec, delta_f_c, label=f"Δφ = {delta_phi}", color=colors[i], linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δf_c(t) (Hz)")
    ax.set_title("Frequency centroid shift Δf_c(t) = f_c,LOSA(t) − f_c,iso(t)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")

    out_path = ROOT / "experiments/outputs_corrected/observability_delta_fc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

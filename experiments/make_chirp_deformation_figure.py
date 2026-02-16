#!/usr/bin/env python3
"""
Physics-first chirp deformation figure: LOSA time-warp changes the chirp trajectory.

Shows paired h_iso(t) vs h_losa(t) and f_c,iso(t) vs f_c,losa(t) with Δφ = 3 rad.
No CWT panels—intuition anchor for the paper before AUROC/Λ plots.

Uses the exact Phase0 chirp generator (tight chirp: f_start=12 Hz, f_end=65 Hz).
Noise-free (SNR → ∞). Constant-accel LOSA model.

---
CAPTION STUB (LaTeX):
\\textbf{Figure X.} Chirp deformation under LOSA.
(a) Strain overlay: isolated (black) and LOSA-modified (red) chirps;
    Δh(t) = h_losa − h_iso (faint). (b) Instantaneous frequency trajectories
    from CWT centroid: f_c,iso(t), f_c,losa(t), and Δf_c(t).
    Δφ_env = 3 rad, Δt_end ≈ N ms, noise-free.
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
from data.synthetic.losa import C_LIGHT, accel_from_delta_phi, apply_losa_constant_accel
from utils.io import load_yaml

from experiments.observability_phase_metric import (
    DEFAULT_CONFIG,
    complex_cwt_pipeline,
    joint_gate,
    compute_freq_centroid,
    compute_losa_dt,
)


DELTA_PHI = 3.0
DOWNSAMPLE_FACTOR = 4


def main():
    cfg = load_yaml(DEFAULT_CONFIG)
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    T = int(data_cfg.get("T", 4096))
    fs = float(data_cfg.get("sample_rate", 1024))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    duration = T / fs

    p0 = cfg.get("phase0_losa", {})
    f_star = float(p0.get("f_star_hz", 40.0))
    syn = cfg.get("synthetic", {})
    f_start = float(syn.get("chirp_f_start", 12.0))
    f_end = float(syn.get("chirp_f_end", 65.0))
    t_peak = float(syn.get("chirp_t_peak", 0.55))
    sigma = float(syn.get("chirp_sigma", 0.10))
    amplitude = float(syn.get("signal_amplitude", 1e-20))

    noise_sigma = 0.0  # noise-free
    # Sanity: confirm Phase0 parameterization
    print(f"noise_sigma = {noise_sigma} (noise-free)")
    print(f"LOSA model: constant accel (Phase0), f_star = {f_star} Hz")

    # 1. Generate paired signals
    h_iso = generate_isolated_chirp(
        T=T,
        sample_rate=fs,
        f_start=f_start,
        f_end=f_end,
        t_peak=t_peak,
        sigma=sigma,
        amplitude=amplitude,
        seed=seed,
    ).astype(np.float64)

    a_los = accel_from_delta_phi(DELTA_PHI, duration, f_star_hz=f_star)
    h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=fs, a_los=a_los)
    h_losa = h_losa.astype(np.float64)

    # 2. dt_end and max |Δt|
    dt_arr = compute_losa_dt(a_los, T, fs)
    dt_end = float(np.abs(dt_arr[-1]))
    dt_max = float(np.max(np.abs(dt_arr)))
    dt_end_ms = dt_end * 1000
    print(f"dt_end = {dt_end:.6e} s = {dt_end_ms:.4f} ms")
    print(f"max |Δt(t)| = {dt_max:.6e} s")

    # 3. Frequency centroid via CWT (same as Phase0 observability)
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    fmin = float(cwt_cfg.get("fmin", 20.0))
    fmax = float(cwt_cfg.get("fmax", 512.0))
    n_scales = int(cwt_cfg.get("target_height", 8))

    C_iso, _, freqs_pywt = complex_cwt_pipeline(
        h_iso, int(fs), DOWNSAMPLE_FACTOR, fmin, fmax, n_scales
    )
    C_losa, _, _ = complex_cwt_pipeline(
        h_losa, int(fs), DOWNSAMPLE_FACTOR, fmin, fmax, n_scales
    )
    P_iso = np.abs(C_iso) ** 2
    P_losa = np.abs(C_losa) ** 2
    gate = joint_gate(P_iso, P_losa)

    f_c_iso = compute_freq_centroid(P_iso, freqs_pywt, gate)
    f_c_losa = compute_freq_centroid(P_losa, freqs_pywt, gate)
    delta_f_c = f_c_losa - f_c_iso

    valid = (gate > 0.5) & np.isfinite(f_c_iso) & np.isfinite(f_c_losa)
    max_delta_fc = float(np.nanmax(np.abs(delta_f_c[valid]))) if np.any(valid) else 0.0
    print(f"max |Δf_c(t)| = {max_delta_fc:.4f} Hz")

    # Time axis
    n_time = h_iso.shape[0]
    t_sec = np.arange(n_time, dtype=np.float64) / fs

    n_time_down = P_iso.shape[1]
    fs_down = fs / DOWNSAMPLE_FACTOR
    t_down = np.arange(n_time_down, dtype=np.float64) / fs_down

    # Normalize strain for clean overlay
    h_iso_norm = h_iso / (np.max(np.abs(h_iso)) + 1e-30)
    h_losa_norm = h_losa / (np.max(np.abs(h_iso)) + 1e-30)
    delta_h = h_losa - h_iso
    delta_h_scale = np.max(np.abs(h_iso)) + 1e-30
    delta_h_norm = delta_h / delta_h_scale  # same units as normalized strain

    # 4. 1×2 figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Strain overlay
    ax = axes[0]
    ax.plot(t_sec, h_iso_norm, "k-", linewidth=1.0, label=r"$h_{\mathrm{iso}}(t)$")
    ax.plot(t_sec, h_losa_norm, "r-", linewidth=1.0, label=r"$h_{\mathrm{LOSA}}(t)$")
    ax.plot(t_sec, delta_h_norm, color="gray", linewidth=0.7, linestyle="--", alpha=0.7, label=r"$\Delta h(t)$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain (norm)")
    ax.set_title("(a) Strain overlay")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # (b) Frequency trajectory overlay
    ax = axes[1]
    ax.plot(t_down, f_c_iso, "k-", linewidth=1.2, label=r"$f_{c,\mathrm{iso}}(t)$")
    ax.plot(t_down, f_c_losa, "r-", linewidth=1.2, label=r"$f_{c,\mathrm{LOSA}}(t)$")
    ax.plot(t_down, delta_f_c, color="gray", linewidth=0.7, linestyle="--", alpha=0.7, label=r"$\Delta f_c(t)$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("(b) Frequency trajectory")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # Annotations
    ann_text = (
        rf"$\Delta\phi_{{\mathrm{{env}}}} = {DELTA_PHI:.0f}$ rad"
        + "\n"
        + rf"$\Delta t_{{\mathrm{{end}}}} \approx {dt_end_ms:.2f}$ ms"
        + "\n"
        + r"SNR $= \infty$"
    )
    ax.annotate(
        ann_text,
        xy=(0.97, 0.15),
        xycoords="axes fraction",
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()

    # 5. Save
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "chirp_deformation.png"
    pdf_path = out_dir / "chirp_deformation.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close()

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()

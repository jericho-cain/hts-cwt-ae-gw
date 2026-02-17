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
import pywt
from scipy.signal import butter, decimate, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import C_LIGHT, accel_from_delta_phi, apply_losa_constant_accel

# CWT/gate/centroid inlined to avoid observability_phase_metric → utils → torch
COLUMN_ENERGY_THRESH = 0.01
COMPLEX_WAVELET = "cmor1.5-1.0"


def _compute_losa_dt(a_los: float, n_samples: int, fs: float, t0: float = 0.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float64) / fs
    return 0.5 * (a_los / C_LIGHT) * (t - t0) ** 2


def _complex_cwt_pipeline(strain, sample_rate, downsample_factor, fmin, fmax, n_scales):
    if downsample_factor > 1:
        x = decimate(strain.astype(np.float64), downsample_factor, zero_phase=True)
        fs = sample_rate / downsample_factor
    else:
        x = strain.astype(np.float64)
        fs = float(sample_rate)
    sos = butter(4, fmin, btype="high", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x)
    whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
    freqs_grid = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    fc = pywt.central_frequency(pywt.ContinuousWavelet(COMPLEX_WAVELET))
    scales = fc * fs / freqs_grid
    coeffs, freqs_pywt = pywt.cwt(whitened, scales, COMPLEX_WAVELET, sampling_period=1 / fs)
    freqs_pywt = np.asarray(freqs_pywt, dtype=np.float64)
    return coeffs.astype(np.complex128), freqs_grid, freqs_pywt


def _joint_gate(P_iso, P_losa):
    colsum_iso = np.sum(P_iso, axis=0)
    colsum_losa = np.sum(P_losa, axis=0)
    thr = COLUMN_ENERGY_THRESH * max(colsum_iso.max(), colsum_losa.max())
    return ((colsum_iso > thr) & (colsum_losa > thr)).astype(np.float64)


def _compute_freq_centroid(P, freqs, gate, eps=1e-30):
    n_time = P.shape[1]
    f_c = np.full(n_time, np.nan, dtype=np.float64)
    for t in range(n_time):
        if gate[t] < 0.5:
            continue
        Pcol = P[:, t] + eps
        denom = np.sum(Pcol)
        if denom < eps:
            continue
        f_c[t] = np.sum(freqs * Pcol) / denom
    return f_c


DELTA_PHI = 3.0
DOWNSAMPLE_FACTOR = 4
# Phase0 tight chirp config (ground_phase0_tight_chirp.yaml)
T = 32768
FS = 1024.0
F_STAR = 40.0
F_START = 12.0
F_END = 65.0
T_PEAK = 0.55
SIGMA = 0.10
AMPLITUDE = 1e-20
SEED = 42
FMIN = 10.0
FMAX = 64.0
N_SCALES = 8


def main():
    duration = T / FS
    print("noise_sigma = 0.0 (noise-free)")
    print(f"LOSA model: constant accel (Phase0), f_star = {F_STAR} Hz")

    # 1. Generate paired signals
    h_iso = generate_isolated_chirp(
        T=T,
        sample_rate=FS,
        f_start=F_START,
        f_end=F_END,
        t_peak=T_PEAK,
        sigma=SIGMA,
        amplitude=AMPLITUDE,
        seed=SEED,
    ).astype(np.float64)

    a_los = accel_from_delta_phi(DELTA_PHI, duration, f_star_hz=F_STAR)
    h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=FS, a_los=a_los)
    h_losa = h_losa.astype(np.float64)

    # 2. dt_end and max |Δt|
    dt_arr = _compute_losa_dt(a_los, T, FS)
    dt_end = float(np.abs(dt_arr[-1]))
    dt_max = float(np.max(np.abs(dt_arr)))
    dt_end_ms = dt_end * 1000
    print(f"dt_end = {dt_end:.6e} s = {dt_end_ms:.4f} ms")
    print(f"max |Δt(t)| = {dt_max:.6e} s")

    # 3. Frequency centroid via CWT (same as Phase0 observability)
    C_iso, _, freqs_pywt = _complex_cwt_pipeline(
        h_iso, int(FS), DOWNSAMPLE_FACTOR, FMIN, FMAX, N_SCALES
    )
    C_losa, _, _ = _complex_cwt_pipeline(
        h_losa, int(FS), DOWNSAMPLE_FACTOR, FMIN, FMAX, N_SCALES
    )
    P_iso = np.abs(C_iso) ** 2
    P_losa = np.abs(C_losa) ** 2
    gate = _joint_gate(P_iso, P_losa)

    f_c_iso = _compute_freq_centroid(P_iso, freqs_pywt, gate)
    f_c_losa = _compute_freq_centroid(P_losa, freqs_pywt, gate)
    delta_f_c = f_c_losa - f_c_iso

    valid = (gate > 0.5) & np.isfinite(f_c_iso) & np.isfinite(f_c_losa)
    max_delta_fc = float(np.nanmax(np.abs(delta_f_c[valid]))) if np.any(valid) else 0.0
    print(f"max |Δf_c(t)| = {max_delta_fc:.4f} Hz")

    # Time axis
    n_time = h_iso.shape[0]
    t_sec = np.arange(n_time, dtype=np.float64) / FS

    n_time_down = P_iso.shape[1]
    fs_down = FS / DOWNSAMPLE_FACTOR
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

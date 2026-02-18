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

import matplotlib.patheffects as pe
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


def _smooth_ma(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving-average smoothing, ignoring NaNs. w must be odd."""
    if w <= 1:
        return x
    w = int(w)
    if w % 2 == 0:
        w += 1
    x2 = x.copy()
    nan_mask = np.isnan(x2)
    x2[nan_mask] = 0.0
    k = np.ones(w, dtype=np.float64)
    num = np.convolve(x2, k, mode="same")
    den = np.convolve((~nan_mask).astype(np.float64), k, mode="same")
    out = np.full_like(x, np.nan, dtype=np.float64)
    good = den > 0
    out[good] = num[good] / den[good]
    return out


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
    # Trim gate edges so Δf_c(t) doesn't include boundary artifacts (CWT edge/gate transitions)
    EDGE_TRIM = 3
    gate_bool = gate > 0.5
    idx = np.where(gate_bool)[0]
    if idx.size > 0:
        i0, i1 = int(idx[0]), int(idx[-1])
        i0 = min(i0 + EDGE_TRIM, gate_bool.size)
        i1 = max(i1 - EDGE_TRIM, -1)
        gate_bool2 = np.zeros_like(gate_bool)
        if i1 > i0:
            gate_bool2[i0 : i1 + 1] = True
    else:
        gate_bool2 = gate_bool.copy()
    gate_for_plot = gate_bool2

    f_c_iso = _compute_freq_centroid(P_iso, freqs_pywt, gate)
    f_c_losa = _compute_freq_centroid(P_losa, freqs_pywt, gate)
    delta_f_c = f_c_losa - f_c_iso

    # --- visualization-only cleanup for Δf_c(t) ---
    SMOOTH_W = 101  # stronger smoothing than PN (Gaussian centroid has more scale jitter)
    PLOT_STRIDE = 10  # more decimation for cleaner dashed line
    # mask first so smoothing doesn't create boundary steps
    delta_f_c_plot = delta_f_c.copy()
    delta_f_c_plot[~gate_for_plot] = np.nan
    delta_f_c_plot = _smooth_ma(delta_f_c_plot, SMOOTH_W)
    # drop boundary samples to avoid turn-on artifacts
    finite = np.isfinite(delta_f_c_plot)
    idx = np.where(finite)[0]
    N_BOUNDARY = 6
    if idx.size > 2 * N_BOUNDARY:
        delta_f_c_plot[idx[:N_BOUNDARY]] = np.nan
        delta_f_c_plot[idx[-N_BOUNDARY:]] = np.nan

    max_delta_fc = float(np.nanmax(np.abs(delta_f_c)))
    median_delta_fc = float(np.nanmedian(np.abs(delta_f_c)))
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

    # Zoom window for panel (a): last 0.5 s before peak so oscillations are visible
    ZOOM_WINDOW = 0.5  # seconds
    t_peak = float(t_sec[np.argmax(np.abs(h_iso_norm))])
    mask_zoom = (t_sec >= t_peak - ZOOM_WINDOW) & (t_sec <= t_peak)

    # 4. Figure: left = strain overlay, right = stacked f_c + Δf_c
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])  # full left column
    ax1 = fig.add_subplot(gs[0, 1])  # top-right (f_c)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)  # bottom-right (Δf_c)

    # --- Plot styles ---
    ISO_COLOR = "#1f4e79"  # deep blue
    LOSA_COLOR = "#b22222"  # crimson
    DELTA_COLOR = "#666666"  # medium gray (secondary, readable)
    iso_style = dict(color=ISO_COLOR, lw=0.9, alpha=1.0, zorder=2)
    losa_style = dict(color=LOSA_COLOR, lw=0.9, alpha=1.0, zorder=3)
    diff_style = dict(color=DELTA_COLOR, lw=0.8, ls="--", alpha=0.9, zorder=1)

    # (a) Strain overlay — zoomed to last 0.5 s before peak (oscillations visible)
    ax0.plot(t_sec[mask_zoom], h_iso_norm[mask_zoom], label=r"$h_{\mathrm{iso}}(t)$", **iso_style)
    ax0.plot(t_sec[mask_zoom], h_losa_norm[mask_zoom], label=r"$h_{\mathrm{LOSA}}(t)$", **losa_style)
    ax0.plot(t_sec[mask_zoom], delta_h_norm[mask_zoom], label=r"$\Delta h(t)$", **diff_style)
    ax0.set_xlim(t_peak - ZOOM_WINDOW, t_peak)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Strain (norm)")
    ax0.set_title("(a) Strain overlay")
    handles, labels = ax0.get_legend_handles_labels()
    order = [
        labels.index(r"$h_{\mathrm{iso}}(t)$"),
        labels.index(r"$h_{\mathrm{LOSA}}(t)$"),
        labels.index(r"$\Delta h(t)$"),
    ]
    ax0.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # --- (b1) Frequency trajectory ---
    ax1.plot(t_down, f_c_iso, color=ISO_COLOR, lw=1.2, label=r"$f_{c,\mathrm{iso}}(t)$")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("(b) Frequency trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_xlim(0, duration)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- (b2) Residual ---
    ax2.plot(
        t_down[::PLOT_STRIDE],
        delta_f_c_plot[::PLOT_STRIDE],
        color=DELTA_COLOR, lw=1.0, ls="--",
        label=r"$\Delta f_c(t)$",
    )
    abs_df = np.abs(delta_f_c_plot)
    finite = np.isfinite(abs_df)
    if np.any(finite):
        df_max = float(np.nanmax(abs_df[finite]))
        ax2.set_ylim(0, max(df_max * 1.1, 0.05))
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$\Delta f_c(t)$ (Hz)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    # Annotations
    ann_text = (
        r"$\Delta\phi_{\rm env} = 3$ rad" + "\n"
        + r"$\Delta t_{{\rm end}} \approx {:.2f}$ ms".format(dt_end * 1e3) + "\n"
        + r"$\max |\Delta f_c| = {:.3f}$ Hz".format(max_delta_fc) + "\n"
        + r"$\mathrm{{median}} |\Delta f_c| = {:.3f}$ Hz".format(median_delta_fc) + "\n"
        + r"noise-free"
    )
    ax1.annotate(
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

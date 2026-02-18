#!/usr/bin/env python3
"""
PN waveform sanity-check figure using LALSimulation.

Generates a single TaylorT4 time-domain inspiral, applies LOSA time-warp,
and produces a 2-panel figure (strain overlay + Δf_c(t)) like chirp_deformation.
Standalone diagnostic—does not touch observability sweeps.

Requires LALSuite (lalsimulation). Install via conda-forge in a separate env:
  micromamba create -n pn-demo -c conda-forge python=3.11 lalsuite
  micromamba run -n pn-demo python experiments/pn_minimal_demo.py

Or: conda create -n pn-demo -c conda-forge python=3.11 lalsuite
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# LALSuite imports (may fail if not in pn-demo env)
try:
    import lal
    import lalsimulation as lalsim
except ImportError as e:
    print("LALSuite not found. Create a conda env with lalsuite:")
    print("  micromamba create -n pn-demo -c conda-forge python=3.11 lalsuite")
    print("  micromamba run -n pn-demo python experiments/pn_minimal_demo.py")
    print(f"Import error: {e}")
    sys.exit(1)

# Rest of imports (no LAL deps, no torch—avoids utils/seed)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, decimate, sosfiltfilt

from data.synthetic.losa import C_LIGHT, accel_from_delta_phi, apply_losa_constant_accel

# CWT/gate/centroid logic inlined to avoid observability_phase_metric → utils → torch
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
    """Δt(t) = 0.5 * (a/c) * (t - t0)^2. Returns (n_samples,) in seconds."""
    t = np.arange(n_samples, dtype=np.float64) / fs
    return 0.5 * (a_los / C_LIGHT) * (t - t0) ** 2


def _complex_cwt_pipeline(strain, sample_rate, downsample_factor, fmin, fmax, n_scales):
    """Same as observability_phase_metric.complex_cwt_pipeline. Returns (C, freqs_grid, freqs_pywt)."""
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
    """Gate: keep t where (colsum_iso > thr) & (colsum_losa > thr)."""
    colsum_iso = np.sum(P_iso, axis=0)
    colsum_losa = np.sum(P_losa, axis=0)
    thr = COLUMN_ENERGY_THRESH * max(colsum_iso.max(), colsum_losa.max())
    return ((colsum_iso > thr) & (colsum_losa > thr)).astype(np.float64)


def _compute_freq_centroid(P, freqs, gate, eps=1e-30):
    """f_c(t) = sum_k(f_k * P[k,t]) / sum_k(P[k,t]) for gated t."""
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


# Match existing 32 s / 1024 Hz setup
DURATION_S = 32.0
FS = 1024.0
T_SAMPLES = int(DURATION_S * FS)
DELTA_T = 1.0 / FS
F_STAR_HZ = 40.0
DELTA_PHI = 3.0
DOWNSAMPLE_FACTOR = 4

# PN parameters: fixed masses, zero spins
M1_SOLAR = 20.0
M2_SOLAR = 20.0
F_LOW = 12.0  # Hz, match synthetic chirp band
DISTANCE_MPC = 100.0


def generate_pn_td_waveform() -> np.ndarray:
    """Generate TaylorT4 time-domain plus polarization, return strain as numpy array."""
    m1 = M1_SOLAR * lal.MSUN_SI
    m2 = M2_SOLAR * lal.MSUN_SI
    distance = DISTANCE_MPC * 1e6 * lal.PC_SI
    approx = lalsim.SimInspiralGetApproximantFromString("TaylorT4")

    # LAL SimInspiralChooseTDWaveform: (m1, m2, S1x,S1y,S1z, S2x,S2y,S2z, r, i,
    #   phiRef, longAscNodes, eccentricity, meanPerAno, deltaT, f_min, f_ref, LALpars, approximant)
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        m1, m2,
        0.0, 0.0, 0.0,  # S1x, S1y, S1z
        0.0, 0.0, 0.0,  # S2x, S2y, S2z
        distance,
        0.0,   # inclination
        0.0,   # phiRef
        0.0,   # longAscNodes
        0.0,   # eccentricity
        0.0,   # meanPerAno
        DELTA_T,
        F_LOW,
        0.0,   # f_ref (0 = waveform end)
        None,  # LALpars
        approx,
    )

    strain = np.array(hp.data.data, dtype=np.float64)
    return strain


def main():
    # 1. Generate PN waveform
    print("Generating TaylorT4 waveform...")
    h_full = generate_pn_td_waveform()

    # Truncate or pad to T_SAMPLES
    if len(h_full) >= T_SAMPLES:
        # Take the last T_SAMPLES (end of inspiral, higher SNR in band)
        h_iso = h_full[-T_SAMPLES:].copy()
    else:
        h_iso = np.zeros(T_SAMPLES, dtype=np.float64)
        h_iso[-len(h_full) :] = h_full

    h_iso = h_iso.astype(np.float64)

    # 2. LOSA: a_|| from Δφ_env via Δt_end = Δφ/(2π f_star), a = 2 c Δt_end / T²
    a_los = accel_from_delta_phi(DELTA_PHI, DURATION_S, f_star_hz=F_STAR_HZ)
    h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=FS, a_los=a_los)
    h_losa = h_losa.astype(np.float64)

    # 3. Δt_end
    dt_arr = _compute_losa_dt(a_los, T_SAMPLES, FS)
    dt_end = float(np.abs(dt_arr[-1]))
    dt_end_ms = dt_end * 1000
    print(f"Δt_end = {dt_end:.6e} s = {dt_end_ms:.4f} ms")

    # 4. CWT + centroid
    fmin = 10.0
    fmax = 128.0
    n_scales = 8

    C_iso, _, freqs_pywt = _complex_cwt_pipeline(
        h_iso, int(FS), DOWNSAMPLE_FACTOR, fmin, fmax, n_scales
    )
    C_losa, _, _ = _complex_cwt_pipeline(
        h_losa, int(FS), DOWNSAMPLE_FACTOR, fmin, fmax, n_scales
    )
    P_iso = np.abs(C_iso) ** 2
    P_losa = np.abs(C_losa) ** 2
    gate = _joint_gate(P_iso, P_losa)
    # Trim gate edges so Δf_c(t) doesn't include boundary artifacts (CWT edge/gate transitions)
    EDGE_TRIM = 10
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
    # Extra erosion for plotting only (removes centroid lock-on transient)
    FS_DOWN = FS / DOWNSAMPLE_FACTOR
    PLOT_ERODE_S = 0.8  # try 0.4–1.2 s for PN
    PLOT_ERODE = int(PLOT_ERODE_S * FS_DOWN)
    gate_plot = gate_for_plot.copy()
    idxp = np.where(gate_plot)[0]
    if idxp.size > 2 * PLOT_ERODE:
        gate_plot[idxp[:PLOT_ERODE]] = False
        gate_plot[idxp[-PLOT_ERODE:]] = False

    f_c_iso = _compute_freq_centroid(P_iso, freqs_pywt, gate)
    f_c_losa = _compute_freq_centroid(P_losa, freqs_pywt, gate)
    delta_f_c = f_c_losa - f_c_iso

    # --- visualization-only cleanup for Δf_c(t) ---
    SMOOTH_W = 31
    PLOT_STRIDE = 4
    delta_f_c_plot = delta_f_c.copy()
    delta_f_c_plot[~gate_plot] = np.nan
    delta_f_c_plot = _smooth_ma(delta_f_c_plot, SMOOTH_W)
    delta_f_c_plot[~gate_plot] = np.nan

    max_delta_fc = float(np.nanmax(np.abs(delta_f_c)))
    median_delta_fc = float(np.nanmedian(np.abs(delta_f_c)))
    print(f"max |Δf_c(t)| = {max_delta_fc:.4f} Hz")

    # 5. Time axes (shift for display: merger at t_target, not right edge)
    t_sec = np.arange(T_SAMPLES, dtype=np.float64) / FS
    n_down = P_iso.shape[1]
    t_down = np.arange(n_down, dtype=np.float64) / (FS / DOWNSAMPLE_FACTOR)
    t_merge = t_sec[np.argmax(np.abs(h_iso))]
    t_target = 24.0
    t_plot_sec = t_sec - t_merge + t_target
    t_plot_down = t_down - t_merge + t_target
    x_min = t_target - 20.0
    x_max = t_target + 8.0

    # 6. Normalize strain for overlay
    scale = np.max(np.abs(h_iso)) + 1e-30
    h_iso_norm = h_iso / scale
    h_losa_norm = h_losa / scale
    delta_h_norm = (h_losa - h_iso) / scale

    # Zoom window for panel (a): last 0.4 s before merger (oscillations visible)
    ZOOM_WINDOW = 0.4  # seconds
    mask_zoom = (t_plot_sec >= t_target - ZOOM_WINDOW) & (t_plot_sec <= t_target)

    # 7. Figure: left = strain overlay, right = stacked f_c + Δf_c
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.05)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)

    # --- Plot styles ---
    ISO_COLOR = "#1f4e79"  # deep blue
    LOSA_COLOR = "#b22222"  # crimson
    DELTA_COLOR = "#666666"  # medium gray (secondary, readable)
    iso_style = dict(color=ISO_COLOR, lw=0.9, alpha=1.0, zorder=2)
    losa_style = dict(color=LOSA_COLOR, lw=0.9, alpha=1.0, zorder=3)
    diff_style = dict(color=DELTA_COLOR, lw=0.8, ls="--", alpha=0.9, zorder=1)

    # (a) Strain overlay — zoomed to last 0.4 s before merger (oscillations visible)
    ax0.plot(t_plot_sec[mask_zoom], h_iso_norm[mask_zoom], label=r"$h_{\mathrm{iso}}(t)$", **iso_style)
    ax0.plot(t_plot_sec[mask_zoom], h_losa_norm[mask_zoom], label=r"$h_{\mathrm{LOSA}}(t)$", **losa_style)
    ax0.plot(t_plot_sec[mask_zoom], delta_h_norm[mask_zoom], label=r"$\Delta h(t)$", **diff_style)
    ax0.set_xlim(t_target - ZOOM_WINDOW, t_target)
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("Strain (norm)")
    ax0.set_title("(a) PN strain overlay (TaylorT4)")
    handles, labels = ax0.get_legend_handles_labels()
    order = [
        labels.index(r"$h_{\mathrm{iso}}(t)$"),
        labels.index(r"$h_{\mathrm{LOSA}}(t)$"),
        labels.index(r"$\Delta h(t)$"),
    ]
    ax0.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", fontsize=9)
    ax0.grid(True, alpha=0.3)

    # --- (b1) Frequency trajectory ---
    f_c_iso_plot = f_c_iso.copy()
    f_c_iso_plot[~gate_plot] = np.nan
    ax1.plot(t_plot_down, f_c_iso_plot, color=ISO_COLOR, lw=1.2, label=r"$f_{c,\mathrm{iso}}(t)$")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title("(b) Frequency trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_xlim(x_min, x_max)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # --- (b2) Residual ---
    # PN: tail mask (after gate masking) — centroid unstable very near merger
    MASK_TAIL_S = 1.0  # bump to 2.0 s if spike remains
    t_end = np.nanmax(t_plot_down[np.isfinite(t_plot_down)])
    delta_f_c_plot[t_plot_down > (t_end - MASK_TAIL_S)] = np.nan
    finite_idx = np.where(np.isfinite(delta_f_c_plot))[0]
    DROP_HEAD = 8
    if finite_idx.size > DROP_HEAD:
        delta_f_c_plot[finite_idx[:DROP_HEAD]] = np.nan
    ax2.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax2.plot(
        t_plot_down[::PLOT_STRIDE],
        delta_f_c_plot[::PLOT_STRIDE],
        color=DELTA_COLOR, lw=1.0, ls="--",
        label=r"$\Delta f_c(t)$",
    )
    abs_df = np.abs(delta_f_c_plot)
    finite = np.isfinite(abs_df)
    if np.any(finite):
        df_ref = np.nanpercentile(abs_df[finite], 99.0)
        df_ref = max(df_ref, 1e-6)
        ax2.set_ylim(-1.2 * df_ref, 1.2 * df_ref)
    yl = ax2.get_ylim()
    if (yl[1] - yl[0]) < 1e-3:
        ax2.set_ylim(-0.1, 0.1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel(r"$\Delta f_c(t)$ (Hz)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)

    ann_text = (
        r"$\Delta\phi_{\rm env} = 3$ rad" + "\n"
        + r"$\Delta t_{{\rm end}} \approx {:.2f}$ ms".format(dt_end * 1e3) + "\n"
        + r"$\max |\Delta f_c| = {:.3f}$ Hz".format(max_delta_fc) + "\n"
        + r"$\mathrm{{median}} |\Delta f_c| = {:.3f}$ Hz".format(median_delta_fc) + "\n"
        + r"TaylorT4, noise-free"
    )
    ax1.annotate(
        ann_text,
        xy=(0.03, 0.15),
        xycoords="axes fraction",
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()

    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "pn_chirp_deformation.png"
    pdf_path = out_dir / "pn_chirp_deformation.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close()

    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()

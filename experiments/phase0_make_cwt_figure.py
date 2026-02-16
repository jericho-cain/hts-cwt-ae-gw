#!/usr/bin/env python3
"""
Paper-ready CWT spectrogram figure: isolated vs LOSA.

Generates a 2×2 panel (or 2×1 if no model) showing:
  1) Log CWT magnitude for isolated chirp
  2) Log CWT magnitude for same chirp after LOSA
  3) ΔS = S_LOSA − S_iso (log-space, lightly time-smoothed)
  4) (Optional) AE reconstruction residual for LOSA sample
  5) Δf_ridge (ridge frequency shift) — "LOSA as time warp" panel

COI masked (light gray). Uses same preprocessing as experiments. Y-axis in Hz (log scale).

Conventions: cwt_clean returns scalogram shape (H, W) = (n_scales, time_len).
Preprocessor with return_before_norm=True returns log10(|W|) where W is CWT
coefficient magnitude (not power). Outputs to figures/.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Journal-style font sizes
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})


def _edges_from_centers(x):
    """Cell edges from center coordinates (linear spacing)."""
    x = np.asarray(x)
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=x.dtype)
    edges[1:-1] = x[:-1] + 0.5 * dx
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


def _log_edges_from_centers(f):
    """Cell edges from log-spaced center frequencies (geometric midpoints)."""
    f = np.asarray(f)
    edges = np.empty(f.size + 1, dtype=f.dtype)
    edges[1:-1] = np.sqrt(f[:-1] * f[1:])
    edges[0] = f[0] * f[0] / edges[1]
    edges[-1] = f[-1] * f[-1] / edges[-2]
    return edges


def _plot_cwt_panel(ax, time_s, freqs, S, cmap, vmin, vmax, ridge=None, fmin=None, fmax=None, chirp_end_hz=None):
    """Plot CWT spectrogram with log-spaced frequency axis (pcolormesh + log y-scale)."""
    t_edges = _edges_from_centers(time_s)
    f_edges = _log_edges_from_centers(freqs)
    T, F = np.meshgrid(t_edges, f_edges)
    pcm = ax.pcolormesh(
        T, F, S, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_yscale("log")
    if fmin is not None and fmax is not None:
        ax.set_ylim(fmin, fmax)
    # Y-ticks: adapt to band (e.g. [10,20,30,40,60] for 64 Hz, add 80,100,120 for 128 Hz)
    base_ticks = [10, 20, 30, 40, 60]
    yticks = [z for z in base_ticks + [80, 100, 120] if fmin <= z <= fmax] if fmax else base_ticks
    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
    if ridge is not None:
        ax.plot(time_s, ridge, "w-", lw=0.5, alpha=0.8)
    if chirp_end_hz is not None and fmin <= chirp_end_hz <= fmax:
        ax.axhline(chirp_end_hz, color="white", ls="--", lw=0.8, alpha=0.9)
    return pcm


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-ready CWT figure (iso vs LOSA)"
    )
    parser.add_argument(
        "--delta_phi",
        type=float,
        default=3.0,
        help="LOSA Δφ in radians (default: 3.0)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures",
        help="Output directory (default: figures)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/ground_phase0_tight_chirp.yaml",
        help="Config path for preprocessing and chirp params",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model checkpoint for residual panel (optional)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print nan/inf stats before plotting",
    )
    parser.add_argument(
        "--noise_free",
        action="store_true",
        default=True,
        help="Use noise-free signal for paper figure (default: True)",
    )
    parser.add_argument(
        "--noise",
        dest="noise_free",
        action="store_false",
        help="Include noise for debugging / matching experiment conditions",
    )
    parser.add_argument(
        "--low_res",
        action="store_true",
        help="Use config CWT resolution (fmax=64, H=8); default is high-res (fmax=128, H=48)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default=None,
        help="Override wavelet for figure (e.g. cmor6-1 for less sidelobes). Default: use config.",
    )
    parser.add_argument(
        "--show_chirp_end",
        action="store_true",
        help="Draw horizontal line at chirp end frequency (f1) on (a)/(b) panels for sanity check",
    )
    parser.add_argument(
        "--debug_freq",
        action="store_true",
        help="Print frequency calibration info (wavelet fc, freqs min/max)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1

    from utils.io import load_yaml
    from data.synthetic.isolated_generator import generate_isolated_chirp
    from data.synthetic.losa import apply_losa_constant_accel, accel_from_delta_phi, C_LIGHT
    from data.synthetic.noise_models import gaussian_noise
    from data.synthetic.datasets import make_phase0_batch

    import copy
    cfg = load_yaml(config_path)
    cfg_fig = copy.deepcopy(cfg)
    high_res = not args.low_res
    if high_res:
        cfg_fig.setdefault("preprocessing", {}).setdefault("cwt", {}).update({
            "fmax": 128.0,
            "fmin": 10.0,
            "target_height": 48,
        })
        logger.info("Using high-res figure: fmax=128 Hz, target_height=48")
        if args.wavelet is None:
            cfg_fig.setdefault("preprocessing", {}).setdefault("cwt", {})["wavelet"] = "cmor6-1"
            logger.info("Figure wavelet: cmor6-1 (less ringy for paper)")
    if args.wavelet is not None:
        cfg_fig.setdefault("preprocessing", {}).setdefault("cwt", {})["wavelet"] = args.wavelet
        logger.info(f"Figure wavelet override: {args.wavelet}")
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    cwt_cfg = cfg_fig.get("preprocessing", {}).get("cwt", {})
    p0 = cfg.get("phase0_losa", cfg.get("synthetic", {}))

    T = int(data_cfg.get("T", 32768))
    fs = float(data_cfg.get("sample_rate", 1024))
    snr = float(data_cfg.get("snr", 5.0))
    noise_sigma = float(data_cfg.get("noise_sigma", 1e-21))
    duration = T / fs
    f_star = float(p0.get("f_star_hz", 40.0))

    # Tight chirp: fixed params
    f0 = float(p0.get("chirp_f_start", [12.0, 12.0])[0])
    f1 = float(p0.get("chirp_f_end", [65.0, 65.0])[0])
    t_peak = 0.55
    sigma_env = 0.10

    a_los = accel_from_delta_phi(args.delta_phi, duration, f_star_hz=f_star)

    # Matched pair: same chirp, same noise, one shared scale factor (from isolated)
    np.random.seed(12345)
    h = generate_isolated_chirp(
        T=T, sample_rate=fs,
        f_start=f0, f_end=f1,
        t_peak=t_peak, sigma=sigma_env,
        amplitude=1e-20, seed=42,
    )
    nse_sigma = 0.0 if args.noise_free else noise_sigma
    nse = gaussian_noise(T, sigma=nse_sigma, seed=999)
    h_norm = np.sqrt(np.mean(h**2)) + 1e-30
    scale = (snr * noise_sigma) / h_norm
    x_iso = (h * scale + nse).astype(np.float32)

    h_losa = apply_losa_constant_accel(h.copy(), sample_rate=fs, a_los=a_los)
    x_losa = (h_losa * scale + nse).astype(np.float32)

    # Get log magnitude (before z-score) for paper-ready visualization
    from experiments.run_experiment import _build_preprocessor
    preprocessor = _build_preprocessor(
        cfg_fig, cwt_norm_mean=None, cwt_norm_std=None, return_before_norm=True
    )
    S_iso = preprocessor.process(x_iso)
    S_losa = preprocessor.process(x_losa)

    if S_iso is None or S_losa is None:
        logger.error("CWT preprocessing returned None")
        return 1

    # Preprocessor with return_before_norm=True returns log10(magnitude) directly
    # (cwt_clean: log_scalogram = np.log10(scalogram + 1e-10)). Use S directly.
    S_iso = np.squeeze(S_iso).astype(np.float64)
    S_losa = np.squeeze(S_losa).astype(np.float64)
    if S_iso.ndim == 3:
        S_iso = S_iso[0]
        S_losa = S_losa[0]

    # Sanitize log arrays (no 10^S inversion — avoids dynamic-range / resampling artifacts)
    for arr in (S_iso, S_losa):
        s_valid = arr[np.isfinite(arr)]
        fill = np.min(s_valid) if len(s_valid) > 0 else -99.0
        arr[:] = np.nan_to_num(arr, nan=fill, neginf=-99.0, posinf=99.0)

    # Axes: frequency (Hz) and time (s)
    fmin = float(cwt_cfg.get("fmin", 10.0))
    fmax = float(cwt_cfg.get("fmax", 64.0))
    H = int(cwt_cfg.get("target_height", 8))
    from preprocessing.cwt import get_cwt_display_freqs, get_cwt_coi
    wavelet = str(cwt_cfg.get("wavelet", "morl"))
    downsample_factor = int(cwt_cfg.get("downsample_factor", 8))
    fs_down = fs / downsample_factor
    n_time = S_iso.shape[1]
    time_s = np.linspace(0, (n_time - 1) / fs_down, n_time)
    freqs = get_cwt_display_freqs(wavelet, fmin, fmax, H, fs_down)

    if args.debug_freq:
        import pywt
        fc = pywt.central_frequency(pywt.ContinuousWavelet(wavelet))
        print(f"wavelet={wavelet} fc={fc:.6f}")
        print(f"freqs (pywt): min={freqs.min():.4f} max={freqs.max():.4f} Hz")

    # Log-power for display (damps sidelobes vs log-magnitude): log10(|W|^2) = 2*log10(|W|)
    S_iso_pow = 2.0 * S_iso
    S_losa_pow = 2.0 * S_losa

    # Light frequency smoothing (display only) to knock down ring edges
    from scipy.ndimage import gaussian_filter
    sigma_disp = (0.7, 0.3)  # (freq bins, time bins)
    S_iso_pow = gaussian_filter(S_iso_pow, sigma=sigma_disp, mode="nearest")
    S_losa_pow = gaussian_filter(S_losa_pow, sigma=sigma_disp, mode="nearest")

    # Mask cone-of-influence (keep full time axis; COI shown as light gray)
    coi = get_cwt_coi(freqs, fs_down, n_time, wavelet=wavelet)
    coi_mask = coi.astype(bool)
    S_iso_plot = S_iso_pow.copy()
    S_losa_plot = S_losa_pow.copy()
    S_iso_plot[:, coi_mask] = np.nan
    S_losa_plot[:, coi_mask] = np.nan

    # Main difference: ΔS = S_losa − S_iso (log-power space)
    dS = S_losa_plot - S_iso_plot
    dS = np.nan_to_num(dS, nan=0.0, posinf=0.0, neginf=0.0)

    # Light time smoothing on ΔS for visibility
    dS_smooth = gaussian_filter(dS, sigma=(0.0, 1.0), mode="nearest")

    # Relative intensity for top panels: S_rel = S - max(S), peak at 0
    valid = np.isfinite(S_iso_plot) & np.isfinite(S_losa_plot)
    S_all_max = np.nanmax(np.concatenate([S_iso_plot[valid], S_losa_plot[valid]]))
    S_iso_rel = S_iso_plot - S_all_max
    S_losa_rel = S_losa_plot - S_all_max
    S_rel_all = np.concatenate([S_iso_rel[valid], S_losa_rel[valid]])
    vmin = np.percentile(S_rel_all, 2) if len(S_rel_all) > 0 else -1.0
    vmax = 0.0

    # Chirp end frequency (for --show_chirp_end sanity check)
    chirp_end_hz = f1 if args.show_chirp_end else None

    # Mask ΔS: only show where signal energy exceeds threshold (avoids "barcode" in quiet regions)
    thresh_signal = -3.0  # log-power rel. to peak; ~15 dB down (2x log-mag)
    mask_signal = (S_iso_rel > thresh_signal) | (S_losa_rel > thresh_signal)
    dS_smooth[~mask_signal] = np.nan

    dlim = np.percentile(np.abs(dS_smooth[np.isfinite(dS_smooth)]), 99)
    dlim = max(dlim, 1e-6)

    # True LOSA time shift Δt(t) = 0.5 * (a_los/c) * t^2 (for panel d; replaces noisy ridge)
    dt_true = 0.5 * (a_los / C_LIGHT) * (time_s ** 2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        for name, A in [
            ("S_iso", S_iso),
            ("S_losa", S_losa),
            ("dS", dS),
        ]:
            print(
                name,
                "nan", np.isnan(A).sum(),
                "inf", np.isinf(A).sum(),
                "min", np.nanmin(A),
                "max", np.nanmax(A),
            )
        # Check for constant columns/rows (padding/zoom artifacts)
        A = np.squeeze(S_iso)
        col_std = np.std(A, axis=0)
        row_std = np.std(A, axis=1)
        print(
            "S_iso col std min/max:", np.min(col_std), np.max(col_std),
            "| row std min/max:", np.min(row_std), np.max(row_std),
        )

    # Colormaps: COI/masked regions shown as light gray
    cmap_vir = plt.get_cmap("viridis").copy()
    cmap_vir.set_bad(color=(0.9, 0.9, 0.9, 1.0))
    cmap_cw = plt.get_cmap("coolwarm").copy()
    cmap_cw.set_bad(color=(0.9, 0.9, 0.9, 1.0))

    # Individual panels (paper-ready): pcolormesh + log y-axis; no ridge overlay on (a)/(b)
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    im1 = _plot_cwt_panel(ax1, time_s, freqs, S_iso_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title("Isolated chirp")
    plt.colorbar(im1, ax=ax1, label=r"$\log_{10}|W|^2$ (rel. peak)")
    plt.tight_layout()
    plt.savefig(outdir / "phase0_cwt_iso.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    im2 = _plot_cwt_panel(ax2, time_s, freqs, S_losa_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_title(r"LOSA ($\Delta\phi=" + f"{args.delta_phi}" + r"$ rad)")
    plt.colorbar(im2, ax=ax2, label=r"$\log_{10}|W|^2$ (rel. peak)")
    plt.tight_layout()
    plt.savefig(outdir / "phase0_cwt_losa.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    im3 = _plot_cwt_panel(ax3, time_s, freqs, dS_smooth, cmap_cw, -dlim, dlim, fmin=fmin, fmax=fmax)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Frequency [Hz]")
    ax3.set_title(r"$\Delta S = S_{\mathrm{LOSA}} - S_{\mathrm{iso}}$")
    plt.colorbar(im3, ax=ax3, label=r"$\Delta S$")
    plt.tight_layout()
    plt.savefig(outdir / "phase0_cwt_diff.png", dpi=300, bbox_inches="tight")
    plt.close()

    resid = None
    if args.model_path:
        model_path = Path(args.model_path)
        if model_path.exists():
            from experiments.run_experiment import fit_cwt_dataset_norm, _build_preprocessor
            x_train, _ = make_phase0_batch(
                n=100, T=T, sample_rate=fs, snr=snr, a_los=0.0,
                noise_sigma=noise_sigma, seed=42,
                f_start_range=(f0, f0), f_end_range=(f1, f1),
                t_peak_range=(t_peak, t_peak), sigma_range=(sigma_env, sigma_env),
            )
            cwt_norm_mean, cwt_norm_std = fit_cwt_dataset_norm(cfg, x_train)
            preprocessor_norm = _build_preprocessor(
                cfg, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std,
            )
            X_losa = preprocessor_norm.process(x_losa)
            if X_losa is not None:
                X_losa = np.squeeze(X_losa)
                if X_losa.ndim == 3:
                    X_losa = X_losa[0]
                import torch
                from models.registry import build_model
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                load_cfg = checkpoint.get("config", cfg)
                model = build_model(load_cfg)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                x_in = torch.from_numpy(X_losa).float().unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    out = model(x_in)
                x_hat = out[0] if isinstance(out, (tuple, list)) else out
                resid = (x_hat - x_in).squeeze().numpy()
                resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
                resid_plot = resid.copy() if resid.ndim == 2 else resid
                if resid_plot.ndim == 2:
                    resid_plot[:, coi_mask] = np.nan

    # Triptych (paper-ready 3-row layout): iso | losa; ΔS; ridge shift
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(constrained_layout=True, figsize=(12, 5.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[1, :], sharex=ax1)
    ax4 = fig.add_subplot(gs[2, :], sharex=ax1)

    im1 = _plot_cwt_panel(ax1, time_s, freqs, S_iso_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
    ax1.set_title("Isolated")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes, va="top", ha="left", fontweight="bold")

    im2 = _plot_cwt_panel(ax2, time_s, freqs, S_losa_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
    ax2.set_title(r"LOSA ($\Delta\phi=" + f"{args.delta_phi}" + r"$ rad)")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.text(0.01, 0.98, "(b)", transform=ax2.transAxes, va="top", ha="left", fontweight="bold")
    plt.setp(ax2.get_yticklabels(), visible=True)

    im3 = _plot_cwt_panel(ax3, time_s, freqs, dS_smooth, cmap_cw, -dlim, dlim, fmin=fmin, fmax=fmax)
    ax3.set_title(r"$\Delta S = S_{\mathrm{LOSA}} - S_{\mathrm{iso}}$ (smoothed)")
    ax3.set_xlabel("")
    ax3.set_ylabel("Frequency [Hz]")
    ax3.text(0.01, 0.98, "(c)", transform=ax3.transAxes, va="top", ha="left", fontweight="bold")

    # Panel (d): true LOSA time shift Δt(t) = 0.5*(a/c)*t² (from generator model)
    ax4.plot(time_s, dt_true * 1e6, "k-", lw=1)
    ax4.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax4.set_ylabel(r"$\Delta t$ [µs]")
    ax4.set_xlabel("Time [s]")
    ax4.text(0.01, 0.98, "(d)", transform=ax4.transAxes, va="top", ha="left", fontweight="bold")

    cbar1 = fig.colorbar(im1, ax=[ax1, ax2], shrink=0.9, pad=0.02)
    cbar1.set_label(r"$\log_{10}|W|^2$ (rel. peak)")
    cbar3 = fig.colorbar(im3, ax=ax3, shrink=0.9, pad=0.02)
    cbar3.set_label(r"$\Delta S$")

    fig.savefig(outdir / "phase0_cwt_losa_triptych.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Combined quad (includes residual if model provided)
    if resid is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        im_a = _plot_cwt_panel(axes[0, 0], time_s, freqs, S_iso_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
        axes[0, 0].set_title("Isolated")
        axes[0, 0].set_ylabel("Frequency [Hz]")
        axes[0, 0].text(0.01, 0.98, "(a)", transform=axes[0, 0].transAxes, va="top", ha="left", fontweight="bold")

        im_b = _plot_cwt_panel(axes[0, 1], time_s, freqs, S_losa_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
        axes[0, 1].set_title(r"LOSA ($\Delta\phi=" + f"{args.delta_phi}" + r"$ rad)")
        axes[0, 1].set_ylabel("Frequency [Hz]")
        axes[0, 1].text(0.01, 0.98, "(b)", transform=axes[0, 1].transAxes, va="top", ha="left", fontweight="bold")

        im_c = _plot_cwt_panel(axes[1, 0], time_s, freqs, dS_smooth, cmap_cw, -dlim, dlim, fmin=fmin, fmax=fmax)
        axes[1, 0].set_title(r"$\Delta S$ (log-power)")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("Frequency [Hz]")
        axes[1, 0].text(0.01, 0.98, "(c)", transform=axes[1, 0].transAxes, va="top", ha="left", fontweight="bold")

        rlim = float(np.nanpercentile(np.abs(resid_plot[np.isfinite(resid_plot)]), 99)) if np.any(np.isfinite(resid_plot)) else 1e-6
        rlim = max(rlim, 1e-6)
        cmap_rdbu = plt.get_cmap("RdBu_r").copy()
        cmap_rdbu.set_bad(color=(0.9, 0.9, 0.9, 1.0))
        # Resid is model resolution (8 bins); use matching freq grid from pywt
        cwt_resid = cfg.get("preprocessing", {}).get("cwt", {})
        H_resid = resid_plot.shape[0]
        freqs_resid = get_cwt_display_freqs(
            str(cwt_resid.get("wavelet", "morl")),
            float(cwt_resid.get("fmin", 10)),
            float(cwt_resid.get("fmax", 64)),
            H_resid,
            fs_down,
        )
        im_d = _plot_cwt_panel(axes[1, 1], time_s, freqs_resid, resid_plot, cmap_rdbu, -rlim, rlim, fmin=10, fmax=64)
        axes[1, 1].set_title("AE residual (LOSA)")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("Frequency [Hz]")
        axes[1, 1].text(0.01, 0.98, "(d)", transform=axes[1, 1].transAxes, va="top", ha="left", fontweight="bold")

        fig.colorbar(im_a, ax=[axes[0, 0], axes[0, 1]], shrink=0.9, pad=0.02, label=r"$\log_{10}|W|^2$ (rel. peak)")
        fig.colorbar(im_c, ax=axes[1, 0], shrink=0.9, pad=0.02, label=r"$\Delta S$")
        fig.colorbar(im_d, ax=axes[1, 1], shrink=0.9, pad=0.02, label="Residual")

        plt.savefig(outdir / "phase0_cwt_quad.png", dpi=300, bbox_inches="tight")
        plt.close()
        plt.imsave(outdir / "phase0_cwt_resid.png", resid, cmap="RdBu_r", origin="lower")
    else:
        # Same layout as triptych (3 rows with ridge shift)
        fig = plt.figure(constrained_layout=True, figsize=(12, 5.5))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[1, :], sharex=ax1)
        ax4 = fig.add_subplot(gs[2, :], sharex=ax1)
        im1 = _plot_cwt_panel(ax1, time_s, freqs, S_iso_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
        ax1.set_title("Isolated")
        ax1.set_ylabel("Frequency [Hz]")
        ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes, va="top", ha="left", fontweight="bold")
        im2 = _plot_cwt_panel(ax2, time_s, freqs, S_losa_rel, cmap_vir, vmin, vmax, ridge=None, fmin=fmin, fmax=fmax, chirp_end_hz=chirp_end_hz)
        ax2.set_title(r"LOSA ($\Delta\phi=" + f"{args.delta_phi}" + r"$ rad)")
        ax2.set_ylabel("Frequency [Hz]")
        ax2.text(0.01, 0.98, "(b)", transform=ax2.transAxes, va="top", ha="left", fontweight="bold")
        im3 = _plot_cwt_panel(ax3, time_s, freqs, dS_smooth, cmap_cw, -dlim, dlim, fmin=fmin, fmax=fmax)
        ax3.set_title(r"$\Delta S = S_{\mathrm{LOSA}} - S_{\mathrm{iso}}$ (smoothed)")
        ax3.set_xlabel("")
        ax3.set_ylabel("Frequency [Hz]")
        ax3.text(0.01, 0.98, "(c)", transform=ax3.transAxes, va="top", ha="left", fontweight="bold")
        ax4.plot(time_s, dt_true * 1e6, "k-", lw=1)
        ax4.axhline(0, color="gray", lw=0.8, linestyle="--")
        ax4.set_ylabel(r"$\Delta t$ [µs]")
        ax4.set_xlabel("Time [s]")
        ax4.text(0.01, 0.98, "(d)", transform=ax4.transAxes, va="top", ha="left", fontweight="bold")
        fig.colorbar(im1, ax=[ax1, ax2], shrink=0.9, pad=0.02, label=r"$\log_{10}|W|^2$ (rel. peak)")
        fig.colorbar(im3, ax=ax3, shrink=0.9, pad=0.02, label=r"$\Delta S$")
        plt.savefig(outdir / "phase0_cwt_quad.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved figures to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

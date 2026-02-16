#!/usr/bin/env python3
"""
Diagnose Δφ=10 turn-down: separate shape change from time shift + boundary effects.

Noise-free, same fixed chirp, same preprocessing. Sweeps [1, 3, 5, 7, 10, 12].
Reports: dt_max, dt_end, tau*, rel_L2 raw vs aligned, mask_fraction.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import C_LIGHT, accel_from_delta_phi, apply_losa_constant_accel
from preprocessing import CWTPreprocessor
from utils.io import load_yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

DELTA_PHI_GRID = [1.0, 3.0, 5.0, 7.0, 10.0, 12.0]
DEFAULT_CONFIG = ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml"


def compute_losa_dt(a_los: float, T: int, fs: float, t0: float = 0.0) -> np.ndarray:
    """Δ t(t) = 0.5 * (a/c) * (t - t0)^2. Returns dt for each sample (seconds)."""
    t = np.arange(T, dtype=np.float64) / fs
    dt = 0.5 * (a_los / C_LIGHT) * (t - t0) ** 2
    return dt


def _build_preprocessor(cfg: dict, return_before_norm: bool = False) -> CWTPreprocessor:
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    return CWTPreprocessor(
        sample_rate=int(cwt_cfg.get("sample_rate", 1024)),
        target_height=int(cwt_cfg.get("target_height", 8)),
        target_width=cwt_cfg.get("target_width", 4096),
        fmin=float(cwt_cfg.get("fmin", 20.0)),
        fmax=float(cwt_cfg.get("fmax", 512.0)),
        wavelet=cwt_cfg.get("wavelet", "morl"),
        downsample_factor=int(cwt_cfg.get("downsample_factor", 1)),
        cwt_norm_mean=None,
        cwt_norm_std=None,
        return_before_norm=return_before_norm,
        use_complex=cwt_cfg.get("use_complex", False),
    )


def compute_metrics(h_iso: np.ndarray, h_losa: np.ndarray, preprocessor_x, preprocessor_s):
    """rel_L2, mean_abs_delta (masked), mean_abs_delta (no mask), mask_fraction."""
    x_iso = preprocessor_x.process(h_iso)
    x_losa = preprocessor_x.process(h_losa)
    if x_iso is None or x_losa is None:
        raise RuntimeError("CWT returned None")
    x_iso = np.asarray(x_iso).ravel()
    x_losa = np.asarray(x_losa).ravel()
    norm_iso = np.linalg.norm(x_iso) + 1e-12
    rel_L2 = float(np.linalg.norm(x_losa - x_iso) / norm_iso)

    s_iso = preprocessor_s.process(h_iso)
    s_losa = preprocessor_s.process(h_losa)
    if s_iso is None or s_losa is None:
        raise RuntimeError("CWT (before norm) returned None")
    s_iso = np.asarray(s_iso).squeeze()
    s_losa = np.asarray(s_losa).squeeze()
    mask = s_iso > (s_iso.max() - 3.0)
    mask_fraction = float(np.mean(mask))
    if mask.sum() == 0:
        mean_abs_delta_masked = 0.0
    else:
        mean_abs_delta_masked = float(np.mean(np.abs(s_losa[mask] - s_iso[mask])))
    mean_abs_delta_nomask = float(np.mean(np.abs(s_losa - s_iso)))
    return rel_L2, mean_abs_delta_masked, mean_abs_delta_nomask, mask_fraction


def estimate_tau_crosscorr(h_iso: np.ndarray, h_losa: np.ndarray, fs: float) -> float:
    """Best global shift (seconds) via cross-correlation. Positive tau = h_losa delayed."""
    cc = correlate(h_iso, h_losa, mode="full")
    lags = correlation_lags(len(h_iso), len(h_losa), mode="full")
    peak_idx = np.argmax(cc)
    lag_samples = int(lags[peak_idx])
    return lag_samples / fs


def time_shift_signal(h: np.ndarray, tau: float, fs: float) -> np.ndarray:
    """Shift h by -tau seconds (advance in time). Uses interpolation."""
    T = len(h)
    t_orig = np.arange(T, dtype=np.float64) / fs
    t_query = t_orig - tau
    return np.interp(t_query, t_orig, h.astype(np.float64), left=0.0, right=0.0).astype(
        np.float32
    )


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

    preprocessor_x = _build_preprocessor(cfg, return_before_norm=False)
    preprocessor_s = _build_preprocessor(cfg, return_before_norm=True)

    results = []
    print("Δφ   | dt_max(s) | dt_end(s) | dt_end_frac | tau*(s)   | rel_L2_raw  | rel_L2_aligned | mean_abs_masked | mean_abs_nomask | mask_frac")
    print("-" * 130)

    for delta_phi in DELTA_PHI_GRID:
        a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)
        h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=fs, a_los=a_los)
        h_losa = h_losa.astype(np.float32)

        # A) LOSA time shift
        dt_arr = compute_losa_dt(a_los, T, fs)
        dt_max = float(np.max(np.abs(dt_arr)))
        dt_end = float(np.abs(dt_arr[-1]))
        dt_end_frac = dt_end / duration

        # B) Time alignment
        tau_star = estimate_tau_crosscorr(h_iso, h_losa, fs)
        h_losa_aligned = time_shift_signal(h_losa, tau_star, fs)

        # Metrics raw
        rel_L2_raw, mean_abs_masked, mean_abs_nomask, mask_fraction = compute_metrics(
            h_iso, h_losa, preprocessor_x, preprocessor_s
        )

        # Metrics aligned
        rel_L2_aligned, _, _, _ = compute_metrics(
            h_iso, h_losa_aligned, preprocessor_x, preprocessor_s
        )

        results.append({
            "delta_phi": delta_phi,
            "dt_max": dt_max,
            "dt_end": dt_end,
            "dt_end_frac": dt_end_frac,
            "tau_star": tau_star,
            "rel_L2_raw": rel_L2_raw,
            "rel_L2_aligned": rel_L2_aligned,
            "mean_abs_delta_masked": mean_abs_masked,
            "mean_abs_delta_nomask": mean_abs_nomask,
            "mask_fraction": mask_fraction,
        })
        print(f"{delta_phi:4.1f} | {dt_max:9.4e} | {dt_end:9.4e} | {dt_end_frac:11.4e} | {tau_star:9.4e} | {rel_L2_raw:11.6e} | {rel_L2_aligned:14.6e} | {mean_abs_masked:15.6e} | {mean_abs_nomask:14.6e} | {mask_fraction:8.4f}")

    out_path = ROOT / "experiments/outputs_corrected/observability_debug_large_delta_phi.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": str(DEFAULT_CONFIG.name), "delta_phi_grid": DELTA_PHI_GRID, "results": results}, f, indent=2)
    print()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

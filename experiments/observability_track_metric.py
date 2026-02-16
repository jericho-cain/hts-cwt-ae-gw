#!/usr/bin/env python3
"""
Chirp-track observability metric: measure change on signal support, not full TF image.

Uses ridge extraction + narrow band around chirp to avoid L2 cancellation from
alternating lobes. Goal: D_track monotonic with Δφ.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import accel_from_delta_phi, apply_losa_constant_accel
from preprocessing import CWTPreprocessor
from utils.io import load_yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

DELTA_PHI_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 10.0, 12.0]
DEFAULT_CONFIG = ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml"
RIDGE_BAND_K = 2  # ±k frequency bins around ridge
GAUSSIAN_SIGMA = 1.0  # smoothing for ridge
COLUMN_ENERGY_THRESH = 0.01  # gate: include col if max_f P > thresh * P.max()


def _build_preprocessor(cfg: dict, return_before_norm: bool = False) -> CWTPreprocessor:
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    return CWTPreprocessor(
        sample_rate=int(cwt_cfg.get("sample_rate", 1024)),
        target_height=int(cwt_cfg.get("target_height", 8)),
        target_width=int(cwt_cfg.get("target_width", 4096)),
        fmin=float(cwt_cfg.get("fmin", 20.0)),
        fmax=float(cwt_cfg.get("fmax", 512.0)),
        wavelet=cwt_cfg.get("wavelet", "morl"),
        downsample_factor=int(cwt_cfg.get("downsample_factor", 1)),
        cwt_norm_mean=None,
        cwt_norm_std=None,
        return_before_norm=return_before_norm,
        use_complex=cwt_cfg.get("use_complex", False),
    )


def extract_ridge_and_gate(
    P: np.ndarray,
    sigma: float = GAUSSIAN_SIGMA,
    thresh: float = COLUMN_ENERGY_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    P: (n_freq, n_time) power map.
    Returns (ridge_idx, gate) where ridge_idx[t]=argmax_f P(f,t), gate[t]=1 if col above thresh.
    """
    P_smooth = gaussian_filter(P.astype(np.float64), sigma=(sigma, sigma), mode="nearest")
    ridge_idx = np.argmax(P_smooth, axis=0).astype(np.int32)  # (n_time,)
    col_max = np.max(P_smooth, axis=0)
    gate = (col_max >= thresh * P_smooth.max()).astype(np.float64)
    return ridge_idx, gate


def band_indices(ridge_idx: np.ndarray, n_freq: int, k: int) -> np.ndarray:
    """
    For each t, return indices [ridge(t)-k, ridge(t)+k] clipped.
    Returns (n_time, 2*k+1) array of frequency indices.
    """
    n_time = ridge_idx.shape[0]
    band = np.zeros((n_time, 2 * k + 1), dtype=np.int32)
    for t in range(n_time):
        r = ridge_idx[t]
        band[t] = np.clip(np.arange(r - k, r + k + 1), 0, n_freq - 1)
    return band


def compute_D_track(
    P_iso: np.ndarray,
    P_losa: np.ndarray,
    ridge_idx: np.ndarray,
    gate: np.ndarray,
    k: int,
    eps: float = 1e-12,
) -> float:
    """
    D_track = (1/T_gate) sum_t [ sum_f in band |P_losa-P_iso| / (sum_f P_iso + eps) ]
    summed only over gated t.
    """
    n_freq, n_time = P_iso.shape
    band = band_indices(ridge_idx, n_freq, k)
    numer = 0.0
    denom = 0.0
    n_gate = 0
    for t in range(n_time):
        if gate[t] < 0.5:
            continue
        n_gate += 1
        f_indices = band[t]
        p_iso_band = P_iso[f_indices, t]
        p_losa_band = P_losa[f_indices, t]
        sum_abs_diff = np.sum(np.abs(p_losa_band - p_iso_band))
        sum_iso = np.sum(p_iso_band) + eps
        numer += sum_abs_diff / sum_iso
    if n_gate == 0:
        return 0.0
    return float(numer / n_gate)


def compute_D_track_S(
    S_iso: np.ndarray,
    S_losa: np.ndarray,
    ridge_idx: np.ndarray,
    gate: np.ndarray,
    k: int,
) -> float:
    """
    D_track_S = (1/T_gate) sum_t (1/|band|) sum_f in band |S_losa - S_iso|
    """
    n_freq, n_time = S_iso.shape
    band = band_indices(ridge_idx, n_freq, k)
    total = 0.0
    n_gate = 0
    band_size = 2 * k + 1
    for t in range(n_time):
        if gate[t] < 0.5:
            continue
        n_gate += 1
        f_indices = band[t]
        s_diff = np.mean(np.abs(S_losa[f_indices, t] - S_iso[f_indices, t]))
        total += s_diff
    if n_gate == 0:
        return 0.0
    return float(total / n_gate)


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

    # Get S_iso once for ridge extraction
    s_iso = preprocessor_s.process(h_iso)
    if s_iso is None:
        raise RuntimeError("CWT returned None")
    s_iso = np.asarray(s_iso).squeeze()
    P_iso = np.power(10.0, s_iso)

    ridge_idx, gate = extract_ridge_and_gate(P_iso)
    n_gated = int(gate.sum())

    results = []
    print("Δφ   | rel_L2_raw  | D_track     | D_track_S   | n_gated")
    print("-" * 60)

    for delta_phi in DELTA_PHI_GRID:
        if delta_phi == 0:
            h_losa = h_iso.copy()
        else:
            a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)
            h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=fs, a_los=a_los)
            h_losa = h_losa.astype(np.float32)

        # rel_L2_raw
        x_iso = preprocessor_x.process(h_iso)
        x_losa = preprocessor_x.process(h_losa)
        if x_iso is None or x_losa is None:
            raise RuntimeError("CWT returned None")
        x_iso = np.asarray(x_iso).ravel()
        x_losa = np.asarray(x_losa).ravel()
        rel_L2_raw = float(np.linalg.norm(x_losa - x_iso) / (np.linalg.norm(x_iso) + 1e-12))

        # S_losa, P_losa
        s_losa = preprocessor_s.process(h_losa)
        if s_losa is None:
            raise RuntimeError("CWT returned None")
        s_losa = np.asarray(s_losa).squeeze()
        P_losa = np.power(10.0, s_losa)

        D_track = compute_D_track(P_iso, P_losa, ridge_idx, gate, RIDGE_BAND_K)
        D_track_S = compute_D_track_S(s_iso, s_losa, ridge_idx, gate, RIDGE_BAND_K)

        results.append({
            "delta_phi": delta_phi,
            "rel_L2_raw": rel_L2_raw,
            "D_track": D_track,
            "D_track_S": D_track_S,
            "n_gated": n_gated,
        })
        print(f"{delta_phi:4.1f} | {rel_L2_raw:11.6e} | {D_track:11.6e} | {D_track_S:11.6e} | {n_gated:7d}")

    out_path = ROOT / "experiments/outputs_corrected/observability_track_metric.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": str(DEFAULT_CONFIG.name),
            "delta_phi_grid": DELTA_PHI_GRID,
            "ridge_band_k": RIDGE_BAND_K,
            "results": results,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")

    # Monotonicity check
    D_tracks = [r["D_track"] for r in results]
    D_track_Ss = [r["D_track_S"] for r in results]
    mono_track = all(D_tracks[i] <= D_tracks[i + 1] for i in range(len(D_tracks) - 1))
    mono_track_S = all(D_track_Ss[i] <= D_track_Ss[i + 1] for i in range(len(D_track_Ss) - 1))
    print()
    if mono_track:
        print("✓ D_track increases monotonically with Δφ.")
    else:
        print("⚠ D_track does NOT increase monotonically.")
    if mono_track_S:
        print("✓ D_track_S increases monotonically with Δφ.")
    else:
        print("⚠ D_track_S does NOT increase monotonically.")


if __name__ == "__main__":
    main()

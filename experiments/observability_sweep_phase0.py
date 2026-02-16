#!/usr/bin/env python3
"""
Observability sweep: does LOSA produce a monotonic, measurable change in the CWT?

Pure representation test — no model, no training, no normalization fitting.
For a single fixed chirp (noise-free), sweep Δφ and measure CWT change.

Answers: Is there even signal in the representation?
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import accel_from_delta_phi, apply_losa_constant_accel
from preprocessing import CWTPreprocessor
from utils.io import load_yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DELTA_PHI_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
DEFAULT_CONFIG = ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml"


def _build_preprocessor(cfg: dict, return_before_norm: bool = False) -> CWTPreprocessor:
    """Build CWTPreprocessor from config (same as run_experiment)."""
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


def main():
    cfg = load_yaml(DEFAULT_CONFIG)
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    T = int(data_cfg.get("T", 4096))
    fs = float(data_cfg.get("sample_rate", 1024))
    snr = float(data_cfg.get("snr", 5.0))
    seed = int(cfg.get("experiment", {}).get("seed", 42))

    p0 = cfg.get("phase0_losa", {})
    f_star = float(p0.get("f_star_hz", 40.0))
    duration = T / fs

    # Fixed chirp params (tight config)
    syn = cfg.get("synthetic", {})
    f_start = float(syn.get("chirp_f_start", 12.0))
    f_end = float(syn.get("chirp_f_end", 65.0))
    t_peak = float(syn.get("chirp_t_peak", 0.55))
    sigma = float(syn.get("chirp_sigma", 0.10))
    amplitude = float(syn.get("signal_amplitude", 1e-20))

    # Generate base chirp (noise-free: no noise added)
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

    # Scale to same nominal amplitude as training (SNR relative to noise_sigma)
    noise_sigma = 1e-21  # reference; we add zero noise
    h_norm = np.sqrt(np.mean(h_iso**2)) + 1e-30
    scale = (snr * noise_sigma) / h_norm
    h_iso = (h_iso * scale).astype(np.float32)

    preprocessor_x = _build_preprocessor(cfg, return_before_norm=False)
    preprocessor_s = _build_preprocessor(cfg, return_before_norm=True)

    results = []
    print("Δφ   | rel_L2     | mean_abs_delta")
    print("-" * 40)

    for delta_phi in DELTA_PHI_GRID:
        if delta_phi == 0:
            a_los = 0.0
            h_losa = h_iso.copy()
        else:
            a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)
            h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=fs, a_los=a_los)
            h_losa = h_losa.astype(np.float32)

        # Metric A: post z-score CWT tensors (same as model input)
        x_iso = preprocessor_x.process(h_iso)
        x_losa = preprocessor_x.process(h_losa)
        if x_iso is None or x_losa is None:
            raise RuntimeError("CWT preprocessing returned None")
        x_iso = np.asarray(x_iso).ravel()
        x_losa = np.asarray(x_losa).ravel()

        norm_iso = np.linalg.norm(x_iso) + 1e-12
        rel_L2 = float(np.linalg.norm(x_losa - x_iso) / norm_iso)

        # Metric B: log-power scalogram S before z-score
        s_iso = preprocessor_s.process(h_iso)
        s_losa = preprocessor_s.process(h_losa)
        if s_iso is None or s_losa is None:
            raise RuntimeError("CWT preprocessing (before norm) returned None")
        s_iso = np.asarray(s_iso).squeeze()
        s_losa = np.asarray(s_losa).squeeze()

        mask = s_iso > (s_iso.max() - 3.0)
        if mask.sum() == 0:
            mean_abs_delta = 0.0
        else:
            mean_abs_delta = float(np.mean(np.abs(s_losa[mask] - s_iso[mask])))

        results.append(
            {"delta_phi": delta_phi, "rel_L2": rel_L2, "mean_abs_delta": mean_abs_delta}
        )
        print(f"{delta_phi:4.1f} | {rel_L2:10.6e} | {mean_abs_delta:12.6e}")

    # Save JSON
    out_path = ROOT / "experiments/outputs_corrected/observability_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": str(DEFAULT_CONFIG.name),
                "noise_free": True,
                "delta_phi_grid": DELTA_PHI_GRID,
                "results": results,
            },
            f,
            indent=2,
        )
    print()
    print(f"Saved {out_path}")

    # Sanity: expect monotonic increase
    rel_L2s = [r["rel_L2"] for r in results]
    deltas = [r["mean_abs_delta"] for r in results]
    monotonic_rel = all(rel_L2s[i] <= rel_L2s[i + 1] for i in range(len(rel_L2s) - 1))
    monotonic_delta = all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))
    print()
    if monotonic_rel and monotonic_delta:
        print("✓ Both metrics increase monotonically with Δφ.")
    else:
        print("⚠ Metrics do NOT increase monotonically — check LOSA calib, representation, preprocessing.")


if __name__ == "__main__":
    main()

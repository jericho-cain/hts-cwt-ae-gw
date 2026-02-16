#!/usr/bin/env python3
"""
LOSA input-tensor diagnostic: does the CWT pipeline preserve LOSA-induced differences?

Generates one isolated and one LOSA (Δφ=10) sample from the SAME chirp, runs both
through the full CWT preprocessing pipeline, and reports:
  - ||X_losa - X_iso|| / ||X_iso|| (relative Frobenius)
  - max absolute difference
  - SSIM-like structural similarity (optional)

If these are tiny (<1-2%), the preprocessing is wiping out the LOSA effect.
"""

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import apply_losa_constant_accel
from data.synthetic.noise_models import gaussian_noise
from preprocessing import CWTPreprocessor
from utils.io import load_yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_losa_tensor_diagnostic(cfg: dict) -> dict:
    """
    Compare CWT tensors for isolated vs LOSA (Δφ=10) from the same chirp.

    Returns dict with rel_diff, max_abs_diff, and interpretation.
    """
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    T = int(data_cfg.get("T", 32768))
    fs = float(data_cfg.get("sample_rate", 1024))
    snr = float(data_cfg.get("snr", 5.0))
    noise_sigma = float(data_cfg.get("noise_sigma", 1e-21))
    p0 = cfg.get("phase0_losa", cfg.get("synthetic", {}))
    f_start_range = tuple(p0.get("chirp_f_start", [10.0, 15.0]))
    f_end_range = tuple(p0.get("chirp_f_end", [55.0, 70.0]))

    duration = T / fs
    delta_phi = 10.0
    f_star = float(cfg.get("phase0_losa", {}).get("f_star_hz", 40.0))
    from data.synthetic.losa import accel_from_delta_phi
    a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)

    # Same chirp params for both
    np.random.seed(12345)
    f0 = float(np.random.uniform(f_start_range[0], f_start_range[1]))
    f1 = float(np.random.uniform(f_end_range[0], f_end_range[1]))
    t_peak = 0.5
    sigma_env = 0.1

    # Isolated: chirp + noise
    h_iso = generate_isolated_chirp(
        T=T, sample_rate=fs,
        f_start=f0, f_end=f1,
        t_peak=t_peak, sigma=sigma_env,
        amplitude=1e-20, seed=42,
    )
    nse = gaussian_noise(T, sigma=noise_sigma, seed=999)
    h_norm = np.sqrt(np.mean(h_iso**2)) + 1e-30
    scale = (snr * noise_sigma) / h_norm
    x_iso = (h_iso * scale + nse).astype(np.float32)

    # LOSA: same chirp with LOSA, same noise
    h_losa = generate_isolated_chirp(
        T=T, sample_rate=fs,
        f_start=f0, f_end=f1,
        t_peak=t_peak, sigma=sigma_env,
        amplitude=1e-20, seed=42,
    )
    h_losa = apply_losa_constant_accel(h_losa, sample_rate=fs, a_los=a_los)
    h_norm_l = np.sqrt(np.mean(h_losa**2)) + 1e-30
    scale_l = (snr * noise_sigma) / h_norm_l
    x_losa = (h_losa * scale_l + nse).astype(np.float32)

    # Preprocess both through full pipeline
    preprocessor = CWTPreprocessor(
        sample_rate=int(cwt_cfg.get("sample_rate", 1024)),
        target_height=int(cwt_cfg.get("target_height", 8)),
        target_width=cwt_cfg.get("target_width"),
        fmin=float(cwt_cfg.get("fmin", 10.0)),
        fmax=float(cwt_cfg.get("fmax", 64.0)),
        wavelet=cwt_cfg.get("wavelet", "morl"),
        downsample_factor=int(cwt_cfg.get("downsample_factor", 8)),
    )

    X_iso = preprocessor.process(x_iso)
    X_losa = preprocessor.process(x_losa)

    if X_iso is None or X_losa is None:
        return {"error": "CWT preprocessing returned None"}

    X_iso = np.asarray(X_iso, dtype=np.float64).ravel()
    X_losa = np.asarray(X_losa, dtype=np.float64).ravel()

    diff = X_losa - X_iso
    nrm_iso = np.sqrt(np.sum(X_iso**2)) + 1e-15
    rel_diff = np.sqrt(np.sum(diff**2)) / nrm_iso
    max_abs_diff = float(np.max(np.abs(diff)))

    result = {
        "relative_diff": float(rel_diff),
        "max_abs_diff": max_abs_diff,
        "delta_phi": delta_phi,
        "X_iso_norm": float(nrm_iso),
        "interpretation": (
            "Pipeline likely preserves LOSA (rel_diff >= 1%)"
            if rel_diff >= 0.01
            else "Pipeline may be wiping out LOSA effect (rel_diff < 1%)"
        ),
    }
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LOSA CWT tensor diagnostic")
    parser.add_argument("--config", type=str, default="experiments/configs/ground_phase0_real.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_yaml(config_path)

    result = run_losa_tensor_diagnostic(cfg)
    if "error" in result:
        logger.error(result["error"])
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("LOSA tensor diagnostic (isolated vs Δφ=10)")
    logger.info("=" * 50)
    logger.info(f"  ||X_losa - X_iso|| / ||X_iso|| = {result['relative_diff']:.4f} ({result['relative_diff']*100:.2f}%)")
    logger.info(f"  max |X_losa - X_iso| = {result['max_abs_diff']:.6f}")
    logger.info(f"  → {result['interpretation']}")
    logger.info("=" * 50)

    save_dir = Path(cfg.get("experiment", {}).get("save_dir", "experiments/outputs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    from utils.io import save_json
    save_json(result, save_dir / "losa_tensor_diagnostic.json")
    logger.info(f"Saved to {save_dir / 'losa_tensor_diagnostic.json'}")


if __name__ == "__main__":
    main()

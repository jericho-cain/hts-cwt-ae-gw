#!/usr/bin/env python3
"""
Smoke-test experiment runner for HTS-CWT-AE-GW.

Runs end-to-end: synthetic data generation -> CWT preprocessing ->
model instantiation -> 1 epoch training. No real detector data.

Usage:
    python -m experiments.run_experiment --config experiments/configs/ground_baseline.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root and src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic import generate_isolated_chirp, gaussian_noise
from models.registry import build_model
from preprocessing import CWTPreprocessor
from utils.io import load_yaml
from utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(cfg: dict) -> np.ndarray:
    """Generate synthetic time series (chirp + noise)."""
    syn_cfg = cfg.get("synthetic", {})
    n_samples = int(syn_cfg.get("n_samples", 16))
    T = int(syn_cfg.get("T", 4096))
    sample_rate = float(syn_cfg.get("sample_rate", 1024))
    noise_sigma = float(syn_cfg.get("noise_sigma", 1e-21))
    signal_amp = float(syn_cfg.get("signal_amplitude", 1e-20))

    logger.info(f"Generating {n_samples} synthetic samples (T={T}, fs={sample_rate})")
    data = []
    for i in range(n_samples):
        sig = generate_isolated_chirp(
            T=T,
            sample_rate=float(sample_rate),
            amplitude=signal_amp,
            seed=42 + i,
        )
        noise = gaussian_noise(T, sigma=noise_sigma, seed=100 + i)
        data.append(sig + noise)
    return np.stack(data, axis=0).astype(np.float32)


def preprocess_cwt(cfg: dict, data: np.ndarray) -> np.ndarray:
    """Apply CWT preprocessing to time series."""
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    sample_rate = cwt_cfg.get("sample_rate", 1024)
    target_height = cwt_cfg.get("target_height", 8)
    target_width = cwt_cfg.get("target_width", 4096)
    fmin = cwt_cfg.get("fmin", 20.0)
    fmax = cwt_cfg.get("fmax", 512.0)
    wavelet = cwt_cfg.get("wavelet", "morl")
    downsample_factor = cwt_cfg.get("downsample_factor", 1)

    preprocessor = CWTPreprocessor(
        sample_rate=sample_rate,
        target_height=target_height,
        target_width=target_width,
        fmin=fmin,
        fmax=fmax,
        wavelet=wavelet,
        downsample_factor=downsample_factor,
    )

    cwt_list = []
    for i in range(data.shape[0]):
        cwt = preprocessor.process(data[i])
        if cwt is not None:
            cwt_list.append(cwt)
    cwt_arr = np.stack(cwt_list, axis=0)
    logger.info(f"CWT preprocessing done: shape={cwt_arr.shape}")
    return cwt_arr


def run_training(cfg: dict, cwt_data: np.ndarray) -> float:
    """Run 1 epoch of training and return average loss."""
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 4)
    lr = train_cfg.get("learning_rate", 0.001)
    loss_fn_name = train_cfg.get("loss_function", "mse")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model = model.to(device)

    if loss_fn_name == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Add channel dim: (N, H, W) -> (N, 1, H, W)
    x = torch.FloatTensor(cwt_data).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, (batch,) in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        x_hat, z = model(batch)
        loss = criterion(x_hat, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        logger.info(f"  Batch {batch_idx + 1}: loss={loss.item():.6f}")

    avg_loss = total_loss / n_batches if n_batches else 0.0
    logger.info(f"Epoch 1 complete. Average loss: {avg_loss:.6f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Run smoke-test experiment (synthetic data only)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/ground_baseline.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    set_seed(42)
    cfg = load_yaml(config_path)

    logger.info("=" * 60)
    logger.info("HTS-CWT-AE-GW Smoke Test (synthetic data only)")
    logger.info("=" * 60)

    data = generate_synthetic_data(cfg)
    cwt_data = preprocess_cwt(cfg, data)
    loss = run_training(cfg, cwt_data)

    logger.info("=" * 60)
    logger.info(f"Smoke test complete. Final loss: {loss:.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

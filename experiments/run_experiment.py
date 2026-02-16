#!/usr/bin/env python3
"""
Smoke-test experiment runner for HTS-CWT-AE-GW.

Runs end-to-end: synthetic data generation -> CWT preprocessing ->
model instantiation -> 1 epoch training. No real detector data.

Phase 0 LOSA evaluation: when phase0_losa.enabled in config, runs
reconstruction error evaluation on isolated + LOSA bins and saves reports.

Usage:
    python -m experiments.run_experiment --config experiments/configs/ground_baseline.yaml
    python -m experiments.run_experiment --config experiments/configs/ground_phase0_losa.yaml
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
from data.synthetic.datasets import make_phase0_batch
from data.synthetic.losa import accel_from_delta_phi
from evaluation.phase0_reports import save_phase0_reports
from models.registry import build_model
from preprocessing import CWTPreprocessor
from utils.io import load_yaml, save_json
from utils.seed import set_seed

C_LIGHT = 299_792_458.0


def eps_to_accel(eps: float, duration_s: float) -> float:
    """Convert epsilon to acceleration for constant-accel LOSA: a = eps * c / T."""
    return float(eps) * C_LIGHT / duration_s


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(cfg: dict) -> np.ndarray:
    """Generate synthetic time series (chirp + noise). Smoke-test path: n_samples."""
    syn_cfg = cfg.get("synthetic", {})
    n_samples = int(syn_cfg.get("n_samples", 16))
    T = int(syn_cfg.get("T", 4096))
    sample_rate = float(syn_cfg.get("sample_rate", 1024))
    noise_sigma = float(syn_cfg.get("noise_sigma", 1e-21))
    signal_amp = float(syn_cfg.get("signal_amplitude", 1e-20))
    f_start_range = syn_cfg.get("chirp_f_start", [30.0, 50.0])
    f_end_range = syn_cfg.get("chirp_f_end", [150.0, 250.0])

    logger.info(f"Generating {n_samples} synthetic samples (T={T}, fs={sample_rate})")
    data = []
    rng = np.random.default_rng(42)
    for i in range(n_samples):
        f0 = float(rng.uniform(f_start_range[0], f_start_range[1]))
        f1 = float(rng.uniform(f_end_range[0], f_end_range[1]))
        sig = generate_isolated_chirp(
            T=T,
            sample_rate=float(sample_rate),
            f_start=f0,
            f_end=f1,
            amplitude=signal_amp,
            seed=42 + i,
        )
        noise = gaussian_noise(T, sigma=noise_sigma, seed=100 + i)
        data.append(sig + noise)
    return np.stack(data, axis=0).astype(np.float32)


def _get_chirp_ranges(cfg: dict) -> tuple[tuple, tuple, tuple, tuple]:
    """Get (f_start_range, f_end_range, t_peak_range, sigma_range) from config."""
    syn_cfg = cfg.get("synthetic", {})
    p0 = cfg.get("phase0_losa", syn_cfg)
    source = p0 if "chirp_f_start" in p0 else syn_cfg

    def to_tuple(v, default_lo: float, default_hi: float):
        if v is None:
            return (default_lo, default_hi)
        if isinstance(v, (int, float)):
            return (float(v), float(v))
        v = list(v) if isinstance(v, (list, tuple)) else [default_lo, default_hi]
        return (float(v[0]), float(v[1]) if len(v) > 1 else float(v[0]))

    if syn_cfg.get("tight_chirp", False):
        fs = to_tuple(syn_cfg.get("chirp_f_start", source.get("chirp_f_start", 12.0)), 12.0, 12.0)
        fe = to_tuple(syn_cfg.get("chirp_f_end", source.get("chirp_f_end", 65.0)), 65.0, 65.0)
        tp = to_tuple(syn_cfg.get("chirp_t_peak", 0.55), 0.55, 0.55)
        sig = to_tuple(syn_cfg.get("chirp_sigma", 0.10), 0.10, 0.10)
        jitter = float(syn_cfg.get("chirp_jitter_pct", 0.0))
        if jitter > 0:
            def jit(lo, hi):
                c = (lo + hi) / 2
                w = abs(c) * jitter
                return (c - w, c + w)
            fs, fe, tp, sig = jit(*fs), jit(*fe), jit(*tp), jit(*sig)
        return (fs, fe, tp, sig)
    f_start = tuple(source.get("chirp_f_start", [10.0, 15.0]))
    f_end = tuple(source.get("chirp_f_end", [55.0, 70.0]))
    return (f_start, f_end, (0.4, 0.7), (0.06, 0.14))


def generate_train_val_test(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate train/val/test isolated batches for real training."""
    syn_cfg = cfg.get("synthetic", {})
    data_cfg = cfg.get("data", syn_cfg)
    n_train = int(syn_cfg.get("n_train", 1000))
    n_val = int(syn_cfg.get("n_val", 200))
    n_test = int(syn_cfg.get("n_test", 200))
    T = int(data_cfg.get("T", 4096))
    fs = float(data_cfg.get("sample_rate", 1024))
    snr = float(data_cfg.get("snr", 5.0))
    noise_sigma = float(data_cfg.get("noise_sigma", 1e-21))
    seed = cfg.get("experiment", {}).get("seed", 42)

    f_start_range, f_end_range, t_peak_range, sigma_range = _get_chirp_ranges(cfg)
    if syn_cfg.get("tight_chirp", False):
        logger.info(f"Tight chirp: f_start={f_start_range}, f_end={f_end_range}, t_peak={t_peak_range}, sigma={sigma_range}")

    logger.info(f"Generating train={n_train}, val={n_val}, test={n_test} isolated samples")
    def _batch(n, s, a=0.0):
        return make_phase0_batch(
            n=n, T=T, sample_rate=fs, snr=snr, a_los=a,
            noise_sigma=noise_sigma, seed=s,
            f_start_range=f_start_range, f_end_range=f_end_range,
            t_peak_range=t_peak_range, sigma_range=sigma_range,
        )
    x_train, _ = _batch(n_train, seed)
    x_val, _ = _batch(n_val, seed + 10000)
    x_test, _ = _batch(n_test, seed + 20000)
    return x_train, x_val, x_test


def generate_phase0_eval_sets(cfg: dict) -> dict:
    """Build isolated + LOSA bin evaluation sets. Uses delta_phi_grid or eps_grid."""
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    T = int(data_cfg.get("T", 4096))
    fs = float(data_cfg.get("sample_rate", 1024))
    duration = T / fs
    snr = float(data_cfg.get("snr", 5.0))
    noise_sigma = float(data_cfg.get("noise_sigma", 1e-21))
    seed = cfg.get("experiment", {}).get("seed", 42)

    f_start_range, f_end_range, t_peak_range, sigma_range = _get_chirp_ranges(cfg)

    p0 = cfg["phase0_losa"]
    n = int(p0["n_per_bin"])
    f_star = float(p0.get("f_star_hz", 40.0))

    if "delta_phi_grid" in p0:
        dphi_grid = [float(x) for x in p0["delta_phi_grid"]]
        a_grid = [
            accel_from_delta_phi(dphi, duration, f_star_hz=f_star)
            for dphi in dphi_grid
        ]
        grid_label = "delta_phi"
        grid_vals = dphi_grid
    else:
        eps_grid = [float(e) for e in p0["eps_grid"]]
        a_grid = [eps_to_accel(eps, duration) for eps in eps_grid]
        grid_label = "eps"
        grid_vals = eps_grid

    sets = {}

    x_iso, _ = make_phase0_batch(
        n=n,
        T=T,
        sample_rate=fs,
        snr=snr,
        a_los=0.0,
        noise_sigma=noise_sigma,
        seed=seed + 1,
        f_start_range=f_start_range,
        f_end_range=f_end_range,
        t_peak_range=t_peak_range,
        sigma_range=sigma_range,
    )
    sets["isolated"] = {"data": x_iso, "eps": 0.0, "a": 0.0, "delta_phi": 0.0}

    for idx, (val, a) in enumerate(zip(grid_vals, a_grid)):
        x_losa, _ = make_phase0_batch(
            n=n,
            T=T,
            sample_rate=fs,
            snr=snr,
            a_los=a,
            noise_sigma=noise_sigma,
            seed=int(seed + val * 1e6),
            f_start_range=f_start_range,
            f_end_range=f_end_range,
            t_peak_range=t_peak_range,
            sigma_range=sigma_range,
        )
        key = f"{grid_label}_{val:.3g}" if isinstance(val, float) else f"{grid_label}_{val}"
        entry = {"data": x_losa, "a": a}
        if grid_label == "delta_phi":
            entry["delta_phi"] = val
            entry["eps"] = 0.0
        else:
            entry["eps"] = val
            entry["delta_phi"] = 0.0
        sets[key] = entry

    return sets


@torch.no_grad()
def compute_latents(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Extract latent vectors for each sample."""
    model.eval()
    latents = []
    for (x,) in dataloader:
        x = x.to(device)
        out = model(x)
        if isinstance(out, (tuple, list)):
            z = out[1]
        elif isinstance(out, dict):
            z = out["latent"]
        else:
            raise ValueError("Model must return (x_hat, latent)")
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def compute_recon_errors(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    return_metrics: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reconstruction error per sample.

    Returns r = |x - x_hat|^2 / |x|^2 (relative energy, dimensionless).
    If return_metrics=True, also returns (r, mse_elem, sse, n_elem) for JSON storage.
    """
    model.eval()
    all_r, all_mse, all_sse = [], [], []

    for (x,) in dataloader:
        x = x.to(device)
        out = model(x)

        if isinstance(out, (tuple, list)):
            x_hat = out[0]
        elif isinstance(out, dict):
            x_hat = out["x_hat"]
        else:
            x_hat = out

        diff = (x_hat - x).reshape(x.shape[0], -1)
        n_elem = diff.shape[1]
        sse = torch.sum(diff * diff, dim=1).cpu().numpy()
        denom = torch.sum((x.reshape(x.shape[0], -1)) ** 2, dim=1).cpu().numpy() + 1e-12
        r = sse / denom
        mse_elem = sse / n_elem
        all_r.append(r)
        all_mse.append(mse_elem)
        all_sse.append(sse)

    r_arr = np.concatenate(all_r, axis=0)
    if not return_metrics:
        return r_arr
    return (
        r_arr,
        np.concatenate(all_mse, axis=0),
        np.concatenate(all_sse, axis=0),
        n_elem,
    )


def fit_mahalanobis(latents: np.ndarray, reg: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Fit μ and Σ^{-1} on training latents. Returns (mu, Sigma_inv)."""
    mu = np.mean(latents, axis=0)
    cov = np.cov(latents, rowvar=False)
    d = cov.shape[0]
    Sigma_inv = np.linalg.inv(cov + reg * np.eye(d))
    return mu, Sigma_inv


def mahalanobis_scores(latents: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance (z - μ)^T Σ^{-1} (z - μ) per sample."""
    diff = latents - mu  # (n, d)
    return np.sum(diff @ Sigma_inv * diff, axis=1).astype(np.float64)


def _build_preprocessor(
    cfg: dict,
    cwt_norm_mean: float | None = None,
    cwt_norm_std: float | None = None,
    return_before_norm: bool = False,
) -> CWTPreprocessor:
    """Build CWTPreprocessor from config."""
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    return CWTPreprocessor(
        sample_rate=int(cwt_cfg.get("sample_rate", 1024)),
        target_height=int(cwt_cfg.get("target_height", 8)),
        target_width=cwt_cfg.get("target_width", 4096),
        fmin=float(cwt_cfg.get("fmin", 20.0)),
        fmax=float(cwt_cfg.get("fmax", 512.0)),
        wavelet=cwt_cfg.get("wavelet", "morl"),
        downsample_factor=int(cwt_cfg.get("downsample_factor", 1)),
        cwt_norm_mean=cwt_norm_mean,
        cwt_norm_std=cwt_norm_std,
        return_before_norm=return_before_norm,
        use_complex=cwt_cfg.get("use_complex", False),
    )


def fit_cwt_dataset_norm(cfg: dict, train_data: np.ndarray) -> tuple[float, float]:
    """Fit CWT normalization mean/std on training data (log-mag or Re/Im)."""
    preprocessor = _build_preprocessor(cfg, return_before_norm=True)
    collected = []
    for i in range(train_data.shape[0]):
        out = preprocessor.process(train_data[i])
        if out is not None:
            collected.append(out)
    if not collected:
        raise RuntimeError("No valid CWT outputs from training data")
    stacked = np.concatenate([a.ravel() for a in collected])
    mean_val = float(np.mean(stacked))
    std_val = float(np.std(stacked)) + 1e-10
    logger.info(f"Fitted CWT dataset norm: mean={mean_val:.6e}, std={std_val:.6e}")
    return mean_val, std_val


def preprocess_cwt(
    cfg: dict,
    data: np.ndarray,
    cwt_norm_mean: float | None = None,
    cwt_norm_std: float | None = None,
) -> np.ndarray:
    """Apply CWT preprocessing to time series."""
    preprocessor = _build_preprocessor(cfg, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std)
    cwt_list = []
    for i in range(data.shape[0]):
        cwt = preprocessor.process(data[i])
        if cwt is not None:
            cwt_list.append(cwt)
    cwt_arr = np.stack(cwt_list, axis=0)
    logger.info(f"CWT preprocessing done: shape={cwt_arr.shape}")
    return cwt_arr


def run_training(
    cfg: dict,
    cwt_train: np.ndarray,
    cwt_val: np.ndarray | None = None,
) -> tuple[list[float], list[float], nn.Module, torch.device]:
    """
    Run training. Returns (train_losses, val_losses, model, device).
    If cwt_val is None, runs single epoch (smoke path).
    """
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 4)
    lr = train_cfg.get("learning_rate", 0.001)
    num_epochs = train_cfg.get("num_epochs", 1)
    patience = train_cfg.get("early_stopping_patience", 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_train = torch.FloatTensor(cwt_train)
    if x_train.ndim == 3:
        x_train = x_train.unsqueeze(1)  # (N, 1, H, W) for magnitude
    train_loader = DataLoader(
        TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = None
    if cwt_val is not None and len(cwt_val) > 0:
        x_val = torch.FloatTensor(cwt_val)
        if x_val.ndim == 3:
            x_val = x_val.unsqueeze(1)
        val_loader = DataLoader(
            TensorDataset(x_val),
            batch_size=batch_size,
            shuffle=False,
        )

    train_losses = []
    val_losses = []
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (batch,) in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / n_batches if n_batches else 0.0
        train_losses.append(avg_train)

        val_loss = 0.0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    x_hat, _ = model(batch)
                    val_loss += criterion(x_hat, batch).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: train_loss={avg_train:.6f}, val_loss={val_loss:.6f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        else:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: train_loss={avg_train:.6f}")

    return train_losses, val_losses, model, device


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment (smoke or real training)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/ground_baseline.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Override experiment.save_dir from config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    seed = 42
    set_seed(seed)
    cfg = load_yaml(config_path)
    if args.save_dir is not None:
        cfg.setdefault("experiment", {})["save_dir"] = args.save_dir
        logger.info(f"Overrode save_dir -> {args.save_dir}")
    if "experiment" in cfg and "seed" in cfg["experiment"]:
        seed = int(cfg["experiment"]["seed"])
        set_seed(seed)

    use_real = "n_train" in cfg.get("synthetic", {})
    cwt_norm_mean, cwt_norm_std = None, None
    use_complex = cfg.get("preprocessing", {}).get("cwt", {}).get("use_complex", False)
    if use_complex:
        cfg.setdefault("model", {})["in_channels"] = 2

    if use_real:
        logger.info("=" * 60)
        logger.info("HTS-CWT-AE-GW Real Training (baseline fit + LOSA eval)")
        logger.info("=" * 60)
        from experiments.losa_tensor_diagnostic import run_losa_tensor_diagnostic
        diag = run_losa_tensor_diagnostic(cfg)
        if "error" not in diag:
            logger.info(f"LOSA tensor diagnostic: rel_diff={diag['relative_diff']:.4f} ({diag['interpretation']})")
            save_dir = Path(cfg.get("experiment", {}).get("save_dir", "experiments/outputs"))
            save_dir.mkdir(parents=True, exist_ok=True)
            save_json(diag, save_dir / "losa_tensor_diagnostic.json")

        x_train, x_val, x_test = generate_train_val_test(cfg)
        norm_mode = cfg.get("preprocessing", {}).get("cwt", {}).get("norm_mode", "per_sample")
        if norm_mode == "dataset":
            cwt_norm_mean, cwt_norm_std = fit_cwt_dataset_norm(cfg, x_train)
        cwt_train = preprocess_cwt(cfg, x_train, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std)
        cwt_val = preprocess_cwt(cfg, x_val, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std)
        cwt_test = preprocess_cwt(cfg, x_test, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std)
        train_losses, val_losses, model, device = run_training(
            cfg, cwt_train, cwt_val
        )
        final_train = train_losses[-1] if train_losses else 0.0
        final_val = val_losses[-1] if val_losses else 0.0
        logger.info(f"Training complete. Final train={final_train:.6f}, val={final_val:.6f}")

        # Baseline fit: recon error on train/val/test isolated
        train_cfg = cfg.get("training", {})
        batch_size = train_cfg.get("batch_size", 32)
        baseline_errs = {}
        for label, cwt_np in [("train", cwt_train), ("val", cwt_val), ("test", cwt_test)]:
            x = torch.from_numpy(cwt_np).float()
            if x.ndim == 3:
                x = x.unsqueeze(1)
            dl = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
            errs = compute_recon_errors(model, dl, device)
            baseline_errs[label] = errs
            mean_err, std_err = float(np.mean(errs)), float(np.std(errs))
            logger.info(f"  Baseline {label} recon error: mean={mean_err:.6f}, std={std_err:.6f}")

        # Save baseline metrics
        save_dir = Path(cfg.get("experiment", {}).get("save_dir", "experiments/outputs"))
        save_dir.mkdir(parents=True, exist_ok=True)
        baseline = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_recon_mean": float(np.mean(baseline_errs["train"])),
            "train_recon_std": float(np.std(baseline_errs["train"])),
            "val_recon_mean": float(np.mean(baseline_errs["val"])),
            "val_recon_std": float(np.std(baseline_errs["val"])),
            "test_recon_mean": float(np.mean(baseline_errs["test"])),
            "test_recon_std": float(np.std(baseline_errs["test"])),
        }
        save_json(baseline, save_dir / "baseline_fit.json")
        # Save model checkpoint with norms so eval scripts use the same preprocessing
        if cfg.get("experiment", {}).get("save_model", True):
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "cwt_norm_mean": cwt_norm_mean,
                "cwt_norm_std": cwt_norm_std,
            }
            torch.save(ckpt, save_dir / "model.pt")
            logger.info(f"Saved model and norms to {save_dir / 'model.pt'}")
        cwt_train_for_latent = cwt_train
    else:
        logger.info("=" * 60)
        logger.info("HTS-CWT-AE-GW Smoke Test (synthetic data only)")
        logger.info("=" * 60)
        data = generate_synthetic_data(cfg)
        cwt_data = preprocess_cwt(cfg, data)
        train_losses, val_losses, model, device = run_training(cfg, cwt_data, None)
        logger.info(f"Smoke test complete. Final loss: {train_losses[-1]:.6f}")
        cwt_train_for_latent = None

    if cfg.get("phase0_losa", {}).get("enabled", False):
        logger.info("Running Phase 0 LOSA evaluation...")
        eval_sets = generate_phase0_eval_sets(cfg)
        recon_stats = {}
        train_cfg = cfg.get("training", {})
        batch_size = train_cfg.get("batch_size", 32)

        mahal_mu, mahal_sinv = None, None
        if cwt_train_for_latent is not None:
            x_tr = torch.from_numpy(cwt_train_for_latent).float()
            if x_tr.ndim == 3:
                x_tr = x_tr.unsqueeze(1)
            dl_tr = DataLoader(TensorDataset(x_tr), batch_size=batch_size, shuffle=False)
            train_latents = compute_latents(model, dl_tr, device)
            mahal_mu, mahal_sinv = fit_mahalanobis(train_latents)
            logger.info(f"Fitted Mahalanobis on train latents: μ shape={mahal_mu.shape}")

        for name, entry in eval_sets.items():
            raw_np = entry["data"]
            cwt_np = preprocess_cwt(cfg, raw_np, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std)
            x = torch.from_numpy(cwt_np).float()
            if x.ndim == 3:
                x = x.unsqueeze(1)  # (N, 1, H, W) for magnitude
            x = x.to(device)
            ds = TensorDataset(x)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            r_arr, mse_elem, sse_arr, n_elem = compute_recon_errors(model, dl, device, return_metrics=True)
            entry_out = {
                "eps": entry.get("eps", 0),
                "delta_phi": entry.get("delta_phi", 0),
                "errs": r_arr,
                "mse_elem": mse_elem,
                "sse": sse_arr,
                "n_elem": n_elem,
            }
            if mahal_mu is not None:
                lats = compute_latents(model, dl, device)
                entry_out["latent_scores"] = mahalanobis_scores(lats, mahal_mu, mahal_sinv)
            recon_stats[name] = entry_out

        save_phase0_reports(cfg, recon_stats)
        logger.info(f"Phase 0 reports saved to {cfg['experiment']['save_dir']}")


if __name__ == "__main__":
    main()

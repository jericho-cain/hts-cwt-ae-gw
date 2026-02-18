#!/usr/bin/env python3
"""
Recon metrics: MSE_elem, SSE, r for H=8 vs H=32.

Prints the three metrics (unambiguous, cross-H comparable) and optionally
trains H=32 if no checkpoint exists.

  MSE_elem = (1/N) sum_i (x_i - x_hat_i)^2   [per-element, stable across shapes]
  SSE      = sum_i (x_i - x_hat_i)^2         [raw sum-squared error]
  r       = |x - x_hat|^2 / |x|^2            [relative energy, dimensionless = current "err"]

Usage:
  python experiments/check_recon_metrics_H8_vs_H32.py
  python experiments/check_recon_metrics_H8_vs_H32.py --h8-model path.pt --h32-model path.pt
  python experiments/check_recon_metrics_H8_vs_H32.py --train-h32-if-missing   # train H32 if no checkpoint
  # Omit --quick for fair comparison (H32 phase diagram didn't save model.pt; run a full H32 experiment first)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def summarize(name, x):
    """Print input-stat summary for sanity checks. For z-scored: std≈1, norm2/elem≈1."""
    x_np = x.detach().cpu().numpy()
    flat = x_np.reshape(x_np.shape[0], -1)
    n_elem_per_sample = flat.shape[1]
    norm2_per_elem = (flat**2).sum(axis=1).mean() / n_elem_per_sample
    print(
        name,
        "shape", x_np.shape,
        "mean", x_np.mean(),
        "std", x_np.std(),
        "min", x_np.min(),
        "max", x_np.max(),
        "norm2/elem", norm2_per_elem,
    )


@torch.no_grad()
def compute_recon_metrics(model, dataloader, device):
    """Return (mse_elem, sse, r) per sample as arrays."""
    model.eval()
    mse_list, sse_list, r_list = [], [], []
    for (x,) in dataloader:
        x = x.to(device)
        out = model(x)
        x_hat = out[0] if isinstance(out, (tuple, list)) else (out["x_hat"] if isinstance(out, dict) else out)
        assert x_hat.shape == x.shape, f"x_hat.shape={x_hat.shape} != x.shape={x.shape}"
        diff = (x_hat - x).reshape(x.shape[0], -1)
        n_elem = diff.shape[1]
        sse = torch.sum(diff * diff, dim=1).cpu().numpy()
        norm_x_sq = torch.sum((x.reshape(x.shape[0], -1)) ** 2, dim=1).cpu().numpy() + 1e-12
        mse_elem = sse / n_elem
        r = sse / norm_x_sq
        mse_list.append(mse_elem)
        sse_list.append(sse)
        r_list.append(r)
    return np.concatenate(mse_list), np.concatenate(sse_list), np.concatenate(r_list)


def main():
    parser = argparse.ArgumentParser(description="MSE_elem, SSE, r for H=8 vs H=32")
    parser.add_argument(
        "--h8-model",
        type=str,
        default="experiments/outputs_corrected/phase_diagram/very_tight/model.pt",
        help="H=8 checkpoint path",
    )
    parser.add_argument(
        "--h32-model",
        type=str,
        default="experiments/outputs_corrected/phase_diagram_H32/very_tight/model.pt",
        help="H=32 checkpoint path",
    )
    parser.add_argument("--train-h32-if-missing", action="store_true", help="Train H=32 if no checkpoint")
    parser.add_argument("--quick", action="store_true", help="Use quick config for H32 train")
    parser.add_argument("--n-eval", type=int, default=500, help="Eval samples per bin")
    args = parser.parse_args()

    h8_path = Path(args.h8_model) if not Path(args.h8_model).is_absolute() else Path(args.h8_model)
    h32_path = Path(args.h32_model) if not Path(args.h32_model).is_absolute() else Path(args.h32_model)
    if not h8_path.is_absolute():
        h8_path = ROOT / h8_path
    if not h32_path.is_absolute():
        h32_path = ROOT / h32_path

    if not h8_path.exists():
        logger.error(f"H=8 model not found: {h8_path}")
        return 1
    if not h32_path.exists() and args.train_h32_if_missing:
        logger.info("Training H=32 model (quick)...")
        from experiments.run_experiment import (
            generate_train_val_test,
            generate_phase0_eval_sets,
            run_training,
            preprocess_cwt,
            fit_cwt_dataset_norm,
            _build_preprocessor,
        )
        from models.registry import build_model
        from utils.seed import set_seed
        from utils.io import load_yaml

        set_seed(42)
        cfg = load_yaml(ROOT / "experiments/configs/ground_phase0_tight_chirp_H32.yaml")
        if args.quick:
            cfg["training"]["num_epochs"] = 2
            cfg["synthetic"]["n_train"] = 300
            cfg["synthetic"]["n_val"] = 50
            cfg["synthetic"]["n_test"] = 50
        x_train, x_val, _ = generate_train_val_test(cfg)
        cwt_norm_mean, cwt_norm_std = fit_cwt_dataset_norm(cfg, x_train)
        cwt_train = preprocess_cwt(cfg, x_train, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std)
        cwt_val = preprocess_cwt(cfg, x_val, cwt_norm_mean=cwt_norm_mean, cwt_norm_std=cwt_norm_std)
        train_losses, val_losses, model, device = run_training(cfg, cwt_train, cwt_val)
        h32_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "cwt_norm_mean": cwt_norm_mean,
            "cwt_norm_std": cwt_norm_std,
        }, h32_path)
        logger.info(f"Saved H=32 model and norms to {h32_path}")
    elif not h32_path.exists():
        logger.error(f"H=32 model not found: {h32_path}. Use --train-h32-if-missing to train.")
        return 1

    from experiments.run_experiment import (
        generate_phase0_eval_sets,
        preprocess_cwt,
        fit_cwt_dataset_norm,
        generate_train_val_test,
    )
    from models.registry import build_model
    from utils.io import load_yaml
    from utils.seed import set_seed

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_h8 = load_yaml(ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml")
    cfg_h32 = load_yaml(ROOT / "experiments/configs/ground_phase0_tight_chirp_H32.yaml")
    cfg_h8["phase0_losa"]["n_per_bin"] = args.n_eval
    cfg_h32["phase0_losa"]["n_per_bin"] = args.n_eval

    ckpt_h8 = torch.load(h8_path, map_location=device, weights_only=False)
    ckpt_h32 = torch.load(h32_path, map_location=device, weights_only=False)
    assert "model_state_dict" in ckpt_h8, "Checkpoint missing model_state_dict"
    assert "model_state_dict" in ckpt_h32, "Checkpoint missing model_state_dict"

    # Use norms from checkpoint if saved; else refit (backward compat for old checkpoints)
    if "cwt_norm_mean" in ckpt_h8 and ckpt_h8["cwt_norm_mean"] is not None:
        cwt_norm_h8 = (ckpt_h8["cwt_norm_mean"], ckpt_h8["cwt_norm_std"])
        logger.info("H=8: using norms from checkpoint")
    else:
        x_train_h8, _, _ = generate_train_val_test(cfg_h8)
        cwt_norm_h8 = fit_cwt_dataset_norm(cfg_h8, x_train_h8)
        logger.warning("H=8: checkpoint has no norms, refitted (may differ from training run)")
    if "cwt_norm_mean" in ckpt_h32 and ckpt_h32["cwt_norm_mean"] is not None:
        cwt_norm_h32 = (ckpt_h32["cwt_norm_mean"], ckpt_h32["cwt_norm_std"])
        logger.info("H=32: using norms from checkpoint")
    else:
        x_train_h32, _, _ = generate_train_val_test(cfg_h32)
        cwt_norm_h32 = fit_cwt_dataset_norm(cfg_h32, x_train_h32)
        logger.warning("H=32: checkpoint has no norms, refitted (may differ from training run)")

    cfg_loaded_h8 = ckpt_h8.get("config", cfg_h8)
    cfg_loaded_h32 = ckpt_h32.get("config", cfg_h32)
    model_h8 = build_model(cfg_loaded_h8)
    model_h32 = build_model(cfg_loaded_h32)
    model_h8.load_state_dict(ckpt_h8["model_state_dict"])
    model_h32.load_state_dict(ckpt_h32["model_state_dict"])
    model_h8 = model_h8.to(device).eval()
    model_h32 = model_h32.to(device).eval()

    eval_sets = generate_phase0_eval_sets(cfg_h8)
    results = {"H8": {}, "H32": {}}
    x_iso_h32, x_losa_h32 = None, None

    for name, entry in eval_sets.items():
        raw = entry["data"]
        delta_phi = entry.get("delta_phi", entry.get("eps", 0))

        cwt_h8 = preprocess_cwt(cfg_h8, raw, *cwt_norm_h8)
        cwt_h32 = preprocess_cwt(cfg_h32, raw, *cwt_norm_h32)
        x_h8 = torch.from_numpy(cwt_h8).float().unsqueeze(1)
        x_h32 = torch.from_numpy(cwt_h32).float().unsqueeze(1)

        # Sanity: input stats (for isolated only, to avoid spam)
        if name == "isolated":
            summarize("H8 x (isolated)", x_h8)
            summarize("H32 x (isolated)", x_h32)
        if name == "isolated":
            x_iso_h32 = x_h32
        elif x_losa_h32 is None and delta_phi > 0:
            x_losa_h32 = x_h32

        dl_h8 = DataLoader(TensorDataset(x_h8), batch_size=32, shuffle=False)
        dl_h32 = DataLoader(TensorDataset(x_h32), batch_size=32, shuffle=False)

        mse_h8, sse_h8, r_h8 = compute_recon_metrics(model_h8, dl_h8, device)
        mse_h32, sse_h32, r_h32 = compute_recon_metrics(model_h32, dl_h32, device)

        results["H8"][name] = {
            "delta_phi": delta_phi,
            "n_elem": 8 * 4096,
            "mse_elem_mean": float(np.mean(mse_h8)),
            "mse_elem_std": float(np.std(mse_h8)),
            "sse_mean": float(np.mean(sse_h8)),
            "r_mean": float(np.mean(r_h8)),
            "r_std": float(np.std(r_h8)),
        }
        results["H32"][name] = {
            "delta_phi": delta_phi,
            "n_elem": 32 * 4096,
            "mse_elem_mean": float(np.mean(mse_h32)),
            "mse_elem_std": float(np.std(mse_h32)),
            "sse_mean": float(np.mean(sse_h32)),
            "r_mean": float(np.mean(r_h32)),
            "r_std": float(np.std(r_h32)),
        }

    # LOSA delta: does LOSA change the model-input tensor?
    if x_iso_h32 is not None and x_losa_h32 is not None:
        iso_mean = x_iso_h32.float().mean(dim=0, keepdim=True)
        losa_mean = x_losa_h32.float().mean(dim=0, keepdim=True)
        delta = torch.norm(losa_mean - iso_mean) / (torch.norm(iso_mean) + 1e-12)
        print("relative delta in model-input space (mean LOSA vs mean isolated):", delta.item())

    print()
    print("=" * 70)
    print("RECON METRICS: H=8 vs H=32 (same eval data, same training budget)")
    print("=" * 70)
    print()
    print("MSE_elem = (1/N) sum(x - x_hat)^2  [per-element, cross-H comparable]")
    print("SSE      = sum(x - x_hat)^2        [raw sum-squared]")
    print("r        = |x - x_hat|^2 / |x|^2   [relative energy = current err]")
    print()

    for name in list(eval_sets.keys()):
        d8 = results["H8"][name]
        d32 = results["H32"][name]
        print(f"--- {name} (Δφ={d8['delta_phi']}) ---")
        print(f"  {'':10} {'H=8':>14} {'H=32':>14} {'H32/H8':>10}")
        print(f"  MSE_elem   {d8['mse_elem_mean']:>14.6e} {d32['mse_elem_mean']:>14.6e} {d32['mse_elem_mean']/d8['mse_elem_mean']:>10.3f}")
        print(f"  SSE        {d8['sse_mean']:>14.2f} {d32['sse_mean']:>14.2f} {d32['sse_mean']/d8['sse_mean']:>10.3f}")
        print(f"  r          {d8['r_mean']:>14.6f} {d32['r_mean']:>14.6f} {d32['r_mean']/d8['r_mean']:>10.3f}")
        print()

    out_path = ROOT / "experiments/outputs_corrected/recon_metrics_H8_vs_H32.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")

    print("Interpretation:")
    print("  - If MSE_elem(H32) < MSE_elem(H8): model truly reconstructs better per coefficient.")
    print("  - If r changes but MSE_elem similar: mostly representation/normalization.")
    print("  - r is dimensionless; comparable across H (given consistent z-score norm).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

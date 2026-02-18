#!/usr/bin/env python3
"""
Check input variance and recon-error scaling: H=8 vs H=32.

Verifies:
- Same normalization (dataset-level) in both
- var(X) of tensor fed to AE post-preprocessing
- Whether mean recon error shift (1.0 -> 0.5) is explained by input scaling

err = ||x_hat - x||^2 / (||x||^2 + eps)
If ||x||^2 scales with n_elements (H*W), then H=32 has 4x larger denom -> 4x smaller err.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from utils.io import load_yaml


def main():
    # Load configs
    cfg_h8 = load_yaml(ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml")
    cfg_h32 = load_yaml(ROOT / "experiments/configs/ground_phase0_tight_chirp_H32.yaml")

    # Generate same train data as experiments (fixed seed)
    from utils.seed import set_seed
    from experiments.run_experiment import generate_train_val_test
    set_seed(42)

    cfg_h8["synthetic"]["n_train"] = 500  # Subset for speed
    cfg_h8["synthetic"]["n_val"] = 100
    cfg_h8["synthetic"]["n_test"] = 100
    x_train, _, _ = generate_train_val_test(cfg_h8)

    # Preprocess both with dataset norm
    from experiments.run_experiment import fit_cwt_dataset_norm, preprocess_cwt

    print("=" * 60)
    print("H=8")
    print("=" * 60)
    mean_h8, std_h8 = fit_cwt_dataset_norm(cfg_h8, x_train)
    cwt_h8 = preprocess_cwt(cfg_h8, x_train, cwt_norm_mean=mean_h8, cwt_norm_std=std_h8)
    x_h8 = cwt_h8.astype(np.float32)
    n_h8 = x_h8.size
    var_h8 = float(np.var(x_h8))
    mean_x_h8 = float(np.mean(x_h8))
    std_x_h8 = float(np.std(x_h8))
    norm_sq_h8 = float(np.sum(x_h8 ** 2)) / x_h8.shape[0]  # per-sample avg ||x||^2
    print(f"  shape: {x_h8.shape}, n_elements/sample: {n_h8 // x_h8.shape[0]}")
    print(f"  post-norm: mean={mean_x_h8:.6e}, std={std_x_h8:.6e}, var={var_h8:.6e}")
    print(f"  avg ||x||^2 per sample: {norm_sq_h8:.2f} (expect ~n_elem if unit var)")

    print()
    print("=" * 60)
    print("H=32")
    print("=" * 60)
    mean_h32, std_h32 = fit_cwt_dataset_norm(cfg_h32, x_train)
    cwt_h32 = preprocess_cwt(cfg_h32, x_train, cwt_norm_mean=mean_h32, cwt_norm_std=std_h32)
    x_h32 = cwt_h32.astype(np.float32)
    n_h32 = x_h32.size
    var_h32 = float(np.var(x_h32))
    mean_x_h32 = float(np.mean(x_h32))
    std_x_h32 = float(np.std(x_h32))
    norm_sq_h32 = float(np.sum(x_h32 ** 2)) / x_h32.shape[0]
    print(f"  shape: {x_h32.shape}, n_elements/sample: {n_h32 // x_h32.shape[0]}")
    print(f"  post-norm: mean={mean_x_h32:.6e}, std={std_x_h32:.6e}, var={var_h32:.6e}")
    print(f"  avg ||x||^2 per sample: {norm_sq_h32:.2f}")

    print()
    print("=" * 60)
    print("Recon error denominator scaling")
    print("=" * 60)
    ratio_elem = (32 * 4096) / (8 * 4096)
    ratio_norm_sq = norm_sq_h32 / norm_sq_h8
    print(f"  n_elements ratio (H32/H8): {ratio_elem:.1f}")
    print(f"  ||x||^2 ratio (H32/H8): {ratio_norm_sq:.2f}")
    print(f"  => err_H32 / err_H8 ≈ {1/ratio_norm_sq:.2f} (err scales ~1/||x||^2)")
    print()
    print("  If err_H8 ≈ 1.0, expected err_H32 ≈ {:.2f}".format(1.0 / ratio_norm_sq))
    print("  (Observed: mean_err H8 ~1.0, H32 ~0.5)")
    print("  Conclusion: {} explain recon error shift".format(
        "Denom scaling DOES" if 0.3 <= 1/ratio_norm_sq <= 0.7 else "Denom scaling does NOT fully"
    ))

    print()
    print("=" * 60)
    print("Pre-norm (log-mag) stats")
    print("=" * 60)
    import experiments.run_experiment as re
    pp_h8 = re._build_preprocessor(cfg_h8, return_before_norm=True)
    pp_h32 = re._build_preprocessor(cfg_h32, return_before_norm=True)
    raw_h8 = []
    raw_h32 = []
    for i in range(min(200, x_train.shape[0])):
        o8 = pp_h8.process(x_train[i])
        o32 = pp_h32.process(x_train[i])
        if o8 is not None:
            raw_h8.append(o8.ravel())
        if o32 is not None:
            raw_h32.append(o32.ravel())
    r8 = np.concatenate(raw_h8)
    r32 = np.concatenate(raw_h32)
    print(f"  H=8  pre-norm: mean={np.mean(r8):.4e}, std={np.std(r8):.4e}")
    print(f"  H=32 pre-norm: mean={np.mean(r32):.4e}, std={np.std(r32):.4e}")
    print(f"  Dataset norm fit: H8 mean={mean_h8:.4e} std={std_h8:.4e}")
    print(f"                   H32 mean={mean_h32:.4e} std={std_h32:.4e}")


if __name__ == "__main__":
    main()

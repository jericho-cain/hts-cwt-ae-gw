#!/usr/bin/env python3
"""
Centroid-track detector baseline: AUROC(S_fc) vs Δφ.

NON-ORACLE: learns μ_iso(t) from isolated train set; scores each single sample by
S_fc = median_t |f_c(t) - μ_iso(t)|. No paired reference.

Sanity: Δφ=0 (iso vs iso) should give AUROC ~ 0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.datasets import make_phase0_batch
from data.synthetic.losa import accel_from_delta_phi
from utils.io import load_yaml

from experiments.observability_lambda_utils import lambda_from, recommended_dphi
from experiments.observability_phase_metric import (
    DEFAULT_CONFIG,
    COLUMN_ENERGY_THRESH,
    complex_cwt_pipeline,
    joint_gate,
    compute_freq_centroid,
)

DELTA_PHI_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 10.0, 12.0]
DOWNSAMPLE_FACTOR = 4


def _get_chirp_ranges(cfg: dict) -> tuple:
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


def gate_single(P: np.ndarray) -> np.ndarray:
    """Gate for single sample: t where colsum > thr * max(colsum)."""
    colsum = np.sum(P, axis=0)
    thr = COLUMN_ENERGY_THRESH * (colsum.max() + 1e-30)
    return (colsum > thr).astype(np.float64)


def get_fc_for_sample(
    strain: np.ndarray,
    complex_cwt_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute f_c(t) and gate for a single strain sample. Returns (f_c, gate)."""
    C, _, freqs = complex_cwt_pipeline(strain, **complex_cwt_kwargs)
    P = np.abs(C) ** 2
    gate = gate_single(P)
    f_c = compute_freq_centroid(P, freqs, gate)
    return f_c, gate


def compute_S_fc(
    f_c: np.ndarray,
    gate: np.ndarray,
    mu_iso: np.ndarray,
) -> float:
    """S_fc = median over gated t of |f_c(t) - μ_iso(t)|."""
    valid = (gate > 0.5) & np.isfinite(f_c) & np.isfinite(mu_iso)
    if not np.any(valid):
        return 0.0
    dev = np.abs(f_c - mu_iso)
    return float(np.median(dev[valid]))


def compute_S_z(
    f_c: np.ndarray,
    gate: np.ndarray,
    mu_iso: np.ndarray,
    sigma_iso: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """S_z = median over gated t of |f_c(t) - μ_iso(t)| / (σ_iso(t) + eps)."""
    valid = (gate > 0.5) & np.isfinite(f_c) & np.isfinite(mu_iso) & np.isfinite(sigma_iso)
    denom = sigma_iso + eps
    valid = valid & (denom > eps)
    if not np.any(valid):
        return 0.0
    z = np.abs(f_c - mu_iso) / denom
    return float(np.median(z[valid]))


SNR_SWEEP_VALS = [5, 10, 20, 40]
DELTA_PHI_SNR_SWEEP = [0.3, 1.0, 3.0]


def run_snr_sweep(
    cfg: dict,
    n_train: int,
    n_per_bin: int,
    complex_cwt_kwargs: dict,
    f_start_range: tuple,
    f_end_range: tuple,
    t_peak_range: tuple,
    sigma_range: tuple,
    duration: float,
    f_star: float,
    noise_sigma: float,
    seed: int,
    T: int,
    fs: float,
    auto_dphi: bool = False,
    lambda_target: str = "0.8",
) -> None:
    """Sweep SNR. Table: rows=SNR, cols=AUROC and Λ for each Δφ."""
    if auto_dphi:
        delta_phi_sweep = {}
        for snr in SNR_SWEEP_VALS:
            dphi_rec = recommended_dphi(snr, lambda_target)
            delta_phi_sweep[snr] = [0.5 * dphi_rec, dphi_rec, 1.5 * dphi_rec]
        dphi_vals_to_use = list(delta_phi_sweep.values())[0]  # for header; varies per snr
    else:
        delta_phi_sweep = {snr: DELTA_PHI_SNR_SWEEP for snr in SNR_SWEEP_VALS}
        dphi_vals_to_use = DELTA_PHI_SNR_SWEEP

    print(f"SNR sweep (ds=4, S_fc only)" + (" [auto_dphi]" if auto_dphi else ""))
    print()
    # Header: SNR | AUROC(Δφ=...) | Λ(Δφ=...) for each
    header_parts = []
    for d in dphi_vals_to_use:
        header_parts.append(f"AUROC({d:.2f})")
        header_parts.append(f"Λ({d:.2f})")
    header = "SNR  | " + " | ".join(header_parts)
    print(header)
    print("-" * len(header))

    results = []
    for snr in SNR_SWEEP_VALS:
        dphi_vals = delta_phi_sweep[snr]
        # Fit μ_iso from train at this SNR
        x_train, _ = make_phase0_batch(
            n=n_train,
            T=T,
            sample_rate=fs,
            snr=snr,
            a_los=0.0,
            noise_sigma=noise_sigma,
            seed=seed,
            f_start_range=f_start_range,
            f_end_range=f_end_range,
            t_peak_range=t_peak_range,
            sigma_range=sigma_range,
        )
        f_c_stack = []
        for i in range(n_train):
            f_c, _ = get_fc_for_sample(x_train[i], complex_cwt_kwargs)
            f_c_stack.append(f_c)
        f_c_stack = np.stack(f_c_stack, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu_iso = np.nanmean(f_c_stack, axis=0)

        # Iso eval: same for all Δφ at this SNR
        x_iso, _ = make_phase0_batch(
            n=n_per_bin,
            T=T,
            sample_rate=fs,
            snr=snr,
            a_los=0.0,
            noise_sigma=noise_sigma,
            seed=seed + 100000,
            f_start_range=f_start_range,
            f_end_range=f_end_range,
            t_peak_range=t_peak_range,
            sigma_range=sigma_range,
        )
        S_iso = np.array([
            compute_S_fc(*get_fc_for_sample(x_iso[i], complex_cwt_kwargs), mu_iso)
            for i in range(n_per_bin)
        ])

        std_S_iso = float(np.std(S_iso))
        row_result = {"snr": snr, "std_S_iso": std_S_iso, "auroc": {}, "lambda": {}}
        print_parts = []
        for delta_phi in dphi_vals:
            a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)
            x_losa, _ = make_phase0_batch(
                n=n_per_bin,
                T=T,
                sample_rate=fs,
                snr=snr,
                a_los=a_los,
                noise_sigma=noise_sigma,
                seed=int(seed + 200000 + delta_phi * 1e6),
                f_start_range=f_start_range,
                f_end_range=f_end_range,
                t_peak_range=t_peak_range,
                sigma_range=sigma_range,
            )
            S_losa = np.array([
                compute_S_fc(*get_fc_for_sample(x_losa[i], complex_cwt_kwargs), mu_iso)
                for i in range(n_per_bin)
            ])
            y = np.concatenate([np.zeros(n_per_bin), np.ones(n_per_bin)])
            s = np.concatenate([S_iso, S_losa])
            auroc = roc_auc_score(y, s)
            lam = lambda_from(delta_phi, snr)
            row_result["auroc"][float(delta_phi)] = auroc
            row_result["lambda"][float(delta_phi)] = lam
            print_parts.append(f"{auroc:.4f}")
            print_parts.append(f"{lam:.1f}")

        results.append(row_result)
        print(f"{snr:4.0f} | " + " | ".join(print_parts))

    out_path = ROOT / "experiments/outputs_corrected/observability_dfc_snr_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(DEFAULT_CONFIG.name),
        "mode": "snr_sweep",
        "delta_phi_values": [list(delta_phi_sweep[s]) for s in SNR_SWEEP_VALS] if auto_dphi else DELTA_PHI_SNR_SWEEP,
        "snr_values": SNR_SWEEP_VALS,
        "n_train": n_train,
        "n_per_bin": n_per_bin,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "results": results,
    }
    if auto_dphi:
        payload["lambda_target"] = lambda_target
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="S_fc AUROC baseline (non-oracle)")
    parser.add_argument("--n_train", type=int, default=2000, help="Isolated samples to fit μ_iso")
    parser.add_argument("--n_per_bin", type=int, default=500, help="Eval samples per bin")
    parser.add_argument("--snr_sweep", action="store_true", help="Sweep SNR × Δφ")
    parser.add_argument("--lambda_target", type=str, choices=["0.8", "0.95"], default="0.8", help="Target AUROC for recommended Δφ")
    parser.add_argument("--auto_dphi", action="store_true", help="Use recommended_dphi(snr,target) × [0.5,1,1.5] per SNR")
    args = parser.parse_args()
    n_train = args.n_train
    n_per_bin = args.n_per_bin

    cfg = load_yaml(DEFAULT_CONFIG)
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    T = int(data_cfg.get("T", 4096))
    fs = float(data_cfg.get("sample_rate", 1024))
    snr = float(data_cfg.get("snr", 5.0))
    noise_sigma = float(data_cfg.get("noise_sigma", 1e-21))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    duration = T / fs
    p0 = cfg.get("phase0_losa", {})
    f_star = float(p0.get("f_star_hz", 40.0))
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    fmin = float(cwt_cfg.get("fmin", 20.0))
    fmax = float(cwt_cfg.get("fmax", 512.0))
    n_scales = int(cwt_cfg.get("target_height", 8))

    f_start_range, f_end_range, t_peak_range, sigma_range = _get_chirp_ranges(cfg)

    complex_cwt_kwargs = {
        "sample_rate": int(fs),
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "fmin": fmin,
        "fmax": fmax,
        "n_scales": n_scales,
    }

    if args.snr_sweep:
        run_snr_sweep(
            cfg=cfg,
            n_train=n_train,
            n_per_bin=n_per_bin,
            complex_cwt_kwargs=complex_cwt_kwargs,
            f_start_range=f_start_range,
            f_end_range=f_end_range,
            t_peak_range=t_peak_range,
            sigma_range=sigma_range,
            duration=duration,
            f_star=f_star,
            noise_sigma=noise_sigma,
            seed=seed,
            T=T,
            fs=fs,
            auto_dphi=args.auto_dphi,
            lambda_target=args.lambda_target,
        )
        return

    # 1. Fit μ_iso(t) from isolated TRAIN set
    x_train, _ = make_phase0_batch(
        n=n_train,
        T=T,
        sample_rate=fs,
        snr=snr,
        a_los=0.0,
        noise_sigma=noise_sigma,
        seed=seed,
        f_start_range=f_start_range,
        f_end_range=f_end_range,
        t_peak_range=t_peak_range,
        sigma_range=sigma_range,
    )
    f_c_stack = []
    for i in range(n_train):
        f_c, _ = get_fc_for_sample(x_train[i], complex_cwt_kwargs)
        f_c_stack.append(f_c)
    f_c_stack = np.stack(f_c_stack, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu_iso = np.nanmean(f_c_stack, axis=0)
        sigma_iso = np.nanstd(f_c_stack, axis=0)
    print(f"Fitted μ_iso(t), σ_iso(t) from {n_train} isolated train samples")

    # 2. Eval: independent iso and LOSA sets (no pairing)
    results = []
    gate_stats = []
    print()
    print("Δφ   | Λ=Δφ×SNR | AUROC(S_fc) | AUROC(S_z) | mean S_iso | mean S_losa | n_gate: iso   | losa")
    print("-" * 95)

    for delta_phi in DELTA_PHI_GRID:
        lam = lambda_from(delta_phi, snr)
        a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)

        # Iso eval: independent draw (seed + 100000)
        x_iso, _ = make_phase0_batch(
            n=n_per_bin,
            T=T,
            sample_rate=fs,
            snr=snr,
            a_los=0.0,
            noise_sigma=noise_sigma,
            seed=seed + 100000,
            f_start_range=f_start_range,
            f_end_range=f_end_range,
            t_peak_range=t_peak_range,
            sigma_range=sigma_range,
        )
        iso_fc_gate = [get_fc_for_sample(x_iso[i], complex_cwt_kwargs) for i in range(n_per_bin)]
        S_iso = np.array([compute_S_fc(fc, g, mu_iso) for fc, g in iso_fc_gate])
        S_z_iso = np.array([compute_S_z(fc, g, mu_iso, sigma_iso) for fc, g in iso_fc_gate])
        n_gate_iso = np.array([int(np.sum(g > 0.5)) for _, g in iso_fc_gate])

        # LOSA eval: independent draw (seed + 200000 + Δφ)
        x_losa, _ = make_phase0_batch(
            n=n_per_bin,
            T=T,
            sample_rate=fs,
            snr=snr,
            a_los=a_los,
            noise_sigma=noise_sigma,
            seed=int(seed + 200000 + delta_phi * 1e6),
            f_start_range=f_start_range,
            f_end_range=f_end_range,
            t_peak_range=t_peak_range,
            sigma_range=sigma_range,
        )
        losa_fc_gate = [get_fc_for_sample(x_losa[i], complex_cwt_kwargs) for i in range(n_per_bin)]
        S_losa = np.array([compute_S_fc(fc, g, mu_iso) for fc, g in losa_fc_gate])
        S_z_losa = np.array([compute_S_z(fc, g, mu_iso, sigma_iso) for fc, g in losa_fc_gate])
        n_gate_losa = np.array([int(np.sum(g > 0.5)) for _, g in losa_fc_gate])

        y = np.concatenate([np.zeros(n_per_bin), np.ones(n_per_bin)])
        auroc_fc = roc_auc_score(y, np.concatenate([S_iso, S_losa]))
        auroc_z = roc_auc_score(y, np.concatenate([S_z_iso, S_z_losa]))
        mean_iso = float(np.mean(S_iso))
        mean_losa = float(np.mean(S_losa))
        n_iso_mean, n_iso_std = float(np.mean(n_gate_iso)), float(np.std(n_gate_iso))
        n_losa_mean, n_losa_std = float(np.mean(n_gate_losa)), float(np.std(n_gate_losa))

        results.append({
            "delta_phi": delta_phi,
            "snr": snr,
            "lambda": lam,
            "auroc_S_fc": auroc_fc,
            "auroc_S_z": auroc_z,
            "mean_S_iso": mean_iso,
            "mean_S_losa": mean_losa,
            "std_S_iso": float(np.std(S_iso)),
            "std_S_losa": float(np.std(S_losa)),
            "n_gate_iso_mean": n_iso_mean,
            "n_gate_iso_std": n_iso_std,
            "n_gate_losa_mean": n_losa_mean,
            "n_gate_losa_std": n_losa_std,
        })
        gate_stats.append({
            "delta_phi": delta_phi,
            "n_gate_iso": {"mean": n_iso_mean, "std": n_iso_std, "min": int(np.min(n_gate_iso)), "max": int(np.max(n_gate_iso))},
            "n_gate_losa": {"mean": n_losa_mean, "std": n_losa_std, "min": int(np.min(n_gate_losa)), "max": int(np.max(n_gate_losa))},
        })
        print(f"{delta_phi:4.1f} | {lam:8.1f} | {auroc_fc:11.4f} | {auroc_z:9.4f} | {mean_iso:9.6f} | {mean_losa:9.6f} | {n_iso_mean:6.0f}±{n_iso_std:.0f} | {n_losa_mean:6.0f}±{n_losa_std:.0f}")

    out_path = ROOT / "experiments/outputs_corrected/observability_dfc_auroc_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": str(DEFAULT_CONFIG.name),
            "mode": "non_oracle",
            "snr": snr,
            "n_train": n_train,
            "n_per_bin": n_per_bin,
            "downsample_factor": DOWNSAMPLE_FACTOR,
            "delta_phi_grid": DELTA_PHI_GRID,
            "results": results,
            "gate_stats": gate_stats,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")
    auroc_fc_0 = next(r["auroc_S_fc"] for r in results if r["delta_phi"] == 0)
    auroc_z_0 = next(r["auroc_S_z"] for r in results if r["delta_phi"] == 0)
    print(f"Sanity: AUROC(S_fc, Δφ=0) = {auroc_fc_0:.4f}, AUROC(S_z, Δφ=0) = {auroc_z_0:.4f} (expect ~0.5)")


if __name__ == "__main__":
    main()

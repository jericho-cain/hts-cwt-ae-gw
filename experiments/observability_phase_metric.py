#!/usr/bin/env python3
"""
Complex CWT phase-based observability metric.

Uses dual-ridge band (midpoint of iso+losa), joint gating, and tests both
compensation signs. Fixes ridge/band mismatch and sign convention.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, decimate, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from data.synthetic.losa import C_LIGHT, accel_from_delta_phi, apply_losa_constant_accel
from utils.io import load_yaml

logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

DELTA_PHI_GRID = [0.0, 0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 10.0, 12.0]
DEFAULT_CONFIG = ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml"
RIDGE_BAND_K = 2
GAUSSIAN_SIGMA = 1.0
COLUMN_ENERGY_THRESH = 0.01
COMPLEX_WAVELET = "cmor1.5-1.0"
LAG_WINDOW_MS = 50  # ±50 ms for per-scale lag search


def compute_losa_dt(a_los: float, n_samples: int, fs: float, t0: float = 0.0) -> np.ndarray:
    """Δt(t) = 0.5 * (a/c) * (t - t0)^2. Returns (n_samples,) in seconds."""
    t = np.arange(n_samples, dtype=np.float64) / fs
    return 0.5 * (a_los / C_LIGHT) * (t - t0) ** 2


def complex_cwt_pipeline(
    strain: np.ndarray,
    sample_rate: int,
    downsample_factor: int,
    fmin: float,
    fmax: float,
    n_scales: int,
    wavelet: str = COMPLEX_WAVELET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same preprocessing as training (downsample, filter, whiten) then complex CWT.
    Returns (C, freqs_grid, freqs_pywt) where:
      - C is complex (n_scales, n_time)
      - freqs_grid = logspace(fmin, fmax, n_scales) used to build scales
      - freqs_pywt = frequencies returned by pywt.cwt (authoritative per-scale)
    """
    if downsample_factor > 1:
        x = decimate(strain.astype(np.float64), downsample_factor, zero_phase=True)
        fs = sample_rate / downsample_factor
    else:
        x = strain.astype(np.float64)
        fs = float(sample_rate)
    sos = butter(4, fmin, btype="high", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x)
    whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
    freqs_grid = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    fc = pywt.central_frequency(pywt.ContinuousWavelet(wavelet))
    scales = fc * fs / freqs_grid
    coeffs, freqs_pywt = pywt.cwt(whitened, scales, wavelet, sampling_period=1 / fs)
    freqs_pywt = np.asarray(freqs_pywt, dtype=np.float64)
    return coeffs.astype(np.complex128), freqs_grid, freqs_pywt


def extract_ridge(P: np.ndarray) -> np.ndarray:
    """P: (n_freq, n_time). Returns ridge_idx (n_time,)."""
    P_smooth = gaussian_filter(
        P.astype(np.float64), sigma=(GAUSSIAN_SIGMA, GAUSSIAN_SIGMA), mode="nearest"
    )
    return np.argmax(P_smooth, axis=0).astype(np.int32)


def midpoint_band_indices(
    ridge_iso: np.ndarray, ridge_losa: np.ndarray, n_freq: int, k: int
) -> np.ndarray:
    """Band centered on round(0.5*(ridge_iso + ridge_losa)) per time."""
    n_time = ridge_iso.shape[0]
    c_idx = np.round(0.5 * (ridge_iso.astype(np.float64) + ridge_losa)).astype(np.int32)
    band = np.zeros((n_time, 2 * k + 1), dtype=np.int32)
    for t in range(n_time):
        c = c_idx[t]
        band[t] = np.clip(np.arange(c - k, c + k + 1), 0, n_freq - 1)
    return band


def joint_gate(P_iso: np.ndarray, P_losa: np.ndarray) -> np.ndarray:
    """Gate: keep t where (colsum_iso > thr) & (colsum_losa > thr)."""
    colsum_iso = np.sum(P_iso, axis=0)
    colsum_losa = np.sum(P_losa, axis=0)
    thr = COLUMN_ENERGY_THRESH * max(colsum_iso.max(), colsum_losa.max())
    return ((colsum_iso > thr) & (colsum_losa > thr)).astype(np.float64)


def compute_freq_centroid(
    P: np.ndarray, freqs: np.ndarray, gate: np.ndarray, eps: float = 1e-30
) -> np.ndarray:
    """f_c(t) = sum_k(f_k * P[k,t]) / sum_k(P[k,t]) for gated t. Returns (n_time,); ungated = nan."""
    n_time = P.shape[1]
    f_c = np.full(n_time, np.nan, dtype=np.float64)
    for t in range(n_time):
        if gate[t] < 0.5:
            continue
        Pcol = P[:, t] + eps
        denom = np.sum(Pcol)
        if denom < eps:
            continue
        f_c[t] = np.sum(freqs * Pcol) / denom
    return f_c


def compute_D_fc(
    P_iso: np.ndarray,
    P_losa: np.ndarray,
    freqs: np.ndarray,
    gate: np.ndarray,
) -> tuple[float, float]:
    """
    Time-warp proxy: D_fc = median_t |Δf_c(t)|, max_fc = max_t |Δf_c(t)|,
    where Δf_c(t) = f_c,LOSA(t) - f_c,iso(t) and f_c = sum_k(f_k P_k)/sum_k(P_k).
    Uses gated times only.
    """
    f_c_iso = compute_freq_centroid(P_iso, freqs, gate)
    f_c_losa = compute_freq_centroid(P_losa, freqs, gate)
    valid = (gate > 0.5) & np.isfinite(f_c_iso) & np.isfinite(f_c_losa)
    if not np.any(valid):
        return 0.0, 0.0
    delta_f_c = f_c_losa - f_c_iso
    abs_delta = np.abs(delta_f_c[valid])
    return float(np.median(abs_delta)), float(np.max(abs_delta))


def compute_D_mag_global(
    C_iso: np.ndarray,
    C_losa_aligned: np.ndarray,
    band: np.ndarray,
    gate: np.ndarray,
    eps: float = 1e-30,
) -> float:
    """Mean |log|C_losa| - log|C_iso|| over gated band. Lag-invariant magnitude metric."""
    n_time = C_iso.shape[1]
    total = 0.0
    count = 0
    for t in range(n_time):
        if gate[t] < 0.5:
            continue
        fi = band[t]
        mag_iso = np.abs(C_iso[fi, t]) + eps
        mag_losa = np.abs(C_losa_aligned[fi, t]) + eps
        log_diff = np.abs(np.log(mag_losa) - np.log(mag_iso))
        total += np.sum(log_diff)
        count += len(fi)
    return total / count if count > 0 else 0.0


def compute_gamma_and_D(
    C_iso: np.ndarray,
    C_losa: np.ndarray,
    band: np.ndarray,
    gate: np.ndarray,
    eps: float = 1e-30,
) -> tuple[float, float]:
    """γ(t) = sum_f C_losa*C_iso* / (sum_f |C_losa||C_iso| + eps). Returns (D_phase, mean|γ|)."""
    n_time = C_iso.shape[1]
    gamma_sum = 0.0
    n_gate = 0
    for t in range(n_time):
        if gate[t] < 0.5:
            continue
        n_gate += 1
        fi = band[t]
        c_iso = C_iso[fi, t]
        c_losa = C_losa[fi, t]
        numer = np.sum(c_losa * np.conj(c_iso))
        denom = np.sum(np.abs(c_losa) * np.abs(c_iso)) + eps
        gamma_t = numer / denom
        gamma_sum += np.abs(gamma_t)
    if n_gate == 0:
        return 1.0, 0.0
    mean_abs_gamma = gamma_sum / n_gate
    return 1.0 - mean_abs_gamma, mean_abs_gamma


def compute_D_comp(
    C_iso: np.ndarray,
    C_losa: np.ndarray,
    dt_arr: np.ndarray,
    freqs: np.ndarray,
    band: np.ndarray,
    gate: np.ndarray,
    sign: int,
    eps: float = 1e-30,
) -> tuple[float, float]:
    """sign: +1 or -1. C_iso_comp = C_iso * exp(sign * 1j * 2π * f * Δt). Returns (D, mean|γ|)."""
    phase_ramp = np.exp(sign * 1j * 2 * np.pi * freqs[:, None] * dt_arr[None, :])
    C_iso_comp = C_iso * phase_ramp
    return compute_gamma_and_D(C_iso_comp, C_losa, band, gate, eps)


def band_to_mask_ht(band: np.ndarray, gate: np.ndarray, n_freq: int, n_time: int) -> np.ndarray:
    """Build (H,T) bool mask: True where (f,t) is in band and gated."""
    mask_ht = np.zeros((n_freq, n_time), dtype=bool)
    for t in range(n_time):
        if gate[t] < 0.5:
            continue
        for fi in band[t]:
            mask_ht[fi, t] = True
    return mask_ht


def best_global_lag(
    Ci: np.ndarray, Cl: np.ndarray, mask_ht: np.ndarray, L: int,
    min_bins: int = 5, eps: float = 1e-30,
) -> tuple[int, float]:
    """
    Lag search using mean per-time coherence magnitude (avoids phase cancellation).
    For each lag k: shift Cl; at each t compute γ(t)=Σ a*_s b_s/(√(Σ|a|²)√(Σ|b|²)); score=mean_t|γ(t)|.
    Only include t with ≥min_bins masked freq bins and nonzero norms.
    """
    n_freq, n_time = Ci.shape

    best_k, best_score = 0, -1e9
    for k in range(-L, L + 1):
        Bk = np.roll(Cl.copy(), k, axis=1)
        if k > 0:
            Bk[:, :k] = 0
        elif k < 0:
            Bk[:, k:] = 0

        gammas = []
        for t in range(n_time):
            idx = mask_ht[:, t]
            if np.sum(idx) < min_bins:
                continue
            a = Ci[idx, t]
            b = Bk[idx, t]
            norm_a = np.sqrt(np.sum(np.abs(a) ** 2))
            norm_b = np.sqrt(np.sum(np.abs(b) ** 2))
            if norm_a < eps or norm_b < eps:
                continue
            num = np.vdot(a, b)  # sum(conj(a)*b)
            den = norm_a * norm_b + eps
            gamma_t = num / den
            gammas.append(np.abs(gamma_t))

        if len(gammas) == 0:
            score = -np.inf
        else:
            score = float(np.mean(gammas))

        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def align_global(
    C_iso: np.ndarray,
    C_losa: np.ndarray,
    band: np.ndarray,
    gate: np.ndarray,
    fs_down: float,
    lag_window_ms: float = LAG_WINDOW_MS,
) -> tuple[np.ndarray, float, float]:
    """Single global lag. Returns (C_losa_aligned, tau_star_ms, corr_score)."""
    n_freq, n_time = C_iso.shape
    L = int(lag_window_ms * fs_down / 1000)
    L = min(L, n_time // 4)

    mask_ht = band_to_mask_ht(band, gate, n_freq, n_time)
    best_k, corr_score = best_global_lag(C_iso, C_losa, mask_ht, L)

    shifted = np.roll(C_losa, best_k, axis=1)
    if best_k > 0:
        shifted[:, :best_k] = 0
    elif best_k < 0:
        shifted[:, best_k:] = 0

    tau_star_ms = best_k * 1000 / fs_down
    return shifted, tau_star_ms, corr_score


def parse_args():
    parser = argparse.ArgumentParser(description="Observability phase metric with global lag")
    parser.add_argument(
        "--downsample_factor",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="CWT time resolution. Default 4 (3.91 ms bins); also 2 or 8.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
    cwt_cfg = cfg.get("preprocessing", {}).get("cwt", {})
    fmin = float(cwt_cfg.get("fmin", 20.0))
    fmax = float(cwt_cfg.get("fmax", 512.0))
    n_scales = int(cwt_cfg.get("target_height", 8))
    downsample_factor = args.downsample_factor
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

    C_iso, freqs_grid, freqs_pywt = complex_cwt_pipeline(
        h_iso, int(fs), downsample_factor, fmin, fmax, n_scales
    )
    n_time = C_iso.shape[1]
    fs_down = fs / downsample_factor
    n_freq = C_iso.shape[0]
    dt_bin_ms = 1000 / fs_down

    P_iso = np.abs(C_iso) ** 2
    ridge_iso = extract_ridge(P_iso)

    results = []
    print(f"dt_bin = {dt_bin_ms:.2f} ms (downsample_factor={downsample_factor}, fs_down={fs_down:.0f} Hz)")
    print("Δφ   | D_fc(Hz) | max|Δf_c|(Hz) | D_raw_global | D_mag_global | tau*(ms) | corr_score")
    print("-" * 95)

    for delta_phi in DELTA_PHI_GRID:
        if delta_phi == 0:
            h_losa = h_iso.copy()
            dt_arr = np.zeros(n_time)
        else:
            a_los = accel_from_delta_phi(delta_phi, duration, f_star_hz=f_star)
            h_losa = apply_losa_constant_accel(h_iso.copy(), sample_rate=fs, a_los=a_los)
            h_losa = h_losa.astype(np.float32)
            dt_arr = compute_losa_dt(a_los, n_time, fs_down)

        C_losa, _, _ = complex_cwt_pipeline(
            h_losa, int(fs), downsample_factor, fmin, fmax, n_scales
        )
        P_losa = np.abs(C_losa) ** 2
        ridge_losa = extract_ridge(P_losa)

        band = midpoint_band_indices(ridge_iso, ridge_losa, n_freq, RIDGE_BAND_K)
        gate = joint_gate(P_iso, P_losa)

        D_fc, max_fc = compute_D_fc(P_iso, P_losa, freqs_pywt, gate)

        D_raw, _ = compute_gamma_and_D(C_iso, C_losa, band, gate)

        C_losa_global, tau_star_ms, corr_score = align_global(
            C_iso, C_losa, band, gate, fs_down, LAG_WINDOW_MS
        )
        D_raw_global, _ = compute_gamma_and_D(C_iso, C_losa_global, band, gate)
        D_mag_global = compute_D_mag_global(C_iso, C_losa_global, band, gate)
        D_comp_global, _ = compute_D_comp(
            C_iso, C_losa_global, dt_arr, freqs_pywt, band, gate, -1
        )

        tau_star_bins = tau_star_ms / dt_bin_ms
        results.append({
            "delta_phi": delta_phi,
            "D_fc": D_fc,
            "max_fc": max_fc,
            "D_raw_global": D_raw_global,
            "D_mag_global": D_mag_global,
            "D_comp_global": D_comp_global,
            "tau_star_ms": tau_star_ms,
            "tau_star_bins": tau_star_bins,
            "corr_score": corr_score,
            "D_raw": D_raw,
        })
        print(f"{delta_phi:4.1f} | {D_fc:8.4f} | {max_fc:12.4f} | {D_raw_global:12.6e} | {D_mag_global:12.6e} | {tau_star_ms:8.2f} | {corr_score:11.6e}")

    out_path = ROOT / f"experiments/outputs_corrected/observability_phase_metric_ds{downsample_factor}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": str(DEFAULT_CONFIG.name),
            "wavelet": COMPLEX_WAVELET,
            "delta_phi_grid": DELTA_PHI_GRID,
            "lag_window_ms": LAG_WINDOW_MS,
            "downsample_factor": downsample_factor,
            "dt_bin_ms": dt_bin_ms,
            "results": results,
        }, f, indent=2)
    print()
    print(f"Saved {out_path}")

    global_helps = sum(1 for r in results if r["D_raw_global"] < r["D_raw"])
    comp_global_ok = all(r["D_comp_global"] <= r["D_raw_global"] for r in results[1:4])
    # Monotonicity in perturbative regime (Δφ ∈ [0, 0.1, 0.3, 1, 3])
    perturb_idx = [i for i, r in enumerate(results) if r["delta_phi"] <= 3]
    D_raw_global_vals = [results[i]["D_raw_global"] for i in perturb_idx]
    D_mag_global_vals = [results[i]["D_mag_global"] for i in perturb_idx]
    D_fc_vals = [results[i]["D_fc"] for i in perturb_idx]
    mono_raw = all(D_raw_global_vals[i] <= D_raw_global_vals[i + 1] for i in range(len(D_raw_global_vals) - 1))
    mono_mag = all(D_mag_global_vals[i] <= D_mag_global_vals[i + 1] for i in range(len(D_mag_global_vals) - 1))
    mono_fc = all(D_fc_vals[i] <= D_fc_vals[i + 1] for i in range(len(D_fc_vals) - 1))
    print()
    print(f"  Global lag improves D_raw at {global_helps}/{len(results)} Δφ points.")
    if comp_global_ok:
        print("✓ D_comp_global ≤ D_raw_global in perturbative regime (Δφ ≲ 1).")
    else:
        print("⚠ D_comp_global not ≤ D_raw_global at small Δφ.")
    print(f"  D_raw_global monotone (Δφ≤3): {'✓' if mono_raw else '✗'}")
    print(f"  D_mag_global monotone (Δφ≤3): {'✓' if mono_mag else '✗'}")
    print(f"  D_fc monotone (Δφ≤3): {'✓' if mono_fc else '✗'}")


if __name__ == "__main__":
    main()

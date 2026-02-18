#!/usr/bin/env python3
"""
>65 Hz verification: confirm chirp band-limits at 65 Hz and trace leakage source.

Check 1: Instantaneous frequency f(t) ≤ 65 Hz
Check 2: FFT of noise-free chirp — negligible content above 65 Hz?
Check 3: STFT vs CWT — which shows >65 leakage? (isolates wavelet vs signal)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.synthetic.isolated_generator import generate_isolated_chirp
from utils.io import load_yaml


def main():
    config_path = ROOT / "experiments/configs/ground_phase0_tight_chirp.yaml"
    cfg = load_yaml(config_path)
    p0 = cfg.get("phase0_losa", cfg.get("synthetic", {}))
    data_cfg = cfg.get("data", cfg.get("synthetic", {}))
    
    T = int(data_cfg.get("T", 32768))
    fs = float(data_cfg.get("sample_rate", 1024))
    f0 = float(p0.get("chirp_f_start", [12.0])[0])
    f1 = float(p0.get("chirp_f_end", [65.0])[0])
    t_peak = 0.55
    sigma_env = 0.10
    
    h = generate_isolated_chirp(
        T=T, sample_rate=fs,
        f_start=f0, f_end=f1,
        t_peak=t_peak, sigma=sigma_env,
        amplitude=1e-20, seed=42,
    )
    
    t = np.arange(T) / fs
    t_norm = t / t[-1]
    f_inst = f0 + (f1 - f0) * t_norm
    
    print("=" * 60)
    print("Check 1: Instantaneous frequency")
    print("=" * 60)
    print(f"  f(t) range: {f_inst.min():.2f} – {f_inst.max():.2f} Hz")
    print(f"  Chirp end (config): {f1} Hz")
    above = np.sum(f_inst > f1)
    print(f"  Samples with f(t) > {f1}: {above} (expected 0 for linear chirp)")
    assert np.all(f_inst <= f1 + 1e-6), "Instantaneous freq exceeds chirp end!"
    print("  ✓ f(t) ≤ 65 Hz\n")
    
    print("=" * 60)
    print("Check 2: FFT of noise-free chirp")
    print("=" * 60)
    fft = np.fft.rfft(h)
    freqs_fft = np.fft.rfftfreq(T, 1/fs)
    power_fft = np.abs(fft) ** 2
    power_fft_norm = power_fft / power_fft.max()
    
    mask_above65 = freqs_fft > 65
    p_above65 = power_fft[mask_above65].sum() / power_fft.sum()
    peak_above65 = np.sqrt(power_fft[mask_above65].max()) if np.any(mask_above65) else 0
    peak_below65 = np.sqrt(power_fft[~mask_above65].max()) if np.any(~mask_above65) else 0
    
    print(f"  Power above 65 Hz: {100*p_above65:.4f}% of total")
    print(f"  Peak |FFT| above 65: {peak_above65:.2e} (vs peak below 65: {peak_below65:.2e})")
    print(f"  Ratio (above/below): {peak_above65/peak_below65:.4f}" if peak_below65 > 0 else "  N/A")
    
    # Show a few bins above 65
    idx65 = np.searchsorted(freqs_fft, 65)
    above_bins = min(5, len(freqs_fft) - idx65 - 1)
    if above_bins > 0:
        f_sample = freqs_fft[idx65:idx65+above_bins]
        print(f"  Sample bins above 65 Hz: freqs={[f'{x:.1f}' for x in f_sample]} Hz")
        print(f"    power_norm={power_fft_norm[idx65:idx65+above_bins].tolist()}")
    print()
    
    print("=" * 60)
    print("Check 3: STFT vs CWT (compare >65 content)")
    print("=" * 60)
    
    from scipy.signal import stft, decimate
    from preprocessing.cwt import cwt_clean
    
    n_fft = 512
    noverlap = n_fft // 2
    f_stft, t_stft, Zxx = stft(h, fs=fs, nperseg=n_fft, noverlap=noverlap)
    S_stft = np.abs(Zxx)
    
    # Match figure pipeline: decimate by 8, then CWT
    downsample = 8
    fs_down = fs / downsample
    h_down = decimate(h.astype(np.float64), downsample, zero_phase=True).astype(np.float32)
    h_down = h_down[:4096]
    log_scalogram, freqs_cwt, _, _ = cwt_clean(
        h_down, fs=fs_down, fmin=10, fmax=128, n_scales=48,
        wavelet="cmor6-1", skip_whitening=True, return_before_norm=True,
    )
    S_cwt = 10.0 ** np.squeeze(log_scalogram)
    
    mask_stft_65 = f_stft > 65
    mask_cwt_65 = freqs_cwt > 65
    
    stft_above = S_stft[mask_stft_65, :].max() if np.any(mask_stft_65) else 0
    stft_below = S_stft[~mask_stft_65, :].max() if np.any(~mask_stft_65) else 1e-30
    cwt_above = S_cwt[mask_cwt_65, :].max() if np.any(mask_cwt_65) else 0
    cwt_below = S_cwt[~mask_cwt_65, :].max() if np.any(~mask_cwt_65) else 1e-30
    
    print(f"  STFT: max mag above 65 / below 65 = {stft_above/stft_below:.4f}")
    print(f"  CWT:  max mag above 65 / below 65 = {cwt_above/cwt_below:.4f}")
    print("  → If CWT ratio > STFT ratio, wavelet bandwidth adds leakage.\n")
    
    print("=" * 60)
    print("Conclusion")
    print("=" * 60)
    if p_above65 < 0.01 and peak_above65 / peak_below65 < 0.1:
        print("  FFT: negligible power above 65 Hz — chirp is band-limited.")
    else:
        print("  FFT: non-trivial power above 65 Hz — check envelope / generator.")
    print("  >65 in CWT figure is likely wavelet PSF leakage (expected).")
    print("  Consider: chirp-end line, tighter color range, or cmor10-1.")


if __name__ == "__main__":
    main()

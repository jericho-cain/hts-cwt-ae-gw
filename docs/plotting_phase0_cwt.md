# Phase0 LOSA CWT Figure — Plotting Reference

Plotting doc for the Phase0 CWT spectrogram figure (iso vs LOSA + ΔS + Δt).

## Current Setup

* **log-power display**: `S_pow = 2*S` where `S = log10|W|`
* **display-only smoothing**: `gaussian_filter(S, sigma=(freq_sigma, time_sigma))`
* **PyWavelets pseudo-frequencies** for plotting (`get_cwt_display_freqs(...)`)
* **High-res mode**: `fmax=60 Hz`, `H≈48` (clips at chirp end, minimal leakage)
* **Default wavelet for figures**: `cmor6-1` unless overridden
* **COI masking**: set to NaN, colormap "bad" = light gray
* **ΔS panel**: ΔP/P (frac_dp) or ΔS (log_ratio); median + Gaussian smoothing; mask thresh -2.0

## Known Issues & Fixes

### 1. Y-axis tick overlap (100/120 collide)

**Cause:** Log-scaled axis with fixed high ticks; 100 and 120 too close in vertical space.

**Fix (Option A):** Drop 120 tick for fmax=128 Hz. Use ticks: `[10, 20, 30, 40, 60, 80, 100]` only.

**Status:** [x] Fixed — see `_plot_cwt_panel` in `phase0_make_cwt_figure.py`

---

### 2. ΔS "connected hotdog" / banded morphology (panel c)

**Expected morphology, not a bug.** Subtracting two highly correlated scalograms with a small time warp (LOSA) yields alternating lobes along the chirp track — matched-filter derivative / time-warp residual pattern. You can soften it but cannot remove it without blurring the effect you want to show.

**Current pipeline:** `--delta_s_mode frac_dp` (default) or `log_ratio`; median (9×9) + Gaussian σ=(1.0, 0.8); dlim×1.4; mask thresh -2.0.

**Alternatives** (if panel c needs to read differently): integrated δ(t)=∫ΔP/P df; ridge-aligned 1D curve; or stronger bandpass/mask around chirp track.

**Caption guidance:** "Alternating lobes in (c) reflect expected CWT-difference morphology for time-warped scalograms, not a plotting artifact."

---

### 3. Apparent signal above ~60–65 Hz

**Important:** May be *expected leakage*, not a true >65 Hz component.

**Causes:**
* Wavelet bandwidth (finite resolution)
* Chirp amplitude envelope / time localization
* Peak-normalization (sidelobes visible)
* LOSA time-warp broadening

**Verification:** `python experiments/check_chirp_bandlimit.py`

1. ✓ **Instantaneous frequency:** f(t) ≤ 65 Hz
2. ✓ **FFT:** negligible power above 65 Hz — chirp band-limited
3. ✓ **STFT vs CWT:** STFT above/below ≈ 0.0002; CWT ≈ 1.0 → wavelet PSF leakage

**Presentation:** fmax=60 (default), `--show_chirp_end`, cmor6-1. Energy above chirp end = wavelet PSF leakage, not true >65 Hz signal.

**Status:** [x] Checks run (`python experiments/check_chirp_bandlimit.py`) — leakage confirmed

---

## Referee-proof mode (`--referee`)

* Unsmooth P for ΔP/P (derivative-like structure preserved)
* eps = 10⁻³ × P_peak
* Light display smoothing σ=(0.5, 0.5)
* Mask thresh -3 (~15 dB)
* **Ridge overlay** on panel (c): black curve = analytic f(t) (chirp truth; not ridge-extracted); legend in **lower right** so it does not cover the panel label "(c)"; legend frame white, framealpha=0.95
* **Δf(t)** on panel (d) second y-axis: Δf = (df/dt)Δt

**Caption (referee):**
> CWT scalograms for a synthetic chirp in isolation and under LOSA time-warping (Δφ=3 rad). (a–b) Log-power scalograms (relative to peak), cmor6-1; gray = cone of influence. (c) Signed fractional power difference ΔP/(P+ε) (masked to signal band); black curve = analytic f(t); alternating lobes aligned with the chirp arise from the first-order CWT response to a small time warp (derivative-like structure). (d) Analytic LOSA time shift Δt(t)=½(a_los/c)t²; dashed = Δf(t)=(df/dt)Δt.

---

## Final-paper configuration

| Setting | Value | Rationale |
|--------|-------|------------|
| fmax | 60 Hz | Clips at chirp end; avoids leakage band |
| fmin | 10 Hz | Below chirp start |
| wavelet | cmor6-1 | Less ringy than morl |
| ticks | [10, 20, 30, 40, 60] | No overlap at fmax=60 |
| panel (c) | ΔP/P (frac_dp) | Fractional power change |
| smoothing | median 9×9 + Gauss σ=(1.0, 0.8) | Softens banding |
| dlim | 99th percentile × 1.4 | Less saturated |
| caption | See §2 above | Describe hotdog as expected |

**Scale–frequency correctness:** `scales = fc * fs / freqs` with `fc = pywt.central_frequency(wavelet)`. Plotting uses `get_cwt_display_freqs(...)`. Training/analysis must use the same mapping if freq bands or masks are used.

---

## Files Touched

* `experiments/phase0_make_cwt_figure.py`
  * `_plot_cwt_panel(...)` — ticks, pcolormesh
  * ΔS computation + smoothing + masking
  * wavelet selection
* `experiments/check_chirp_bandlimit.py` — >65 Hz verification script
* `src/preprocessing/cwt.py`
  * `cwt_clean(...)`, `get_cwt_display_freqs(...)`, `get_cwt_coi(...)`

---

## Suggested Action Order

1. [x] Fix ticks (done)
2. [x] Run >65 Hz verification checks (done — leakage confirmed)
3. [x] ΔS modes: `--delta_s_mode log_ratio` | `frac_dp` — compare and pick
4. [x] Analytic f(t) explicit labeling in panel (c) legend (not ridge-extracted)
5. [x] Legend placement: lower right; white frame (avoids overlap with "(c)" label)

---

## Final figure (referee layout)

`--referee` produces the paper-ready layout. Outputs: `figures/phase0_cwt_losa_triptych.png`, `figures/phase0_cwt_quad.png`. Caption: `docs/phase0_cwt_figure_caption.tex`.

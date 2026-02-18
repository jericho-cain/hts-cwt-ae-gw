# TODO: Recovering AUROC with Corrected CWT

The corrected scale formula (`scales = fc * fs / freqs`) collapsed AUROC for very_tight (0.87 → 0.41 @ Δφ=3). This list prioritizes actions to get good detection while keeping the physically correct CWT.

---

## Quick wins (low effort, try first)

### 1. [x] Use latent Mahalanobis instead of recon error for phase diagram
**Observation:** For corrected very_tight, `auroc_latent_vs_bin` @ Δφ=10 = 0.72 vs `auroc_vs_bin` = 0.30. The latent space has discriminative signal; recon error collapsed.

**Action:** Add `--metric latent` to `run_phase_diagram.py`; plot `auroc_latent_vs_bin` when available. Or switch the default phase diagram to latent AUROC.

**Effort:** Small. Phase diagram already computes latent metrics; just need to plot them.

---

### 2. [ ] Run full training (no --quick) for phase diagram
**Hypothesis:** Quick config uses 2 epochs, 200 train. The AE may need more training to learn the corrected CWT representation.

**Action:**
```bash
python -m experiments.run_phase_diagram --output-base experiments/outputs_corrected
```
(omit `--quick`). Full config: 30 epochs, 10k samples.

**Effort:** ~30–60 min per spread level (6 levels total).

---

### 3. [x] Increase CWT frequency resolution (H=16 or 32)
**Hypothesis:** 8 scales over 10–64 Hz is very coarse; LOSA time-warp creates subtle ridge shifts that get averaged away.

**Action:** Use `ground_phase0_tight_chirp_H32.yaml`. Run: `--config experiments/configs/ground_phase0_tight_chirp_H32.yaml --output-suffix _H32`. **Result:** very_tight recon AUROC recovered to 0.71/0.64 (vs 0.50/0.41 at H=8).

**Effort:** Medium. Config + model compatibility check. Run phase diagram with H=32 base config.

---

## Medium effort

### 4. [ ] Increase model capacity for corrected CWT
**Hypothesis:** Corrected scalograms may need a deeper/wider AE to learn a good representation.

**Action:** Try `latent_dim: 64`, `lstm_hidden: 128`, or add one conv layer. Compare AUROC on very_tight.

**Effort:** Medium. New config, retrain, compare.

---

### 5. [ ] Try complex (Re+Im) CWT instead of magnitude-only
**Hypothesis:** Phase info from complex CWT may capture LOSA time-warp more robustly.

**Action:** Use `use_complex: true` (e.g. `ground_phase0_runC_complex.yaml`). Model must have `in_channels: 2`.

**Effort:** Medium. Config exists; ensure phase diagram / eval path supports it.

---

### 6. [ ] Try cmor6-1 wavelet instead of morl
**Hypothesis:** Different time–frequency tradeoff; cmor has cleaner sidelobes. May change scalogram structure favorably.

**Action:** Set `wavelet: cmor6-1` in preprocessing. Verify PyWavelets supports it.

**Effort:** Small. Config change only.

---

## Larger / exploratory

### 7. [ ] Ridge-based or derivative features
**Hypothesis:** LOSA manifests as ridge shift. Explicit ridge extraction or ridge-aligned features could improve separability.

**Action:** Add optional preprocessing: ridge track extraction → use ridge-centered or ridge-gradient features as AE input. Research/implement.

**Effort:** High.

---

### 8. [ ] Two-stage pipeline
**Hypothesis:** Coarse CWT for initial detection; fine-grained analysis (e.g. ridge, STFT) for characterization.

**Action:** Design and implement. Out of scope for quick recovery.

**Effort:** High.

---

### 9. [ ] Tune CWT fmin/fmax for chirp band
**Hypothesis:** Chirp is 12–65 Hz. Perhaps narrower band (e.g. 15–60 Hz) with same H gives better per-bin resolution.

**Action:** Sweep fmin, fmax in config; compare very_tight AUROC.

**Effort:** Low. Config sweep.

---

## Validation / sanity checks (after H32 experiment)

### A. [x] Recon error scaling (mean ~1.0 vs ~0.5)
**Red flag:** H=8 mean recon error ~1.0, H=32 ~0.5. Suggests input scaling changed, not just "better representation."

**Results (check_input_variance_H8_vs_H32.py):**
- Same normalization ✓: both post-norm mean≈0, std=1, var=1
- ||x||^2 ratio (H32/H8) = 4.0 (exactly n_elements ratio)
- If err scaled purely as 1/||x||^2: expected err_H32 ≈ 0.25 when err_H8=1.0
- Observed: err_H32 ~0.5 (not 0.25)
- **Conclusion:** Denom scaling does NOT fully explain the shift. A pure scaling story would give err≈0.25; we get 0.5, so the AE at H=32 is doing relatively better per-element reconstruction (MSE doesn't scale 1:1 with n_elements) — the improvement is partly genuine representation quality, not just arithmetic.

**Airtight check (check_recon_metrics_H8_vs_H32.py):** Now computes MSE_elem, SSE, r separately. Phase0 JSON stores mean_mse_elem, mean_sse, n_elem. Re-run with properly trained H=32 checkpoint (phase diagram didn't save model.pt; train one with full config) for fair MSE_elem comparison.

---

### B. [ ] Δφ ordering / monotonicity
**Red flag:** AUROC(Δφ=1) > AUROC(Δφ=3) in some runs. Physics suggests larger deformation → easier to detect → AUROC(3) ≥ AUROC(1).

**Common causes when reversed:**
- Saturation/clipping (percentile clip, log eps floor, masking)
- Distribution overlap: larger Δφ pushes into regions where synthetic variability or preprocessing dominates
- Score nonlinearity: recon error saturates if AE outputs something generic for both classes

**Action:** After H32 experiment, check monotonicity across full Δφ grid (0.1, 0.3, 1, 3, 10) in phase0_summary.json. Plot AUROC vs Δφ for very_tight; expect roughly monotonic increase. If not, inspect clipping/saturation in preprocessing and AE output.

---

## Ablation studies (after red flags A and B are addressed)

### Ablation 1: H=32 train, evaluate H=32 vs collapsed-to-8
**Goal:** Prove “frequency resolution is the lever.”

**Design:**
- Train AE at H=32.
- Compute H=32 scalogram at eval time.
- Downproject to 8 bins (e.g. average groups of 4 bins).
- Score both ways (H=32 input vs collapsed-to-8) without retraining (or retrain a tiny head).

**Interpretation:** If separability disappears when you collapse H=32→8, that’s strong evidence the information is in the higher-res frequency structure.

---

### Ablation 2: Time resolution instead of frequency
**Goal:** Test whether the benefit is “frequency resolution” specifically or “representation resolution” more generally.

**Design:** Keep H fixed (e.g. H=8 or H=32), change time resolution instead. If time axis is coarse (large downsample_factor), the time-warp may be sub-bin.

**Action:** One run with smaller `downsample_factor` (e.g. 4 instead of 8), same H. Compare AUROC.

**Interpretation:** If time resolution helps similarly, the story becomes “representation resolution” more generally, not just frequency.

---

## Summary priority

| # | Action                          | Effort | Expected impact      |
|---|---------------------------------|--------|----------------------|
| 1 | Use latent AUROC for phase diag | Small  | Likely helpful       |
| 2 | Full training (no --quick)      | Time   | Possible recovery    |
| 3 | H=32 frequency resolution       | Medium | Possible recovery    |
| 6 | cmor6-1 wavelet                 | Small  | Unknown             |
| 4 | Larger model                    | Medium | Possible             |
| 5 | Complex CWT                    | Medium | Unknown              |

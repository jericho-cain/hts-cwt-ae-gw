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

### 3. [ ] Increase CWT frequency resolution (H=16 or 32)
**Hypothesis:** 8 scales over 10–64 Hz is very coarse; LOSA time-warp creates subtle ridge shifts that get averaged away.

**Action:** Use `ground_phase0_runB_H32.yaml`-style config: `target_height: 32`, `input_height: 32`. Requires model arch supports H=32.

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

## Summary priority

| # | Action                          | Effort | Expected impact      |
|---|---------------------------------|--------|----------------------|
| 1 | Use latent AUROC for phase diag | Small  | Likely helpful       |
| 2 | Full training (no --quick)      | Time   | Possible recovery    |
| 3 | H=32 frequency resolution       | Medium | Possible recovery    |
| 6 | cmor6-1 wavelet                 | Small  | Unknown             |
| 4 | Larger model                    | Medium | Possible             |
| 5 | Complex CWT                    | Medium | Unknown              |

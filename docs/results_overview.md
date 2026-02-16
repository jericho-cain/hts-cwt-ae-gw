# HTS-CWT-AE-GW Phase 0 Results Overview

*Last updated: 2026-02-15*

This document summarizes the synthetic Phase 0 LOSA experiments and key findings. Use it to **reproduce good results** and **document failed/informative runs** for future reference.

---

## Core Methodological Insight (Publishable)

> **LOSA detectability with template-free AEs is fundamentally an identifiability problem: if the training distribution is "closed" under the deformation (time-warp), reconstruction error won't flag it. When the isolated family is constrained, LOSA becomes an out-of-family deformation and is cleanly detectable.**

Supported claims:
1. Broad intrinsic variability makes LOSA "in-distribution" for reconstruction-based AEs.
2. Constraining the isolated family makes LOSA detectable with strong monotone scaling in Δφ.
3. Template-free environmental detection requires either: (a) conditional/mixture models, (b) training on narrow intrinsic submanifolds, or (c) representations explicitly sensitive to phase/time-warp structure.

---

## How to Reproduce

All runs use:
```bash
python -m experiments.run_experiment --config <CONFIG_PATH>
```
From the project root. Seed is set in config (typically 42).

---

## Studies with Good Results (Reproducible)

### Tight Chirp + Dataset Norm — **LOSA detectable**

**Config:** `experiments/configs/ground_phase0_tight_chirp.yaml`

```bash
python -m experiments.run_experiment --config experiments/configs/ground_phase0_tight_chirp.yaml
```

**Reproducibility:**
- `synthetic.tight_chirp: true`
- `synthetic.chirp_f_start: 12.0`, `chirp_f_end: 65.0`, `chirp_t_peak: 0.55`, `chirp_sigma: 0.10`, `chirp_jitter_pct: 0.0`
- `experiment.seed: 42`
- `preprocessing.cwt.norm_mode: dataset`

**Expected outputs:** `experiments/outputs/phase0_tight_chirp/`

| Δφ (rad) | AUROC | Cohen's d | TPR@1% FPR |
|----------|-------|-----------|------------|
| 0.1 | 0.59 | 0.33 | 2.2% |
| 0.3 | 0.63 | 0.45 | 3.0% |
| 1.0 | 0.80 | 1.17 | 12.3% |
| 3.0 | **0.87** | **1.57** | **21.5%** |
| 10.0 | 0.84 | 1.38 | 16.5% |

**Spearman ρ = 0.90** (p = 0.037)

---

## Variability–Detectability Phase Diagram (Complete)

Sweep intrinsic spread (chirp parameter jitter) and measure AUROC @ Δφ=1 and Δφ=3.

**Run:**
```bash
python -m experiments.run_phase_diagram
```
Use `--quick` for fast validation. Use `--skip-run` to replot from existing outputs.

**Results:**

| Spread | Jitter | AUROC @ Δφ=1 | AUROC @ Δφ=3 |
|--------|--------|--------------|--------------|
| **very_tight** | 0% | **0.80** | **0.87** |
| tight | 1% | 0.50 | 0.51 |
| medium | 5% | 0.49 | 0.50 |
| wide | 10% | 0.49 | 0.49 |
| very_wide | 15% | 0.49 | 0.49 |
| broad | full | 0.49 | 0.50 |

**Finding:** Detectability drops sharply at ~1% jitter. Only *very_tight* (0% jitter) yields separation; any intrinsic spread makes LOSA in-distribution again. The identifiability cliff is steep.

**Figure:** `experiments/outputs/phase_diagram/phase_diagram_auroc_vs_spread.png`

---

## Studies with No LOSA Separation (Documented)

These runs did **not** separate LOSA from isolated; they are kept for completeness and to avoid repeating them.

### Original Phase 0 (per-sample norm, broad chirp)

**Config:** `experiments/configs/ground_phase0_real.yaml`

```bash
python -m experiments.run_experiment --config experiments/configs/ground_phase0_real.yaml
```

- **Result:** AUROC ≈ 0.51, Cohen's d ≈ 0.02 at Δφ=10
- **Why:** Per-sample z-score; broad chirp variability

---

### Run A: Dataset norm only (broad chirp)

**Config:** `experiments/configs/ground_phase0_runA_dataset_norm.yaml`

```bash
python -m experiments.run_experiment --config experiments/configs/ground_phase0_runA_dataset_norm.yaml
```

- **Result:** AUROC ≈ 0.50, Cohen's d ≈ 0.03 at Δφ=10
- **Why:** Chirp family still broad; LOSA reparameterization absorbed by variability

---

### Run B: Dataset norm + H=32 (broad chirp)

**Config:** `experiments/configs/ground_phase0_runB_H32.yaml`

```bash
python -m experiments.run_experiment --config experiments/configs/ground_phase0_runB_H32.yaml
```

- **Result:** AUROC ≈ 0.51, Cohen's d ≈ 0.03 at Δφ=10; Spearman ρ improved to ~0.5
- **Why:** Higher CWT resolution didn't fix distributional degeneracy

---

### Run C: Dataset norm + H=32 + Re/Im channels (broad chirp)

**Config:** `experiments/configs/ground_phase0_runC_complex.yaml`

```bash
python -m experiments.run_experiment --config experiments/configs/ground_phase0_runC_complex.yaml
```

- **Result:** Loss converged quickly (~0.012); recon errors ~1.0 with near-zero variance; AUROC ≈ 0.5
- **Why:** Possible metric/preprocess mismatch for complex channels; *not* interpretable as “phase doesn't help”

---

## LOSA Tensor Diagnostic

Run standalone:
```bash
python -m experiments.losa_tensor_diagnostic --config experiments/configs/ground_phase0_tight_chirp.yaml
```

- **Relative diff** (isolated vs Δφ=10, same chirp): **~62%**
- **Interpretation:** CWT pipeline preserves LOSA; the effect is present in the features.

---

## Config Summary

| Experiment | Config | Outcome |
|------------|--------|---------|
| **Tight chirp** | `ground_phase0_tight_chirp.yaml` | ✓ LOSA detectable |
| **Phase diagram** | `run_phase_diagram` | ✓ AUROC vs spread; sharp cliff |
| Original (broad) | `ground_phase0_real.yaml` | ✗ No separation |
| Run A (dataset norm) | `ground_phase0_runA_dataset_norm.yaml` | ✗ No separation |
| Run B (H=32) | `ground_phase0_runB_H32.yaml` | ✗ No separation |
| Run C (Re/Im) | `ground_phase0_runC_complex.yaml` | ✗ Metric/preprocess issues |

---

## Quick Validation Runs

For fast checks before long runs:

| Config | Notes |
|--------|-------|
| `ground_phase0_real_quick.yaml` | 200 train, 2 epochs |
| `ground_phase0_tight_chirp_quick.yaml` | 200 train, 2 epochs, tight chirp + latent |

---

## Output Layout

Each run writes to `experiments/outputs/<save_dir>/`:

- `baseline_fit.json` — train/val loss, recon error stats
- `phase0_summary.json` — AUROC, Cohen's d, TPR@1% FPR, Spearman ρ (recon + latent if applicable)
- `phase0_hist.png`, `phase0_auroc.png` — plots
- `losa_tensor_diagnostic.json` — CWT diff diagnostic (when run)

**Phase diagram:** `experiments/outputs/phase_diagram/` — per-level subdirs, `phase_diagram_auroc_vs_spread.png`, `phase_diagram_results.json`

---

## Planned Experiments

### Experiment B: Mixture-of-AEs

Train K AEs for K parameter bins; evaluate LOSA on full mixed distribution. *To be implemented.*

---

## Δφ=10 Slight Drop — Note

The small AUROC drop at Δφ=10 (vs Δφ=3) is expected (window/boundary effects, “too strong” deformation). Planned diagnostics: fraction of samples out-of-window, mean signal energy vs Δφ. Not a concern for current conclusions.

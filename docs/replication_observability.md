# Replication Guide: Observability & Chirp Deformation Studies

**Purpose:** Reproduce the LOSA (line-of-sight acceleration) observability experiments and chirp deformation figure for referee review or follow-on analysis.

**Last updated:** 2025-02

---

## Handoff / AI Context

**Use these phrases when handing off to a future session:**

- "Replicate the observability studies: run SNR sweep → scaling law, lambda fit, heatmap; then chirp deformation figure. See `docs/replication_observability.md`."
- "The paper has a chirp deformation figure (intuition anchor) before the AUROC/Λ plots—no CWT panels, just strain overlay + frequency trajectory."
- "Λ = Δφ × SNR collapses detectability. Fitted thresholds Λ₀.8 ≈ 19, Λ₀.95 ≈ 27 live in `observability_lambda_utils.py`."
- "Referee wants X" → use the Referee / Extension Ideas section; add bootstrap CIs, more SNRs, or alternate chirp params as requested.
- "outputs_corrected" = folder for corrected-CWT pipeline outputs (vs older outputs/). All observability JSONs and PNGs go there.

**Jargon:**

| Term | Meaning |
|------|---------|
| LOSA | Line-of-sight acceleration; Romer delay Δt(t) = 0.5(a/c)t² |
| Δφ | Cumulative phase distortion (rad) at f_star=40 Hz; controls LOSA strength |
| f_c(t) | Frequency centroid of CWT power: Σ f_k P_{k,t} / Σ P_{k,t} |
| S_fc | Detector: median_t \|f_c(t) − μ_iso(t)\| |
| μ_iso | Learned from isolated (no-LOSA) training; non-oracle baseline |
| Phase0 | Synthetic chirp + LOSA pipeline; tight chirp = fixed f_start/f_end |
| joint_gate | Keep times where both iso and LOSA CWT have power above threshold |

**Gotchas:**

- `observability_dfc_*` scripts all read from `observability_dfc_snr_sweep.json`; run the SNR sweep first.
- With `--auto_dphi`, `delta_phi_values` in the JSON becomes a list-of-lists (one per SNR). Scaling law, lambda fit, heatmap handle this via `r["auroc"].keys()`.
- Chirp deformation is noise-free; no SNR scaling. Uses same chirp generator as Phase0.
- Config `ground_phase0_tight_chirp.yaml` has T=32768, duration ≈32 s. Downsample factor 4 in observability (not 8 as in some training configs).

---

## Scientific Context

We study whether LOSA-induced time warps are detectable in a CWT-based centroid statistic:

- **Statistic:** S_fc = median_t |f_c(t) − μ_iso(t)|, where f_c(t) is the frequency centroid of CWT power and μ_iso(t) is learned from isolated (no-LOSA) training data.
- **Key result:** AUROC(S_fc) collapses onto a single control parameter **Λ = Δφ × SNR**.
- **Fitted thresholds:** Λ₀.8 ≈ 18.97, Λ₀.95 ≈ 26.95 (from sigmoid fit).

All experiments use the **Phase0 tight chirp** (fixed f_start=12 Hz, f_end=65 Hz, t_peak=0.55, sigma=0.10) and the **constant-accel LOSA model** parameterized by Δφ (cumulative phase shift at f_star=40 Hz).

---

## Prerequisites

```bash
# From repo root
pip install -r requirements.txt
```

Required: `numpy`, `scipy`, `PyWavelets`, `scikit-learn`, `matplotlib`, `PyYAML`.

**Config:** All scripts read `experiments/configs/ground_phase0_tight_chirp.yaml`. Do not modify it for replication unless you document changes.

---

## Quick Replication (≈5–10 min)

Reduce sample counts for fast verification. Full runs use larger `n_train` and `n_per_bin`.

### 1. SNR sweep (produces the JSON used by scaling law, lambda fit, heatmap)

```bash
python experiments/observability_dfc_auroc_baseline.py --snr_sweep --n_train 500 --n_per_bin 150
```

**Output:** `experiments/outputs_corrected/observability_dfc_snr_sweep.json`

Columns: SNR × Δφ grid (default Δφ ∈ {0.3, 1, 3}), AUROC and Λ per cell.

### 2. Downstream plots (read the JSON from step 1)

```bash
python experiments/observability_dfc_scaling_law.py
python experiments/observability_dfc_lambda_fit.py
python experiments/observability_dfc_snr_heatmap.py
```

**Outputs:**
- `experiments/outputs_corrected/observability_dfc_scaling_law.png` — AUROC vs Λ
- `experiments/outputs_corrected/observability_dfc_lambda_fit.png` — Sigmoid fit, Λ₀.8, Λ₀.95
- `experiments/outputs_corrected/observability_dfc_snr_heatmap.png` — AUROC(SNR, Δφ) heatmap

### 3. Chirp deformation figure (paper intuition panel)

```bash
python experiments/make_chirp_deformation_figure.py
```

**Outputs:** `figures/chirp_deformation.png`, `figures/chirp_deformation.pdf`

---

## Full Replication (≈1–2 hr)

Use production sample sizes for publication-quality curves.

```bash
# SNR sweep (longest step)
python experiments/observability_dfc_auroc_baseline.py --snr_sweep --n_train 2000 --n_per_bin 500

# Plots
python experiments/observability_dfc_scaling_law.py
python experiments/observability_dfc_lambda_fit.py
python experiments/observability_dfc_snr_heatmap.py

# Chirp deformation
python experiments/make_chirp_deformation_figure.py
```

---

## Full Pipeline Summary

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `observability_dfc_auroc_baseline.py --snr_sweep` | Config | `observability_dfc_snr_sweep.json` |
| 2 | `observability_dfc_scaling_law.py` | Step 1 JSON | `observability_dfc_scaling_law.png` |
| 3 | `observability_dfc_lambda_fit.py` | Step 1 JSON | `observability_dfc_lambda_fit.png` |
| 4 | `observability_dfc_snr_heatmap.py` | Step 1 JSON | `observability_dfc_snr_heatmap.png` |
| 5 | `make_chirp_deformation_figure.py` | Config | `figures/chirp_deformation.{png,pdf}` |

---

## Optional Experiments

### Δφ sweep (no SNR sweep)

```bash
python experiments/observability_dfc_auroc_baseline.py --n_train 2000 --n_per_bin 500
```

Output: `observability_dfc_auroc_baseline.json` (AUROC vs Δφ at fixed SNR from config).

### Auto Δφ sweep (uses Λ thresholds to pick Δφ per SNR)

```bash
python experiments/observability_dfc_auroc_baseline.py --snr_sweep --auto_dphi --lambda_target 0.8 --n_train 500 --n_per_bin 150
```

Uses `recommended_dphi(snr, "0.8")` × [0.5, 1, 1.5] as Δφ values per SNR.

### Δf_c(t) diagnostic

```bash
python experiments/observability_delta_fc_plot.py
```

Output: `experiments/outputs_corrected/observability_delta_fc.png` — Δf_c(t) vs time for Δφ ∈ {0, 0.3, 1, 3}.

### LOSA time-shift diagnostics

```bash
python experiments/observability_debug_large_delta_phi.py
```

Output: `observability_debug_large_delta_phi.json` — dt_max, dt_end, rel_L2, etc.

---

## Key Implementation Details

- **CWT:** Complex CWT (cmor1.5-1.0), downsample_factor=4, fmin/fmax from config.
- **Gate:** `joint_gate(P_iso, P_losa)` — keep times where both have sufficient power.
- **Centroid:** f_c(t) = Σ f_k P_{k,t} / Σ P_{k,t} over gated t.
- **Λ utils:** `experiments/observability_lambda_utils.py` — `lambda_from(dphi, snr)`, `recommended_dphi(snr, target)`.

---

## Referee / Extension Ideas

1. **Vary f_star:** Change `phase0_losa.f_star_hz` in config and re-run; note how Λ thresholds shift.
2. **Different SNR grid:** Edit `SNR_SWEEP_VALS` in `observability_dfc_auroc_baseline.py` (default [5, 10, 20, 40]).
3. **More Δφ points:** Use `--auto_dphi` or modify `DELTA_PHI_SNR_SWEEP`.
4. **Bootstrap confidence intervals:** Add to lambda_fit and scaling_law scripts; currently point estimates only.
5. **Alternative chirp parameters:** Use a different config (e.g. broad chirp) and document.

---

## PN Waveform Sanity Figure (optional)

A standalone figure using LALSimulation TaylorT4 validates that LOSA deformation behaves similarly for post-Newtonian waveforms. **Requires LALSuite** in a separate env:

```bash
# Create env (Python 3.11 recommended for LALSuite wheels)
micromamba create -n pn-demo -c conda-forge python=3.11 lalsuite
# Or: conda create -n pn-demo -c conda-forge python=3.11 lalsuite

micromamba run -n pn-demo python experiments/pn_minimal_demo.py
```

**Outputs:** `figures/pn_chirp_deformation.png`, `figures/pn_chirp_deformation.pdf`

Prints Δt_end (ms) and max |Δf_c| (Hz) for the caption. Does not depend on the repo `.venv`; keep `.venv` for main pipeline.

---

## Sanity Checks

- **Δφ=0:** AUROC(S_fc) ≈ 0.5 (isolated vs isolated).
- **Λ scaling:** AUROC curves from different (SNR, Δφ) should overlap when plotted vs Λ.
- **Chirp figure:** `make_chirp_deformation_figure.py` prints dt_end (ms), max |Δf_c| (Hz), confirms noise_sigma=0.

# HTS-CWT-AE-GW

[![arXiv](https://img.shields.io/badge/arXiv-2602.17725-b31b1b.svg)](https://arxiv.org/abs/2602.17725)

Synthetic framework for studying line-of-sight acceleration (LOSA) phase modulation in gravitational-wave-like chirp signals using time-frequency methods.

This repository accompanies the manuscript in `docs/main.pdf`, with the primary result that detectability of LOSA deformation is governed by a one-parameter scaling:

\[
\Lambda = \Delta\phi_{\mathrm{env}} \times \mathrm{SNR},
\]

using a template-free centroid-based statistic derived from the continuous wavelet transform (CWT).

No real detector strain data are distributed in this repository; all reported experiments are synthetic.

## Scope

The current publication-facing pipeline is the centroid observability analysis:

- Construct CWT power maps from isolated and LOSA-modulated chirps
- Compute centroid trajectory \( f_c(t) \)
- Score each sample with \( S_{fc} = \mathrm{median}_t |f_c(t) - \mu_{\mathrm{iso}}(t)| \)
- Evaluate AUROC across \((\Delta\phi_{\mathrm{env}}, \mathrm{SNR})\) and test collapse versus \(\Lambda\)

Autoencoder components are retained in the codebase for internal follow-on work, but they are not required for reproducing the main manuscript figures or conclusions.

## Reproducing Main Results

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the observability SNR sweep (produces the core JSON for downstream plots):

```bash
python experiments/observability_dfc_auroc_baseline.py --snr_sweep --n_train 2000 --n_per_bin 500
```

Generate scaling-law and heatmap figures:

```bash
python experiments/observability_dfc_scaling_law.py
python experiments/observability_dfc_lambda_fit.py
python experiments/observability_dfc_snr_heatmap.py
```

Generate the chirp deformation figure:

```bash
python experiments/make_chirp_deformation_figure.py
```

For a shorter validation run, see `docs/replication_observability.md`.

## Repository Layout

```
hts-cwt-ae-gw/
├── README.md
├── docs/
│   ├── main.pdf
│   ├── replication_observability.md
│   └── results_overview.md
├── experiments/
│   ├── observability_*.py
│   ├── make_chirp_deformation_figure.py
│   ├── run_experiment.py
│   └── configs/
├── src/
│   ├── data/synthetic/
│   ├── preprocessing/
│   ├── evaluation/
│   ├── training/
│   ├── models/
│   └── utils/
└── tests/
```

## Notes on Autoencoder Modules

Autoencoder training and phase-0 anomaly workflows remain available (`experiments/run_experiment.py`, `src/models/`, and related configs). These paths are preserved intentionally for private comparative studies (e.g., latent-space and reconstruction-based diagnostics), including future autoencoder experiments.

## Citation and License

If you use the manuscript associated with this repository, cite:

```bibtex
@article{cain2026scaling,
  title = {Scaling Laws for Template-Free Detection of Environmental Phase Modulation in Gravitational-Wave Signals},
  author = {Cain, Jericho},
  journal = {arXiv preprint arXiv:2602.17725},
  year = {2026},
  url = {https://arxiv.org/abs/2602.17725}
}
```

See `CITATION.cff` for software and manuscript citation metadata and `LICENSE` for licensing terms (MIT).

# HTS-CWT-AE-GW

Synthetic-only study for **Hierarchical Triple Signatures (HTS)** and LOSA perturbations in gravitational-wave anomaly detection. This repository provides reusable CWT (Continuous Wavelet Transform) autoencoder infrastructure—architecture-agnostic and ready for future HTS/LOSA modeling.

**No real detector data included.** All experiments use synthetic data.

## Overview

- **CWT preprocessing**: Time–frequency scalograms with configurable wavelet, frequency range, and normalization
- **Autoencoder models**: CNN-based backbone (LSTM-style) with model registry for easy backbone swapping
- **Training scaffolding**: Training loop, callbacks, and evaluation metrics
- **Synthetic data**: Chirp-like signals + Gaussian noise for smoke tests and baselines

Future work will add LOSA modeling and eccentric Kepler solvers. LISA-specific pipelines and real LIGO strain are out of scope for this repo.

## Quickstart

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run smoke test** (synthetic data → CWT → model → 1 epoch training):
   ```bash
   python -m experiments.run_experiment --config experiments/configs/ground_baseline.yaml
   ```

   Expected output:
   ```
   HTS-CWT-AE-GW Smoke Test (synthetic data only)
   Generating 16 synthetic samples (T=4096, fs=1024)
   CWT preprocessing done: shape=(16, 8, 4096)
   Batch 1: loss=...
   Epoch 1 complete. Average loss: ...
   Smoke test complete.
   ```

3. **Verify imports**:
   ```bash
   python -c "import sys; sys.path.insert(0,'src'); import models; import preprocessing; import training; print('OK')"
   ```

## Project Structure

```
hts-cwt-ae-gw/
├── README.md
├── requirements.txt
├── src/
│   ├── data/synthetic/     # Isolated chirp + noise generators
│   ├── models/             # Registry, ae_base, backbones, cwt_autoencoder
│   ├── preprocessing/      # CWT, whitening, normalization
│   ├── training/           # Trainer scaffolding
│   ├── evaluation/         # Metrics, anomaly detection
│   ├── experiments/        # run_experiment, configs
│   └── utils/              # seed, io
├── experiments/
│   ├── configs/
│   │   └── ground_baseline.yaml
│   └── run_experiment.py
└── tests/
```

## Configuration

See `experiments/configs/ground_baseline.yaml` for model, preprocessing, training, and synthetic data parameters. The model registry uses `model.name` and `model.backbone` to instantiate models.

## License

MIT — see [LICENSE](LICENSE).

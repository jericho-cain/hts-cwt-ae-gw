# Contributing to HTS-CWT-AE-GW

Thank you for your interest in contributing! This document provides guidelines for contributing to the synthetic-only HTS/LOSA study.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes following the guidelines below

## Development Environment

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9 or higher
- Required packages listed in `requirements.txt`

### Setup
```bash
# Clone your fork
git clone <repo-url>
cd hts-cwt-ae-gw

# Install dependencies
pip install -r requirements.txt

# Run smoke test
python -m experiments.run_experiment --config experiments/configs/ground_baseline.yaml

# Run tests
python -m pytest tests/ -v
```

## Code Style and Standards

### Python Code
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized

### Testing
- All new code must include appropriate tests
- Run the full test suite before submitting: `python -m pytest tests/ -v`
- Tests should be deterministic and not depend on external services

### Documentation
- Update README.md if you add new features
- Add docstrings for new functions and classes

## Project Structure

```
src/
├── data/synthetic/   # Synthetic signal generators
├── preprocessing/    # CWT, whitening, normalization
├── models/           # Registry, backbones, autoencoders
├── training/         # Trainer scaffolding
├── evaluation/       # Metrics, anomaly detection
├── experiments/      # run_experiment, configs
└── utils/            # seed, io

tests/                # Test suite
experiments/configs/  # Experiment configurations
```

## Submitting Changes

### Pull Request Process
1. Ensure your branch is up to date with the main branch
2. Make sure all tests pass
3. Update documentation as needed
4. Submit a pull request with a clear description of your changes

### Pull Request Guidelines
- Use clear, descriptive commit messages
- Reference any related issues in your PR description
- Ensure your PR description explains what changes were made and why
- Keep PRs focused on a single feature or bugfix

## Areas for Contribution

### High Priority
- Additional test coverage for core modules
- Performance optimizations
- Documentation improvements
- Bug fixes

### Medium Priority
- Additional evaluation metrics
- Support for other detectors (L1, V1)
- Enhanced visualization tools
- Configuration validation improvements

### Research Areas
- Alternative preprocessing methods
- Different model architectures
- Advanced thresholding strategies
- Real-time detection capabilities

## Reporting Issues

When reporting bugs or requesting features:

1. Check existing issues to avoid duplicates
2. Provide clear reproduction steps for bugs
3. Include relevant system information (Python version, OS, etc.)
4. For feature requests, explain the use case and expected behavior

## Code of Conduct

- Be respectful and constructive in all interactions
- Focus on technical merit and scientific accuracy
- Help maintain a welcoming environment for all contributors

## Questions?

If you have questions about contributing, please:
1. Check the documentation in the README
2. Review existing issues and pull requests
3. Open a new issue with your question

Thank you for contributing to gravitational wave detection research!

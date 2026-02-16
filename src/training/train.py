"""
Training entry point and lightweight interface.

Re-exports CWTModelTrainer for compatibility. For synthetic experiments,
use experiments.run_experiment instead.
"""

from .trainer import CWTModelTrainer

__all__ = ["CWTModelTrainer"]

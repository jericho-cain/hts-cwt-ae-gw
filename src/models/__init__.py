"""
Models module for HTS-CWT-AE-GW.

Architecture-agnostic autoencoder models with registry for backbone swapping.
"""

from .cwtlstm import (
    CWT_LSTM_Autoencoder,
    SimpleCWTAutoencoder,
    create_model,
    save_model,
    load_model,
)
from .registry import build_model

__all__ = [
    'CWT_LSTM_Autoencoder',
    'SimpleCWTAutoencoder',
    'create_model',
    'save_model',
    'load_model',
    'build_model',
]

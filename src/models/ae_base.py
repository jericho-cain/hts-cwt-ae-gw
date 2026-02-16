"""
Base class / protocol for CWT autoencoder models.

Architecture-agnostic interface: all backbones must implement
forward, encode, and decode.
"""

from typing import Dict, Any, Tuple, Protocol, runtime_checkable

import torch


@runtime_checkable
class AutoencoderBase(Protocol):
    """Protocol for autoencoder models."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (x_hat, z) - reconstructed input and latent representation
        """
        ...

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        ...

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        ...

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata (params, architecture name, etc.)."""
        ...

"""
Model registry for architecture-agnostic model construction.

Use build_model(cfg) with cfg["model"]["name"] and cfg["model"]["backbone"]
to instantiate models. Easy to add new backbones (e.g., transformer) later.
"""

from typing import Dict, Any

from .cwtlstm import CWT_LSTM_Autoencoder, SimpleCWTAutoencoder


_REGISTRY: Dict[str, type] = {
    "cwt_lstm": CWT_LSTM_Autoencoder,
    "simple_cwt": SimpleCWTAutoencoder,
}


def build_model(cfg: Dict[str, Any]):
    """
    Build model from configuration.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Full config. Uses:
        - cfg["model"]["name"]: "cwt_autoencoder" (main model type)
        - cfg["model"]["backbone"]: "lstm" or "simple" (backbone variant)
        - cfg["model"]["input_height"], input_width, latent_dim, etc.

    Returns
    -------
    nn.Module
        Initialized model
    """
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "cwt_autoencoder")
    backbone = model_cfg.get("backbone", "lstm")

    # Map backbone to registry key
    backbone_key = "cwt_lstm" if backbone == "lstm" else "simple_cwt"
    if backbone_key not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone: {backbone}. Supported: lstm, simple. "
            f"Registry keys: {list(_REGISTRY.keys())}"
        )

    model_cls = _REGISTRY[backbone_key]

    # Build kwargs from config
    kwargs = {
        "input_height": model_cfg.get("input_height", 8),
        "input_width": model_cfg.get("input_width", 4096),
        "latent_dim": model_cfg.get("latent_dim", 32),
    }

    if model_cls == CWT_LSTM_Autoencoder:
        kwargs["lstm_hidden"] = model_cfg.get("lstm_hidden", 64)
        kwargs["dropout"] = model_cfg.get("dropout", 0.0)
    elif model_cls == SimpleCWTAutoencoder:
        kwargs = {
            "height": model_cfg.get("input_height", 8),
            "width": model_cfg.get("input_width", 4096),
            "latent_dim": model_cfg.get("latent_dim", 64),
            "dropout": model_cfg.get("dropout", 0.1),
        }

    return model_cls(**kwargs)

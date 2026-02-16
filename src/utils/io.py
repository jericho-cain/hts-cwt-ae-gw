"""I/O utilities."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save dict to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

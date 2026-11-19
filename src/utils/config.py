"""
Centralized config loading with validation.

All scripts in this repo load configs through this module. Validation is
deliberately strict — a missing or misnamed config field should fail loud
at load time rather than producing silent runtime errors hours into an
expensive cloud job.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a config file is missing required fields or has invalid values."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Raises ConfigError if missing or unparseable."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML in {path}: {e}") from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(cfg).__name__}")
    validate_config(cfg)
    return cfg


REQUIRED_SECTIONS = ("experiment", "huggingface", "corpus", "inference", "topology", "analysis")


def validate_config(cfg: dict[str, Any]) -> None:
    """Check that required top-level sections exist. Raises ConfigError on missing."""
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"Config missing required sections: {missing}")
    # Validate experiment.random_seed is set
    if "random_seed" not in cfg["experiment"]:
        raise ConfigError("Config missing experiment.random_seed (required for reproducibility)")
    # Validate huggingface token is set (allow placeholder during dev)
    token = cfg["huggingface"].get("token", "")
    if not token or token == "hf_...":
        # Allow placeholder during pipeline development; warn loudly at inference time
        pass


def get_hf_token(cfg: dict[str, Any]) -> str:
    """Get Hugging Face token from config or HF_TOKEN env var, with clear error if missing."""
    token = cfg.get("huggingface", {}).get("token", "")
    if not token or token == "hf_...":
        token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise ConfigError(
            "Hugging Face token required. Set huggingface.token in config or HF_TOKEN env var. "
            "Get a token at https://huggingface.co/settings/tokens"
        )
    return token


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it doesn't exist; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

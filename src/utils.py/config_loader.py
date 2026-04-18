"""
config_loader.py
─────────────────
Loads .env and settings.yaml into a single config object.
Access any setting with: cfg.paths.raw_data, cfg.embedding.model_name, etc.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigNode:
    """
    Converts a nested dict into attribute access.
    config["paths"]["raw_data"] → config.paths.raw_data
    """
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def load_config(config_path: str = "config/settings.yaml") -> ConfigNode:
    """
    Load .env first (so env vars are available), then YAML config.
    Environment variables always override YAML values.
    """
    # Step 1: Load .env file into os.environ
    load_dotenv()

    # Step 2: Read YAML
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Did you run from the project root directory?"
        )

    with open(config_file, "r") as f:
        raw_config = yaml.safe_load(f)

    # Step 3: Inject environment-specific overrides
    app_env = os.getenv("APP_ENV", "development")
    raw_config["app_env"] = app_env

    # Resolve log level based on environment
    log_section = raw_config.get("logging", {})
    if app_env == "production":
        log_section["level"] = log_section.get("level_production", "INFO")
    else:
        log_section["level"] = log_section.get("level_development", "DEBUG")

    # Step 4: Attach secrets from environment (never put in YAML)
    raw_config["secrets"] = {
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY_API_KEY", ""),
    }

    return ConfigNode(raw_config)


# Global singleton — import this in other modules
cfg = load_config()
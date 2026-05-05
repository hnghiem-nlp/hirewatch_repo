"""Configuration loading utilities."""

import json
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """Get the project root directory."""
    # Assumes this file is at src/utils/config.py
    return Path(__file__).parent.parent.parent


def load_settings(settings_path: str = None) -> Dict[str, Any]:
    """Load settings from config/settings.json."""
    if settings_path is None:
        settings_path = get_project_root() / "config" / "settings.json"

    with open(settings_path, "r") as f:
        return json.load(f)


def load_secrets(secrets_path: str = None) -> Dict[str, str]:
    """Load key=value pairs from config/secrets."""
    if secrets_path is None:
        secrets_path = get_project_root() / "config" / "secrets"

    secrets = {}
    with open(secrets_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                secrets[key.strip()] = val.strip().strip('"')
    return secrets


def get_root_dir() -> Path:
    """Get root_dir from config/secrets, falling back to project root."""
    secrets = load_secrets()
    return Path(secrets.get("root_dir", str(get_project_root())))


def get_data_path(data_file: str) -> Path:
    """Get full path to a data file."""
    settings = load_settings()
    return get_project_root() / settings["paths"].get(data_file, data_file)

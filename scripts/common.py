"""Common utilities shared across scripts."""

import os
from pathlib import Path
from typing import Dict

import yaml


def load_config(path: str | os.PathLike = "config.yaml") -> Dict:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML config file (default: config.yaml)
        
    Returns:
        Dictionary with configuration, or empty dict if file not found
    """
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg if cfg is not None else {}
    except FileNotFoundError:
        print(f"Warning: Config file not found at {path}")
        return {}


def ensure_sumo_home() -> str:
    """Ensure SUMO_HOME environment variable is set.
    
    Returns:
        Path to SUMO_HOME
        
    Raises:
        RuntimeError: If SUMO_HOME cannot be determined
    """
    if "SUMO_HOME" in os.environ:
        return os.environ["SUMO_HOME"]
    
    # Try common installation paths
    common_paths = [
        "/opt/homebrew/opt/sumo/share/sumo",  # macOS with Homebrew
        "/usr/share/sumo",                     # Linux
        "/opt/sumo/share/sumo",                # Linux alternate
        "/Program Files/SUMO",                 # Windows
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            os.environ["SUMO_HOME"] = path
            return path
    
    raise RuntimeError(
        "SUMO_HOME not set and cannot find SUMO installation.\n"
        "Please set: export SUMO_HOME=/path/to/sumo/share/sumo"
    )


def create_output_dir(path: str | Path = "outputs") -> Path:
    """Create output directory if it doesn't exist.
    
    Args:
        path: Output directory path
        
    Returns:
        Path object to output directory
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

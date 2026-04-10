"""Common utilities shared across scripts."""

import os
from pathlib import Path
from typing import Dict, Optional

import torch
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

    common_paths = [
        "/opt/homebrew/opt/sumo/share/sumo",          # macOS Homebrew
        "/usr/share/sumo",                             # Linux apt
        "/opt/sumo/share/sumo",                        # Linux alternate
        r"C:\Program Files (x86)\Eclipse\Sumo",        # Windows default (MSI)
        r"C:\Program Files\Eclipse\Sumo",              # Windows alternate
    ]

    for path in common_paths:
        if os.path.exists(path):
            os.environ["SUMO_HOME"] = path
            return path

    raise RuntimeError(
        "SUMO_HOME not set and cannot find SUMO installation.\n"
        "Windows: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo\n"
        "Linux/macOS: export SUMO_HOME=/usr/share/sumo"
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


def load_dqn_agent(
    model_path: Path,
    state_dim: int,
    action_dim: int,
    device: Optional[str] = None,
):
    """Load a trained DQN agent from a checkpoint file.

    Handles both raw state_dict checkpoints and wrapped dict checkpoints
    (keys: 'model_state_dict' or 'q_state_dict').

    Args:
        model_path: Path to the .pt checkpoint file.
        state_dim:  Input state dimension of the network.
        action_dim: Number of discrete actions.
        device:     Torch device string ('cpu', 'cuda', ...). Defaults to 'cpu'.

    Returns:
        Loaded DQNAgent in eval mode, or None if loading fails.
    """
    from src.dqn.agent import DQNAgent, AgentConfig  # local import to avoid circular deps

    map_device = device or "cpu"
    try:
        agent_cfg = AgentConfig(state_dim=state_dim, action_dim=action_dim)
        agent = DQNAgent(agent_cfg)

        checkpoint = torch.load(model_path, map_location=map_device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                agent.q.load_state_dict(checkpoint["model_state_dict"])
            elif "q_state_dict" in checkpoint:
                agent.q.load_state_dict(checkpoint["q_state_dict"])
            else:
                agent.q.load_state_dict(checkpoint)
        else:
            agent.q.load_state_dict(checkpoint)

        agent.q.eval()
        print(f"✓ DQN model loaded from {model_path}")
        return agent
    except Exception as e:
        print(f"⚠ Could not load model from {model_path}: {e}")
        return None

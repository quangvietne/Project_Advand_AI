#!/usr/bin/env python3
"""Direct SUMO traffic simulator launcher - no descriptions, just GUI."""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.common import load_config, ensure_sumo_home, load_dqn_agent

ensure_sumo_home()

import numpy as np
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights
from src.baseline import FixedTimeController, FixedTimeConfig


def run_simulation(mode: str = "dqn", steps: int = 3600) -> None:
    """Run SUMO simulation with GUI visualization.

    Args:
        mode:  'dqn' | 'baseline' | 'fixed' | 'demo'
        steps: Maximum simulation steps (seconds).
    """
    config = load_config()
    sumo_cfg = config.get("sumo", {})
    vn_cfg = config.get("vn_weights", {})

    _action_dur = sumo_cfg.get("action_duration", 5)
    env_config = EnvConfig(
        sumocfg_path=sumo_cfg.get("sumocfg_path", "data/scenarios/hn_sample/config.sumocfg"),
        tls_id=sumo_cfg.get("tls_id", "c"),
        phases=sumo_cfg.get("phases", [0, 1, 2, 3]),
        action_duration=_action_dur,
        min_phase_steps=max(1, sumo_cfg.get("min_phase_duration", 30) // _action_dur),
        max_phase_steps=max(2, sumo_cfg.get("max_phase_duration", 120) // _action_dur),
        max_steps=steps,
        gui=True,  # GUI mode is the purpose of this script
        vn_weights=VNWeights(
            motorcycle=vn_cfg.get("motorcycle", 0.5),
            car=vn_cfg.get("car", 1.5),
            bus=vn_cfg.get("bus", 2.0),
            truck=vn_cfg.get("truck", 2.0),
        ),
    )

    print(f"\nStarting SUMO Traffic Simulator...")
    print(f"Mode: {mode} | Steps: {steps} | Config: {env_config.sumocfg_path}")

    env = SumoMDPEnv(env_config)
    state = env.reset()
    action_dim = env.action_dim

    # Load DQN agent if requested
    agent = None
    if mode == "dqn":
        model_path = Path("outputs/dqn_vn_tls.pt")
        if model_path.exists():
            agent = load_dqn_agent(model_path, env.state_dim, action_dim)
        if agent is None:
            print("Warning: DQN model not found or failed to load, falling back to random.")

    # Build fixed-time phase sequence for baseline/fixed modes
    phase_seq: list = []
    if mode in ("fixed", "baseline"):
        green_steps = max(1, 55 // _action_dur)
        yellow_steps = max(1, 5 // _action_dur)
        phase_seq = (
            [0] * green_steps + [1] * yellow_steps +
            [2] * green_steps + [3] * yellow_steps
        )

    print("\nRunning simulation...\n")
    total_reward = 0.0
    step = 0
    for step in range(steps):
        if mode == "dqn" and agent is not None:
            action = agent.act(state, eps=0.0)
        elif phase_seq:
            action = phase_seq[step % len(phase_seq)]
        else:
            action = np.random.randint(action_dim)

        state, reward, done, info = env.step(action)
        total_reward += reward

        # Small delay so the GUI renders at a visible speed
        time.sleep(0.05)

        if (step + 1) % 500 == 0:
            print(f"Step {step + 1}/{steps} | Reward: {reward:.2f} | Total: {total_reward:.2f}")

        if done:
            break

    print(f"\nSimulation completed!")
    print(f"Total steps : {step + 1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Avg reward  : {total_reward / (step + 1):.2f}")

    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    run_simulation(mode, steps)

#!/usr/bin/env python3
"""Direct SUMO traffic simulator launcher - no descriptions, just GUI."""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set SUMO_HOME before importing
from scripts.common import load_config, ensure_sumo_home

ensure_sumo_home()

import time
import yaml
import torch
import numpy as np
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights
from src.dqn.agent import DQNAgent, AgentConfig
from src.baseline import FixedTimeController, FixedTimeConfig


def run_simulation(mode: str = "dqn", steps: int = 3600) -> None:
    """Run SUMO simulation with GUI visualization."""
    config = load_config()
    
    # Enable GUI mode for realistic 2D visualization
    gui_mode = True
    
    # Extract config values with safe defaults
    sumo_cfg = config.get('sumo', {})
    vn_cfg = config.get('vn_weights', {})
    
    # Create environment with GUI visualization
    _action_dur = sumo_cfg.get('action_duration', 5)
    env_config = EnvConfig(
        sumocfg_path=sumo_cfg.get('sumocfg_path', 'data/scenarios/hn_sample/config.sumocfg'),
        tls_id=sumo_cfg.get('tls_id', 'c'),
        phases=sumo_cfg.get('phases', [0, 1, 2, 3]),
        action_duration=_action_dur,
        min_phase_steps=max(1, sumo_cfg.get('min_phase_duration', 30) // _action_dur),
        max_phase_steps=max(2, sumo_cfg.get('max_phase_duration', 120) // _action_dur),
        max_steps=steps,
        gui=gui_mode,
        vn_weights=VNWeights(
            motorcycle=vn_cfg.get('motorcycle', 0.5),
            car=vn_cfg.get('car', 1.5),
            bus=vn_cfg.get('bus', 2.0),
            truck=vn_cfg.get('truck', 2.0),
        ),
    )
    
    print("\n🚦 Starting SUMO Traffic Simulator...")
    print(f"Mode: {mode}")
    print(f"Steps: {steps}")
    print(f"Config: {env_config.sumocfg_path}")
    
    env = SumoMDPEnv(env_config)
    state = env.reset()
    state_dim = state.shape[0] if isinstance(state, np.ndarray) else len(state)
    action_dim = len(sumo_cfg.get('phases', [0, 1, 2, 3]))
    
    # Load model if DQN mode and model exists
    agent: DQNAgent | None = None
    if mode == "dqn":
        model_path = Path("outputs/dqn_vn_tls.pt")
        if model_path.exists():
            try:
                agent_cfg = AgentConfig(state_dim=state_dim, action_dim=action_dim)
                agent = DQNAgent(agent_cfg)
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    agent.q.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent.q.load_state_dict(checkpoint)
                agent.q.eval()
            except Exception:
                agent = None
    
    # Build phase sequence for fixed-time / baseline mode
    # SUMO phases: 0=NS-green, 1=NS-yellow, 2=EW-green, 3=EW-yellow
    # action_duration=5s → green 55s=11 steps, yellow 5s=1 step
    phase_seq: list = []
    if mode in ("fixed", "baseline"):
        action_dur = sumo_cfg.get('action_duration', 5)
        green_steps = max(1, 55 // action_dur)   # 11 steps × 5s = 55s xanh
        yellow_steps = max(1, 5 // action_dur)    # 1 step  × 5s = 5s  vàng → đỏ = 60s
        phase_seq = (
            [0] * green_steps + [1] * yellow_steps +   # Bắc-Nam: xanh → vàng
            [2] * green_steps + [3] * yellow_steps      # Đông-Tây: xanh → vàng
        )

    # Run simulation
    print("\n⏱️  Running simulation...\n")
    total_reward = 0
    step = 0
    for step in range(steps):
        if mode == "demo":
            # Random action
            action = np.random.randint(action_dim)
        elif mode == "dqn" and agent is not None:
            # DQN action
            action = agent.act(state, eps=0.0)
        elif phase_seq:
            # Fixed-time: cycle through pre-computed phase sequence
            action = phase_seq[step % len(phase_seq)]
        else:
            # Fallback to random
            action = np.random.randint(action_dim)
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Small delay to allow GUI to render (50ms per step = realistic speed)
        time.sleep(0.05)
        
        # Progress update every 500 steps
        if (step + 1) % 500 == 0:
            print(f"Step {step + 1}/{steps} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
        
        if done:
            break
    
    print(f"\n✅ Simulation completed!")
    print(f"Total steps: {step + 1}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward/(step+1):.2f}")
    
    env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dqn"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
    run_simulation(mode, steps)

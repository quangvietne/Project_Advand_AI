"""Test the SUMO environment with random actions."""
from __future__ import annotations

import os
import sys

import numpy as np

from src.env.sumo_env import SumoMDPEnv, EnvConfig, VNWeights


def test_env(gui: bool = False, steps: int = 100) -> None:
    """Run a quick test of the environment with random actions."""
    
    # Check SUMO_HOME
    if "SUMO_HOME" not in os.environ:
        print("❌ SUMO_HOME not set!")
        print("Set it with: export SUMO_HOME=/path/to/sumo/share/sumo")
        sys.exit(1)
    
    scenario_path = "data/scenarios/hn_sample/config.sumocfg"
    if not os.path.exists(scenario_path):
        print(f"❌ Scenario not found: {scenario_path}")
        print("Run: python src/utils/generate_scenario.py")
        sys.exit(1)
    
    cfg = EnvConfig(
        sumocfg_path=scenario_path,
        tls_id="c",
        phases=[0, 1, 2, 3],  # Adjust based on your TLS program - int phases
        step_length=1.0,
        action_duration=5,
        min_phase_steps=6,   # 30s min (6 × 5s)
        max_phase_steps=24,  # 120s max (24 × 5s)
        max_steps=steps,
        warmup_steps=10,
        gui=gui,
        vn_weights=VNWeights(),
        reward_type="queue_delay",
    )
    
    env = SumoMDPEnv(cfg)
    
    try:
        print(f"🚦 Starting SUMO environment test ({'GUI' if gui else 'headless'})...")
        state = env.reset()
        print(f"✓ State dimension: {env.state_dim}")
        print(f"✓ Action dimension: {env.action_dim}")
        print(f"✓ Initial state shape: {state.shape}")
        
        total_reward = 0.0
        for step in range(steps // 5):  # action_duration=5
            action = np.random.randint(env.action_dim)
            s2, r, done, info = env.step(action)
            total_reward += r
            
            if step % 10 == 0:
                print(f"  Step {info['step']}: phase={info['phase']}, reward={r:.2f}, done={done}")
            
            if done:
                print(f"✓ Episode finished at step {info['step']}")
                break
        
        print(f"✓ Total reward: {total_reward:.2f}")
        print("✓ Environment test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    parser.add_argument("--steps", type=int, default=100, help="Max steps")
    args = parser.parse_args()
    
    test_env(gui=args.gui, steps=args.steps)

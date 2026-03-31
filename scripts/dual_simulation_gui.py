#!/usr/bin/env python3
"""
Dual Simulation GUI - Run DQN and Fixed-Time side-by-side with real-time metrics display

Usage:
    python scripts/dual_simulation_gui.py --episodes 1 --update-interval 100
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from scripts.common import load_config, ensure_sumo_home
ensure_sumo_home()

from src.baseline import FixedTimeController, FixedTimeConfig
from src.dqn.agent import DQNAgent, AgentConfig
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights


class SimulationMetrics:
    """Track metrics for a running simulation."""
    
    def __init__(self, name: str, window_size: int = 50):
        self.name = name
        self.window_size = window_size
        self.queues = deque(maxlen=window_size)
        self.speeds = deque(maxlen=window_size)
        self.wait_times = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.vehicles_passed_list = deque(maxlen=window_size)
        
        self.total_reward = 0.0
        self.total_vehicles = 0
        self.step_count = 0
        self.is_running = False
        self.lock = threading.Lock()
        
    def update(self, info: Dict, reward: float):
        """Update metrics from step info."""
        with self.lock:
            self.queues.append(info.get("queue_length", 0))
            self.speeds.append(info.get("avg_speed", 0))
            self.wait_times.append(info.get("avg_wait", 0))
            self.rewards.append(reward)
            self.vehicles_passed_list.append(info.get("vehicles_passed", 0))
            self.total_reward += reward
            self.total_vehicles += info.get("vehicles_passed", 0)
            self.step_count += 1
    
    def get_current_stats(self) -> Dict:
        """Get current statistics."""
        with self.lock:
            return {
                "avg_queue": float(np.mean(self.queues)) if self.queues else 0,
                "max_queue": float(np.max(self.queues)) if self.queues else 0,
                "min_queue": float(np.min(self.queues)) if self.queues else 0,
                "avg_speed": float(np.mean(self.speeds)) if self.speeds else 0,
                "avg_wait": float(np.mean(self.wait_times)) if self.wait_times else 0,
                "total_reward": self.total_reward,
                "total_vehicles": self.total_vehicles,
                "step_count": self.step_count,
            }
    
    def start(self):
        """Mark simulation as running."""
        with self.lock:
            self.is_running = True
    
    def stop(self):
        """Mark simulation as stopped."""
        with self.lock:
            self.is_running = False


def run_dqn_simulation(
    env_config: EnvConfig,
    agent: DQNAgent,
    metrics: SimulationMetrics,
    max_steps: int = 3600,
) -> None:
    """Run DQN simulation in a separate thread."""
    try:
        metrics.start()
        env = SumoMDPEnv(env_config)
        state = env.reset()
        
        step = 0
        while step < max_steps:
            action = agent.act(state, eps=0.0)
            next_state, reward, done, info = env.step(action)
            metrics.update(info, reward)
            state = next_state
            step += 1
            
            if done:
                break
        
        env.close()
    except Exception as e:
        print(f"❌ DQN simulation error: {e}")
    finally:
        metrics.stop()


def run_fixed_time_simulation(
    env_config: EnvConfig,
    metrics: SimulationMetrics,
    max_steps: int = 3600,
) -> None:
    """Run Fixed-Time simulation in a separate thread."""
    try:
        metrics.start()
        env = SumoMDPEnv(env_config)
        state = env.reset()
        
        controller = FixedTimeController(
            FixedTimeConfig(
                green_duration=55,
                yellow_duration=5,
            )
        )
        
        step = 0
        while step < max_steps:
            action = controller.get_action()
            next_state, reward, done, info = env.step(action)
            metrics.update(info, reward)
            state = next_state
            step += 1
            
            if done:
                break
        
        env.close()
    except Exception as e:
        print(f"❌ Fixed-Time simulation error: {e}")
    finally:
        metrics.stop()


def print_metrics_header():
    """Print header for metrics display."""
    print("\n" + "="*140)
    print(f"{'Metric':<25} | {'DQN':<35} | {'Fixed-Time':<35} | {'Difference':<35}")
    print("-"*140)


def print_metrics_row(metric_name: str, dqn_val: float, fixed_val: float, is_better_lower: bool = True):
    """Print a metrics row."""
    if is_better_lower:
        diff = fixed_val - dqn_val
        dqn_color = "✓" if dqn_val < fixed_val else " "
        fixed_color = "✓" if fixed_val < dqn_val else " "
    else:
        diff = dqn_val - fixed_val
        dqn_color = "✓" if dqn_val > fixed_val else " "
        fixed_color = "✓" if fixed_val > dqn_val else " "
    
    print(
        f"{metric_name:<25} | "
        f"{dqn_color} {dqn_val:>12.2f}              | "
        f"{fixed_color} {fixed_val:>12.2f}              | "
        f"{diff:>+12.2f}"
    )


def print_current_metrics(dqn_metrics: SimulationMetrics, fixed_metrics: SimulationMetrics):
    """Print current metrics side-by-side."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("\n" + "="*140)
    print(f"{'DQN vs Fixed-Time Real-Time Comparison':^140}")
    print("="*140)
    
    dqn_stats = dqn_metrics.get_current_stats()
    fixed_stats = fixed_metrics.get_current_stats()
    
    # Status
    dqn_status = "🟢 RUNNING" if dqn_metrics.is_running else "🔴 STOPPED"
    fixed_status = "🟢 RUNNING" if fixed_metrics.is_running else "🔴 STOPPED"
    
    print(f"\n{dqn_status:<35} {fixed_status:<35}")
    print(f"Step: {dqn_stats['step_count']:<30} Step: {fixed_stats['step_count']:<30}")
    
    print_metrics_header()
    
    # Key metrics (lower is better)
    print_metrics_row("Avg Queue Length (vehicles)", dqn_stats['avg_queue'], fixed_stats['avg_queue'], is_better_lower=True)
    print_metrics_row("Max Queue Length", dqn_stats['max_queue'], fixed_stats['max_queue'], is_better_lower=True)
    print_metrics_row("Avg Wait Time (s)", dqn_stats['avg_wait'], fixed_stats['avg_wait'], is_better_lower=True)
    
    # Key metrics (higher is better)
    print_metrics_row("Avg Speed (km/h)", dqn_stats['avg_speed'], fixed_stats['avg_speed'], is_better_lower=False)
    print_metrics_row("Total Vehicles Passed", dqn_stats['total_vehicles'], fixed_stats['total_vehicles'], is_better_lower=False)
    print_metrics_row("Total Reward", dqn_stats['total_reward'], fixed_stats['total_reward'], is_better_lower=False)
    
    print("\n" + "="*140)
    print(f"{'Lower is better for: Queue, Wait Time | Higher is better for: Speed, Vehicles, Reward':^140}")
    print("="*140)
    
    # Summary
    print(f"\n{'Summary':<25} | {'DQN':<35} | {'Fixed-Time':<35}")
    print("-"*95)
    print(f"{'Total Steps':<25} | {dqn_stats['step_count']:<35} | {fixed_stats['step_count']:<35}")
    
    # Winner
    dqn_score = 0
    fixed_score = 0
    
    if dqn_stats['avg_queue'] < fixed_stats['avg_queue']: dqn_score += 1
    else: fixed_score += 1
    
    if dqn_stats['avg_speed'] > fixed_stats['avg_speed']: dqn_score += 1
    else: fixed_score += 1
    
    if dqn_stats['total_vehicles'] > fixed_stats['total_vehicles']: dqn_score += 1
    else: fixed_score += 1
    
    winner = "🏆 DQN" if dqn_score > fixed_score else "🏆 Fixed-Time" if fixed_score > dqn_score else "🤝 Tie"
    print(f"\n{'Current Winner':<25} | {winner:>35}")
    print("="*95 + "\n")


def run_dual_comparison(
    env_config: EnvConfig,
    model_path: Optional[Path] = None,
    num_episodes: int = 1,
    update_interval: int = 100,
) -> None:
    """Run dual simulation with real-time metrics display.
    
    Args:
        env_config: Environment configuration
        model_path: Path to trained DQN model
        num_episodes: Number of episodes to run
        update_interval: Steps between metric updates (in milliseconds)
    """
    print("\n" + "="*140)
    print(f"{'Dual Simulation - DQN vs Fixed-Time':^140}")
    print("="*140)
    
    # Load DQN agent
    agent = None
    if model_path and model_path.exists():
        try:
            print(f"\n📦 Loading DQN model from {model_path}...")
            num_phases = len(env_config.phases)
            state_dim = num_phases * 4
            
            agent_cfg = AgentConfig(state_dim=state_dim, action_dim=num_phases)
            agent = DQNAgent(agent_cfg)
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    agent.q.load_state_dict(checkpoint['model_state_dict'])
                elif 'q_state_dict' in checkpoint:
                    agent.q.load_state_dict(checkpoint['q_state_dict'])
                else:
                    agent.q.load_state_dict(checkpoint)
            else:
                agent.q.load_state_dict(checkpoint)
            agent.q.eval()
            print(f"✓ Model loaded successfully\n")
        except Exception as e:
            print(f"⚠ Could not load model: {e}\n")
            agent = None
    
    if agent is None:
        print("❌ No DQN model available. Cannot run dual comparison.\n")
        return
    
    # Run episodes
    for ep in range(num_episodes):
        print(f"\n{'='*140}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*140}\n")
        
        # Create metrics objects
        dqn_metrics = SimulationMetrics("DQN")
        fixed_metrics = SimulationMetrics("Fixed-Time")
        
        # Start simulations in separate threads
        print("🚀 Starting simulations...\n")
        dqn_thread = threading.Thread(
            target=run_dqn_simulation,
            args=(env_config, agent, dqn_metrics, env_config.max_steps),
            daemon=True
        )
        fixed_thread = threading.Thread(
            target=run_fixed_time_simulation,
            args=(env_config, fixed_metrics, env_config.max_steps),
            daemon=True
        )
        
        dqn_thread.start()
        fixed_thread.start()
        
        # Monitor metrics while simulations run
        update_count = 0
        while dqn_metrics.is_running or fixed_metrics.is_running:
            try:
                if update_count % 5 == 0:  # Update display every 5 cycles
                    print_current_metrics(dqn_metrics, fixed_metrics)
                
                time.sleep(update_interval / 1000.0)
                update_count += 1
            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                break
        
        # Wait for threads to finish
        dqn_thread.join(timeout=10)
        fixed_thread.join(timeout=10)
        
        # Final metrics
        print("\n" + "="*140)
        print("EPISODE COMPLETE")
        print("="*140)
        print_current_metrics(dqn_metrics, fixed_metrics)
    
    print("✓ All episodes completed!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dual simulation with real-time metrics display"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="data/scenarios/hn_sample/config.sumocfg",
        help="Path to SUMO scenario config",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained DQN model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=100,
        help="Update interval in milliseconds",
    )
    
    args = parser.parse_args()
    
    # Create environment config
    env_config = EnvConfig(
        sumocfg_path=args.scenario,
        tls_id="c",
        phases=[0, 1, 2, 3],
        action_duration=5,
        min_phase_steps=6,   # 30s min (6 × 5s)
        max_phase_steps=24,  # 120s max (24 × 5s)
        max_steps=3600,
        warmup_steps=60,
        gui=False,
        vn_weights=VNWeights(motorcycle=0.5, car=1.5, bus=2.0, truck=2.0),
    )
    
    # Run dual comparison
    run_dual_comparison(env_config, args.model_path, args.episodes, args.update_interval)


if __name__ == "__main__":
    main()

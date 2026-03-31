#!/usr/bin/env python3
"""
Comparison script: DQN Agent vs Fixed-Time Controller

Evaluates both strategies on the same traffic scenario and generates
comparison metrics and plots.

Usage:
    python scripts/compare_strategies.py --model-path outputs/model.pt --num-episodes 5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from scripts.common import load_config, ensure_sumo_home

# Ensure SUMO is available
ensure_sumo_home()

from src.baseline import FixedTimeController, FixedTimeConfig
from src.dqn.agent import DQNAgent, AgentConfig
from src.dqn.model import DQN
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights


class ComparisonRunner:
    """Runs DQN and Fixed-Time strategies and collects metrics."""

    def __init__(
        self,
        env_config: EnvConfig,
        num_episodes: int = 20,
        model_path: Optional[Path] = None,
    ):
        """Initialize comparison runner.
        
        Args:
            env_config: SUMO environment configuration
            num_episodes: Number of episodes to run
            model_path: Path to trained DQN model (optional)
        """
        self.env_config = env_config
        self.num_episodes = num_episodes
        self.model_path = model_path

        # Storage for metrics (expanded with additional SUMO metrics)
        self.dqn_metrics: Dict[str, List[float]] = {
            "total_wait": [],
            "avg_queue": [],
            "max_queue": [],
            "throughput": [],
            "episode_reward": [],
            "avg_speed": [],
            "avg_occupancy": [],
            "avg_halting_vehicles": [],
        }
        self.fixed_metrics: Dict[str, List[float]] = {
            "total_wait": [],
            "avg_queue": [],
            "max_queue": [],
            "throughput": [],
            "episode_reward": [],
            "avg_speed": [],
            "avg_occupancy": [],
            "avg_halting_vehicles": [],
        }

    def run_dqn_episode(self, agent: DQNAgent, episode: int = 0) -> Dict:
        """Run one episode with DQN agent.
        
        Args:
            agent: Trained DQN agent
            episode: Episode number (for seed)
            
        Returns:
            Dictionary with metrics
        """
        env = SumoMDPEnv(self.env_config)
        state = env.reset()

        total_reward = 0.0
        queues = []
        speeds = []
        occupancies = []
        halting_vehicles_list = []
        vehicles_passed_total = 0

        while True:
            # DQN action (greedy, no exploration)
            action = agent.act(state, eps=0.0)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            queue_length = info.get("queue_length", 0.0)
            queues.append(queue_length)
            speeds.append(info.get("avg_speed", 0.0))
            occupancies.append(info.get("occupancy", 0.0))
            halting_vehicles_list.append(info.get("halting_vehicles", 0))
            vehicles_passed_total += info.get("vehicles_passed", 0)

            state = next_state
            if done:
                break

        env.close()

        metrics = {
            "total_wait": -total_reward if self.env_config.reward_type == "queue_delay" else 0.0,
            "avg_queue": float(np.mean(queues)) if queues else 0.0,
            "max_queue": float(np.max(queues)) if queues else 0.0,
            "throughput": float(vehicles_passed_total),
            "episode_reward": float(total_reward),
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "avg_occupancy": float(np.mean(occupancies)) if occupancies else 0.0,
            "avg_halting_vehicles": float(np.mean(halting_vehicles_list)) if halting_vehicles_list else 0.0,
        }
        return metrics

    def run_fixed_time_episode(self, episode: int = 0) -> Dict:
        """Run one episode with fixed-time controller.
        
        Args:
            episode: Episode number (for seed)
            
        Returns:
            Dictionary with metrics
        """
        env = SumoMDPEnv(self.env_config)
        state = env.reset()

        controller = FixedTimeController(
            FixedTimeConfig(
                green_duration=55,
                yellow_duration=5,
            )
        )

        total_reward = 0.0
        queues = []
        speeds = []
        occupancies = []
        halting_vehicles_list = []
        vehicles_passed_total = 0

        while True:
            # Fixed-time action
            action = controller.get_action()
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            queue_length = info.get("queue_length", 0.0)
            queues.append(queue_length)
            speeds.append(info.get("avg_speed", 0.0))
            occupancies.append(info.get("occupancy", 0.0))
            halting_vehicles_list.append(info.get("halting_vehicles", 0))
            vehicles_passed_total += info.get("vehicles_passed", 0)

            state = next_state
            if done:
                break

        env.close()

        metrics = {
            "total_wait": -total_reward if self.env_config.reward_type == "queue_delay" else 0.0,
            "avg_queue": float(np.mean(queues)) if queues else 0.0,
            "max_queue": float(np.max(queues)) if queues else 0.0,
            "throughput": float(vehicles_passed_total),
            "episode_reward": float(total_reward),
            "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
            "avg_occupancy": float(np.mean(occupancies)) if occupancies else 0.0,
            "avg_halting_vehicles": float(np.mean(halting_vehicles_list)) if halting_vehicles_list else 0.0,
        }
        return metrics

    def run_comparison(self, agent: Optional[DQNAgent] = None) -> None:
        """Run full comparison and print results.
        
        Args:
            agent: Optional trained DQN agent. If None, will try to load from model_path.
        """
        # Load agent if not provided
        if agent is None:
            if self.model_path and self.model_path.exists():
                print(f"Loading model from {self.model_path}")
                # Load your model (adjust as needed for your architecture)
                try:
                    num_phases = len(self.env_config.phases)
                    # Detect actual state_dim from environment instead of hardcoding
                    _tmp_env = SumoMDPEnv(self.env_config)
                    _tmp_state = _tmp_env.reset()
                    actual_state_dim = _tmp_state.shape[0]
                    _tmp_env.close()
                    model = DQN(
                        state_dim=actual_state_dim,
                        action_dim=num_phases,
                    )
                    # Load weights
                    agent_cfg = AgentConfig(
                        state_dim=actual_state_dim,
                        action_dim=num_phases,
                    )
                    agent = DQNAgent(agent_cfg)
                    # Load trained weights into agent (train.py saves q.state_dict() directly)
                    checkpoint = torch.load(self.model_path, map_location=agent.device)
                    if isinstance(checkpoint, dict) and "q_state_dict" in checkpoint:
                        agent.q.load_state_dict(checkpoint["q_state_dict"])
                    else:
                        agent.q.load_state_dict(checkpoint)
                    agent.q.eval()
                except Exception as e:
                    print(f"Could not load model: {e}. Running fixed-time only.")
                    agent = None
            else:
                print("No model path provided. Running fixed-time only.")
                agent = None

        print(f"\n{'='*60}")
        print(f"COMPARISON: DQN vs Fixed-Time Controller")
        print(f"{'='*60}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Scenario: {self.env_config.sumocfg_path}")
        print()

        # Run DQN episodes
        if agent:
            print("Running DQN episodes...")
            for ep in tqdm(range(self.num_episodes), desc="DQN"):
                metrics = self.run_dqn_episode(agent, ep)
                for key, val in metrics.items():
                    self.dqn_metrics[key].append(val)

        # Run Fixed-Time episodes
        print("\nRunning Fixed-Time episodes...")
        for ep in tqdm(range(self.num_episodes), desc="Fixed-Time"):
            metrics = self.run_fixed_time_episode(ep)
            for key, val in metrics.items():
                self.fixed_metrics[key].append(val)

        # Print results
        self._print_results()

    def _print_results(self) -> None:
        """Print comparison results table."""
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}\n")

        metrics_to_show = [
            "avg_queue", 
            "max_queue", 
            "avg_speed", 
            "avg_occupancy",
            "avg_halting_vehicles",
            "total_wait", 
            "throughput",
            "episode_reward"
        ]

        print(f"{'Metric':<28} | {'DQN (mean ± std)':<20} | {'Fixed-Time (mean ± std)':<20} | {'Improvement':<12}")
        print("-" * 85)

        for metric in metrics_to_show:
            dqn_vals = self.dqn_metrics.get(metric, [])
            fixed_vals = self.fixed_metrics.get(metric, [])

            if dqn_vals and fixed_vals:
                dqn_mean = np.mean(dqn_vals)
                dqn_std = np.std(dqn_vals)
                fixed_mean = np.mean(fixed_vals)
                fixed_std = np.std(fixed_vals)

                # Calculate improvement (better is lower for wait/queue, higher for speed/reward/throughput)
                if metric in ["episode_reward", "throughput", "avg_speed"]:
                    improvement_pct = ((dqn_mean - fixed_mean) / abs(fixed_mean) * 100) if fixed_mean != 0 else 0
                else:
                    improvement_pct = ((fixed_mean - dqn_mean) / fixed_mean * 100) if fixed_mean != 0 else 0

                metric_name = metric.replace("_", " ").title()
                print(
                    f"{metric_name:<28} | "
                    f"{dqn_mean:>6.2f}±{dqn_std:<6.2f}   | "
                    f"{fixed_mean:>6.2f}±{fixed_std:<6.2f}   | "
                    f"{improvement_pct:>+6.1f}%"
                )
            elif fixed_vals:
                fixed_mean = np.mean(fixed_vals)
                fixed_std = np.std(fixed_vals)
                print(
                    f"{metric.replace('_', ' ').title():<28} | "
                    f"{'N/A':<20} | "
                    f"{fixed_mean:>6.2f}±{fixed_std:<6.2f}   | "
                    f"{'N/A':<12}"
                )

        print(f"\n{'='*60}")
        print("INTERPRETATION:")
        print("  - Lower Avg Queue = better traffic flow")
        print("  - Lower Max Queue = prevents bottlenecks")
        print("  - Higher Avg Speed = better traffic efficiency")
        print("  - Lower Occupancy = less congestion")
        print("  - Lower Halting Vehicles = fewer stops")
        print("  - Lower Total Wait = better for drivers")
        print("  - Higher Throughput = more vehicles passed")
        print("  - Higher Reward = better learning performance")
        print(f"{'='*60}\n")

    def save_results(self, output_dir: Path = Path("outputs")) -> None:
        """Save comparison results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as text
        with open(output_dir / "comparison_results.txt", "w") as f:
            f.write("DQN Metrics:\n")
            for key, vals in self.dqn_metrics.items():
                if vals:
                    f.write(f"  {key}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}\n")

            f.write("\nFixed-Time Metrics:\n")
            for key, vals in self.fixed_metrics.items():
                if vals:
                    f.write(f"  {key}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}\n")

        # Save raw data as numpy
        np.savez(
            output_dir / "comparison_metrics.npz",
            dqn_metrics=np.array([self.dqn_metrics[k] for k in sorted(self.dqn_metrics.keys())], dtype=object),
            fixed_metrics=np.array([self.fixed_metrics[k] for k in sorted(self.fixed_metrics.keys())], dtype=object),
        )

        print(f"Results saved to {output_dir}")


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare DQN vs Fixed-Time traffic light control"
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
        "--num-episodes",
        type=int,
        default=20,
        help="Number of episodes per strategy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without SUMO GUI",
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
        gui=not args.no_gui,
        vn_weights=VNWeights(motorcycle=0.5, car=1.5, bus=2.0, truck=2.0),
    )

    # Run comparison
    runner = ComparisonRunner(env_config, args.num_episodes, args.model_path)
    runner.run_comparison()
    runner.save_results(args.output_dir)


if __name__ == "__main__":
    main()

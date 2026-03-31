#!/usr/bin/env python3
"""
Side-by-side comparison of DQN vs Fixed-Time strategies
Collects metrics and displays them in parallel tables.

Usage:
    python scripts/parallel_comparison.py --model-path outputs/dqn_vn_tls.pt --episodes 1
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from scripts.common import load_config, ensure_sumo_home
ensure_sumo_home()

from src.baseline import FixedTimeController, FixedTimeConfig
from src.dqn.agent import DQNAgent, AgentConfig
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights


class EpisodeMetrics:
    """Track comprehensive metrics for an episode."""
    
    def __init__(self):
        self.steps = []
        self.queues = []
        self.speeds = []
        self.wait_times = []
        self.rewards = []
        self.vehicles_passed = []
        self.phases = []
        self.total_steps = 0
        self.total_reward = 0.0
        
    def add_step(self, info: Dict, reward: float, phase: int):
        """Record a step's metrics."""
        self.steps.append(info.get("step", 0))
        self.queues.append(info.get("queue_length", 0))
        self.speeds.append(info.get("avg_speed", 0))
        self.wait_times.append(info.get("avg_wait", 0))
        self.vehicles_passed.append(info.get("vehicles_passed", 0))
        self.rewards.append(reward)
        self.phases.append(phase)
        self.total_reward += reward
        self.total_steps += 1
        
    def summarize(self) -> Dict:
        """Get summary statistics."""
        return {
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "total_vehicles": sum(self.vehicles_passed),
            "avg_queue": np.mean(self.queues) if self.queues else 0,
            "max_queue": np.max(self.queues) if self.queues else 0,
            "min_queue": np.min(self.queues) if self.queues else 0,
            "avg_speed": np.mean(self.speeds) if self.speeds else 0,
            "max_speed": np.max(self.speeds) if self.speeds else 0,
            "min_speed": np.min(self.speeds) if self.speeds else 0,
            "avg_wait": np.mean(self.wait_times) if self.wait_times else 0,
            "max_wait": np.max(self.wait_times) if self.wait_times else 0,
            "std_queue": np.std(self.queues) if self.queues else 0,
            "std_speed": np.std(self.speeds) if self.speeds else 0,
        }


def run_episode(
    env_config: EnvConfig,
    agent: Optional[DQNAgent] = None,
    verbose: bool = False,
) -> EpisodeMetrics:
    """Run a single episode and collect metrics.
    
    Args:
        env_config: Environment configuration
        agent: DQN agent (if None, uses fixed-time)
        verbose: Print progress
        
    Returns:
        EpisodeMetrics object with collected data
    """
    env = SumoMDPEnv(env_config)
    state = env.reset()
    
    metrics = EpisodeMetrics()
    
    # Initialize controller if using fixed-time
    if agent is None:
        controller = FixedTimeController(
            FixedTimeConfig(
                green_duration=55,
                yellow_duration=5,
            )
        )
        strategy_name = "Fixed-Time"
    else:
        strategy_name = "DQN"
    
    step_count = 0
    while True:
        # Get action
        if agent is not None:
            action = agent.act(state, eps=0.0)
        else:
            action = controller.get_action()
        
        # Execute step
        next_state, reward, done, info = env.step(action)
        
        # Record metrics
        metrics.add_step(info, reward, action)
        step_count += 1
        
        if verbose and step_count % 50 == 0:
            print(f"  [{strategy_name}] Step {step_count}: Queue={info.get('queue_length', 0):.1f}, "
                  f"Speed={info.get('avg_speed', 0):.1f}")
        
        state = next_state
        if done:
            break
    
    env.close()
    return metrics


def format_metric_table(dqn_metrics: Dict, fixed_metrics: Dict) -> str:
    """Format a side-by-side comparison table."""
    table = ""
    table += "\n" + "="*100 + "\n"
    table += f"{'METRIC':<25} {'DQN':<20} {'FIXED-TIME':<20} {'DIFFERENCE':<20} {'WINNER':<10}\n"
    table += "="*100 + "\n"
    
    # Metrics to compare (lower is better for queue/wait, higher for speed/reward)
    comparisons = [
        ("Avg Queue Length", "avg_queue", "lower"),
        ("Max Queue Length", "max_queue", "lower"),
        ("Std Dev Queue", "std_queue", "lower"),
        ("Avg Speed (km/h)", "avg_speed", "higher"),
        ("Max Speed (km/h)", "max_speed", "higher"),
        ("Avg Wait Time", "avg_wait", "lower"),
        ("Total Vehicles Passed", "total_vehicles", "higher"),
        ("Total Reward", "total_reward", "higher"),
        ("Total Steps", "total_steps", "neutral"),
    ]
    
    for display_name, key, comparison_type in comparisons:
        dqn_val = dqn_metrics.get(key, 0)
        fixed_val = fixed_metrics.get(key, 0)
        
        if comparison_type == "neutral":
            diff = dqn_val - fixed_val
            winner = "--"
        elif comparison_type == "lower":
            diff = fixed_val - dqn_val
            if dqn_val < fixed_val:
                winner = "DQN ✓"
            elif dqn_val > fixed_val:
                winner = "Fixed ✓"
            else:
                winner = "TIE"
        else:  # higher
            diff = dqn_val - fixed_val
            if dqn_val > fixed_val:
                winner = "DQN ✓"
            elif dqn_val < fixed_val:
                winner = "Fixed ✓"
            else:
                winner = "TIE"
        
        table += (f"{display_name:<25} {dqn_val:>18.2f}  {fixed_val:>18.2f}  "
                 f"{diff:>18.2f}  {winner:>10}\n")
    
    table += "="*100 + "\n"
    return table


def run_parallel_comparison(
    env_config: EnvConfig,
    model_path: Optional[Path] = None,
    num_episodes: int = 1,
    save_results: bool = True,
) -> None:
    """Run parallel comparison and display results.
    
    Args:
        env_config: Environment configuration
        model_path: Path to trained DQN model
        num_episodes: Number of episodes to run
        save_results: Whether to save results to file
    """
    print("\n" + "="*100)
    print("PARALLEL METRICS COMPARISON: DQN vs Fixed-Time Controller")
    print("="*100)
    print(f"Episodes: {num_episodes}")
    print(f"Scenario: {env_config.sumocfg_path}")
    print(f"Max Steps per Episode: {env_config.max_steps}")
    print("="*100 + "\n")
    
    # Load DQN agent if available
    agent = None
    if model_path and model_path.exists():
        try:
            print(f"Loading DQN model from {model_path}...")
            num_phases = len(env_config.phases)
            state_dim = num_phases * 4
            
            agent_cfg = AgentConfig(state_dim=state_dim, action_dim=num_phases)
            agent = DQNAgent(agent_cfg)
            
            # Load model weights
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                # Try different possible keys for state dict
                if 'model_state_dict' in checkpoint:
                    agent.q.load_state_dict(checkpoint['model_state_dict'])
                elif 'q_state_dict' in checkpoint:
                    agent.q.load_state_dict(checkpoint['q_state_dict'])
                else:
                    # Assume it's directly the state dict
                    agent.q.load_state_dict(checkpoint)
            else:
                agent.q.load_state_dict(checkpoint)
            agent.q.eval()
            print(f"✓ Model loaded successfully\n")
        except Exception as e:
            print(f"⚠ Could not load model: {e}. Running fixed-time only.\n")
            agent = None
    else:
        print("⚠ No model path provided. Running fixed-time only.\n")
    
    # Run episodes and collect results
    all_dqn_results = []
    all_fixed_results = []
    
    print("Running episodes...\n")
    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}:")
        
        # Run DQN
        if agent:
            print("  [DQN] Running...", end="", flush=True)
            dqn_metrics = run_episode(env_config, agent=agent, verbose=False)
            all_dqn_results.append(dqn_metrics.summarize())
            print(f" ✓ (Vehicles: {dqn_metrics.summarize()['total_vehicles']})")
        
        # Run Fixed-Time
        print("  [Fixed-Time] Running...", end="", flush=True)
        fixed_metrics = run_episode(env_config, agent=None, verbose=False)
        all_fixed_results.append(fixed_metrics.summarize())
        print(f" ✓ (Vehicles: {fixed_metrics.summarize()['total_vehicles']})")
        
        # Show episode comparison
        if agent:
            print("\n  Episode Summary:")
            print(format_metric_table(all_dqn_results[-1], all_fixed_results[-1]))
    
    # Calculate averages
    if all_dqn_results:
        avg_dqn = {}
        for key in all_dqn_results[0].keys():
            avg_dqn[key] = np.mean([r[key] for r in all_dqn_results])
    
    avg_fixed = {}
    for key in all_fixed_results[0].keys():
        avg_fixed[key] = np.mean([r[key] for r in all_fixed_results])
    
    # Print final results
    print("\n" + "="*100)
    print(f"FINAL RESULTS (Average over {num_episodes} episodes)")
    print("="*100)
    
    if all_dqn_results:
        print(format_metric_table(avg_dqn, avg_fixed))
    else:
        print("\nFixed-Time Results Only:")
        print("-"*100)
        for key, val in avg_fixed.items():
            print(f"{key:<30} {val:>15.2f}")
        print("-"*100)
    
    # Save results
    if save_results:
        output_dir = Path("outputs/parallel_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "scenario": env_config.sumocfg_path,
            "episodes": num_episodes,
            "dqn_results": all_dqn_results if all_dqn_results else None,
            "fixed_results": all_fixed_results,
        }
        
        # Save as JSON
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save as text
        with open(output_dir / "comparison_results.txt", "w") as f:
            f.write("="*100 + "\n")
            f.write("PARALLEL METRICS COMPARISON\n")
            f.write("="*100 + "\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Scenario: {results['scenario']}\n")
            f.write(f"Episodes: {results['episodes']}\n")
            f.write("="*100 + "\n\n")
            
            if all_dqn_results:
                f.write(format_metric_table(avg_dqn, avg_fixed))
            
            f.write("\nDETAILED RESULTS:\n")
            f.write("-"*100 + "\n")
            for ep, (dqn_result, fixed_result) in enumerate(
                zip(all_dqn_results if all_dqn_results else [None]*len(all_fixed_results), 
                    all_fixed_results)
            ):
                f.write(f"\nEpisode {ep + 1}:\n")
                if dqn_result:
                    f.write(format_metric_table(dqn_result, fixed_result))
        
        print(f"\n✓ Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Side-by-side metrics comparison for traffic signal control"
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
        "--no-save",
        action="store_true",
        help="Don't save results to files",
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
    
    # Run comparison
    run_parallel_comparison(env_config, args.model_path, args.episodes, not args.no_save)


if __name__ == "__main__":
    main()

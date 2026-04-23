#!/usr/bin/env python3
"""
Real-time metrics monitor for comparing DQN vs Fixed-Time strategies
Displays detailed metrics during simulation execution.

Usage:
    python scripts/monitor_metrics.py --model-path outputs/dqn_vn_tls.pt --episodes 2
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm import tqdm

from scripts.common import load_config, ensure_sumo_home
ensure_sumo_home()

from src.baseline import FixedTimeController, FixedTimeConfig
from src.dqn.agent import DQNAgent, AgentConfig
from src.dqn.model import DQN
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MetricsCollector:
    """Collect and format metrics during episode execution."""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.metrics = {
            "queue_length": deque(maxlen=100),
            "avg_speed": deque(maxlen=100),
            "wait_time": deque(maxlen=100),
            "vehicles_passed": deque(maxlen=100),
            "halting_vehicles": deque(maxlen=100),
            "occupancy": deque(maxlen=100),
            "reward": deque(maxlen=100),
        }
        self.episode_stats = {}
        
    def update(self, info: Dict, reward: float = 0.0):
        """Update metrics from step info."""
        self.metrics["queue_length"].append(info.get("queue_length", 0))
        self.metrics["avg_speed"].append(info.get("avg_speed", 0))
        self.metrics["wait_time"].append(info.get("avg_wait", 0))
        self.metrics["vehicles_passed"].append(info.get("vehicles_passed", 0))
        self.metrics["reward"].append(reward)
        
    def get_summary(self) -> Dict:
        """Get aggregated metrics summary."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_max"] = float(np.max(values))
                summary[f"{key}_min"] = float(np.min(values))
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}=== {self.strategy_name} Summary ==={Colors.ENDC}")
        print(f"{'Metric':<30} {'Mean':<12} {'Min':<12} {'Max':<12}")
        print("-" * 66)
        
        for key, val in sorted(summary.items()):
            if "_mean" in key:
                metric_name = key.replace("_mean", "").replace("_", " ").title()
                mean = summary.get(key, 0)
                min_val = summary.get(key.replace("_mean", "_min"), 0)
                max_val = summary.get(key.replace("_mean", "_max"), 0)
                print(f"{metric_name:<30} {mean:>10.2f}  {min_val:>10.2f}  {max_val:>10.2f}")


def run_episode_with_monitoring(
    env_config: EnvConfig,
    agent: Optional[DQNAgent] = None,
    strategy_name: str = "Unknown",
    verbose: bool = True,
    phase_schedule: Optional[List] = None,
) -> Dict:
    """Run a single episode with detailed metrics monitoring.
    
    Args:
        env_config: Environment configuration
        agent: DQN agent (if None, uses fixed-time)
        strategy_name: Name of strategy for display
        verbose: Print real-time metrics
        
    Returns:
        Dictionary with episode metrics
    """
    env = SumoMDPEnv(env_config)
    state = env.reset()
    
    collector = MetricsCollector(strategy_name)
    total_reward = 0.0
    step_count = 0
    
    # Initialize controller if using fixed-time
    if agent is None:
        # Dùng 160s schedule bất đối xứng (EW:NS = 2:1) từ config, khớp với báo cáo
        _schedule = phase_schedule or [(2, 100), (3, 5), (0, 50), (1, 5)]
        controller = FixedTimeController(
            FixedTimeConfig(
                phase_schedule=[tuple(p) for p in _schedule],
                action_duration=env_config.action_duration,
            )
        )
    
    if verbose:
        print(f"\n{Colors.BOLD}{Colors.OKBLUE}Starting {strategy_name} Episode{Colors.ENDC}")
        print(f"{'Step':<8} {'Phase':<8} {'Queue':<10} {'Speed':<10} {'Wait':<10} {'Veh Pass':<12} {'Reward':<12}")
        print("-" * 80)
    
    while True:
        # Get action
        if agent is not None:
            action = agent.act(state, eps=0.0)
        else:
            action = controller.get_action()
        
        # Execute step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Collect metrics
        collector.update(info, reward)
        
        # Print real-time metrics (every 5 steps)
        if verbose and step_count % 5 == 0:
            queue = info.get("queue_length", 0)
            speed = info.get("avg_speed", 0)
            wait = info.get("avg_wait", 0)
            veh_pass = info.get("vehicles_passed", 0)
            
            # Color coding for queue length
            if queue < 5:
                queue_color = Colors.OKGREEN
            elif queue < 15:
                queue_color = Colors.WARNING
            else:
                queue_color = Colors.FAIL
            
            print(
                f"{step_count:<8} {action:<8} "
                f"{queue_color}{queue:>8.1f}{Colors.ENDC}  "
                f"{speed:>8.1f}  "
                f"{wait:>8.1f}  "
                f"{veh_pass:>10}  "
                f"{reward:>10.1f}"
            )
        
        state = next_state
        if done:
            break
    
    env.close()
    
    # Print summary
    if verbose:
        collector.print_summary()
    
    # Return final metrics
    summary = collector.get_summary()
    summary["total_reward"] = total_reward
    summary["total_steps"] = step_count
    summary["total_vehicles_passed"] = sum(collector.metrics["vehicles_passed"])
    summary["avg_queue"] = summary.get("queue_length_mean", 0)
    summary["total_wait"] = summary.get("wait_time_mean", 0) * step_count
    
    return summary


def compare_strategies_realtime(
    env_config: EnvConfig,
    model_path: Optional[Path] = None,
    num_episodes: int = 2,
    ft_schedule: Optional[List] = None,
) -> None:
    """Compare DQN and Fixed-Time strategies with real-time monitoring.
    
    Args:
        env_config: Environment configuration
        model_path: Path to trained DQN model
        num_episodes: Number of episodes to run
    """
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}REALTIME METRICS COMPARISON: DQN vs Fixed-Time{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{'='*80}{Colors.ENDC}")
    print(f"Episodes: {num_episodes}")
    print(f"Scenario: {env_config.sumocfg_path}")
    print(f"{'='*80}\n")
    
    # Load DQN agent if model provided
    agent = None
    if model_path and model_path.exists():
        try:
            print(f"{Colors.OKGREEN}Loading DQN model from {model_path}{Colors.ENDC}\n")
            num_phases = len(env_config.phases)
            # Resolve state_dim thực tế từ env (lanes × 4 features = 8 × 4 = 32)
            _tmp_env = SumoMDPEnv(env_config)
            state_dim = _tmp_env.reset().shape[0]
            _tmp_env.close()

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
            print(f"{Colors.OKGREEN}✓ Model loaded successfully\n{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.WARNING}Could not load model: {e}. Running fixed-time only.\n{Colors.ENDC}")
            agent = None
    
    # Storage for all episodes
    all_dqn_results = []
    all_fixed_results = []
    
    # Run episodes
    for ep in range(num_episodes):
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}")
        print(f"EPISODE {ep + 1}/{num_episodes}")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        # Run DQN if available
        if agent:
            dqn_result = run_episode_with_monitoring(
                env_config,
                agent=agent,
                strategy_name=f"DQN Agent (Episode {ep + 1})",
                verbose=True,
                phase_schedule=ft_schedule,
            )
            all_dqn_results.append(dqn_result)
        else:
            dqn_result = None

        # Run Fixed-Time
        fixed_result = run_episode_with_monitoring(
            env_config,
            agent=None,
            strategy_name=f"Fixed-Time Controller (Episode {ep + 1})",
            verbose=True,
            phase_schedule=ft_schedule,
        )
        all_fixed_results.append(fixed_result)
    
    # Final comparison table
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}")
    print(f"FINAL COMPARISON (Average across {num_episodes} episodes)")
    print(f"{'='*80}{Colors.ENDC}\n")
    
    if all_dqn_results:
        dqn_avg_queue = np.mean([r.get("avg_queue", 0) for r in all_dqn_results])
        dqn_avg_wait = np.mean([r.get("total_wait", 0) for r in all_dqn_results])
        dqn_avg_speed = np.mean([r.get("avg_speed_mean", 0) for r in all_dqn_results])
        dqn_total_reward = np.mean([r.get("total_reward", 0) for r in all_dqn_results])
        dqn_veh_passed = np.mean([r.get("total_vehicles_passed", 0) for r in all_dqn_results])
    
    fixed_avg_queue = np.mean([r.get("avg_queue", 0) for r in all_fixed_results])
    fixed_avg_wait = np.mean([r.get("total_wait", 0) for r in all_fixed_results])
    fixed_avg_speed = np.mean([r.get("avg_speed_mean", 0) for r in all_fixed_results])
    fixed_total_reward = np.mean([r.get("total_reward", 0) for r in all_fixed_results])
    fixed_veh_passed = np.mean([r.get("total_vehicles_passed", 0) for r in all_fixed_results])
    
    print(f"{'Metric':<25} {'DQN':<18} {'Fixed-Time':<18} {'Improvement':<15}")
    print("-" * 76)
    
    metrics_display = [
        ("Avg Queue Length", "avg_queue", "lower_better"),
        ("Avg Speed (km/h)", "avg_speed_mean", "higher_better"),
        ("Total Vehicles Passed", "total_vehicles_passed", "higher_better"),
        ("Total Reward", "total_reward", "higher_better"),
    ]
    
    if all_dqn_results:
        for display_name, metric_key, comparison_type in metrics_display:
            dqn_val = dqn_avg_queue if metric_key == "avg_queue" else \
                     dqn_avg_speed if metric_key == "avg_speed_mean" else \
                     dqn_veh_passed if metric_key == "total_vehicles_passed" else \
                     dqn_total_reward
            fixed_val = fixed_avg_queue if metric_key == "avg_queue" else \
                       fixed_avg_speed if metric_key == "avg_speed_mean" else \
                       fixed_veh_passed if metric_key == "total_vehicles_passed" else \
                       fixed_total_reward
            
            # Calculate improvement percentage
            if comparison_type == "lower_better":
                if fixed_val != 0:
                    improvement = ((fixed_val - dqn_val) / fixed_val * 100)
                else:
                    improvement = 0
                color = Colors.OKGREEN if improvement > 0 else Colors.FAIL
            else:  # higher_better
                if fixed_val != 0:
                    improvement = ((dqn_val - fixed_val) / fixed_val * 100)
                else:
                    improvement = 0
                color = Colors.OKGREEN if improvement > 0 else Colors.FAIL
            
            print(
                f"{display_name:<25} "
                f"{dqn_val:>16.2f}  "
                f"{fixed_val:>16.2f}  "
                f"{color}{improvement:>+12.1f}%{Colors.ENDC}"
            )
    else:
        print(f"{'DQN':<25} {'Not available':<18}")
        print(f"{'Avg Queue Length':<25} {'N/A':<18} {fixed_avg_queue:>16.2f}")
        print(f"{'Avg Speed (km/h)':<25} {'N/A':<18} {fixed_avg_speed:>16.2f}")
        print(f"{'Total Vehicles Passed':<25} {'N/A':<18} {fixed_veh_passed:>16.2f}")
    
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}✓ Comparison complete!{Colors.ENDC}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time metrics monitoring for traffic signal control"
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
        default=2,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without SUMO GUI",
    )
    
    args = parser.parse_args()

    # Đọc từ config.yaml để đảm bảo nhất quán với train.py và báo cáo
    cfg = load_config()
    sumo_cfg = cfg.get("sumo", {})
    _action_dur = sumo_cfg.get("action_duration", 5)

    env_config = EnvConfig(
        sumocfg_path=args.scenario or sumo_cfg.get("sumocfg_path", "data/scenarios/hn_sample/config.sumocfg"),
        tls_id=sumo_cfg.get("tls_id", "c"),
        phases=sumo_cfg.get("phases", [0, 1, 2, 3]),
        action_duration=_action_dur,
        min_phase_steps=max(1, sumo_cfg.get("min_phase_duration", 5) // _action_dur),
        max_phase_steps=max(2, sumo_cfg.get("max_phase_duration", 140) // _action_dur),
        max_steps=sumo_cfg.get("max_steps", 3600),
        warmup_steps=sumo_cfg.get("warmup_steps", 60),
        gui=not args.no_gui,
        vn_weights=VNWeights(**cfg.get("vn_weights", {})),
        phase_green_min={
            int(k): max(1, v // _action_dur)
            for k, v in sumo_cfg.get("phase_green_min", {}).items()
        },
        phase_green_max={
            int(k): max(1, v // _action_dur)
            for k, v in sumo_cfg.get("phase_green_max", {}).items()
        },
    )

    ft_schedule = sumo_cfg.get("fixed_time_phase_schedule", [(2, 100), (3, 5), (0, 50), (1, 5)])

    # Run comparison
    compare_strategies_realtime(env_config, args.model_path, args.episodes, ft_schedule)


if __name__ == "__main__":
    main()

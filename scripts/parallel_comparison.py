#!/usr/bin/env python3
"""
Side-by-side comparison of DQN vs Fixed-Time strategies.
Collects metrics and displays them in parallel tables.

Usage:
    python scripts/parallel_comparison.py --model-path outputs/dqn_vn_tls.pt --episodes 1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.common import load_config, ensure_sumo_home, load_dqn_agent

ensure_sumo_home()

from src.baseline import FixedTimeController, FixedTimeConfig
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights


class EpisodeMetrics:
    """Track comprehensive metrics for one episode."""

    def __init__(self):
        self.queues: List[float] = []
        self.speeds: List[float] = []
        self.wait_times: List[float] = []
        self.rewards: List[float] = []
        self.vehicles_passed: List[int] = []
        self.total_steps = 0
        self.total_reward = 0.0

    def add_step(self, info: Dict, reward: float) -> None:
        self.queues.append(info.get("queue_length", 0))
        self.speeds.append(info.get("avg_speed", 0))
        self.wait_times.append(info.get("avg_wait", 0))
        self.vehicles_passed.append(info.get("vehicles_passed", 0))
        self.rewards.append(reward)
        self.total_reward += reward
        self.total_steps += 1

    def summarize(self) -> Dict:
        return {
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "total_vehicles": sum(self.vehicles_passed),
            "avg_queue": float(np.mean(self.queues)) if self.queues else 0.0,
            "max_queue": float(np.max(self.queues)) if self.queues else 0.0,
            "min_queue": float(np.min(self.queues)) if self.queues else 0.0,
            "avg_speed": float(np.mean(self.speeds)) if self.speeds else 0.0,
            "max_speed": float(np.max(self.speeds)) if self.speeds else 0.0,
            "min_speed": float(np.min(self.speeds)) if self.speeds else 0.0,
            "avg_wait": float(np.mean(self.wait_times)) if self.wait_times else 0.0,
            "max_wait": float(np.max(self.wait_times)) if self.wait_times else 0.0,
            "std_queue": float(np.std(self.queues)) if self.queues else 0.0,
            "std_speed": float(np.std(self.speeds)) if self.speeds else 0.0,
        }


def run_episode(env_config: EnvConfig, agent=None, verbose: bool = False) -> EpisodeMetrics:
    """Run a single episode and collect metrics.

    Args:
        env_config: Environment configuration.
        agent:      DQN agent. If None, uses FixedTimeController.
        verbose:    Print step-level progress.

    Returns:
        EpisodeMetrics with collected data.
    """
    env = SumoMDPEnv(env_config)
    state = env.reset()
    metrics = EpisodeMetrics()

    controller = None
    strategy_name = "DQN" if agent is not None else "Fixed-Time"
    if agent is None:
        controller = FixedTimeController(FixedTimeConfig(green_duration=55, yellow_duration=5))

    step_count = 0
    while True:
        action = agent.act(state, eps=0.0) if agent is not None else controller.get_action()
        next_state, reward, done, info = env.step(action)
        metrics.add_step(info, reward)
        step_count += 1

        if verbose and step_count % 50 == 0:
            print(f"  [{strategy_name}] Step {step_count}: "
                  f"Queue={info.get('queue_length', 0):.1f}  "
                  f"Speed={info.get('avg_speed', 0):.2f}")

        state = next_state
        if done:
            break

    env.close()
    return metrics


def format_metric_table(dqn_metrics: Dict, fixed_metrics: Dict) -> str:
    """Format a side-by-side comparison table string."""
    W = 100
    table = "\n" + "=" * W + "\n"
    table += f"{'METRIC':<25} {'DQN':>18}  {'FIXED-TIME':>18}  {'DIFFERENCE':>18}  {'WINNER':<10}\n"
    table += "=" * W + "\n"

    comparisons = [
        ("Avg Queue Length",      "avg_queue",      "lower"),
        ("Max Queue Length",      "max_queue",      "lower"),
        ("Std Dev Queue",         "std_queue",      "lower"),
        ("Avg Speed (m/s)",       "avg_speed",      "higher"),
        ("Max Speed (m/s)",       "max_speed",      "higher"),
        ("Avg Wait Time",         "avg_wait",       "lower"),
        ("Total Vehicles Passed", "total_vehicles", "higher"),
        ("Total Reward",          "total_reward",   "higher"),
        ("Total Steps",           "total_steps",    "neutral"),
    ]

    for label, key, cmp_type in comparisons:
        dv = dqn_metrics.get(key, 0)
        fv = fixed_metrics.get(key, 0)

        if cmp_type == "neutral":
            diff, winner = dv - fv, "--"
        elif cmp_type == "lower":
            diff = fv - dv
            winner = "DQN ✓" if dv < fv else ("Fixed ✓" if dv > fv else "TIE")
        else:
            diff = dv - fv
            winner = "DQN ✓" if dv > fv else ("Fixed ✓" if dv < fv else "TIE")

        table += f"{label:<25} {dv:>18.2f}  {fv:>18.2f}  {diff:>18.2f}  {winner:<10}\n"

    table += "=" * W + "\n"
    return table


def run_parallel_comparison(
    env_config: EnvConfig,
    model_path: Optional[Path] = None,
    num_episodes: int = 1,
    save_results: bool = True,
) -> None:
    """Run comparison and display results."""
    print("\n" + "=" * 100)
    print("PARALLEL METRICS COMPARISON: DQN vs Fixed-Time Controller")
    print("=" * 100)
    print(f"Episodes : {num_episodes}")
    print(f"Scenario : {env_config.sumocfg_path}")
    print(f"Max Steps: {env_config.max_steps}")
    print("=" * 100 + "\n")

    # Load DQN agent — state_dim is lanes×4; resolve from a quick env reset
    agent = None
    if model_path and model_path.exists():
        _tmp = SumoMDPEnv(env_config)
        state_dim = _tmp.reset().shape[0]
        _tmp.close()
        agent = load_dqn_agent(model_path, state_dim, len(env_config.phases))
    else:
        print("⚠ No model path provided. Running Fixed-Time only.\n")

    all_dqn: List[Dict] = []
    all_fixed: List[Dict] = []

    print("Running episodes...\n")
    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}:")

        if agent:
            print("  [DQN] Running...", end="", flush=True)
            dqn_summary = run_episode(env_config, agent=agent).summarize()
            all_dqn.append(dqn_summary)
            print(f" done  (vehicles: {dqn_summary['total_vehicles']})")

        print("  [Fixed-Time] Running...", end="", flush=True)
        fixed_summary = run_episode(env_config, agent=None).summarize()
        all_fixed.append(fixed_summary)
        print(f" done  (vehicles: {fixed_summary['total_vehicles']})")

        if agent:
            print("\n  Episode Summary:")
            print(format_metric_table(all_dqn[-1], all_fixed[-1]))

    # Aggregate averages
    def _avg(results: List[Dict]) -> Dict:
        return {k: float(np.mean([r[k] for r in results])) for k in results[0]}

    avg_fixed = _avg(all_fixed)
    avg_dqn = _avg(all_dqn) if all_dqn else None

    print("\n" + "=" * 100)
    print(f"FINAL RESULTS (Average over {num_episodes} episode(s))")
    print("=" * 100)
    if avg_dqn:
        print(format_metric_table(avg_dqn, avg_fixed))
    else:
        print("\nFixed-Time Results Only:")
        for k, v in avg_fixed.items():
            print(f"  {k:<30} {v:>12.2f}")

    if save_results:
        output_dir = Path("outputs/parallel_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": datetime.now().isoformat(),
            "scenario": env_config.sumocfg_path,
            "episodes": num_episodes,
            "dqn_results": all_dqn if all_dqn else None,
            "fixed_results": all_fixed,
        }
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(payload, f, indent=2)

        with open(output_dir / "comparison_results.txt", "w") as f:
            f.write("=" * 100 + "\nPARALLEL METRICS COMPARISON\n" + "=" * 100 + "\n")
            f.write(f"Timestamp: {payload['timestamp']}\n")
            f.write(f"Scenario : {payload['scenario']}\n")
            f.write(f"Episodes : {payload['episodes']}\n" + "=" * 100 + "\n\n")
            if avg_dqn:
                f.write(format_metric_table(avg_dqn, avg_fixed))
            f.write("\nDETAILED RESULTS:\n" + "-" * 100 + "\n")
            for ep, fixed_r in enumerate(all_fixed):
                f.write(f"\nEpisode {ep + 1}:\n")
                if all_dqn:
                    f.write(format_metric_table(all_dqn[ep], fixed_r))

        print(f"\n✓ Results saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side metrics comparison")
    parser.add_argument("--scenario", type=str,
                        default="data/scenarios/hn_sample/config.sumocfg")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    sumo_cfg = cfg.get("sumo", {})
    _action_dur = sumo_cfg.get("action_duration", 5)

    env_config = EnvConfig(
        sumocfg_path=args.scenario or sumo_cfg.get("sumocfg_path", args.scenario),
        tls_id=sumo_cfg.get("tls_id", "c"),
        phases=sumo_cfg.get("phases", [0, 1, 2, 3]),
        action_duration=_action_dur,
        min_phase_steps=max(1, sumo_cfg.get("min_phase_duration", 30) // _action_dur),
        max_phase_steps=max(2, sumo_cfg.get("max_phase_duration", 120) // _action_dur),
        max_steps=sumo_cfg.get("max_steps", 3600),
        warmup_steps=sumo_cfg.get("warmup_steps", 60),
        gui=False,
        vn_weights=VNWeights(**cfg.get("vn_weights", {})),
    )

    run_parallel_comparison(env_config, args.model_path, args.episodes, not args.no_save)


if __name__ == "__main__":
    main()

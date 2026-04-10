#!/usr/bin/env python3
"""
Comparison script: DQN Agent vs Fixed-Time Controller.

Evaluates both strategies on the same traffic scenario and generates
comparison metrics.

Usage:
    python scripts/compare_strategies.py --model-path outputs/dqn_vn_tls.pt --num-episodes 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from scripts.common import load_config, ensure_sumo_home, load_dqn_agent

ensure_sumo_home()

from src.baseline import FixedTimeController, FixedTimeConfig
from src.dqn.agent import DQNAgent
from src.env.sumo_env import EnvConfig, SumoMDPEnv, VNWeights


class ComparisonRunner:
    """Runs DQN and Fixed-Time strategies and collects metrics."""

    def __init__(
        self,
        env_config: EnvConfig,
        num_episodes: int = 20,
        model_path: Optional[Path] = None,
    ):
        self.env_config = env_config
        self.num_episodes = num_episodes
        self.model_path = model_path

        _keys = ["total_wait", "avg_queue", "max_queue", "throughput",
                 "episode_reward", "avg_speed", "avg_occupancy", "avg_halting_vehicles"]
        self.dqn_metrics:   Dict[str, List[float]] = {k: [] for k in _keys}
        self.fixed_metrics: Dict[str, List[float]] = {k: [] for k in _keys}

    # ------------------------------------------------------------------
    def _run_episode(self, agent: Optional[DQNAgent] = None) -> Dict:
        """Run one episode with either DQN agent or Fixed-Time controller.

        Args:
            agent: Trained DQN agent, or None to use FixedTimeController.

        Returns:
            Dictionary with episode metrics.
        """
        env = SumoMDPEnv(self.env_config)
        state = env.reset()

        controller = None
        if agent is None:
            controller = FixedTimeController(
                FixedTimeConfig(green_duration=55, yellow_duration=5)
            )

        total_reward = 0.0
        queues, speeds, occupancies, halting_list = [], [], [], []
        vehicles_passed_total = 0

        while True:
            action = agent.act(state, eps=0.0) if agent is not None else controller.get_action()
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            queues.append(info.get("queue_length", 0.0))
            speeds.append(info.get("avg_speed", 0.0))
            occupancies.append(info.get("occupancy", 0.0))
            halting_list.append(info.get("halting_vehicles", 0))
            vehicles_passed_total += info.get("vehicles_passed", 0)

            state = next_state
            if done:
                break

        env.close()

        return {
            "total_wait": -total_reward if self.env_config.reward_type == "queue_delay" else 0.0,
            "avg_queue":            float(np.mean(queues))    if queues    else 0.0,
            "max_queue":            float(np.max(queues))     if queues    else 0.0,
            "throughput":           float(vehicles_passed_total),
            "episode_reward":       float(total_reward),
            "avg_speed":            float(np.mean(speeds))    if speeds    else 0.0,
            "avg_occupancy":        float(np.mean(occupancies)) if occupancies else 0.0,
            "avg_halting_vehicles": float(np.mean(halting_list)) if halting_list else 0.0,
        }

    # ------------------------------------------------------------------
    def run_comparison(self, agent: Optional[DQNAgent] = None) -> None:
        """Run full comparison and print results.

        Args:
            agent: Optional pre-loaded DQN agent.
                   If None, tries to load from self.model_path.
        """
        if agent is None and self.model_path and self.model_path.exists():
            # Resolve actual state_dim from a quick environment reset
            _tmp = SumoMDPEnv(self.env_config)
            state_dim = _tmp.reset().shape[0]
            _tmp.close()
            agent = load_dqn_agent(self.model_path, state_dim, len(self.env_config.phases))

        print(f"\n{'=' * 60}")
        print("COMPARISON: DQN vs Fixed-Time Controller")
        print(f"{'=' * 60}")
        print(f"Episodes: {self.num_episodes} | Scenario: {self.env_config.sumocfg_path}\n")

        if agent:
            print("Running DQN episodes...")
            for ep in tqdm(range(self.num_episodes), desc="DQN"):
                metrics = self._run_episode(agent)
                for k, v in metrics.items():
                    self.dqn_metrics[k].append(v)

        print("\nRunning Fixed-Time episodes...")
        for ep in tqdm(range(self.num_episodes), desc="Fixed-Time"):
            metrics = self._run_episode(agent=None)
            for k, v in metrics.items():
                self.fixed_metrics[k].append(v)

        self._print_results()

    # ------------------------------------------------------------------
    def _print_results(self) -> None:
        """Print comparison results table."""
        print(f"\n{'=' * 60}\nRESULTS SUMMARY\n{'=' * 60}\n")

        metrics_to_show = [
            "avg_queue", "max_queue", "avg_speed", "avg_occupancy",
            "avg_halting_vehicles", "total_wait", "throughput", "episode_reward",
        ]

        print(f"{'Metric':<28} | {'DQN (mean±std)':<22} | {'Fixed-Time (mean±std)':<22} | {'Improvement'}")
        print("-" * 90)

        for metric in metrics_to_show:
            dqn_vals   = self.dqn_metrics.get(metric, [])
            fixed_vals = self.fixed_metrics.get(metric, [])

            if dqn_vals and fixed_vals:
                dm, ds = np.mean(dqn_vals), np.std(dqn_vals)
                fm, fs = np.mean(fixed_vals), np.std(fixed_vals)

                if metric in ("episode_reward", "throughput", "avg_speed"):
                    improvement = ((dm - fm) / abs(fm) * 100) if fm != 0 else 0.0
                else:
                    improvement = ((fm - dm) / fm * 100) if fm != 0 else 0.0

                name = metric.replace("_", " ").title()
                print(f"{name:<28} | {dm:>8.2f}±{ds:<8.2f}   | {fm:>8.2f}±{fs:<8.2f}   | {improvement:>+6.1f}%")

            elif fixed_vals:
                fm, fs = np.mean(fixed_vals), np.std(fixed_vals)
                name = metric.replace("_", " ").title()
                print(f"{name:<28} | {'N/A':<22} | {fm:>8.2f}±{fs:<8.2f}   | {'N/A'}")

        print(f"\n{'=' * 60}")
        print("INTERPRETATION:")
        print("  Lower Queue / Wait / Halting = better flow")
        print("  Higher Speed / Throughput / Reward = better performance")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    def save_results(self, output_dir: Path = Path("outputs")) -> None:
        """Save comparison results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "comparison_results.txt", "w") as f:
            for label, metrics in [("DQN", self.dqn_metrics), ("Fixed-Time", self.fixed_metrics)]:
                f.write(f"{label} Metrics:\n")
                for k, vals in metrics.items():
                    if vals:
                        f.write(f"  {k}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}\n")
                f.write("\n")

        np.savez(
            output_dir / "comparison_metrics.npz",
            dqn_metrics=np.array(
                [self.dqn_metrics[k] for k in sorted(self.dqn_metrics)], dtype=object
            ),
            fixed_metrics=np.array(
                [self.fixed_metrics[k] for k in sorted(self.fixed_metrics)], dtype=object
            ),
        )
        print(f"Results saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DQN vs Fixed-Time traffic control")
    parser.add_argument("--scenario",     type=str,  default="data/scenarios/hn_sample/config.sumocfg")
    parser.add_argument("--model-path",   type=Path, default=None)
    parser.add_argument("--num-episodes", type=int,  default=20)
    parser.add_argument("--output-dir",   type=Path, default=Path("outputs/comparison"))
    parser.add_argument("--no-gui",       action="store_true")
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
        gui=not args.no_gui,
        vn_weights=VNWeights(**cfg.get("vn_weights", {})),
    )

    runner = ComparisonRunner(env_config, args.num_episodes, args.model_path)
    runner.run_comparison()
    runner.save_results(args.output_dir)


if __name__ == "__main__":
    main()

"""Plotting utilities for training and comparison visualization."""

from pathlib import Path
from typing import List, Optional

import numpy as np


def save_plot_data(
    data: List[float],
    filepath: Path,
    label: str = "data",
) -> None:
    """Save plot data to CSV file for external visualization.
    
    Args:
        data: List of values to save
        filepath: Path to save CSV file
        label: Column label
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, data, delimiter=",", header=label, fmt="%.4f")
    print(f"Saved {label} data to {filepath}")


def generate_training_summary(
    rewards: List[float],
    queue_lengths: List[float],
    waiting_times: Optional[List[float]] = None,
    output_dir: Path = Path("outputs"),
) -> None:
    """Generate training summary statistics and save data files.
    
    Args:
        rewards: List of episode rewards
        queue_lengths: List of episode average queue lengths
        waiting_times: Optional list of episode total waiting times
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    save_plot_data(rewards, output_dir / "rewards.csv", "episode_reward")
    save_plot_data(queue_lengths, output_dir / "queue_lengths.csv", "avg_queue_length")
    if waiting_times:
        save_plot_data(waiting_times, output_dir / "waiting_times.csv", "total_wait_time")

    # Save statistics
    with open(output_dir / "training_summary.txt", "w") as f:
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("Rewards:\n")
        f.write(f"  Mean: {np.mean(rewards):.4f}\n")
        f.write(f"  Std:  {np.std(rewards):.4f}\n")
        f.write(f"  Max:  {np.max(rewards):.4f}\n")
        f.write(f"  Min:  {np.min(rewards):.4f}\n\n")

        f.write("Queue Lengths (vehicles):\n")
        f.write(f"  Mean: {np.mean(queue_lengths):.4f}\n")
        f.write(f"  Std:  {np.std(queue_lengths):.4f}\n")
        f.write(f"  Max:  {np.max(queue_lengths):.4f}\n")
        f.write(f"  Min:  {np.min(queue_lengths):.4f}\n\n")

        if waiting_times:
            f.write("Waiting Times (seconds):\n")
            f.write(f"  Mean: {np.mean(waiting_times):.4f}\n")
            f.write(f"  Std:  {np.std(waiting_times):.4f}\n")
            f.write(f"  Max:  {np.max(waiting_times):.4f}\n")
            f.write(f"  Min:  {np.min(waiting_times):.4f}\n")

    print(f"Training summary saved to {output_dir / 'training_summary.txt'}")


def plot_comparison(
    dqn_data: dict,
    fixed_data: dict,
    output_dir: Path = Path("outputs"),
) -> None:
    """Generate comparison visualization data.
    
    Saves data in formats that can be easily visualized with external tools.
    
    Args:
        dqn_data: Dictionary of DQN metrics
        fixed_data: Dictionary of fixed-time metrics
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison.txt", "w") as f:
        f.write("STRATEGY COMPARISON\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Metric':<25} {'DQN':<20} {'Fixed-Time':<20}\n")
        f.write("-" * 65 + "\n")

        for metric in dqn_data.keys():
            dqn_vals = dqn_data[metric]
            fixed_vals = fixed_data[metric]

            if dqn_vals and fixed_vals:
                dqn_mean = np.mean(dqn_vals)
                fixed_mean = np.mean(fixed_vals)

                metric_name = metric.replace("_", " ").title()
                f.write(f"{metric_name:<25} {dqn_mean:<20.2f} {fixed_mean:<20.2f}\n")

    print(f"Comparison data saved to {output_dir / 'comparison.txt'}")


def print_training_progress(
    episode: int,
    total_episodes: int,
    episode_reward: float,
    avg_queue: float,
    total_wait: float,
    epsilon: float,
) -> None:
    """Print formatted training progress.
    
    Args:
        episode: Current episode number
        total_episodes: Total episodes to train
        episode_reward: Reward for current episode
        avg_queue: Average queue length
        total_wait: Total waiting time
        epsilon: Current epsilon value
    """
    pct = (episode / total_episodes) * 100
    print(
        f"[{episode:4d}/{total_episodes:4d} ({pct:5.1f}%)] "
        f"R: {episode_reward:8.2f} | "
        f"Q: {avg_queue:6.2f} | "
        f"W: {total_wait:7.1f}s | "
        f"ε: {epsilon:.3f}"
    )

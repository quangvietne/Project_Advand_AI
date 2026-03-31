from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.dqn.agent import DQNAgent, AgentConfig
from src.dqn.replay_buffer import ReplayBuffer
from src.env.sumo_env import SumoMDPEnv, EnvConfig, VNWeights
from src.utils import LinearEpsilon
from scripts.common import load_config, ensure_sumo_home, create_output_dir


# Ensure SUMO is available
ensure_sumo_home()


def main(cfg_path: str = "config.yaml") -> None:
    cfg = load_config(cfg_path)

    _action_dur = cfg["sumo"].get("action_duration", 5)
    env_cfg = EnvConfig(
        sumocfg_path=cfg["sumo"]["sumocfg_path"],
        tls_id=cfg["sumo"]["tls_id"],
        phases=cfg["sumo"]["phases"],
        step_length=cfg["sumo"].get("step_length", 1.0),
        action_duration=_action_dur,
        min_phase_steps=max(1, cfg["sumo"].get("min_phase_duration", 30) // _action_dur),
        max_phase_steps=max(2, cfg["sumo"].get("max_phase_duration", 120) // _action_dur),
        max_steps=cfg["sumo"].get("max_steps", 3600),
        warmup_steps=cfg["sumo"].get("warmup_steps", 0),
        gui=cfg["sumo"].get("gui", False),
        vn_weights=VNWeights(**cfg.get("vn_weights", {})),
        reward_type=cfg.get("reward", {}).get("type", "queue_delay"),
    )
    env = SumoMDPEnv(env_cfg)

    state = env.reset()
    agent_cfg = AgentConfig(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=cfg["train"].get("gamma", 0.99),
        lr=cfg["train"].get("lr", 1e-3),
        batch_size=cfg["train"].get("batch_size", 64),
        tau=cfg["train"].get("tau", 1.0),
        target_update_interval=cfg["train"].get("target_update_interval", 1000),
        double_dqn=cfg["train"].get("double_dqn", True),
        grad_clip=cfg["train"].get("grad_clip", 10.0),
    )
    agent = DQNAgent(agent_cfg)
    buffer = ReplayBuffer(cfg["train"].get("replay_size", 100_000), state_shape=(env.state_dim,))
    eps_sched = LinearEpsilon(
        start=cfg["explore"].get("eps_start", 1.0),
        end=cfg["explore"].get("eps_end", 0.05),
        steps=cfg["explore"].get("eps_steps", 100_000),
    )

    total_steps = cfg["train"].get("total_steps", 200_000)
    start_train = cfg["train"].get("start_train", 10_000)
    train_every = cfg["train"].get("train_every", 1)
    batch_size = agent_cfg.batch_size

    s = state
    pbar = tqdm(range(total_steps), desc="Training", ncols=100)
    episode_reward = 0.0

    try:
        for t in pbar:
            eps = eps_sched.step()
            a = agent.act(s, eps)
            s2, r, d, info = env.step(a)
            buffer.push(s, a, r, s2, d)
            s = s2
            episode_reward += r

            if len(buffer) >= start_train and t % train_every == 0:
                batch = buffer.sample(batch_size)
                stats = agent.update(batch)
            else:
                stats = {"loss": float("nan")}

            if d:
                pbar.set_postfix({"R": f"{episode_reward:.1f}", "eps": f"{eps:.2f}", "loss": f"{stats['loss']:.3f}"})
                s = env.reset()
                episode_reward = 0.0

        # save weights
        out = create_output_dir(cfg.get("output_dir", "outputs"))
        torch.save(agent.q.state_dict(), out / "dqn_vn_tls.pt")
    finally:
        env.close()


if __name__ == "__main__":
    main()

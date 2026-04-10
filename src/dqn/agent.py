from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import DQN


@dataclass
class AgentConfig:
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    tau: float = 1.0  # if <1 uses soft update, if >=1 uses hard update period (int)
    target_update_interval: int = 1000
    double_dqn: bool = True
    grad_clip: float | None = 10.0


class DQNAgent:
    def __init__(self, cfg: AgentConfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q = DQN(cfg.state_dim, cfg.action_dim).to(self.device)
        self.q_target = DQN(cfg.state_dim, cfg.action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.crit = nn.SmoothL1Loss(reduction="mean")
        self.step_count = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(self.cfg.action_dim)
        s = torch.from_numpy(state.astype(np.float32)).to(self.device)
        q = self.q(s.unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def update(self, batch) -> dict:
        s, a, r, s2, d = [x.to(self.device) for x in batch]

        # Current Q(s,a)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.cfg.double_dqn:
                a2 = self.q(s2).argmax(dim=1)
                q2 = self.q_target(s2).gather(1, a2.unsqueeze(1)).squeeze(1)
            else:
                q2 = self.q_target(s2).max(dim=1).values
            target = r + (1.0 - d) * self.cfg.gamma * q2

        loss = self.crit(q, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.optim.step()  # type: ignore

        self.step_count += 1
        if self.cfg.tau < 1.0:
            # soft update
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)
        else:
            # hard update every N steps
            if self.step_count % int(self.cfg.target_update_interval) == 0:
                self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": float(loss.item())}

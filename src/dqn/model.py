from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop(x)
        return x


class DQN(nn.Module):
    """Feed-forward DQN with optional dueling heads.

    - state_dim: flattened state size
    - action_dim: number of discrete actions (signal phases)
    - hidden_layers: e.g., [128, 128]
    - dueling: if True, splits into value and advantage streams
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: Sequence[int] = (128, 128),
        dueling: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_layers:
            layers.append(MLPBlock(last, h, dropout=dropout))
            last = h
        self.backbone = nn.Sequential(*layers)

        self.dueling = dueling
        if dueling:
            self.val_head = nn.Sequential(nn.Linear(last, last), nn.ReLU(), nn.Linear(last, 1))
            self.adv_head = nn.Sequential(nn.Linear(last, last), nn.ReLU(), nn.Linear(last, action_dim))
        else:
            self.head = nn.Linear(last, action_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Kaiming uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = self.backbone(x)
        if self.dueling:
            v = self.val_head(z)  # [B, 1]
            a = self.adv_head(z)  # [B, A]
            q = v + a - a.mean(dim=1, keepdim=True)
            return q
        else:
            return self.head(z)

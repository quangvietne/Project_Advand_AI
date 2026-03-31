from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    d: bool


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.state_shape = state_shape
        self.storage: List[Transition] = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.storage)

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, d: bool) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(Transition(s.copy(), a, r, s2.copy(), d))
        else:
            self.storage[self.idx] = Transition(s.copy(), a, r, s2.copy(), d)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.storage, batch_size)
        s = torch.from_numpy(np.stack([b.s for b in batch]).astype(np.float32))
        a = torch.tensor([b.a for b in batch], dtype=torch.long)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32)
        s2 = torch.from_numpy(np.stack([b.s2 for b in batch]).astype(np.float32))
        d = torch.tensor([b.d for b in batch], dtype=torch.float32)
        return s, a, r, s2, d

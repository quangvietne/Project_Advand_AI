from __future__ import annotations


class LinearEpsilon:
    """Linear epsilon decay schedule for exploration."""

    def __init__(
        self, start: float = 1.0, end: float = 0.05, steps: int = 50_000
    ) -> None:
        self.start = start
        self.end = end
        self.steps = max(1, steps)
        self.t = 0

    def value(self, t: int | None = None) -> float:
        """Get epsilon at step t."""
        if t is None:
            t = self.t
        frac = min(1.0, max(0.0, t / self.steps))
        return self.start + frac * (self.end - self.start)

    def step(self) -> float:
        """Advance and return current epsilon."""
        v = self.value(self.t)
        self.t += 1
        return v

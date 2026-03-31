"""Fixed-time traffic light controller for baseline comparison."""

from dataclasses import dataclass
from typing import Optional, cast

@dataclass
class FixedTimeConfig:
    """Configuration for fixed-time controller."""
    green_duration: int = 55  # seconds (đèn xanh 55s)
    yellow_duration: int = 5   # seconds (đèn vàng 5s → đèn đỏ = 55+5=60s)
    phases_cycle: Optional[list] = None  # 4-phase cycle

    def __post_init__(self):
        if self.phases_cycle is None:
            self.phases_cycle = [0, 1, 2, 3]  # NS-straight, NS-left, EW-straight, EW-left
        # Ensure phases_cycle is always a list after initialization
        assert isinstance(self.phases_cycle, list), "phases_cycle must be a list"


class FixedTimeController:
    """
    Fixed-time traffic light controller.
    
    Cycles through 4 fixed phases:
    - Phase 0: North-South straight/right
    - Phase 1: North-South left
    - Phase 2: East-West straight/right
    - Phase 3: East-West left
    
    Each phase is green for `green_duration` steps, then yellow for `yellow_duration`.
    """

    def __init__(self, config: FixedTimeConfig) -> None:
        """Initialize the fixed-time controller.
        
        Args:
            config: Configuration with green_duration and yellow_duration.
        """
        self.green_duration = config.green_duration
        self.yellow_duration = config.yellow_duration
        self.cycle_length = self.green_duration + self.yellow_duration
        # Cast to list since __post_init__ guarantees it's never None
        phases = cast(list, config.phases_cycle)
        self.total_cycle_length = self.cycle_length * len(phases)
        self.phases_cycle: list = phases
        self.step_counter = 0

    def get_action(self) -> int:
        """
        Get the next action (phase index) based on fixed schedule.
        
        Returns:
            Phase index (0-3) for the current step.
        """
        # Determine which phase in the cycle (each phase gets green_duration + yellow_duration steps)
        phase_index = (self.step_counter // self.cycle_length) % len(self.phases_cycle)
        action = self.phases_cycle[phase_index]
        
        self.step_counter += 1
        return action

    def reset(self) -> None:
        """Reset the controller for a new episode."""
        self.step_counter = 0

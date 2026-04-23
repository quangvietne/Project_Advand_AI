"""Fixed-time traffic light controller for baseline comparison."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, cast


@dataclass
class FixedTimeConfig:
    """Configuration for fixed-time controller.

    Hai chế độ:
    1. Legacy (uniform): dùng green_duration + yellow_duration + phases_cycle
       — mỗi pha nhận cùng thời gian xanh/vàng.
    2. Explicit schedule: dùng phase_schedule = [(phase_idx, seconds), ...]
       — mỗi pha có thời gian riêng; ưu tiên hơn legacy khi được cung cấp.

    action_duration (giây/bước MDP) cần khớp với EnvConfig.action_duration
    để quy đổi giây → số bước chính xác.
    """
    # Legacy params (backward compatible)
    green_duration: int = 55        # giây xanh mỗi pha (legacy)
    yellow_duration: int = 5        # giây vàng mỗi pha (legacy)
    phases_cycle: Optional[List[int]] = None  # thứ tự pha (legacy)

    # Explicit schedule: [(phase_idx, duration_seconds), ...]
    # Nếu cung cấp → ưu tiên hơn legacy params
    phase_schedule: Optional[List[Tuple[int, int]]] = None

    # Giây mỗi bước MDP — phải khớp với EnvConfig.action_duration
    action_duration: int = 5

    def __post_init__(self):
        if self.phase_schedule is None and self.phases_cycle is None:
            self.phases_cycle = [0, 1, 2, 3]
        if self.phases_cycle is not None:
            assert isinstance(self.phases_cycle, list), "phases_cycle must be a list"


class FixedTimeController:
    """Fixed-time traffic light controller.

    Chu kỳ mặc định mới (160s, tỉ lệ EW:NS = 2:1):
      Phase 2 (EW xanh): 100s
      Phase 3 (EW vàng):   5s
      Phase 0 (NS xanh):  50s
      Phase 1 (NS vàng):   5s

    Hỗ trợ cả hai chế độ:
    - Explicit schedule (khuyến nghị): truyền phase_schedule vào FixedTimeConfig
    - Legacy uniform: dùng green_duration / yellow_duration / phases_cycle
    """

    def __init__(self, config: FixedTimeConfig) -> None:
        self.action_duration = config.action_duration

        if config.phase_schedule is not None:
            # Chế độ schedule tường minh: chuyển giây → số bước MDP
            self._sequence: List[int] = []
            for phase_idx, duration_s in config.phase_schedule:
                steps = max(1, round(duration_s / self.action_duration))
                self._sequence.extend([phase_idx] * steps)
        else:
            # Chế độ legacy: tất cả pha dùng chung green + yellow duration
            phases = cast(List[int], config.phases_cycle)
            cycle_steps = config.green_duration + config.yellow_duration
            self._sequence = []
            for p in phases:
                self._sequence.extend([p] * cycle_steps)

        self._total = len(self._sequence)
        self.step_counter = 0

    # ------------------------------------------------------------------
    def get_action(self) -> int:
        """Trả về phase index cho bước MDP hiện tại."""
        action = self._sequence[self.step_counter % self._total]
        self.step_counter += 1
        return action

    def reset(self) -> None:
        """Reset bộ đếm về đầu chu kỳ."""
        self.step_counter = 0

    @property
    def cycle_seconds(self) -> int:
        """Tổng thời gian một chu kỳ (giây)."""
        return self._total * self.action_duration

    @property
    def cycle_steps(self) -> int:
        """Tổng số bước MDP một chu kỳ."""
        return self._total

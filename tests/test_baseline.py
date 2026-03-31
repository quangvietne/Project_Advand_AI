#!/usr/bin/env python3
"""Test baseline fixed-time controller."""

from src.baseline import FixedTimeController, FixedTimeConfig


def test_fixed_time_controller():
    """Test that fixed-time controller cycles through phases correctly."""
    config = FixedTimeConfig(
        green_duration=3,
        yellow_duration=1,
        phases_cycle=[0, 1, 2, 3],
    )
    controller = FixedTimeController(config)

    print("Fixed-Time Controller Test")
    print("=" * 50)
    print(f"Config: green={config.green_duration}s, yellow={config.yellow_duration}s")
    print(f"Cycle: {config.phases_cycle}")
    print()

    # Run one full cycle (4 phases × 4 steps each = 16 steps)
    actions = []
    for step in range(20):
        action = controller.get_action()
        actions.append(action)
        print(f"Step {step:2d}: Phase {action}")

    print()
    print("Expected pattern: [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,0,0,0,0]")
    print(f"Actual pattern:   {actions}")
    print()

    # Verify pattern
    expected = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0]
    if actions == expected:
        print("✅ PASS: Controller cycles correctly")
    else:
        print("❌ FAIL: Pattern mismatch")

    # Test reset
    controller.reset()
    action_after_reset = controller.get_action()
    if action_after_reset == 0:
        print("✅ PASS: Reset works correctly")
    else:
        print("❌ FAIL: Reset broken")


if __name__ == "__main__":
    test_fixed_time_controller()

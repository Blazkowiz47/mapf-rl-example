"""Tests for continuous kinematics and swappable dynamics rules."""

import numpy as np
import pytest

from dynamics import AccelerationDynamicsRule, VelocityDynamicsRule
from environments import PointMass2DEnv


def test_acceleration_rule_applies_constant_acceleration_kinematics() -> None:
    rule = AccelerationDynamicsRule(max_acceleration=3.0, max_speed=4.0)
    position, velocity = rule.update(
        np.asarray((0.0, 0.0), dtype=np.float32),
        np.asarray((1.0, 0.0), dtype=np.float32),
        np.asarray((2.0, -1.0), dtype=np.float32),
        dt=0.5,
    )

    assert position.tolist() == pytest.approx([0.75, -0.125])
    assert velocity.tolist() == pytest.approx([2.0, -0.5])


def test_velocity_rule_uses_the_action_as_next_velocity() -> None:
    rule = VelocityDynamicsRule(max_speed=3.0)
    position, velocity = rule.update(
        np.asarray((0.0, 0.0), dtype=np.float32),
        np.asarray((-2.0, 2.0), dtype=np.float32),
        np.asarray((2.0, -1.0), dtype=np.float32),
        dt=0.25,
    )

    assert position.tolist() == pytest.approx([0.5, -0.25])
    assert velocity.tolist() == pytest.approx([2.0, -1.0])


def test_point_mass_exposes_state_and_uses_selected_rule() -> None:
    environment = PointMass2DEnv(
        dynamics={
            "name": "acceleration",
            "max_acceleration": 2.0,
            "max_speed": 3.0,
        },
        dt=0.5,
        world_limit=5.0,
        start_position=[0.0, 0.0],
        goal_position=[2.0, 0.0],
        goal_radius=0.1,
        render_size=64,
    )
    observation, reset_info = environment.reset(seed=2026)
    next_observation, reward, terminated, truncated, info = environment.step(
        np.asarray((2.0, 0.0), dtype=np.float32)
    )
    frame = environment.render()

    assert observation.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    assert next_observation.tolist() == pytest.approx([0.25, 0.0, 1.0, 0.0, 2.0, 0.0])
    assert reward > 0.0
    assert terminated is False
    assert truncated is False
    assert reset_info["dynamics_rule"] == "AccelerationDynamicsRule"
    assert info["time"] == pytest.approx(0.5)
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8


def test_point_mass_rejects_actions_outside_the_rule_bounds() -> None:
    environment = PointMass2DEnv(
        dynamics={"name": "velocity", "max_speed": 1.0},
    )
    environment.reset()

    with pytest.raises(ValueError, match="outside"):
        environment.step(np.asarray((1.1, 0.0), dtype=np.float32))

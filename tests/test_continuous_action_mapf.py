"""Tests for the SAC-specific continuous MAPF environment."""

import numpy as np
import pytest

import rules  # noqa: F401
from environments import ContinuousActionMAPFEnv


def _environment() -> ContinuousActionMAPFEnv:
    return ContinuousActionMAPFEnv(
        scenario={
            "name": "continuous_crossing",
            "width": 5,
            "height": 5,
            "max_steps": 12,
            "starts": [[0, 0], [4, 4]],
            "goals": [[4, 4], [0, 0]],
            "walls": [[1, 2], [3, 2]],
        },
        interaction_rule={"name": "example_exclusive_cell"},
        action_dead_zone=0.2,
    )


def test_continuous_controls_decode_to_separate_agent_moves() -> None:
    environment = _environment()

    assert (
        environment.decode_action(
            np.asarray(((-0.8, 0.1), (0.1, 0.9)), dtype=np.float32)
        )
        == 1 + 5 * 2
    )
    assert (
        environment.decode_action(
            np.asarray(((0.8, 0.1), (0.1, -0.9)), dtype=np.float32)
        )
        == 3 + 5 * 4
    )
    assert (
        environment.decode_action(
            np.asarray(((0.1, 0.1), (-0.1, 0.1)), dtype=np.float32)
        )
        == 0
    )

    with pytest.raises(ValueError, match="outside"):
        environment.decode_action(
            np.asarray(((2.0, 0.0), (0.0, 0.0)), dtype=np.float32)
        )


def test_continuous_environment_delegates_to_mapf_physics() -> None:
    environment = _environment()
    observation, _ = environment.reset(seed=2026)
    next_observation, reward, terminated, truncated, info = environment.step(
        np.asarray(((0.0, 0.9), (0.0, -0.9)), dtype=np.float32)
    )

    assert environment.observation_space.contains(observation)
    assert environment.observation_space.contains(next_observation)
    assert environment.world.positions[0].tolist() == [[0, 1], [4, 3]]
    assert reward > 0.0
    assert terminated is False
    assert truncated is False
    assert info["episode_collisions"] == 0

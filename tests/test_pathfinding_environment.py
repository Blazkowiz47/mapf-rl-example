"""Tests for the procedural single-agent pathfinding environment."""

import numpy as np

from pathfinding_environment import ProceduralPathfindingEnv


def test_procedural_environment_is_seeded_and_returns_rgb_grids() -> None:
    environment = ProceduralPathfindingEnv(
        grid_size=128,
        observation_size=64,
        wall_count=12,
        minimum_goal_distance=20,
        maximum_goal_distance=80,
    )

    first_observation, first_info = environment.reset(seed=2026)
    first_grid = environment.get_grid_rgb()
    first_walls = environment.world.wall_mask.copy()
    first_start = environment.world.positions.copy()
    first_goal = environment.world.goal_positions.copy()
    second_observation, second_info = environment.reset(seed=2026)

    assert first_observation.shape == (64, 64, 3)
    assert first_observation.dtype == np.uint8
    assert first_grid.shape == (128, 128, 3)
    assert first_grid.dtype == np.uint8
    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(first_walls, environment.world.wall_mask)
    assert np.array_equal(first_start, environment.world.positions)
    assert np.array_equal(first_goal, environment.world.goal_positions)
    assert first_info == second_info
    assert np.any(np.all(first_grid == (255, 0, 0), axis=2))
    assert np.any(np.all(first_grid == (0, 0, 255), axis=2))


def test_reward_tracks_progress_and_goal_ends_episode() -> None:
    environment = ProceduralPathfindingEnv(
        grid_size=64,
        observation_size=32,
        max_steps=8,
        move_stride=1,
        wall_count=0,
        minimum_goal_distance=1,
        maximum_goal_distance=1,
        step_reward=-0.01,
        progress_reward=0.1,
        goal_reward=1.0,
    )
    _, reset_info = environment.reset(seed=7)
    start = environment.world.positions[0, 0]
    goal = environment.world.goal_positions[0]
    delta = goal - start
    action = (
        1
        if delta[0] > 0
        else 0
        if delta[0] < 0
        else 3
        if delta[1] > 0
        else 2
    )

    observation, reward, terminated, truncated, info = environment.step(
        action
    )

    assert reset_info["distance_to_goal"] == 1
    assert observation.shape == (32, 32, 3)
    assert reward > 1.0
    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    assert info["distance_to_goal"] == 0
    assert info["path_length"] == 1
    assert np.any(np.all(observation == (255, 0, 0), axis=2))
    assert np.any(np.all(observation == (0, 0, 255), axis=2))


def test_invalid_move_is_penalized() -> None:
    environment = ProceduralPathfindingEnv(
        grid_size=64,
        observation_size=32,
        move_stride=1,
        wall_count=0,
        minimum_goal_distance=4,
        maximum_goal_distance=20,
        collision_reward=-1.0,
    )
    environment.reset(seed=9)
    environment.world.positions[0, 0] = (0, 0)

    _, reward, terminated, truncated, info = environment.step(0)

    assert reward < -1.0
    assert terminated is False
    assert truncated is False
    assert info["episode_boundary_collisions"] == 1


def test_goal_distance_must_fit_sampling_area_and_episode_budget() -> None:
    try:
        ProceduralPathfindingEnv(
            grid_size=64,
            minimum_goal_distance=100,
            maximum_goal_distance=100,
        )
    except ValueError as error:
        assert "sampling margin" in str(error)
    else:
        raise AssertionError("Expected an impossible distance to be rejected")

    try:
        ProceduralPathfindingEnv(
            grid_size=128,
            max_steps=4,
            move_stride=2,
            minimum_goal_distance=8,
            maximum_goal_distance=20,
        )
    except ValueError as error:
        assert "within max_steps" in str(error)
    else:
        raise AssertionError("Expected an unreachable budget to be rejected")

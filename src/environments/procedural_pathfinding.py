"""Procedural single-agent pathfinding environment for the ViT example."""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import gymnasium as gym
import numpy as np
from dl_robotics import GridScenario, GridWorldBatch
from numpy.typing import NDArray

UInt8Image = NDArray[np.uint8]


class ProceduralPathfindingEnv(gym.Env[np.ndarray, int]):
    """Navigate one red agent to one blue goal in a procedural grid."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 8,
    }
    _ROBOTICS_ACTIONS = np.asarray([1, 3, 4, 2], dtype=np.int32)

    def __init__(
        self,
        *,
        grid_size: int = 1000,
        observation_size: int = 256,
        max_steps: int = 128,
        move_stride: int = 8,
        wall_count: int = 100,
        minimum_goal_distance: int = 160,
        maximum_goal_distance: int = 480,
        step_reward: float = -0.01,
        progress_reward: float = 0.02,
        collision_reward: float = -0.25,
        goal_reward: float = 5.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        for name, value, minimum in (
            ("grid_size", grid_size, 64),
            ("observation_size", observation_size, 32),
            ("max_steps", max_steps, 1),
            ("move_stride", move_stride, 1),
            ("wall_count", wall_count, 0),
            ("minimum_goal_distance", minimum_goal_distance, 1),
            ("maximum_goal_distance", maximum_goal_distance, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if minimum_goal_distance > maximum_goal_distance:
            raise ValueError(
                "minimum_goal_distance cannot exceed maximum_goal_distance"
            )
        margin = max(8, grid_size // 20)
        maximum_sampled_distance = 2 * (grid_size - 2 * margin - 1)
        if maximum_goal_distance > maximum_sampled_distance:
            raise ValueError(
                "maximum_goal_distance exceeds the largest distance available "
                "inside the sampling margin"
            )
        if maximum_goal_distance > max_steps * move_stride:
            raise ValueError(
                "maximum_goal_distance must be reachable within max_steps"
            )
        rewards = np.asarray(
            [
                step_reward,
                progress_reward,
                collision_reward,
                goal_reward,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(rewards).all():
            raise ValueError("rewards must be finite")
        if progress_reward < 0:
            raise ValueError("progress_reward cannot be negative")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")

        self.grid_size = grid_size
        self.observation_size = observation_size
        self.max_steps = max_steps
        self.move_stride = move_stride
        self.wall_count = wall_count
        self.minimum_goal_distance = minimum_goal_distance
        self.maximum_goal_distance = maximum_goal_distance
        self.step_reward = float(step_reward)
        self.progress_reward = float(progress_reward)
        self.collision_reward = float(collision_reward)
        self.goal_reward = float(goal_reward)
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(observation_size, observation_size, 3),
            dtype=np.uint8,
        )
        self.scenario: GridScenario | None = None
        self.world: GridWorldBatch | None = None
        self.episode_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Replace the previous maze with a newly generated reachable task."""
        return self._reset(seed=seed, options=options)

    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        margin = max(8, self.grid_size // 20)
        start = tuple(
            int(value)
            for value in self.np_random.integers(
                margin,
                self.grid_size - margin,
                size=2,
            )
        )
        goal: tuple[int, int] | None = None
        for _ in range(10_000):
            candidate = tuple(
                int(value)
                for value in self.np_random.integers(
                    margin,
                    self.grid_size - margin,
                    size=2,
                )
            )
            distance = abs(candidate[0] - start[0]) + abs(
                candidate[1] - start[1]
            )
            if (
                self.minimum_goal_distance
                <= distance
                <= self.maximum_goal_distance
            ):
                goal = candidate
                break
        if goal is None:
            raise RuntimeError("Could not sample a goal at the configured distance")

        blocked = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.bool_,
        )
        minimum_wall_length = max(8, self.grid_size // 40)
        maximum_wall_length = max(
            minimum_wall_length + 1,
            self.grid_size // 8,
        )
        minimum_wall_thickness = max(
            2,
            self.grid_size // self.observation_size,
        )
        maximum_wall_thickness = max(
            minimum_wall_thickness + 1,
            minimum_wall_thickness * 3,
        )
        for _ in range(self.wall_count):
            length = int(
                self.np_random.integers(
                    minimum_wall_length,
                    maximum_wall_length,
                )
            )
            thickness = int(
                self.np_random.integers(
                    minimum_wall_thickness,
                    maximum_wall_thickness,
                )
            )
            row = int(
                self.np_random.integers(
                    margin,
                    self.grid_size - margin,
                )
            )
            column = int(
                self.np_random.integers(
                    margin,
                    self.grid_size - margin,
                )
            )
            if self.np_random.random() < 0.5:
                blocked[
                    row : min(self.grid_size, row + thickness),
                    column : min(self.grid_size, column + length),
                ] = True
            else:
                blocked[
                    row : min(self.grid_size, row + length),
                    column : min(self.grid_size, column + thickness),
                ] = True

        corridor_radius = max(self.move_stride, minimum_wall_thickness)
        if self.np_random.random() < 0.5:
            blocked[
                max(0, start[0] - corridor_radius) : min(
                    self.grid_size,
                    start[0] + corridor_radius + 1,
                ),
                min(start[1], goal[1]) : max(start[1], goal[1]) + 1,
            ] = False
            blocked[
                min(start[0], goal[0]) : max(start[0], goal[0]) + 1,
                max(0, goal[1] - corridor_radius) : min(
                    self.grid_size,
                    goal[1] + corridor_radius + 1,
                ),
            ] = False
        else:
            blocked[
                min(start[0], goal[0]) : max(start[0], goal[0]) + 1,
                max(0, start[1] - corridor_radius) : min(
                    self.grid_size,
                    start[1] + corridor_radius + 1,
                ),
            ] = False
            blocked[
                max(0, goal[0] - corridor_radius) : min(
                    self.grid_size,
                    goal[0] + corridor_radius + 1,
                ),
                min(start[1], goal[1]) : max(start[1], goal[1]) + 1,
            ] = False
        blocked[start] = False
        blocked[goal] = False
        walls = tuple(
            tuple(int(value) for value in coordinate)
            for coordinate in np.argwhere(blocked)
        )
        self.scenario = GridScenario(
            name=f"procedural_pathfinding_{seed}",
            width=self.grid_size,
            height=self.grid_size,
            starts=(start,),
            goals=(goal,),
            walls=walls,
            max_steps=self.max_steps * self.move_stride,
        )
        self.world = GridWorldBatch(self.scenario)
        self.episode_steps = 0
        initial_distance = abs(start[0] - goal[0]) + abs(
            start[1] - goal[1]
        )
        return self.get_grid_rgb(self.observation_size), {
            "scenario": self.scenario.name,
            "is_success": False,
            "distance_to_goal": initial_distance,
            "shortest_path_length": initial_distance,
            "path_length": 0,
            "excess_path_length": 0,
            "episode_collisions": 0,
            "episode_boundary_collisions": 0,
            "episode_wall_collisions": 0,
            "episode_actor_collisions": 0,
            "reached_agents": 0,
            "total_agents": 1,
            "makespan": 0,
            "sum_of_costs": 0,
        }

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Move up, down, left, or right and return the resized RGB grid."""
        return self._step(action)

    def _step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.world is None or self.scenario is None:
            raise RuntimeError("reset must be called before step")
        if isinstance(action, bool) or not isinstance(
            action,
            (int, np.integer),
        ):
            raise TypeError("action must be an integer")
        if not self.action_space.contains(int(action)):
            raise ValueError("action must be in [0, 3]")

        position = self.world.positions[0, 0]
        goal = self.world.goal_positions[0]
        before_distance = int(np.abs(position - goal).sum())
        boundary_collisions = 0
        wall_collisions = 0
        actor_collisions = 0
        for _ in range(self.move_stride):
            events = self.world.step(
                np.asarray(
                    [[self._ROBOTICS_ACTIONS[int(action)]]],
                    dtype=np.int32,
                )
            )
            boundary_collisions += int(events.boundary_collisions[0])
            wall_collisions += int(events.wall_collisions[0])
            actor_collisions += int(events.actor_collisions[0])
            if self.world.reached[0, 0] or events.collisions[0] > 0:
                break

        self.episode_steps += 1
        position = self.world.positions[0, 0]
        after_distance = int(np.abs(position - goal).sum())
        success = bool(self.world.reached[0, 0])
        collisions = (
            boundary_collisions + wall_collisions + actor_collisions
        )
        reward = (
            self.step_reward
            + self.progress_reward * (before_distance - after_distance)
            + self.collision_reward * collisions
            + self.goal_reward * success
        )
        truncated = self.episode_steps >= self.max_steps and not success
        path_length = int(self.world.path_lengths[0, 0])
        shortest_path_length = int(
            np.abs(
                self.world.start_positions[0]
                - self.world.goal_positions[0]
            ).sum()
        )
        return (
            self.get_grid_rgb(self.observation_size),
            float(reward),
            success,
            truncated,
            {
                "scenario": self.scenario.name,
                "is_success": success,
                "distance_to_goal": after_distance,
                "shortest_path_length": shortest_path_length,
                "path_length": path_length,
                "excess_path_length": max(
                    0,
                    path_length - shortest_path_length,
                ),
                "episode_collisions": int(
                    self.world.boundary_collision_counts[0]
                    + self.world.wall_collision_counts[0]
                    + self.world.actor_collision_counts[0]
                ),
                "episode_boundary_collisions": int(
                    self.world.boundary_collision_counts[0]
                ),
                "episode_wall_collisions": int(
                    self.world.wall_collision_counts[0]
                ),
                "episode_actor_collisions": int(
                    self.world.actor_collision_counts[0]
                ),
                "reached_agents": int(success),
                "total_agents": 1,
                "makespan": self.episode_steps,
                "sum_of_costs": min(
                    self.episode_steps,
                    self.max_steps,
                ),
            },
        )

    def get_grid_rgb(self, output_size: int | None = None) -> UInt8Image:
        """Return the current grid as an RGB matrix at the requested size."""
        return self._get_grid_rgb(output_size)

    def _get_grid_rgb(self, output_size: int | None = None) -> UInt8Image:
        if self.world is None:
            raise RuntimeError("reset must be called before rendering")
        size = self.grid_size if output_size is None else output_size
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("output_size must be an integer or None")
        if size <= 0:
            raise ValueError("output_size must be positive")

        if size == self.grid_size:
            wall_pixels = self.world.wall_mask
        else:
            wall_density = cv2.resize(
                self.world.wall_mask.astype(np.uint8) * 255,
                (size, size),
                interpolation=cv2.INTER_AREA,
            )
            wall_pixels = wall_density >= 32
        frame = np.full((size, size, 3), 255, dtype=np.uint8)
        frame[wall_pixels] = (42, 52, 68)

        goal_row, goal_column = self.world.goal_positions[0]
        agent_row, agent_column = self.world.positions[0, 0]
        scale = (size - 1) / (self.grid_size - 1)
        goal_center = (
            round(int(goal_column) * scale),
            round(int(goal_row) * scale),
        )
        agent_center = (
            round(int(agent_column) * scale),
            round(int(agent_row) * scale),
        )
        radius = max(2, size // 128)
        cv2.circle(frame, goal_center, radius, (0, 0, 255), -1)
        cv2.circle(frame, agent_center, radius, (255, 0, 0), -1)
        cv2.circle(
            frame,
            goal_center,
            radius + 2,
            (0, 0, 255),
            2,
        )
        return frame

    def render(self) -> UInt8Image:
        """Return the current 1000×1000 RGB grid by default."""
        return self._render()

    def _render(self) -> UInt8Image:
        return self.get_grid_rgb()

    def close(self) -> None:
        """Release the current generated world."""
        self._close()

    def _close(self) -> None:
        self.world = None
        self.scenario = None


if "ProceduralPathfinding-v0" not in gym.registry:
    gym.register(
        id="ProceduralPathfinding-v0",
        entry_point=ProceduralPathfindingEnv,
    )


__all__ = ["ProceduralPathfindingEnv"]

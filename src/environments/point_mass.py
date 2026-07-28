"""Continuous 2D point-mass navigation with configurable dynamics."""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import gymnasium as gym
import numpy as np

from dynamics import PointMassDynamicsRule, make_dynamics_rule


class PointMass2DEnv(gym.Env[np.ndarray, np.ndarray]):
    """Move a point to a fixed goal using velocity or acceleration control."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        *,
        dynamics: dict[str, Any],
        dt: float = 0.1,
        world_limit: float = 5.0,
        start_position: tuple[float, float] | list[float] = (-4.0, -4.0),
        goal_position: tuple[float, float] | list[float] = (4.0, 4.0),
        goal_radius: float = 0.3,
        max_steps: int = 200,
        step_reward: float = -0.01,
        progress_reward: float = 1.0,
        goal_reward: float = 10.0,
        boundary_reward: float = -0.25,
        render_size: int = 256,
        render_mode: str | None = None,
    ) -> None:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(world_limit) or world_limit <= 0.0:
            raise ValueError("world_limit must be finite and positive")
        if not np.isfinite(goal_radius) or goal_radius <= 0.0:
            raise ValueError("goal_radius must be finite and positive")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("max_steps must be an integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if isinstance(render_size, bool) or not isinstance(render_size, int):
            raise TypeError("render_size must be an integer")
        if render_size <= 0:
            raise ValueError("render_size must be positive")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")

        self.dynamics: PointMassDynamicsRule = make_dynamics_rule(dynamics)
        self.dt = float(dt)
        self.world_limit = float(world_limit)
        self.start_position = self._coordinate(
            start_position,
            name="start_position",
        )
        self.goal_position = self._coordinate(
            goal_position,
            name="goal_position",
        )
        if np.any(np.abs(self.start_position) > self.world_limit):
            raise ValueError("start_position must be inside the world")
        if np.any(np.abs(self.goal_position) > self.world_limit):
            raise ValueError("goal_position must be inside the world")
        self.goal_radius = float(goal_radius)
        self.max_steps = max_steps
        self.step_reward = float(step_reward)
        self.progress_reward = float(progress_reward)
        self.goal_reward = float(goal_reward)
        self.boundary_reward = float(boundary_reward)
        rewards = np.asarray(
            [
                self.step_reward,
                self.progress_reward,
                self.goal_reward,
                self.boundary_reward,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(rewards).all():
            raise ValueError("rewards must be finite")
        self.render_size = render_size
        self.render_mode = render_mode
        self.action_space = self.dynamics.action_space
        self.observation_space = gym.spaces.Box(
            low=np.asarray(
                [
                    -self.world_limit,
                    -self.world_limit,
                    -self.dynamics.max_speed,
                    -self.dynamics.max_speed,
                    -self.world_limit,
                    -self.world_limit,
                ],
                dtype=np.float32,
            ),
            high=np.asarray(
                [
                    self.world_limit,
                    self.world_limit,
                    self.dynamics.max_speed,
                    self.dynamics.max_speed,
                    self.world_limit,
                    self.world_limit,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.position = self.start_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.episode_steps = 0

    @staticmethod
    def _coordinate(
        value: tuple[float, float] | list[float],
        *,
        name: str,
    ) -> np.ndarray:
        coordinate = np.asarray(value, dtype=np.float32)
        if coordinate.shape != (2,) or not np.isfinite(coordinate).all():
            raise ValueError(f"{name} must contain two finite values")
        return coordinate

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            (self.position, self.velocity, self.goal_position)
        ).astype(np.float32)

    def _info(
        self,
        *,
        distance: float,
        boundary_collision: bool,
    ) -> dict[str, Any]:
        return {
            "is_success": distance <= self.goal_radius,
            "distance_to_goal": distance,
            "speed": float(np.linalg.norm(self.velocity)),
            "boundary_collision": boundary_collision,
            "time": self.episode_steps * self.dt,
            "dynamics_rule": type(self.dynamics).__name__,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset position, velocity, and elapsed simulation time."""
        super().reset(seed=seed)
        del options
        self.position = self.start_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.episode_steps = 0
        distance = float(np.linalg.norm(self.goal_position - self.position))
        return self._observation(), self._info(
            distance=distance,
            boundary_collision=False,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Integrate one configured time interval and score goal progress."""
        controls = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(controls):
            raise ValueError("Point-mass action is outside the configured space")
        previous_distance = float(np.linalg.norm(self.goal_position - self.position))
        next_position, next_velocity = self.dynamics.update(
            self.position,
            self.velocity,
            controls,
            dt=self.dt,
        )
        clipped_position = np.clip(
            next_position,
            -self.world_limit,
            self.world_limit,
        )
        boundary_axes = clipped_position != next_position
        boundary_collision = bool(boundary_axes.any())
        if boundary_collision:
            next_velocity = next_velocity.copy()
            next_velocity[boundary_axes] = 0.0
        self.position = clipped_position.astype(np.float32)
        self.velocity = next_velocity.astype(np.float32)
        self.episode_steps += 1

        distance = float(np.linalg.norm(self.goal_position - self.position))
        terminated = distance <= self.goal_radius
        truncated = self.episode_steps >= self.max_steps and not terminated
        reward = (
            self.step_reward
            + self.progress_reward * (previous_distance - distance)
            + self.boundary_reward * int(boundary_collision)
            + self.goal_reward * int(terminated)
        )
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            self._info(
                distance=distance,
                boundary_collision=boundary_collision,
            ),
        )

    def render(self) -> np.ndarray:
        """Render the point, goal, and current velocity as an RGB image."""
        frame = np.full(
            (self.render_size, self.render_size, 3),
            255,
            dtype=np.uint8,
        )

        def pixel(coordinate: np.ndarray) -> tuple[int, int]:
            normalized = (coordinate + self.world_limit) / (2.0 * self.world_limit)
            column = round(float(normalized[0]) * (self.render_size - 1))
            row = round(float(1.0 - normalized[1]) * (self.render_size - 1))
            return column, row

        goal_pixel = pixel(self.goal_position)
        position_pixel = pixel(self.position)
        radius = max(4, self.render_size // 40)
        cv2.circle(frame, goal_pixel, radius, (0, 0, 255), 2)
        cv2.circle(frame, position_pixel, radius, (255, 0, 0), -1)
        velocity_tip = np.clip(
            self.position + self.velocity * self.dt * 4.0,
            -self.world_limit,
            self.world_limit,
        )
        cv2.arrowedLine(
            frame,
            position_pixel,
            pixel(velocity_tip),
            (24, 120, 24),
            2,
            tipLength=0.25,
        )
        return frame

    def close(self) -> None:
        """Release environment resources."""


if "PointMassKinematics-v0" not in gym.registry:
    gym.register(
        id="PointMassKinematics-v0",
        entry_point=PointMass2DEnv,
    )


__all__ = ["PointMass2DEnv"]

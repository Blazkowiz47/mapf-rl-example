"""Shape-controlled RGB observations for procedural pathfinding."""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np
from dl_robotics import (
    GridObservationBuilder,
    GridScenario,
    GridWorldBatch,
    register_observation_builder,
)


@register_observation_builder("example_pathfinding_rgb")
class PathfindingRGBObservationBuilder(GridObservationBuilder):
    """Draw hollow blue goals and solid red actors for model input."""

    def __init__(self, *, output_size: int = 256) -> None:
        if isinstance(output_size, bool) or not isinstance(output_size, int):
            raise TypeError("output_size must be an integer")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.output_size = output_size
        self._scenario: GridScenario | None = None
        self._background: np.ndarray | None = None

    def _observation_space(
        self,
        scenario: GridScenario,
    ) -> gym.Space[np.ndarray]:
        del scenario
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.output_size, self.output_size, 3),
            dtype=np.uint8,
        )

    def _build(self, world: GridWorldBatch) -> np.ndarray:
        if self._scenario is not world.scenario or self._background is None:
            if self.output_size == world.scenario.width == world.scenario.height:
                wall_pixels = world.wall_mask
            else:
                wall_density = cv2.resize(
                    world.wall_mask.astype(np.uint8) * 255,
                    (self.output_size, self.output_size),
                    interpolation=cv2.INTER_AREA,
                )
                wall_pixels = wall_density >= 32
            self._background = np.full(
                (self.output_size, self.output_size, 3),
                255,
                dtype=np.uint8,
            )
            self._background[wall_pixels] = (42, 52, 68)
            goal_row, goal_column = world.goal_positions[0]
            row_scale = (self.output_size - 1) / max(
                world.scenario.height - 1,
                1,
            )
            column_scale = (self.output_size - 1) / max(
                world.scenario.width - 1,
                1,
            )
            goal_center = (
                round(int(goal_column) * column_scale),
                round(int(goal_row) * row_scale),
            )
            radius = max(4, self.output_size // 64)
            cv2.circle(
                self._background,
                goal_center,
                radius,
                (0, 0, 255),
                max(2, radius // 2),
                lineType=cv2.LINE_AA,
            )
            self._scenario = world.scenario

        row_scale = (self.output_size - 1) / max(
            world.scenario.height - 1,
            1,
        )
        column_scale = (self.output_size - 1) / max(
            world.scenario.width - 1,
            1,
        )
        radius = max(4, self.output_size // 64)
        observations = []
        for world_index in range(world.num_worlds):
            frame = self._background.copy()
            actor_row, actor_column = world.positions[world_index, 0]
            actor_center = (
                round(int(actor_column) * column_scale),
                round(int(actor_row) * row_scale),
            )
            cv2.circle(
                frame,
                actor_center,
                radius,
                (255, 0, 0),
                -1,
                lineType=cv2.LINE_AA,
            )
            observations.append(frame)
        return np.stack(observations)


__all__ = ["PathfindingRGBObservationBuilder"]

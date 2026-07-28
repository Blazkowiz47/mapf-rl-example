"""Discrete joint-position observations for tabular MAPF learning."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from dl_robotics import (
    GridObservationBuilder,
    GridScenario,
    GridWorldBatch,
    register_observation_builder,
)


@register_observation_builder("example_tabular_mapf")
class TabularMAPFObservationBuilder(GridObservationBuilder):
    """Encode ordered actor positions as one finite discrete state."""

    def observation_space(self, scenario: GridScenario) -> gym.Space[int]:
        """Return one state for every ordered combination of actor cells."""
        cell_count = scenario.width * scenario.height
        return gym.spaces.Discrete(cell_count**scenario.num_agents)

    def build(self, world: GridWorldBatch) -> np.ndarray:
        """Encode each world lane using actor-index order and base cell count."""
        cell_count = world.scenario.width * world.scenario.height
        cell_indices = (
            world.positions[..., 0] * world.scenario.width + world.positions[..., 1]
        )
        states = np.zeros(world.num_worlds, dtype=np.int64)
        multiplier = 1
        for actor_index in range(world.scenario.num_agents):
            states += cell_indices[:, actor_index].astype(np.int64) * multiplier
            multiplier *= cell_count
        return states


__all__ = ["TabularMAPFObservationBuilder"]

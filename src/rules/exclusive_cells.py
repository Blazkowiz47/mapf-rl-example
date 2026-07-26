"""Explicit copy of the package's default exclusive-cell behavior."""

from __future__ import annotations

import numpy as np
from dl_robotics import GridScenario, InteractionRule, register_interaction_rule
from numpy.typing import NDArray


@register_interaction_rule("example_exclusive_cell")
class ExampleExclusiveCellRule(InteractionRule):
    """Reject shared targets, edge swaps, and moves into stationary actors."""

    def resolve(
        self,
        scenario: GridScenario,
        positions: NDArray[np.int32],
        desired_positions: NDArray[np.int32],
        blocked: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Repeat the default behavior so experiments can edit it locally."""
        del scenario
        resolved = desired_positions.copy()
        actor_collisions = np.zeros(positions.shape[0], dtype=np.int32)
        for world_index in range(positions.shape[0]):
            current = positions[world_index]
            desired = resolved[world_index]
            rejected = blocked[world_index].copy()
            changed = True
            while changed:
                changed = False
                candidate = desired.copy()
                candidate[rejected] = current[rejected]
                for actor_index in range(current.shape[0]):
                    for other_index in range(
                        actor_index + 1,
                        current.shape[0],
                    ):
                        same_target = np.array_equal(
                            candidate[actor_index],
                            candidate[other_index],
                        )
                        edge_swap = np.array_equal(
                            candidate[actor_index],
                            current[other_index],
                        ) and np.array_equal(
                            candidate[other_index],
                            current[actor_index],
                        )
                        if same_target or edge_swap:
                            for index in (actor_index, other_index):
                                if not rejected[index] and not np.array_equal(
                                    desired[index],
                                    current[index],
                                ):
                                    rejected[index] = True
                                    changed = True
            resolved[world_index, rejected] = current[rejected]
            actor_collisions[world_index] = int(
                np.logical_and(rejected, ~blocked[world_index]).sum()
            )
        return resolved, actor_collisions

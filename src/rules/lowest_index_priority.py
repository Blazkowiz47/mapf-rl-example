"""Example rule that gives lower-index actors movement priority."""

from __future__ import annotations

import numpy as np
from dl_robotics import GridScenario, InteractionRule, register_interaction_rule
from numpy.typing import NDArray


@register_interaction_rule("lowest_index_priority")
class LowestIndexPriorityRule(InteractionRule):
    """Let the lowest-index moving actor win a shared destination."""

    def resolve(
        self,
        scenario: GridScenario,
        positions: NDArray[np.int32],
        desired_positions: NDArray[np.int32],
        blocked: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Resolve shared targets by actor index while keeping cells exclusive."""
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
                        actor_moves = not np.array_equal(
                            desired[actor_index],
                            current[actor_index],
                        )
                        other_moves = not np.array_equal(
                            desired[other_index],
                            current[other_index],
                        )
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
                        newly_rejected: tuple[int, ...] = ()
                        if edge_swap:
                            newly_rejected = (actor_index, other_index)
                        elif same_target and actor_moves and other_moves:
                            newly_rejected = (other_index,)
                        elif same_target and actor_moves:
                            newly_rejected = (actor_index,)
                        elif same_target and other_moves:
                            newly_rejected = (other_index,)
                        for index in newly_rejected:
                            if not rejected[index]:
                                rejected[index] = True
                                changed = True
            resolved[world_index, rejected] = current[rejected]
            actor_collisions[world_index] = int(
                np.logical_and(rejected, ~blocked[world_index]).sum()
            )
        return resolved, actor_collisions

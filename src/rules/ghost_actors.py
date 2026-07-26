"""Example rule that allows actors to overlap and pass through each other."""

from __future__ import annotations

import numpy as np
from dl_robotics import GridScenario, InteractionRule, register_interaction_rule
from numpy.typing import NDArray


@register_interaction_rule("ghost_actors")
class GhostActorsRule(InteractionRule):
    """Ignore actor-to-actor conflicts while retaining wall and boundary rules."""

    def resolve(
        self,
        scenario: GridScenario,
        positions: NDArray[np.int32],
        desired_positions: NDArray[np.int32],
        blocked: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Accept every position already validated by the world geometry."""
        del scenario, positions, blocked
        return (
            desired_positions.copy(),
            np.zeros(desired_positions.shape[0], dtype=np.int32),
        )

"""Velocity-controlled point-mass dynamics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import PointMassDynamicsRule, register_dynamics_rule


@register_dynamics_rule("velocity")
class VelocityDynamicsRule(PointMassDynamicsRule):
    """Treat the policy action as the velocity for the next time interval."""

    def __init__(self, *, max_speed: float = 3.0) -> None:
        super().__init__(max_speed=max_speed)
        self._action_space = gym.spaces.Box(
            low=-self.max_speed,
            high=self.max_speed,
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        """Return bounded x/y velocity commands."""
        return self._action_space

    def update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        action: np.ndarray,
        *,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance position using the newly commanded velocity."""
        del velocity
        next_velocity = np.asarray(action, dtype=np.float32)
        next_position = position + next_velocity * dt
        return next_position.astype(np.float32), next_velocity.copy()


__all__ = ["VelocityDynamicsRule"]

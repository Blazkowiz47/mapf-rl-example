"""Acceleration-controlled point-mass dynamics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import PointMassDynamicsRule, register_dynamics_rule


@register_dynamics_rule("acceleration")
class AccelerationDynamicsRule(PointMassDynamicsRule):
    """Integrate bounded acceleration with constant-acceleration kinematics."""

    def __init__(
        self,
        *,
        max_acceleration: float = 2.0,
        max_speed: float = 3.0,
    ) -> None:
        super().__init__(max_speed=max_speed)
        if not np.isfinite(max_acceleration) or max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be finite and positive")
        self.max_acceleration = float(max_acceleration)
        self._action_space = gym.spaces.Box(
            low=-self.max_acceleration,
            high=self.max_acceleration,
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        """Return bounded x/y acceleration commands."""
        return self._action_space

    def update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        action: np.ndarray,
        *,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply `p + v*dt + 0.5*a*dt²` and `v + a*dt`."""
        acceleration = np.asarray(action, dtype=np.float32)
        next_position = position + velocity * dt + 0.5 * acceleration * dt**2
        next_velocity = np.clip(
            velocity + acceleration * dt,
            -self.max_speed,
            self.max_speed,
        )
        return next_position.astype(np.float32), next_velocity.astype(np.float32)


__all__ = ["AccelerationDynamicsRule"]

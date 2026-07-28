"""Dynamics-rule contract for the continuous point-mass example."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np


class PointMassDynamicsRule(ABC):
    """Map a continuous control command to the next kinematic state."""

    def __init__(self, *, max_speed: float) -> None:
        if not np.isfinite(max_speed) or max_speed <= 0.0:
            raise ValueError("max_speed must be finite and positive")
        self.max_speed = float(max_speed)

    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        """Return the continuous control space for this rule."""

    @abstractmethod
    def update(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        action: np.ndarray,
        *,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the next position and velocity."""


DYNAMICS_RULES: dict[str, type[PointMassDynamicsRule]] = {}


def register_dynamics_rule(name: str):
    """Register one named point-mass dynamics rule."""

    def decorator(
        rule_class: type[PointMassDynamicsRule],
    ) -> type[PointMassDynamicsRule]:
        if not isinstance(rule_class, type) or not issubclass(
            rule_class,
            PointMassDynamicsRule,
        ):
            raise TypeError("Dynamics rules must inherit PointMassDynamicsRule")
        if name in DYNAMICS_RULES:
            raise ValueError(f"Dynamics rule '{name}' is already registered")
        DYNAMICS_RULES[name] = rule_class
        return rule_class

    return decorator


def make_dynamics_rule(config: dict[str, Any]) -> PointMassDynamicsRule:
    """Create a registered rule from a YAML-compatible mapping."""
    if not isinstance(config, dict):
        raise TypeError("dynamics must be a mapping")
    rule_config = dict(config)
    name = rule_config.pop("name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("dynamics.name must be a non-empty string")
    if name not in DYNAMICS_RULES:
        raise NotImplementedError(
            f"Dynamics rule '{name}' not found. Available rules: "
            f"{sorted(DYNAMICS_RULES)}"
        )
    return DYNAMICS_RULES[name](**rule_config)


__all__ = [
    "DYNAMICS_RULES",
    "PointMassDynamicsRule",
    "make_dynamics_rule",
    "register_dynamics_rule",
]

"""Continuous-action adapter for the discrete MAPF teaching task."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from dl_robotics import GridMAPFEnvironment


class ContinuousActionMAPFEnv(GridMAPFEnvironment):
    """Quantize per-agent movement vectors into the existing grid actions."""

    def __init__(
        self,
        *,
        scenario: dict[str, Any],
        interaction_rule: str | dict[str, Any] | None = None,
        rewards: dict[str, float] | None = None,
        render: dict[str, Any] | None = None,
        observation_builder: str | dict[str, Any] | None = None,
        action_dead_zone: float = 0.2,
        render_mode: str | None = None,
    ) -> None:
        if not 0.0 <= action_dead_zone < 1.0:
            raise ValueError("action_dead_zone must be in [0, 1)")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        config: dict[str, Any] = {
            "scenario": scenario,
            "interaction_rule": interaction_rule,
            "rewards": {} if rewards is None else rewards,
            "render": {} if render is None else render,
        }
        if observation_builder is not None:
            config["observation_builder"] = observation_builder
        super().__init__(config)
        self.action_dead_zone = float(action_dead_zone)
        self.render_mode = render_mode
        self.discrete_action_space = self.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.scenario.num_agents, 2),
            dtype=np.float32,
        )

    def decode_action(self, action: np.ndarray) -> int:
        """Convert `[vertical, horizontal]` controls into one joint action."""
        controls = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(controls):
            raise ValueError("Continuous MAPF action is outside the configured space")

        joint_action = 0
        multiplier = 1
        for vertical, horizontal in controls:
            if max(abs(float(vertical)), abs(float(horizontal))) < (
                self.action_dead_zone
            ):
                actor_action = 0
            elif abs(float(vertical)) >= abs(float(horizontal)):
                actor_action = 1 if vertical < 0.0 else 3
            else:
                actor_action = 4 if horizontal < 0.0 else 2
            joint_action += actor_action * multiplier
            multiplier *= 5
        return joint_action

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Quantize the continuous command and advance the grid world."""
        return super().step(self.decode_action(action))


if "ContinuousActionMAPF-v0" not in gym.registry:
    gym.register(
        id="ContinuousActionMAPF-v0",
        entry_point=ContinuousActionMAPFEnv,
    )


__all__ = ["ContinuousActionMAPFEnv"]

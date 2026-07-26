"""Rate-limited W&B logging for the two-billion-step pathfinding run."""

from __future__ import annotations

from typing import Any, ClassVar

from dl_core.core import config_field, register_callback
from dl_wandb.callbacks.wandb import WandbCallback


@register_callback("sampled_wandb")
class SampledWandbCallback(WandbCallback):
    """Use dl-wandb while sampling high-frequency RL callback events."""

    CONFIG_FIELDS: ClassVar[list[dict[str, Any]]] = (
        WandbCallback.CONFIG_FIELDS
        + [
            config_field(
                "episode_log_frequency",
                "int",
                "Log every Nth completed training episode.",
                default=1000,
            ),
            config_field(
                "update_log_frequency",
                "int",
                "Log every Nth DQN update.",
                default=100,
            ),
        ]
    )

    def __init__(
        self,
        *,
        episode_log_frequency: int = 1000,
        update_log_frequency: int = 100,
        **kwargs: Any,
    ):
        if (
            isinstance(episode_log_frequency, bool)
            or not isinstance(episode_log_frequency, int)
            or episode_log_frequency <= 0
        ):
            raise ValueError("episode_log_frequency must be a positive integer")
        if (
            isinstance(update_log_frequency, bool)
            or not isinstance(update_log_frequency, int)
            or update_log_frequency <= 0
        ):
            raise ValueError("update_log_frequency must be a positive integer")
        super().__init__(**kwargs)
        self.episode_log_frequency = episode_log_frequency
        self.update_log_frequency = update_log_frequency

    def _on_episode_end(
        self,
        episode: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        if episode % self.episode_log_frequency == 0:
            super()._on_episode_end(episode, logs)

    def _on_update_end(
        self,
        update: int,
        logs: dict[str, Any] | None = None,
    ) -> None:
        if update % self.update_log_frequency == 0:
            super()._on_update_end(update, logs)


__all__ = ["SampledWandbCallback"]

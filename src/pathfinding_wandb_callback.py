"""Rate-limited W&B logging for the two-billion-step pathfinding run."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from dl_core.core import config_field, register_callback
from dl_wandb.callbacks.wandb import WandbCallback

import wandb


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
            config_field(
                "dense_update_count",
                "int",
                "Log every update through this initial update count.",
                default=100,
            ),
            config_field(
                "watch_log_frequency",
                "int",
                "DQN updates between W&B parameter and gradient histograms.",
                default=500,
            ),
        ]
    )

    def __init__(
        self,
        *,
        episode_log_frequency: int = 1000,
        update_log_frequency: int = 100,
        dense_update_count: int = 100,
        watch_log_frequency: int = 500,
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
        if (
            isinstance(dense_update_count, bool)
            or not isinstance(dense_update_count, int)
            or dense_update_count < 0
        ):
            raise ValueError("dense_update_count must be a non-negative integer")
        if (
            isinstance(watch_log_frequency, bool)
            or not isinstance(watch_log_frequency, int)
            or watch_log_frequency <= 0
        ):
            raise ValueError("watch_log_frequency must be a positive integer")
        super().__init__(**kwargs)
        self.episode_log_frequency = episode_log_frequency
        self.update_log_frequency = update_log_frequency
        self.dense_update_count = dense_update_count
        self.watch_log_frequency = watch_log_frequency

    def on_training_start(
        self,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """Start W&B and watch online-network weights and gradients."""
        self._on_training_start(logs)

    def _on_training_start(
        self,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().on_training_start(logs)
        if self.run is not None:
            online_model = self.trainer.models["online"]
            self._watched_modules = [
                module
                for module in online_model.modules()
                if any(
                    parameter.requires_grad
                    for parameter in module.parameters(recurse=False)
                )
            ]
            wandb.watch(
                self._watched_modules,
                log="all",
                log_freq=self.watch_log_frequency,
                log_graph=False,
            )

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
        if (
            update <= self.dense_update_count
            or update % self.update_log_frequency == 0
        ):
            update_logs = dict(logs or {})
            trainable_parameters = [
                parameter
                for parameter in self.trainer.models["online"].parameters()
                if parameter.requires_grad
            ]
            if trainable_parameters:
                update_logs["model/weight_l2_norm"] = float(
                    torch.stack(
                        [
                            parameter.detach().float().square().sum()
                            for parameter in trainable_parameters
                        ]
                    )
                    .sum()
                    .sqrt()
                    .item()
                )
                gradients = [
                    parameter.grad.detach()
                    for parameter in trainable_parameters
                    if parameter.grad is not None
                ]
                if gradients:
                    update_logs["model/gradient_l2_norm"] = float(
                        torch.stack(
                            [
                                gradient.float().square().sum()
                                for gradient in gradients
                            ]
                        )
                        .sum()
                        .sqrt()
                        .item()
                    )
            super()._on_update_end(update, update_logs)


__all__ = ["SampledWandbCallback"]

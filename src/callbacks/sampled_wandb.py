"""Rate-limited W&B logging for the reference pathfinding run."""

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
                "DQN updates between W&B model histogram samples.",
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
        self._manual_weight_histograms = False

    def on_training_start(
        self,
        logs: dict[str, Any] | None = None,
    ) -> None:
        """Start W&B and configure online-model monitoring."""
        self._on_training_start(logs)

    def _on_training_start(
        self,
        logs: dict[str, Any] | None = None,
    ) -> None:
        super().on_training_start(logs)
        if self.run is not None:
            online_model = self.trainer.models["online"]
            self._manual_weight_histograms = (
                getattr(online_model, "_compiled_call_impl", None) is not None
            )
            if self._manual_weight_histograms:
                return
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
        update_logs = dict(logs or {})
        if (
            update <= self.dense_update_count
            or update % self.update_log_frequency == 0
        ):
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
            super()._on_update_end(update, update_logs)

        if (
            self._manual_weight_histograms
            and self.run is not None
            and update % self.watch_log_frequency == 0
        ):
            histogram_logs: dict[str, Any] = {
                "global_step": float(update_logs.get("global_step", update))
            }
            for name, parameter in self.trainer.models[
                "online"
            ].named_parameters():
                if parameter.requires_grad:
                    histogram_logs[f"model/weights/{name}"] = wandb.Histogram(
                        parameter.detach().float().cpu().numpy()
                    )
            if not self._rl_step_metric_defined:
                wandb.define_metric("global_step")
                self._rl_step_metric_defined = True
            for metric_name in histogram_logs:
                if (
                    metric_name == "global_step"
                    or metric_name in self._rl_metric_names
                ):
                    continue
                wandb.define_metric(
                    metric_name,
                    step_metric="global_step",
                )
                self._rl_metric_names.add(metric_name)
            wandb.log(histogram_logs)


__all__ = ["SampledWandbCallback"]

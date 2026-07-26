"""Tests for sampled W&B logging in the long pathfinding run."""

from unittest.mock import Mock

from pathfinding_wandb_callback import SampledWandbCallback


def test_sampled_wandb_limits_episode_and_update_events() -> None:
    callback = SampledWandbCallback(
        episode_log_frequency=10,
        update_log_frequency=4,
    )
    callback._log_rl_metrics = Mock()

    callback.on_episode_end(9, {"global_step": 90})
    callback.on_episode_end(10, {"global_step": 100})
    callback.on_update_end(3, {"global_step": 30})
    callback.on_update_end(4, {"global_step": 40})

    assert callback._log_rl_metrics.call_count == 2
    assert callback._log_rl_metrics.call_args_list[0].kwargs == {
        "fallback_step": 10
    }
    assert callback._log_rl_metrics.call_args_list[1].kwargs == {
        "fallback_step": 4
    }

"""Tests for sampled W&B logging in the long pathfinding run."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from dl_core.core import EpisodeContext, EpisodeRecord, EpisodeResult

import wandb
from callbacks import SampledWandbCallback
from episode_managers import PathfindingEpisodeManager


def test_sampled_wandb_watches_model_and_limits_high_frequency_events(
    monkeypatch,
) -> None:
    callback = SampledWandbCallback(
        episode_log_frequency=10,
        update_log_frequency=4,
        dense_update_count=2,
        watch_log_frequency=7,
    )
    online_model = torch.nn.Linear(2, 2)
    callback.trainer = SimpleNamespace(
        accelerator=None,
        config={},
        models={"online": online_model},
        artifact_manager=None,
    )
    monkeypatch.setattr(
        wandb,
        "init",
        Mock(return_value=SimpleNamespace()),
    )
    watch = Mock()
    monkeypatch.setattr(wandb, "watch", watch)

    callback.on_training_start()

    watch.assert_called_once_with(
        [online_model],
        log="all",
        log_freq=7,
        log_graph=False,
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
    logged_update = callback._log_rl_metrics.call_args_list[1].args[0]
    assert logged_update["model/weight_l2_norm"] > 0


def test_pathfinding_manager_reports_shortest_path_gap_and_efficiency() -> None:
    manager = PathfindingEpisodeManager({"media_format": "none"})
    context = EpisodeContext(
        episode_id="evaluation-0001",
        episode=1,
        environment_index=0,
        phase="evaluation",
        seed=2026,
        initial_observation=0,
        reset_info={},
        start_global_step=0,
    )
    record = EpisodeRecord(context=context)
    result = EpisodeResult(
        episode=1,
        episode_return=2.0,
        length=10,
        terminated=True,
        truncated=False,
        final_info={
            "is_success": True,
            "shortest_path_length": 50,
            "path_length": 65,
            "excess_path_length": 15,
            "distance_to_goal": 0,
        },
        episode_id="evaluation-0001",
        environment_index=0,
        seed=2026,
        completion_reason="terminated",
    )

    metrics = manager.summarize_episode(
        record,
        result,
        length=10,
        episode_return=2.0,
        reward_square_sum=1.0,
        reward_min=-0.1,
        reward_max=1.0,
    )

    assert metrics["episode/success"] == 1.0
    assert metrics["pathfinding/expected_shortest_path"] == 50.0
    assert metrics["pathfinding/total_steps_taken"] == 65.0
    assert metrics["pathfinding/step_difference"] == 15.0
    assert metrics["pathfinding/excess_path_length"] == 15.0
    assert metrics["pathfinding/path_efficiency"] == 50 / 65

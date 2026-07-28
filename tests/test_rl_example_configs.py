"""Short integration coverage for each standalone RL trainer example."""

import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from dl_core import load_builtin_components
from dl_core.core import TRAINER_REGISTRY
from dl_core.trainers import (
    DreamerTrainer,
    PPOTrainer,
    QLearningTrainer,
    SACTrainer,
)

import bootstrap  # noqa: F401


@pytest.mark.parametrize(
    ("config_name", "trainer_class", "captures_evaluation", "captures_media"),
    [
        ("mapf_q_learning.yaml", QLearningTrainer, True, False),
        ("mapf_dreamer.yaml", DreamerTrainer, True, True),
        ("mapf_ppo.yaml", PPOTrainer, True, True),
        ("mapf_sac.yaml", SACTrainer, True, True),
        ("point_mass_acceleration_sac.yaml", SACTrainer, False, False),
        ("point_mass_velocity_ppo.yaml", PPOTrainer, False, False),
    ],
)
def test_rl_example_performs_short_training(
    config_name: str,
    trainer_class: type,
    captures_evaluation: bool,
    captures_media: bool,
    tmp_path: Path,
) -> None:
    load_builtin_components()
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "configs" / config_name).read_text())
    trainer_name = next(iter(config["trainer"]))
    public_trainer_config = config["trainer"][trainer_name]
    assert public_trainer_config["total_timesteps"] == 100_000
    assert public_trainer_config["show_progress"] is True
    assert public_trainer_config["checkpoint_frequency_steps"] == 25_000
    expected_accelerator = (
        "cpu" if config_name == "mapf_q_learning.yaml" else "single_gpu"
    )
    assert config["accelerator"]["type"] == expected_accelerator
    config["accelerator"] = {"type": "cpu"}
    config["runtime"]["output_dir"] = str(tmp_path / "artifacts")
    config["runtime"]["name"] = f"test_{Path(config_name).stem}"
    smoke_overrides = {
        "mapf_q_learning.yaml": {
            "total_timesteps": 64,
            "evaluation_frequency": 4,
            "evaluation_episodes": 1,
            "show_progress": False,
            "epsilon_decay_steps": 64,
        },
        "mapf_dreamer.yaml": {
            "total_timesteps": 64,
            "evaluation_frequency": 4,
            "evaluation_episodes": 1,
            "show_progress": False,
            "buffer_size": 256,
            "batch_size": 8,
            "sequence_length": 6,
            "burn_in": 2,
            "learning_starts": 32,
            "train_frequency": 8,
        },
        "mapf_ppo.yaml": {
            "total_timesteps": 64,
            "evaluation_frequency": 4,
            "evaluation_episodes": 1,
            "show_progress": False,
            "rollout_steps": 4,
            "update_epochs": 2,
            "minibatch_size": 8,
        },
        "mapf_sac.yaml": {
            "total_timesteps": 64,
            "evaluation_frequency": 4,
            "evaluation_episodes": 1,
            "show_progress": False,
            "buffer_size": 256,
            "batch_size": 8,
            "learning_starts": 8,
        },
        "point_mass_acceleration_sac.yaml": {
            "total_timesteps": 32,
            "evaluation_episodes": 0,
            "show_progress": False,
            "buffer_size": 512,
            "batch_size": 16,
            "learning_starts": 16,
        },
        "point_mass_velocity_ppo.yaml": {
            "total_timesteps": 32,
            "evaluation_episodes": 0,
            "show_progress": False,
            "rollout_steps": 8,
            "update_epochs": 2,
            "minibatch_size": 16,
        },
    }
    config["trainer"][trainer_name].update(smoke_overrides[config_name])

    assert TRAINER_REGISTRY.get_class(trainer_name) is trainer_class
    trainer = TRAINER_REGISTRY.get(trainer_name, config)
    trainer.setup()
    try:
        trainer.perform_training()
    finally:
        trainer.close()

    assert trainer.global_step == smoke_overrides[config_name]["total_timesteps"]
    assert trainer.update_step > 0
    if isinstance(trainer, DreamerTrainer):
        artifact_root = tmp_path / "artifacts"
        metric_files = list(
            artifact_root.rglob("dreamer_world_model_loss.jsonl")
        )
        assert len(metric_files) == 1
        losses = [
            json.loads(line)["value"]
            for line in metric_files[0].read_text().splitlines()
        ]
        assert losses
        assert all(math.isfinite(loss) for loss in losses)
        checkpoints = list(artifact_root.rglob("latest.pth"))
        assert len(checkpoints) == 1
        checkpoint = torch.load(
            checkpoints[0],
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["algorithm_state"]["checkpoint_replay_buffer"] is True
        assert checkpoint["algorithm_state"]["replay_buffer"] is not None
    if captures_evaluation:
        artifact_root = tmp_path / "artifacts"
        assert list(artifact_root.rglob("*.npz"))
        assert bool(list(artifact_root.rglob("*.gif"))) is captures_media

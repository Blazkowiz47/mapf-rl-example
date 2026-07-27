"""Tests for the procedural pathfinding ViT Q-network."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import yaml
from dl_core import load_builtin_components
from dl_core.trainers import DQNTrainer
from torchvision.models.vision_transformer import (
    interpolate_embeddings as interpolate_vit_embeddings,
)

import environments  # noqa: F401
from models import ViTB16QNetwork, vit_q_network


def test_vit_q_network_returns_four_q_values() -> None:
    model = ViTB16QNetwork(
        {
            "input_dim": 256 * 256 * 3,
            "action_dim": 4,
            "pretrained": False,
            "trainable_blocks": 12,
        }
    )

    with torch.no_grad():
        output = model(torch.zeros(1, 256, 256, 3))

    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()
    assert all(
        parameter.requires_grad
        for parameter in model.network.heads.parameters()
    )
    assert model.network.encoder.layers[0].ln_1.weight.requires_grad


def test_dqn_injects_dimensions_and_loads_interpolated_pretrained_weights(
    monkeypatch,
    tmp_path: Path,
) -> None:
    interpolated_state = interpolate_vit_embeddings(
        image_size=256,
        patch_size=16,
        model_state={
            "encoder.pos_embedding": torch.zeros(1, 197, 768),
        },
    )
    assert interpolated_state["encoder.pos_embedding"].shape == (
        1,
        257,
        768,
    )

    class TinyVisionTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.Sequential(torch.nn.Identity())
            self.encoder.ln = torch.nn.LayerNorm(3)
            self.heads = torch.nn.Linear(3, 4)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.heads(images.mean(dim=(2, 3)))

        def load_state_dict(
            self,
            state_dict,
            strict: bool = True,
            assign: bool = False,
        ):
            del state_dict, strict, assign
            return SimpleNamespace(
                missing_keys=[
                    "heads.head.weight",
                    "heads.head.bias",
                ],
                unexpected_keys=[],
            )

    tiny_factory = Mock(return_value=TinyVisionTransformer())
    pretrained_weights = Mock()
    pretrained_weights.get_state_dict.return_value = {
        "encoder.pos_embedding": torch.zeros(1, 197, 768)
    }
    interpolation = Mock(return_value={})
    monkeypatch.setattr(vit_q_network, "vit_b_16", tiny_factory)
    monkeypatch.setattr(
        vit_q_network,
        "ViT_B_16_Weights",
        SimpleNamespace(IMAGENET1K_V1=pretrained_weights),
    )
    monkeypatch.setattr(
        vit_q_network,
        "interpolate_embeddings",
        interpolation,
    )
    load_builtin_components()
    environment = {
        "name": "gymnasium",
        "id": "ProceduralPathfinding-v0",
        "kwargs": {
            "grid_size": 64,
            "observation_size": 256,
            "max_steps": 8,
            "move_stride": 1,
            "wall_count": 0,
            "minimum_goal_distance": 1,
            "maximum_goal_distance": 1,
        },
    }
    trainer = DQNTrainer(
        {
            "seed": 2026,
            "runtime": {
                "output_dir": str(tmp_path),
                "name": "vit_setup_test",
            },
            "accelerator": {"type": "cpu"},
            "environment": environment,
            "evaluation_environment": environment,
            "models": {
                "q_network": {
                    "name": "vit_b_16_q_network",
                    "pretrained": True,
                    "trainable_blocks": 0,
                }
            },
            "optimizers": {"name": "adam", "lr": 0.001},
            "trainer": {
                "dqn": {
                    "total_timesteps": 1,
                    "max_episode_steps": 1,
                    "buffer_size": 2,
                    "batch_size": 1,
                    "learning_starts": 1,
                    "train_frequency": 1,
                    "target_update_frequency": 1,
                }
            },
            "episode_managers": {"standard": {"capture_phases": []}},
            "callbacks": {},
        }
    )

    trainer.setup()
    try:
        assert isinstance(trainer.models["online"], ViTB16QNetwork)
        assert tiny_factory.call_args.kwargs == {
            "weights": None,
            "image_size": 256,
            "num_classes": 4,
        }
        pretrained_weights.get_state_dict.assert_called_once_with(
            progress=True,
            check_hash=True,
        )
        interpolation.assert_called_once()
        assert interpolation.call_args.kwargs["image_size"] == 256
        assert interpolation.call_args.kwargs["patch_size"] == 16
        assert interpolation.call_args.kwargs["reset_heads"] is True
    finally:
        trainer.close()


def test_pathfinding_config_defines_the_requested_reference_run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "pathfinding_vit_dqn.yaml").read_text()
    )

    assert config["trainer"]["dqn"]["total_timesteps"] == 2_000_000
    assert config["trainer"]["dqn"]["checkpoint_frequency"] == 0
    assert config["trainer"]["dqn"]["checkpoint_frequency_steps"] == 100_000
    assert config["trainer"]["dqn"]["show_progress"] is True
    assert config["models"]["q_network"]["name"] == "vit_b_16_q_network"
    assert config["environment"]["num_envs"] == 32
    assert config["trainer"]["dqn"]["batch_size"] == 512
    assert config["trainer"]["dqn"]["train_frequency"] == 256
    assert config["trainer"]["dqn"]["actor_model_copies"] == 2
    assert config["trainer"]["dqn"]["actor_model_sync_frequency"] == 25
    assert "vectorization_mode" not in config["environment"]
    assert config["environment"]["kwargs"]["grid_size"] == 1000
    assert config["environment"]["kwargs"]["observation_size"] == 256
    assert config["trainer"]["dqn"]["buffer_size"] == 4096
    assert config["tracking"]["backend"] == "wandb"
    assert "pathfinding" in config["episode_managers"]
    assert config["callbacks"]["sampled_wandb"]["dense_update_count"] == 100
    assert config["callbacks"]["sampled_wandb"]["watch_log_frequency"] == 500

    pressure_config = yaml.safe_load(
        (project_root / "configs" / "pathfinding_vit_256_envs.yaml").read_text()
    )
    assert pressure_config["environment"]["num_envs"] == 256
    assert pressure_config["trainer"]["dqn"]["batch_size"] == 512
    assert pressure_config["trainer"]["dqn"]["train_frequency"] == 256
    assert pressure_config["trainer"]["dqn"]["actor_model_copies"] == 2
    assert pressure_config["trainer"]["dqn"]["actor_model_sync_frequency"] == 25
    assert pressure_config["runtime"]["name"] != config["runtime"]["name"]
    assert (
        pressure_config["tracking"]["run_name"]
        != config["tracking"]["run_name"]
    )

    pipelined_config = yaml.safe_load(
        (project_root / "configs" / "pathfinding_vit_pipelined.yaml").read_text()
    )
    assert pipelined_config["environment"] == config["environment"]
    assert pipelined_config["evaluation_environment"] == config[
        "evaluation_environment"
    ]
    assert pipelined_config["models"] == config["models"]
    assert pipelined_config["optimizers"] == config["optimizers"]
    assert pipelined_config["accelerator"] == config["accelerator"]
    pipelined_dqn = pipelined_config["trainer"]["dqn"]
    assert {
        key: value
        for key, value in pipelined_dqn.items()
        if key != "overlap_environment_steps"
    } == config["trainer"]["dqn"]
    assert pipelined_dqn["overlap_environment_steps"] is True
    assert pipelined_dqn["checkpoint_frequency"] == 0
    assert pipelined_dqn["checkpoint_frequency_steps"] == 100_000
    assert pipelined_dqn["show_progress"] is True
    assert pipelined_config["tracking"]["backend"] == "wandb"
    assert pipelined_config["tracking"]["experiment_name"] == config[
        "tracking"
    ]["experiment_name"]
    for setting in (
        "project",
        "job_type",
        "log_config",
        "episode_log_frequency",
        "update_log_frequency",
        "dense_update_count",
        "watch_log_frequency",
    ):
        assert pipelined_config["callbacks"]["sampled_wandb"][setting] == config[
            "callbacks"
        ]["sampled_wandb"][setting]
    assert pipelined_config["runtime"]["name"] not in {
        config["runtime"]["name"],
        pressure_config["runtime"]["name"],
    }
    assert pipelined_config["tracking"]["run_name"] not in {
        config["tracking"]["run_name"],
        pressure_config["tracking"]["run_name"],
    }

    large_batch_config = yaml.safe_load(
        (
            project_root
            / "configs"
            / "pathfinding_vit_pipelined_b1024.yaml"
        ).read_text()
    )
    assert large_batch_config.keys() == pipelined_config.keys()
    for section in (
        "seed",
        "deterministic",
        "accelerator",
        "environment",
        "evaluation_environment",
        "models",
        "optimizers",
        "episode_managers",
    ):
        assert large_batch_config[section] == pipelined_config[section]

    large_batch_dqn = large_batch_config["trainer"]["dqn"]
    assert large_batch_dqn["batch_size"] == 1024
    assert large_batch_dqn["train_frequency"] == 512
    assert large_batch_dqn["batch_size"] == (
        2 * large_batch_dqn["train_frequency"]
    )
    assert {
        key: value
        for key, value in large_batch_dqn.items()
        if key not in {"batch_size", "train_frequency"}
    } == {
        key: value
        for key, value in pipelined_dqn.items()
        if key not in {"batch_size", "train_frequency"}
    }

    for section, metadata_keys in (
        ("runtime", {"name"}),
        ("experiment", {"name", "description"}),
        ("tracking", {"run_name"}),
    ):
        assert {
            key: value
            for key, value in large_batch_config[section].items()
            if key not in metadata_keys
        } == {
            key: value
            for key, value in pipelined_config[section].items()
            if key not in metadata_keys
        }

    assert (
        large_batch_config["callbacks"]["local_metric_tracker"]
        == pipelined_config["callbacks"]["local_metric_tracker"]
    )
    large_batch_wandb = large_batch_config["callbacks"]["sampled_wandb"]
    pipelined_wandb = pipelined_config["callbacks"]["sampled_wandb"]
    assert {
        key: value
        for key, value in large_batch_wandb.items()
        if key not in {"tags", "notes"}
    } == {
        key: value
        for key, value in pipelined_wandb.items()
        if key not in {"tags", "notes"}
    }
    assert large_batch_wandb["tags"] == [
        *pipelined_wandb["tags"],
        "batch-1024",
    ]
    assert large_batch_wandb["notes"]

    previous_runtime_names = {
        config["runtime"]["name"],
        pressure_config["runtime"]["name"],
        pipelined_config["runtime"]["name"],
    }
    previous_run_names = {
        config["tracking"]["run_name"],
        pressure_config["tracking"]["run_name"],
        pipelined_config["tracking"]["run_name"],
    }
    assert large_batch_config["runtime"]["name"] not in previous_runtime_names
    assert (
        large_batch_config["experiment"]["name"]
        == large_batch_config["runtime"]["name"]
    )
    assert (
        large_batch_config["tracking"]["run_name"] not in previous_run_names
    )

    compiled_config = yaml.safe_load(
        (
            project_root
            / "configs"
            / "pathfinding_vit_compiled.yaml"
        ).read_text()
    )
    assert compiled_config.keys() == pipelined_config.keys()
    for section in (
        "seed",
        "deterministic",
        "environment",
        "evaluation_environment",
        "models",
        "optimizers",
        "trainer",
        "episode_managers",
    ):
        assert compiled_config[section] == pipelined_config[section]
    assert compiled_config["accelerator"] == {
        **pipelined_config["accelerator"],
        "compile_models": True,
        "compile_mode": "default",
    }
    for section, metadata_keys in (
        ("runtime", {"name"}),
        ("experiment", {"name", "description"}),
        ("tracking", {"run_name"}),
    ):
        assert {
            key: value
            for key, value in compiled_config[section].items()
            if key not in metadata_keys
        } == {
            key: value
            for key, value in pipelined_config[section].items()
            if key not in metadata_keys
        }
    compiled_wandb = compiled_config["callbacks"]["sampled_wandb"]
    assert {
        key: value
        for key, value in compiled_wandb.items()
        if key not in {"tags", "notes"}
    } == {
        key: value
        for key, value in pipelined_wandb.items()
        if key not in {"tags", "notes"}
    }
    assert compiled_wandb["tags"] == [
        *pipelined_wandb["tags"],
        "compiled",
    ]
    assert (
        compiled_config["callbacks"]["local_metric_tracker"]
        == pipelined_config["callbacks"]["local_metric_tracker"]
    )
    assert compiled_config["runtime"]["name"] not in {
        *previous_runtime_names,
        large_batch_config["runtime"]["name"],
    }
    assert compiled_config["experiment"]["name"] == compiled_config[
        "runtime"
    ]["name"]
    assert compiled_config["tracking"]["run_name"] not in {
        *previous_run_names,
        large_batch_config["tracking"]["run_name"],
    }

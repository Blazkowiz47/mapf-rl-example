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

import pathfinding_environment  # noqa: F401
import vit_q_network
from vit_q_network import ViTB16QNetwork


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


def test_pathfinding_config_defines_the_requested_long_run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "pathfinding_vit_dqn.yaml").read_text()
    )

    assert config["trainer"]["dqn"]["total_timesteps"] == 2_000_000_000
    assert config["models"]["q_network"]["name"] == "vit_b_16_q_network"
    assert config["environment"]["kwargs"]["grid_size"] == 1000
    assert config["environment"]["kwargs"]["observation_size"] == 256
    assert config["trainer"]["dqn"]["buffer_size"] == 4096
    assert config["trainer"]["dqn"]["batch_size"] == 128

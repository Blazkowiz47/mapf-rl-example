"""ViT-B/16 Q-network for RGB pathfinding observations."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from dl_core.core import config_field, register_model
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings


@register_model("vit_b_16_q_network")
class ViTB16QNetwork(nn.Module):
    """Map a 256×256 RGB grid to Q-values for four movement actions."""

    CONFIG_FIELDS: ClassVar[list[dict[str, Any]]] = [
        config_field(
            "input_dim",
            "int",
            "Flattened 256×256 RGB observation dimension.",
            required=True,
        ),
        config_field(
            "action_dim",
            "int",
            "Number of discrete movement actions; must be four.",
            required=True,
        ),
        config_field(
            "pretrained",
            "bool",
            "Initialize the encoder from ImageNet-1K ViT-B/16 weights.",
            default=True,
        ),
        config_field(
            "trainable_blocks",
            "int",
            "Number of final transformer encoder blocks to fine-tune.",
            default=2,
        ),
    ]

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        input_dim = int(config["input_dim"])
        action_dim = int(config["action_dim"])
        pretrained = bool(config.get("pretrained", True))
        trainable_blocks = int(config.get("trainable_blocks", 2))
        if input_dim != 256 * 256 * 3:
            raise ValueError(
                "vit_b_16_q_network requires 256×256 RGB observations"
            )
        if action_dim != 4:
            raise ValueError(
                "vit_b_16_q_network requires four movement actions"
            )
        if not 0 <= trainable_blocks <= 12:
            raise ValueError("trainable_blocks must be between 0 and 12")
        if not pretrained and trainable_blocks != 12:
            raise ValueError(
                "pretrained=False requires trainable_blocks=12 so the "
                "randomly initialized encoder remains trainable"
            )

        self.network = vit_b_16(
            weights=None,
            image_size=256,
            num_classes=action_dim,
        )
        if pretrained:
            state = ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(
                progress=True,
                check_hash=True,
            )
            state = interpolate_embeddings(
                image_size=256,
                patch_size=16,
                model_state=state,
                reset_heads=True,
            )
            incompatible = self.network.load_state_dict(state, strict=False)
            expected_missing = {
                "heads.head.weight",
                "heads.head.bias",
            }
            if (
                set(incompatible.missing_keys) != expected_missing
                or incompatible.unexpected_keys
            ):
                raise RuntimeError(
                    "ImageNet ViT-B/16 weights did not match the model"
                )

        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        if trainable_blocks:
            for block in self.network.encoder.layers[-trainable_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        for parameter in self.network.encoder.ln.parameters():
            parameter.requires_grad_(True)
        for parameter in self.network.heads.parameters():
            parameter.requires_grad_(True)

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return four Q-values for each RGB grid in the batch."""
        return self._forward(observations)

    def _forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4 or observations.shape[1:] != (
            256,
            256,
            3,
        ):
            raise ValueError(
                "ViT observations must have shape [batch, 256, 256, 3]"
            )
        images = observations.permute(0, 3, 1, 2).contiguous()
        images = images / 255.0
        images = (images - self.image_mean) / self.image_std
        return self.network(images)


__all__ = ["ViTB16QNetwork"]

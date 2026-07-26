"""Register robotics and local pathfinding components with dl-core."""

import dl_robotics
import dl_wandb

import pathfinding_environment
import pathfinding_wandb_callback
import vit_q_network

__all__ = [
    "dl_robotics",
    "dl_wandb",
    "pathfinding_environment",
    "pathfinding_wandb_callback",
    "vit_q_network",
]

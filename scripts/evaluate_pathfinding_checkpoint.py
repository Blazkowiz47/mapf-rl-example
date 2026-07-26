"""Evaluate a ViT-DQN checkpoint and render held-out pathfinding episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from dl_robotics import write_animation

from environments import ProceduralPathfindingEnv
from models import ViTB16QNetwork


def main() -> None:
    """Run deterministic checkpoint evaluation and write trajectory media."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pathfinding_vit_dqn.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pathfinding_evaluation"),
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument(
        "--format",
        choices=("gif", "mp4", "both"),
        default="both",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Playback rate in unsampled environment steps per second.",
    )
    parser.add_argument("--frame-stride", type=int, default=4)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.render_size <= 0:
        parser.error("--render-size must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    config: dict[str, Any] = yaml.safe_load(args.config.read_text())
    evaluation_config = config["evaluation_environment"]
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("trainer_type") != "reinforcement_learning":
        raise ValueError("checkpoint is not a dl-core RL checkpoint")
    model_states = checkpoint.get("models_state_dict", {})
    if "online" not in model_states:
        raise ValueError("checkpoint does not contain the online DQN model")
    online_state = model_states["online"]
    del model_states
    del checkpoint
    model = ViTB16QNetwork(
        {
            "input_dim": 256 * 256 * 3,
            "action_dim": 4,
            "pretrained": False,
            "trainable_blocks": 12,
        }
    )
    model.load_state_dict(online_state)
    del online_state
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    environment = ProceduralPathfindingEnv(
        **evaluation_config.get("kwargs", {})
    )
    reports: list[dict[str, Any]] = []
    try:
        for episode in range(args.episodes):
            observation, _ = environment.reset(seed=args.seed + episode)
            frames = [environment.get_grid_rgb(args.render_size)]
            episode_return = 0.0
            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}
            steps = 0
            while not (terminated or truncated):
                observations = torch.as_tensor(
                    observation,
                    device=device,
                ).unsqueeze(0)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    action = int(
                        model(observations).argmax(dim=1).item()
                    )
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    final_info,
                ) = environment.step(action)
                steps += 1
                episode_return += reward
                if (
                    steps % args.frame_stride == 0
                    or terminated
                    or truncated
                ):
                    frames.append(
                        environment.get_grid_rgb(args.render_size)
                    )

            stem = (
                f"episode_{episode:02d}_seed_{args.seed + episode}_"
                f"{'success' if terminated else 'truncated'}"
            )
            formats = (
                ("gif", "mp4")
                if args.format == "both"
                else (args.format,)
            )
            files = [
                str(
                    write_animation(
                        args.output_dir / f"{stem}.{media_format}",
                        frames,
                        fps=max(
                            1,
                            round(args.fps / args.frame_stride),
                        ),
                    )
                )
                for media_format in formats
            ]
            reports.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "is_success": bool(final_info.get("is_success", False)),
                    "return": episode_return,
                    "steps": steps,
                    "rendered_frames": len(frames),
                    "distance_to_goal": final_info.get("distance_to_goal"),
                    "path_length": final_info.get("path_length"),
                    "shortest_path_length": final_info.get(
                        "shortest_path_length"
                    ),
                    "collisions": final_info.get("episode_collisions"),
                    "files": files,
                }
            )
    finally:
        environment.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "render_size": args.render_size,
                "frame_stride": args.frame_stride,
                "playback_fps": max(
                    1,
                    round(args.fps / args.frame_stride),
                ),
                "episodes": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print(report_path.read_text(), end="")


if __name__ == "__main__":
    main()

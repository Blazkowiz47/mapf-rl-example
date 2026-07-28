"""Evaluate every compact 100k RL example from its trainer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from dl_core import load_builtin_components
from dl_core.core import TRAINER_REGISTRY

import bootstrap  # noqa: F401


def main() -> None:
    """Run fixed-seed deterministic evaluations and write one JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/sweeps"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluations/core_0_1_0_20260728/summary.json"
        ),
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=50_000)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    load_builtin_components()
    project_root = Path(__file__).resolve().parents[1]
    summaries: dict[str, dict[str, object]] = {}
    for run_name, config_name in (
        ("mapf_q_learning_100k", "mapf_q_learning.yaml"),
        ("mapf_dqn_100k", "mapf_dqn.yaml"),
        ("mapf_ppo_100k", "mapf_ppo.yaml"),
        ("mapf_sac_100k", "mapf_sac.yaml"),
        (
            "point_mass_acceleration_sac_100k",
            "point_mass_acceleration_sac.yaml",
        ),
        (
            "point_mass_velocity_ppo_100k",
            "point_mass_velocity_ppo.yaml",
        ),
        ("mapf_dreamer_100k", "mapf_dreamer.yaml"),
    ):
        config_path = project_root / "configs" / config_name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["_config_path"] = str(config_path)
        config.pop("sweep_file", None)
        config.pop("auto_resume_local", None)
        config["accelerator"] = {"type": "cpu"}
        config["runtime"] = {
            "output_dir": str(args.output.parent / "trainer_artifacts"),
            "name": f"evaluate_{run_name}",
        }
        trainer_name = next(iter(config["trainer"]))
        trainer_config = config["trainer"][trainer_name]
        trainer_config["evaluation_episodes"] = args.episodes
        trainer_config["show_progress"] = False
        trainer = TRAINER_REGISTRY.get(trainer_name, config)
        trainer.setup()
        checkpoint_dir = (
            args.artifact_root
            / run_name
            / run_name
            / "final"
            / "checkpoints"
        )
        try:
            evaluation_environment = (
                trainer.evaluation_environment.environment.unwrapped
            )
            if hasattr(evaluation_environment, "world"):
                start_distance = float(
                    np.abs(
                        evaluation_environment.world.start_positions
                        - evaluation_environment.world.goal_positions
                    ).sum()
                )
            else:
                start_distance = float(
                    np.linalg.norm(
                        evaluation_environment.goal_position
                        - evaluation_environment.start_position
                    )
                )
            candidate_summaries = []
            for checkpoint_path in sorted(checkpoint_dir.glob("step_*.pth")):
                trainer.load_checkpoint(str(checkpoint_path))
                for model in trainer.models.values():
                    model.eval()
                episode_returns = []
                episode_lengths = []
                successes = []
                collisions = []
                goals_reached = []
                final_distances = []
                with torch.no_grad():
                    for episode in range(args.episodes):
                        result = trainer.run_episode(
                            training=False,
                            episode=args.seed + episode,
                        )
                        episode_returns.append(result.episode_return)
                        episode_lengths.append(result.length)
                        successes.append(
                            bool(result.final_info["is_success"])
                        )
                        if hasattr(evaluation_environment, "world"):
                            world = evaluation_environment.world
                            final_distances.append(
                                float(
                                    np.abs(
                                        world.positions[0]
                                        - world.goal_positions
                                    ).sum()
                                )
                            )
                            collisions.append(
                                int(
                                    result.final_info[
                                        "episode_collisions"
                                    ]
                                )
                            )
                            goals_reached.append(
                                int(result.final_info["reached_agents"])
                            )
                        else:
                            final_distances.append(
                                float(
                                    result.final_info[
                                        "distance_to_goal"
                                    ]
                                )
                            )

                summary: dict[str, float | int | bool | str] = {
                    "checkpoint": checkpoint_path.name,
                    "transitions": int(trainer.global_step),
                    "updates": int(trainer.update_step),
                    "mean_return": float(np.mean(episode_returns)),
                    "mean_length": float(np.mean(episode_lengths)),
                    "success_rate": float(np.mean(successes)),
                    "start_distance": start_distance,
                    "final_distance": float(np.mean(final_distances)),
                    "moved_closer": bool(
                        np.mean(final_distances) < start_distance
                    ),
                }
                if collisions:
                    summary["collisions"] = round(
                        float(np.mean(collisions))
                    )
                    summary["goals_reached"] = round(
                        float(np.mean(goals_reached))
                    )
                else:
                    summary["goal_reached"] = bool(all(successes))
                candidate_summaries.append(summary)
            if not candidate_summaries:
                raise FileNotFoundError(
                    f"No step checkpoints found in {checkpoint_dir}"
                )
            summaries[run_name] = {
                "selected": max(
                    candidate_summaries,
                    key=lambda candidate: (
                        float(candidate["success_rate"]),
                        float(candidate["mean_return"]),
                        -float(candidate["final_distance"]),
                        -float(candidate["mean_length"]),
                        int(candidate["transitions"]),
                    ),
                ),
                "candidates": candidate_summaries,
            }
        finally:
            trainer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "episodes_per_checkpoint": args.episodes,
                "seed_start": args.seed,
                "selection_order": [
                    "success_rate",
                    "mean_return",
                    "lowest_final_distance",
                    "shortest_episode",
                    "latest_transition",
                ],
                "models": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

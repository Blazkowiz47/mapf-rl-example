"""Render one deterministic episode from every compact RL policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from dl_core import load_builtin_components
from dl_core.core import TRAINER_REGISTRY
from dl_robotics import write_animation

import bootstrap  # noqa: F401


def main() -> None:
    """Load selected checkpoints and write one GIF per trainer example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/sweeps"),
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path("pretrained/evaluations.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/media/rl_methods"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/results/rl_method_visualizations.json"),
    )
    parser.add_argument("--episode-index", type=int, default=50_000)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    load_builtin_components()
    project_root = Path(__file__).resolve().parents[1]
    evaluation = json.loads(
        (project_root / args.evaluation_summary).read_text(
            encoding="utf-8"
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for slug, run_name, config_name in (
        (
            "mapf_q_learning",
            "mapf_q_learning_100k",
            "mapf_q_learning.yaml",
        ),
        ("mapf_dqn", "mapf_dqn_100k", "mapf_dqn.yaml"),
        ("mapf_ppo", "mapf_ppo_100k", "mapf_ppo.yaml"),
        ("mapf_sac", "mapf_sac_100k", "mapf_sac.yaml"),
        ("mapf_dreamer", "mapf_dreamer_100k", "mapf_dreamer.yaml"),
        (
            "point_mass_sac",
            "point_mass_acceleration_sac_100k",
            "point_mass_acceleration_sac.yaml",
        ),
        (
            "point_mass_ppo",
            "point_mass_velocity_ppo_100k",
            "point_mass_velocity_ppo.yaml",
        ),
    ):
        config_path = project_root / "configs" / config_name
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["_config_path"] = str(config_path)
        config.pop("sweep_file", None)
        config.pop("auto_resume_local", None)
        config["accelerator"] = {"type": "cpu"}
        config["runtime"] = {
            "output_dir": "artifacts/visualize_rl_policies",
            "name": slug,
        }
        trainer_name = next(iter(config["trainer"]))
        config["trainer"][trainer_name]["show_progress"] = False
        trainer = TRAINER_REGISTRY.get(trainer_name, config)
        trainer.setup()
        checkpoint_name = evaluation["models"][run_name]["selected"][
            "checkpoint"
        ]
        checkpoint_path = (
            args.artifact_root
            / run_name
            / run_name
            / "final"
            / "checkpoints"
            / checkpoint_name
        )
        try:
            trainer.load_checkpoint(str(checkpoint_path))
            for model in trainer.models.values():
                model.eval()

            environment = trainer.evaluation_environment
            episode_seed = (
                trainer.seed + args.episode_index + 1_000_000
            )
            observations, _ = environment.reset_batch([episode_seed])
            observation = environment.batch_item(observations, 0)
            frames = [np.asarray(environment.render()).copy()]
            policy_state = trainer.initialize_policy_state(
                1,
                evaluation=True,
            )
            episode_return = 0.0
            final_info = {}
            steps = 0
            terminated = False
            truncated = False
            with torch.inference_mode():
                while not (terminated or truncated):
                    action_output = trainer.select_action_with_state(
                        observation,
                        policy_state,
                        deterministic=True,
                    )
                    policy_state = action_output.policy_state
                    (
                        next_observations,
                        rewards,
                        terminated_batch,
                        truncated_batch,
                        lane_infos,
                        _,
                    ) = environment.step_batch([action_output.action])
                    terminated = bool(terminated_batch[0])
                    truncated = bool(truncated_batch[0])
                    steps += 1
                    episode_return += float(rewards[0])
                    final_info = lane_infos[0]
                    if terminated or truncated:
                        final_info = dict(
                            final_info.get("final_info", final_info)
                        )
                    frames.append(
                        np.asarray(environment.render()).copy()
                    )
                    observation = environment.batch_item(
                        next_observations,
                        0,
                    )
                    if (
                        steps >= trainer.max_episode_steps
                        and not (terminated or truncated)
                    ):
                        truncated = True

            output_path = write_animation(
                args.output_dir / f"{slug}.gif",
                frames,
                fps=args.fps,
            )
            reports[slug] = {
                "run_name": run_name,
                "trainer": trainer_name,
                "checkpoint": checkpoint_name,
                "seed": episode_seed,
                "return": episode_return,
                "steps": steps,
                "is_success": bool(
                    final_info.get("is_success", False)
                ),
                "distance_to_goal": final_info.get(
                    "distance_to_goal"
                ),
                "goals_reached": final_info.get("reached_agents"),
                "collisions": final_info.get("episode_collisions"),
                "gif": str(output_path),
            }
            print(output_path)
        finally:
            trainer.close()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(reports, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.report)


if __name__ == "__main__":
    main()

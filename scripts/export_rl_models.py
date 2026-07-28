"""Export compact inference models from completed RL trainer checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import torch


class ExportSpec(NamedTuple):
    """Describe one completed run and its inference component."""

    run_name: str
    config_name: str
    policy_components: tuple[str, ...]


EXPORT_SPECS = (
    ExportSpec(
        "mapf_q_learning_100k",
        "mapf_q_learning.yaml",
        ("q_table",),
    ),
    ExportSpec("mapf_dqn_100k", "mapf_dqn.yaml", ("online",)),
    ExportSpec("mapf_ppo_100k", "mapf_ppo.yaml", ("policy",)),
    ExportSpec("mapf_sac_100k", "mapf_sac.yaml", ("actor",)),
    ExportSpec(
        "point_mass_acceleration_sac_100k",
        "point_mass_acceleration_sac.yaml",
        ("actor",),
    ),
    ExportSpec(
        "point_mass_velocity_ppo_100k",
        "point_mass_velocity_ppo.yaml",
        ("policy",),
    ),
    ExportSpec(
        "mapf_dreamer_100k",
        "mapf_dreamer.yaml",
        ("world_model", "actor"),
    ),
)


def export_model(
    spec: ExportSpec,
    *,
    artifact_root: Path,
    output_dir: Path,
    checkpoint_name: str,
) -> Path:
    """Write one tensor-only inference artifact."""
    checkpoint_path = (
        artifact_root
        / spec.run_name
        / spec.run_name
        / "final"
        / "checkpoints"
        / checkpoint_name
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if spec.policy_components == ("q_table",):
        policy_state_dicts = {
            "q_table": {
                "q_table": torch.as_tensor(
                    checkpoint["algorithm_state"]["q_table"]
                ).clone()
            }
        }
    else:
        policy_state_dicts = {
            component: {
                name: value.detach().cpu()
                for name, value in checkpoint["models_state_dict"][
                    component
                ].items()
            }
            for component in spec.policy_components
        }

    export = {
        "format_version": 2,
        "run_name": spec.run_name,
        "trainer_name": checkpoint["trainer_name"],
        "config_path": f"configs/{spec.config_name}",
        "checkpoint": checkpoint_name,
        "global_step": int(checkpoint["global_step"]),
        "update_step": int(checkpoint["update_step"]),
        "policy_components": list(spec.policy_components),
        "policy_state_dicts": policy_state_dicts,
    }
    output_path = output_dir / f"{spec.run_name}.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(export, output_path)
    return output_path


def main() -> None:
    """Export all completed example policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/sweeps"),
        help="Directory containing completed sweep runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained"),
        help="Destination for compact model files.",
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=Path(
            "artifacts/evaluations/core_0_1_0_20260728/summary.json"
        ),
        help="Fixed-seed checkpoint comparison produced by the evaluator.",
    )
    args = parser.parse_args()
    evaluation = json.loads(
        args.evaluation_summary.read_text(encoding="utf-8")
    )

    for spec in EXPORT_SPECS:
        checkpoint_name = evaluation["models"][spec.run_name]["selected"][
            "checkpoint"
        ]
        output_path = export_model(
            spec,
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
            checkpoint_name=checkpoint_name,
        )
        print(output_path)


if __name__ == "__main__":
    main()

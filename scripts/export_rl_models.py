"""Export compact inference models from completed RL trainer checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import torch


class ExportSpec(NamedTuple):
    """Describe one completed run and its inference component."""

    run_name: str
    config_name: str
    policy_component: str


EXPORT_SPECS = (
    ExportSpec("mapf_q_learning_100k", "mapf_q_learning.yaml", "q_table"),
    ExportSpec("mapf_dqn_100k", "mapf_dqn.yaml", "online"),
    ExportSpec("mapf_ppo_100k", "mapf_ppo.yaml", "policy"),
    ExportSpec("mapf_sac_100k", "mapf_sac.yaml", "actor"),
    ExportSpec(
        "point_mass_acceleration_sac_100k",
        "point_mass_acceleration_sac.yaml",
        "actor",
    ),
    ExportSpec(
        "point_mass_velocity_ppo_100k",
        "point_mass_velocity_ppo.yaml",
        "policy",
    ),
)


def export_model(
    spec: ExportSpec,
    *,
    artifact_root: Path,
    output_dir: Path,
) -> Path:
    """Write one tensor-only inference artifact."""
    checkpoint_path = (
        artifact_root
        / spec.run_name
        / spec.run_name
        / "final"
        / "checkpoints"
        / "latest.pth"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if spec.policy_component == "q_table":
        policy_state_dict = {
            "q_table": torch.as_tensor(
                checkpoint["algorithm_state"]["q_table"]
            ).clone()
        }
    else:
        policy_state_dict = {
            name: value.detach().cpu()
            for name, value in checkpoint["models_state_dict"][
                spec.policy_component
            ].items()
        }

    export = {
        "format_version": 1,
        "run_name": spec.run_name,
        "trainer_name": checkpoint["trainer_name"],
        "config_path": f"configs/{spec.config_name}",
        "global_step": int(checkpoint["global_step"]),
        "update_step": int(checkpoint["update_step"]),
        "policy_component": spec.policy_component,
        "policy_state_dict": policy_state_dict,
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
    args = parser.parse_args()

    for spec in EXPORT_SPECS:
        output_path = export_model(
            spec,
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
        )
        print(output_path)


if __name__ == "__main__":
    main()

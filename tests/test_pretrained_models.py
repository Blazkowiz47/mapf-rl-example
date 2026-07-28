"""Integrity checks for the compact trained-policy exports."""

import json
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    ("run_name", "trainer_name", "policy_components"),
    [
        ("mapf_q_learning_100k", "q_learning", ["q_table"]),
        ("mapf_dqn_100k", "dqn", ["online"]),
        ("mapf_ppo_100k", "ppo", ["policy"]),
        ("mapf_sac_100k", "sac", ["actor"]),
        ("point_mass_acceleration_sac_100k", "sac", ["actor"]),
        ("point_mass_velocity_ppo_100k", "ppo", ["policy"]),
        (
            "mapf_dreamer_100k",
            "dreamer",
            ["world_model", "actor"],
        ),
    ],
)
def test_pretrained_model_is_tensor_only_and_complete(
    run_name: str,
    trainer_name: str,
    policy_components: list[str],
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifact = torch.load(
        project_root / "pretrained" / f"{run_name}.pt",
        map_location="cpu",
        weights_only=True,
    )
    evaluation = json.loads(
        (project_root / "pretrained" / "evaluations.json").read_text(
            encoding="utf-8"
        )
    )

    assert artifact["format_version"] == 2
    assert artifact["run_name"] == run_name
    assert artifact["trainer_name"] == trainer_name
    assert artifact["policy_components"] == policy_components
    assert artifact["checkpoint"] == (
        f"step_{artifact['global_step']:012d}.pth"
    )
    assert artifact["checkpoint"] == evaluation["models"][run_name][
        "selected"
    ]["checkpoint"]
    assert 0 < artifact["global_step"] <= 100_000
    assert artifact["update_step"] > 0
    assert list(artifact["policy_state_dicts"]) == policy_components
    for state_dict in artifact["policy_state_dicts"].values():
        assert state_dict
        assert all(
            isinstance(value, torch.Tensor)
            for value in state_dict.values()
        )


def test_policy_gallery_matches_selected_checkpoints() -> None:
    project_root = Path(__file__).resolve().parents[1]
    evaluation = json.loads(
        (project_root / "pretrained" / "evaluations.json").read_text(
            encoding="utf-8"
        )
    )
    gallery = json.loads(
        (
            project_root
            / "docs"
            / "results"
            / "rl_method_visualizations.json"
        ).read_text(encoding="utf-8")
    )

    assert len(gallery) == 7
    for result in gallery.values():
        assert result["checkpoint"] == evaluation["models"][
            result["run_name"]
        ]["selected"]["checkpoint"]
        gif = project_root / result["gif"]
        assert gif.read_bytes().startswith(b"GIF")

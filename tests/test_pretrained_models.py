"""Integrity checks for the compact trained-policy exports."""

from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    ("run_name", "trainer_name", "policy_component"),
    [
        ("mapf_q_learning_100k", "q_learning", "q_table"),
        ("mapf_dqn_100k", "dqn", "online"),
        ("mapf_ppo_100k", "ppo", "policy"),
        ("mapf_sac_100k", "sac", "actor"),
        ("point_mass_acceleration_sac_100k", "sac", "actor"),
        ("point_mass_velocity_ppo_100k", "ppo", "policy"),
    ],
)
def test_pretrained_model_is_tensor_only_and_complete(
    run_name: str,
    trainer_name: str,
    policy_component: str,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifact = torch.load(
        project_root / "pretrained" / f"{run_name}.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert artifact["format_version"] == 1
    assert artifact["run_name"] == run_name
    assert artifact["trainer_name"] == trainer_name
    assert artifact["policy_component"] == policy_component
    assert artifact["global_step"] == 100_000
    assert artifact["update_step"] > 0
    assert artifact["policy_state_dict"]
    assert all(
        isinstance(value, torch.Tensor)
        for value in artifact["policy_state_dict"].values()
    )

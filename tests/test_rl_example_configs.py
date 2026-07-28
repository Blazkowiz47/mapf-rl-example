"""Short integration coverage for each standalone RL trainer example."""

from pathlib import Path

import pytest
import yaml
from dl_core import load_builtin_components
from dl_core.trainers import PPOTrainer, QLearningTrainer, SACTrainer

import bootstrap  # noqa: F401


@pytest.mark.parametrize(
    ("config_name", "trainer_class", "captures_evaluation", "captures_media"),
    [
        ("mapf_q_learning.yaml", QLearningTrainer, True, False),
        ("mapf_ppo.yaml", PPOTrainer, True, True),
        ("mapf_sac.yaml", SACTrainer, True, True),
        ("point_mass_acceleration_sac.yaml", SACTrainer, False, False),
        ("point_mass_velocity_ppo.yaml", PPOTrainer, False, False),
    ],
)
def test_rl_example_performs_short_training(
    config_name: str,
    trainer_class: type,
    captures_evaluation: bool,
    captures_media: bool,
    tmp_path: Path,
) -> None:
    load_builtin_components()
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "configs" / config_name).read_text())
    trainer_name = next(iter(config["trainer"]))
    config["runtime"]["output_dir"] = str(tmp_path / "artifacts")
    config["runtime"]["name"] = f"test_{Path(config_name).stem}"
    if not captures_evaluation:
        config["trainer"][trainer_name]["total_timesteps"] = 32
        config["trainer"][trainer_name]["evaluation_episodes"] = 0

    trainer = trainer_class(config)
    trainer.setup()
    try:
        trainer.perform_training()
    finally:
        trainer.close()

    expected_timesteps = 64 if captures_evaluation else 32
    assert trainer.global_step == expected_timesteps
    assert trainer.update_step > 0
    if captures_evaluation:
        artifact_root = tmp_path / "artifacts"
        assert list(artifact_root.rglob("*.npz"))
        assert bool(list(artifact_root.rglob("*.gif"))) is captures_media

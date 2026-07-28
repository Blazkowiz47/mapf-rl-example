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
    smoke_timesteps = 64 if captures_evaluation else 32
    trainer_config = config["trainer"][trainer_name]
    trainer_config["total_timesteps"] = smoke_timesteps
    trainer_config["show_progress"] = False
    if trainer_name == "sac":
        trainer_config["buffer_size"] = 256
        trainer_config["batch_size"] = 8
        trainer_config["learning_starts"] = 8
    elif trainer_name == "ppo":
        trainer_config["rollout_steps"] = 4
        trainer_config["update_epochs"] = 2
        trainer_config["minibatch_size"] = 8
    elif trainer_name == "q_learning":
        trainer_config["epsilon_decay_steps"] = 64
    if not captures_evaluation:
        trainer_config["evaluation_episodes"] = 0
    else:
        trainer_config["evaluation_frequency"] = 4
        trainer_config["evaluation_episodes"] = 1

    trainer = trainer_class(config)
    trainer.setup()
    try:
        trainer.perform_training()
    finally:
        trainer.close()

    assert trainer.global_step == smoke_timesteps
    assert trainer.update_step > 0
    if captures_evaluation:
        artifact_root = tmp_path / "artifacts"
        assert list(artifact_root.rglob("*.npz"))
        assert bool(list(artifact_root.rglob("*.gif"))) is captures_media

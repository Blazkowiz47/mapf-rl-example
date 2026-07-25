"""End-to-end MAPF training example."""

import json
from pathlib import Path

import dl_robotics
import numpy as np
import yaml
from dl_core import load_builtin_components
from dl_core.trainers import DQNTrainer
from dl_robotics import GridMAPFEnvironment


def test_crossing_scenario_has_a_collision_free_solution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "mapf_dqn.yaml").read_text()
    )
    environment = GridMAPFEnvironment(config["evaluation_environment"])
    _, info = environment.reset(seed=2026)
    assert info["is_success"] is False

    try:
        # Actor 0 goes along the top/right edges while actor 1 uses bottom/left.
        # Joint actions use base-5 digits with actor 0 as the least-significant.
        for action in [22] * 4 + [8] * 4:
            _, _, terminated, truncated, info = environment.step(action)
    finally:
        environment.close()

    assert terminated is True
    assert truncated is False
    assert info["is_success"] is True
    assert info["episode_collisions"] == 0
    assert info["makespan"] == 8
    assert info["sum_of_costs"] == 16
    assert info["path_length"] == 16


def test_vector_dqn_trains_and_captures_evaluation_episode(
    tmp_path: Path,
) -> None:
    assert dl_robotics.__version__
    load_builtin_components()
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "mapf_dqn.yaml").read_text()
    )
    config["runtime"]["output_dir"] = str(tmp_path / "artifacts")
    config["runtime"]["name"] = "test_mapf_dqn"
    trainer = DQNTrainer(config)
    trainer.setup()

    try:
        trainer.perform_training()
    finally:
        trainer.close()

    artifact_root = tmp_path / "artifacts"
    assert trainer.global_step == 64
    assert trainer.collector_step == 16
    assert len(trainer.replay_buffer) == 64
    # Updates are scheduled at transitions 8, 12, ..., 64.
    assert trainer.update_step == 15

    animations = sorted(artifact_root.rglob("*.gif"))
    trajectories = sorted(artifact_root.rglob("*.npz"))
    metric_files = list(artifact_root.rglob("episodes_robotics.jsonl"))
    assert len(animations) == 2
    assert len(trajectories) == 2
    assert len(metric_files) == 1
    assert all(path.parent.name == "evaluation" for path in animations)

    with np.load(trajectories[0], allow_pickle=False) as trajectory:
        assert trajectory["observations"].shape[0] == (
            trajectory["actions"].shape[0] + 1
        )
        metadata = json.loads(str(trajectory["metadata_json"]))
    assert metadata["phase"] == "evaluation"
    assert "robotics/collisions" in metadata["metrics"]

    episode_metrics = [
        json.loads(line)
        for line in metric_files[0].read_text().splitlines()
        if line
    ]
    assert any(metrics["phase"] == "evaluation" for metrics in episode_metrics)

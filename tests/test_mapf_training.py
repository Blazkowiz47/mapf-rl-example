"""End-to-end MAPF training example."""

import json
from pathlib import Path

import dl_robotics
import numpy as np
import yaml
from dl_core import load_builtin_components
from dl_core.trainers import DQNTrainer
from dl_robotics import (
    GridMAPFEnvironment,
    GridScenario,
    astar_path,
    bfs_path,
    dijkstra_path,
)

from mapf_baselines import main as print_baselines


def test_classical_baseline_report(capsys) -> None:
    print_baselines()

    report = json.loads(capsys.readouterr().out)

    assert report["scenario"] == "two_agent_crossing"
    assert report["makespan_lower_bound"] == 8
    assert report["sum_of_costs_lower_bound"] == 16
    assert [agent["shortest_moves"] for agent in report["agents"]] == [8, 8]
    assert set(report["agents"][0]["paths"]) == {
        "astar",
        "dijkstra",
        "bfs",
        "dfs",
    }


def test_crossing_scenario_has_a_collision_free_solution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "mapf_dqn.yaml").read_text()
    )
    scenario = GridScenario.from_config(
        config["evaluation_environment"]["scenario"]
    )
    shortest_moves = []
    for start, goal in zip(scenario.starts, scenario.goals, strict=True):
        planner_moves = {
            len(planner(scenario, start, goal)) - 1
            for planner in (astar_path, dijkstra_path, bfs_path)
        }
        assert len(planner_moves) == 1
        shortest_moves.append(planner_moves.pop())

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
    assert info["makespan"] == max(shortest_moves)
    assert info["sum_of_costs"] == sum(shortest_moves)
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

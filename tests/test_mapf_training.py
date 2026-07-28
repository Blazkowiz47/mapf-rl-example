"""End-to-end MAPF training example."""

import json
import subprocess
import sys
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

import bootstrap  # noqa: F401
from scenarios import make_two_agent_crossing_scenario


def test_classical_baseline_report() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "print_classical_baselines.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    report = json.loads(result.stdout)

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


def test_classical_planner_visualizations(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(
                project_root
                / "scripts"
                / "visualize_classical_planners.py"
            ),
            "--output-dir",
            str(tmp_path),
            "--format",
            "gif",
            "--grid-size",
            "128",
            "--render-size",
            "128",
            "--max-frames",
            "12",
            "--fps",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    report = json.loads(result.stdout)

    assert report["scenario"] == "procedural_128_2030"
    assert report["logical_grid"] == [128, 128]
    assert report["rendered_frames"] == 12
    assert report["wall_cells"] > 0
    assert len({tuple(start) for start in report["starts"]}) == 3
    assert len({tuple(goal) for goal in report["goals"]}) == 3
    assert len(report["files"]) == 5
    assert {Path(path).name for path in report["files"]} == {
        "astar.gif",
        "dijkstra.gif",
        "bfs.gif",
        "dfs.gif",
        "comparison.gif",
    }
    assert all(Path(path).stat().st_size > 0 for path in report["files"])
    assert report["algorithms"]["astar"]["route_moves"] == (
        report["algorithms"]["dijkstra"]["route_moves"]
    )
    assert report["algorithms"]["astar"]["route_moves"] == (
        report["algorithms"]["bfs"]["route_moves"]
    )
    assert all(
        algorithm["collisions"] == 0
        for algorithm in report["algorithms"].values()
    )
    assert all(
        algorithm["wall_entries"] == 0
        for algorithm in report["algorithms"].values()
    )


def test_crossing_scenario_has_a_collision_free_solution() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project_root / "configs" / "mapf_dqn.yaml").read_text()
    )
    configured_scenario = GridScenario.from_config(
        config["evaluation_environment"]["scenario"]
    )
    scenario = make_two_agent_crossing_scenario()
    assert scenario == configured_scenario
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
    assert config["accelerator"]["type"] == "single_gpu"
    config["accelerator"] = {"type": "cpu"}
    config["runtime"]["output_dir"] = str(tmp_path / "artifacts")
    config["runtime"]["name"] = "test_mapf_dqn"
    config["trainer"]["dqn"].update(
        {
            "total_timesteps": 64,
            "evaluation_frequency": 4,
            "evaluation_episodes": 1,
            "show_progress": False,
            "buffer_size": 256,
            "batch_size": 8,
            "learning_starts": 8,
            "target_update_frequency": 32,
            "epsilon_decay_steps": 64,
        }
    )
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

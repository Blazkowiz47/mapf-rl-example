"""Print classical single-agent baselines for the example MAPF scenario."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import yaml
from dl_robotics import (
    GridScenario,
    astar_path,
    bfs_path,
    dfs_path,
    dijkstra_path,
)


def main() -> None:
    """Print paths and exact static lower bounds as JSON."""
    _main()


def _main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "mapf_dqn.yaml"
    if config_path.is_file():
        config_text = config_path.read_text()
    else:
        config_text = (
            files("mapf_baselines_data")
            .joinpath("mapf_dqn.yaml")
            .read_text(encoding="utf-8")
        )
    config = yaml.safe_load(config_text)
    scenario = GridScenario.from_config(
        config["evaluation_environment"]["scenario"]
    )
    planners = {
        "astar": astar_path,
        "dijkstra": dijkstra_path,
        "bfs": bfs_path,
        "dfs": dfs_path,
    }
    agents = []
    shortest_moves = []
    for agent_index, (start, goal) in enumerate(
        zip(scenario.starts, scenario.goals, strict=True)
    ):
        paths = {
            name: planner(scenario, start, goal)
            for name, planner in planners.items()
        }
        optimal_moves = {
            len(paths[name]) - 1
            for name in ("astar", "dijkstra", "bfs")
        }
        if len(optimal_moves) != 1:
            raise RuntimeError("Optimal planners disagreed on path length")
        move_count = optimal_moves.pop()
        shortest_moves.append(move_count)
        agents.append(
            {
                "agent": agent_index,
                "start": start,
                "goal": goal,
                "shortest_moves": move_count,
                "paths": paths,
            }
        )

    print(
        json.dumps(
            {
                "scenario": scenario.name,
                "agents": agents,
                "makespan_lower_bound": max(shortest_moves),
                "sum_of_costs_lower_bound": sum(shortest_moves),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

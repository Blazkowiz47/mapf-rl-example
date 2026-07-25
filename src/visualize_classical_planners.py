"""Animate multi-agent routes from the classical grid planners."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
from dl_robotics import (
    GridPath,
    GridRenderer,
    GridScenario,
    GridWorldBatch,
    astar_path,
    bfs_path,
    dfs_path,
    dijkstra_path,
    write_animation,
)

Planner = Callable[[GridScenario, tuple[int, int], tuple[int, int]], GridPath]
_COLORS = (
    (37, 99, 235),
    (220, 38, 38),
    (22, 163, 74),
)
_ACTION_BY_DELTA = {
    (0, 0): 0,
    (-1, 0): 1,
    (0, 1): 2,
    (1, 0): 3,
    (0, -1): 4,
}


def main(argv: Sequence[str] | None = None) -> None:
    """Create individual and combined classical-planner animations."""
    _main(argv)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Animate A*, Dijkstra, BFS, and DFS multi-agent routes."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/classical_planners"),
    )
    parser.add_argument(
        "--format",
        choices=("gif", "mp4", "both"),
        default="gif",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--cell-size", type=int, default=32)
    args = parser.parse_args(argv)

    lane_walls = ((0, 4), (1, 3), (1, 8), (2, 8))
    walls = tuple(
        (row + row_offset, column)
        for row_offset in (0, 4, 8)
        for row, column in lane_walls
    ) + tuple(
        (row, column)
        for row in (3, 7)
        for column in range(9)
    )
    scenario = GridScenario(
        name="three_agent_lanes",
        width=9,
        height=11,
        starts=((0, 7), (4, 7), (8, 7)),
        goals=((1, 0), (5, 0), (9, 0)),
        walls=walls,
        max_steps=30,
    )
    planners: dict[str, tuple[str, Planner]] = {
        "astar": ("A*", astar_path),
        "dijkstra": ("Dijkstra", dijkstra_path),
        "bfs": ("BFS", bfs_path),
        "dfs": ("DFS", dfs_path),
    }
    frame_sets: dict[str, list[np.ndarray]] = {}
    report: dict[str, object] = {
        "scenario": scenario.name,
        "algorithms": {},
        "files": [],
    }
    output_formats = (
        ("gif", "mp4") if args.format == "both" else (args.format,)
    )

    for name, (label, planner) in planners.items():
        frames, paths, collisions = _render_algorithm(
            scenario,
            label,
            planner,
            cell_size=args.cell_size,
            fps=args.fps,
        )
        frame_sets[name] = frames
        report["algorithms"][name] = {
            "route_moves": [len(path) - 1 for path in paths],
            "collisions": collisions,
        }
        for suffix in output_formats:
            output_path = args.output_dir / f"{name}.{suffix}"
            write_animation(output_path, frames, fps=args.fps)
            report["files"].append(str(output_path))

    comparison_frames = []
    frame_count = max(len(frames) for frames in frame_sets.values())
    for frame_index in range(frame_count):
        astar = frame_sets["astar"][min(frame_index, len(frame_sets["astar"]) - 1)]
        dijkstra = frame_sets["dijkstra"][
            min(frame_index, len(frame_sets["dijkstra"]) - 1)
        ]
        bfs = frame_sets["bfs"][min(frame_index, len(frame_sets["bfs"]) - 1)]
        dfs = frame_sets["dfs"][min(frame_index, len(frame_sets["dfs"]) - 1)]
        comparison_frames.append(
            np.concatenate(
                (
                    np.concatenate((astar, dijkstra), axis=1),
                    np.concatenate((bfs, dfs), axis=1),
                ),
                axis=0,
            )
        )
    for suffix in output_formats:
        output_path = args.output_dir / f"comparison.{suffix}"
        write_animation(output_path, comparison_frames, fps=args.fps)
        report["files"].append(str(output_path))

    print(json.dumps(report, indent=2))


def _render_algorithm(
    scenario: GridScenario,
    label: str,
    planner: Planner,
    *,
    cell_size: int,
    fps: int,
) -> tuple[list[np.ndarray], tuple[GridPath, ...], int]:
    paths = tuple(
        planner(scenario, start, goal)
        for start, goal in zip(scenario.starts, scenario.goals, strict=True)
    )
    world = GridWorldBatch(scenario)
    renderer = GridRenderer(cell_size=cell_size)
    max_moves = max(len(path) - 1 for path in paths)
    frames = []
    collisions = 0

    for step in range(max_moves + 1):
        frame = renderer.render_world(world)
        for agent_index, path in enumerate(paths):
            points = np.asarray(
                [
                    (
                        column * cell_size + cell_size // 2,
                        row * cell_size + cell_size // 2,
                    )
                    for row, column in path
                ],
                dtype=np.int32,
            )
            cv2.polylines(
                frame,
                [points],
                False,
                _COLORS[agent_index],
                max(1, cell_size // 12),
                cv2.LINE_AA,
            )
        titled_frame = np.full(
            (frame.shape[0] + 36, frame.shape[1], 3),
            248,
            dtype=np.uint8,
        )
        titled_frame[36:] = frame
        cv2.putText(
            titled_frame,
            f"{label} | step {step:02d}/{max_moves:02d}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 41, 59),
            1,
            cv2.LINE_AA,
        )
        frames.append(titled_frame)

        if step == max_moves:
            continue
        actions = np.zeros(scenario.num_agents, dtype=np.int32)
        for agent_index, path in enumerate(paths):
            if step + 1 >= len(path):
                continue
            delta = (
                path[step + 1][0] - path[step][0],
                path[step + 1][1] - path[step][1],
            )
            actions[agent_index] = _ACTION_BY_DELTA[delta]
        events = world.step(actions[np.newaxis, :])
        collisions += int(events.collisions[0])

    if collisions or not world.reached[0].all():
        raise RuntimeError(
            f"{label} routes did not complete collision-free: "
            f"collisions={collisions}"
        )
    frames.extend(frames[-1].copy() for _ in range(fps))
    return frames, paths, collisions


if __name__ == "__main__":
    main()

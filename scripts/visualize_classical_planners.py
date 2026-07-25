"""Stream large procedural multi-agent planner visualizations."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from dl_robotics import (
    Coordinate,
    GridPath,
    GridScenario,
    GridWorldBatch,
    astar_path,
    bfs_path,
    dfs_path,
    dijkstra_path,
)
from numpy.typing import NDArray

Planner = Callable[[GridScenario, Coordinate, Coordinate], GridPath]
IntArray = NDArray[np.int32]
UInt8Image = NDArray[np.uint8]
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
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
    """Create streamed A*, Dijkstra, BFS, and DFS animations."""
    _main(argv)


def _main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Animate three simultaneous actors in a seeded procedural maze."
        )
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
    parser.add_argument("--grid-size", type=int, default=2000)
    parser.add_argument("--render-size", type=int, default=400)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    for label, value, minimum in (
        ("grid-size", args.grid_size, 64),
        ("render-size", args.render_size, 64),
        ("max-frames", args.max_frames, 2),
        ("fps", args.fps, 1),
    ):
        if value < minimum:
            parser.error(f"--{label} must be at least {minimum}")

    rng = np.random.default_rng(args.seed)
    blocked = np.zeros(
        (args.grid_size, args.grid_size),
        dtype=np.bool_,
    )
    wall_count = max(24, args.grid_size * 13 // 100)
    minimum_length = max(4, args.grid_size // 30)
    maximum_length = max(minimum_length + 1, args.grid_size // 8)
    margin = max(8, args.grid_size // 20)
    thickness_minimum = 1 if args.grid_size < 400 else 2
    thickness_maximum = max(
        thickness_minimum + 1,
        args.grid_size // 400 + 2,
    )
    for _ in range(wall_count):
        length = int(rng.integers(minimum_length, maximum_length))
        thickness = int(
            rng.integers(thickness_minimum, thickness_maximum)
        )
        row = int(rng.integers(margin, args.grid_size - margin))
        column = int(rng.integers(margin, args.grid_size - margin))
        if rng.random() < 0.5:
            blocked[
                row : min(args.grid_size, row + thickness),
                column : min(args.grid_size, column + length),
            ] = True
        else:
            blocked[
                row : min(args.grid_size, row + length),
                column : min(args.grid_size, column + thickness),
            ] = True

    starts: list[Coordinate] = []
    minimum_start_distance = args.grid_size // 3
    for _ in range(10_000):
        if len(starts) == 3:
            break
        candidate = tuple(
            int(value)
            for value in rng.integers(
                margin,
                args.grid_size - margin,
                size=2,
            )
        )
        if blocked[candidate] or any(
            abs(candidate[0] - start[0])
            + abs(candidate[1] - start[1])
            < minimum_start_distance
            for start in starts
        ):
            continue
        starts.append(candidate)
    if len(starts) != 3:
        raise RuntimeError(
            "Could not place three separated actors in the generated maze"
        )

    minimum_traversal = max(60, args.grid_size * 7 // 20)
    maximum_traversal = max(
        minimum_traversal + 1,
        minimum_traversal + args.grid_size // 5,
    )
    goals = []
    forbidden = set(starts)
    for start in starts:
        target_step = int(
            rng.integers(minimum_traversal, maximum_traversal)
        )
        goal = _select_dfs_goal(
            blocked,
            start,
            target_step,
            forbidden,
        )
        goals.append(goal)
        forbidden.add(goal)

    walls = tuple(
        tuple(int(value) for value in coordinate)
        for coordinate in np.argwhere(blocked)
    )
    scenario = GridScenario(
        name=f"procedural_{args.grid_size}_{args.seed}",
        width=args.grid_size,
        height=args.grid_size,
        starts=tuple(starts),
        goals=tuple(goals),
        walls=walls,
        max_steps=args.grid_size * 4,
    )
    planners: dict[str, tuple[str, Planner]] = {
        "astar": ("A*", astar_path),
        "dijkstra": ("Dijkstra", dijkstra_path),
        "bfs": ("BFS", bfs_path),
        "dfs": ("DFS", dfs_path),
    }
    plans: dict[str, tuple[GridPath, ...]] = {}
    schedules: dict[str, IntArray] = {}
    report: dict[str, object] = {
        "scenario": scenario.name,
        "logical_grid": [scenario.height, scenario.width],
        "seed": args.seed,
        "wall_cells": len(scenario.walls),
        "starts": scenario.starts,
        "goals": scenario.goals,
        "algorithms": {},
        "files": [],
    }
    for name, (_, planner) in planners.items():
        planning_started = time.perf_counter()
        paths = tuple(
            planner(scenario, start, goal)
            for start, goal in zip(
                scenario.starts,
                scenario.goals,
                strict=True,
            )
        )
        delays = _schedule_paths(paths)
        positions, collisions = _execute_schedule(
            scenario,
            paths,
            delays,
        )
        plans[name] = paths
        schedules[name] = positions
        report["algorithms"][name] = {
            "route_moves": [len(path) - 1 for path in paths],
            "start_delays": delays,
            "schedule_steps": len(positions) - 1,
            "collisions": collisions,
            "planning_seconds": round(
                time.perf_counter() - planning_started,
                4,
            ),
        }

    wall_density = cv2.resize(
        blocked.astype(np.uint8) * 255,
        (args.render_size, args.render_size),
        interpolation=cv2.INTER_AREA,
    )
    wall_pixels = cv2.dilate(
        (wall_density > 0).astype(np.uint8),
        np.ones((2, 2), dtype=np.uint8),
    ).astype(bool)
    base = np.full(
        (args.render_size, args.render_size, 3),
        248,
        dtype=np.uint8,
    )
    base[wall_pixels] = (42, 52, 68)
    route_bases = {
        name: _render_route_base(
            base,
            scenario,
            paths,
            args.render_size,
        )
        for name, paths in plans.items()
    }

    output_formats = (
        ("gif", "mp4") if args.format == "both" else (args.format,)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    for name in (*planners, "comparison"):
        for suffix in output_formats:
            output_path = args.output_dir / f"{name}.{suffix}"
            writers[(name, suffix)] = _open_writer(
                output_path,
                suffix,
                args.fps,
            )
            report["files"].append(str(output_path))

    schedule_steps = max(
        len(positions) - 1
        for positions in schedules.values()
    )
    frame_steps = np.unique(
        np.linspace(
            0,
            schedule_steps,
            min(args.max_frames, schedule_steps + 1),
            dtype=np.int32,
        )
    )
    try:
        for step in frame_steps:
            frames = {}
            for name, (label, _) in planners.items():
                positions = schedules[name]
                algorithm_step = min(int(step), len(positions) - 1)
                frames[name] = _render_frame(
                    route_bases[name],
                    scenario,
                    positions[algorithm_step],
                    label,
                    algorithm_step,
                    len(positions) - 1,
                    args.render_size,
                    args.seed,
                )
                for suffix in output_formats:
                    writers[(name, suffix)].append_data(
                        _prepare_frame(frames[name], suffix)
                    )

            comparison = np.concatenate(
                (
                    np.concatenate(
                        (frames["astar"], frames["dijkstra"]),
                        axis=1,
                    ),
                    np.concatenate(
                        (frames["bfs"], frames["dfs"]),
                        axis=1,
                    ),
                ),
                axis=0,
            )
            for suffix in output_formats:
                writers[("comparison", suffix)].append_data(
                    _prepare_frame(comparison, suffix)
                )
    finally:
        for writer in writers.values():
            writer.close()

    report["rendered_frames"] = len(frame_steps)
    report["render_size"] = [
        args.render_size,
        args.render_size,
    ]
    print(json.dumps(report, indent=2))


def _select_dfs_goal(
    blocked: NDArray[np.bool_],
    start: Coordinate,
    target_step: int,
    forbidden: set[Coordinate],
) -> Coordinate:
    height, width = blocked.shape
    frontier = [start]
    visited = {start}
    traversed = 0
    while frontier:
        current = frontier.pop()
        traversed += 1
        if traversed >= target_step and current not in forbidden:
            return current
        neighbors = []
        for row_delta, column_delta in _DIRECTIONS:
            neighbor = (
                current[0] + row_delta,
                current[1] + column_delta,
            )
            if (
                0 <= neighbor[0] < height
                and 0 <= neighbor[1] < width
                and not blocked[neighbor]
                and neighbor not in visited
            ):
                neighbors.append(neighbor)
        for neighbor in reversed(neighbors):
            visited.add(neighbor)
            frontier.append(neighbor)
    raise RuntimeError(f"No reachable goal found from {start}")


def _schedule_paths(paths: tuple[GridPath, ...]) -> list[int]:
    delays = [0]
    maximum_delay = sum(len(path) - 1 for path in paths)
    for path in paths[1:]:
        for delay in range(maximum_delay + 1):
            if all(
                not _paths_conflict(
                    path,
                    delay,
                    other_path,
                    other_delay,
                )
                for other_path, other_delay in zip(
                    paths[: len(delays)],
                    delays,
                    strict=True,
                )
            ):
                delays.append(delay)
                break
        else:
            raise RuntimeError(
                "No collision-free start-delay schedule was found"
            )
    return delays


def _paths_conflict(
    path: GridPath,
    delay: int,
    other_path: GridPath,
    other_delay: int,
) -> bool:
    horizon = max(
        delay + len(path),
        other_delay + len(other_path),
    )
    for step in range(horizon):
        position = _position_at(path, delay, step)
        other_position = _position_at(other_path, other_delay, step)
        if position == other_position:
            return True
        if step == 0:
            continue
        previous = _position_at(path, delay, step - 1)
        other_previous = _position_at(
            other_path,
            other_delay,
            step - 1,
        )
        if position == other_previous and other_position == previous:
            return True
    return False


def _position_at(
    path: GridPath,
    delay: int,
    step: int,
) -> Coordinate:
    if step <= delay:
        return path[0]
    return path[min(step - delay, len(path) - 1)]


def _execute_schedule(
    scenario: GridScenario,
    paths: tuple[GridPath, ...],
    delays: list[int],
) -> tuple[IntArray, int]:
    horizon = max(
        delay + len(path) - 1
        for path, delay in zip(paths, delays, strict=True)
    )
    world = GridWorldBatch(scenario)
    positions = np.empty(
        (horizon + 1, scenario.num_agents, 2),
        dtype=np.int32,
    )
    collisions = 0
    for step in range(horizon + 1):
        expected = np.asarray(
            [
                _position_at(path, delay, step)
                for path, delay in zip(paths, delays, strict=True)
            ],
            dtype=np.int32,
        )
        if not np.array_equal(world.positions[0], expected):
            raise RuntimeError(
                f"Schedule diverged from planned positions at step {step}"
            )
        positions[step] = expected
        if step == horizon:
            continue
        next_positions = np.asarray(
            [
                _position_at(path, delay, step + 1)
                for path, delay in zip(paths, delays, strict=True)
            ],
            dtype=np.int32,
        )
        deltas = next_positions - expected
        actions = np.asarray(
            [_ACTION_BY_DELTA[tuple(delta)] for delta in deltas],
            dtype=np.int32,
        )
        events = world.step(actions[np.newaxis, :])
        collisions += int(events.collisions[0])

    if collisions or not world.reached[0].all():
        raise RuntimeError(
            "The simultaneous schedule did not complete collision-free"
        )
    return positions, collisions


def _render_route_base(
    base: UInt8Image,
    scenario: GridScenario,
    paths: tuple[GridPath, ...],
    render_size: int,
) -> UInt8Image:
    frame = base.copy()
    scale = (render_size - 1) / (scenario.width - 1)
    for agent_index, path in enumerate(paths):
        points = np.asarray(
            [
                (
                    round(column * scale),
                    round(row * scale),
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
            max(1, render_size // 500),
            cv2.LINE_AA,
        )
    return frame


def _render_frame(
    route_base: UInt8Image,
    scenario: GridScenario,
    positions: IntArray,
    label: str,
    step: int,
    final_step: int,
    render_size: int,
    seed: int,
) -> UInt8Image:
    frame = route_base.copy()
    scale = (render_size - 1) / (scenario.width - 1)
    radius = max(4, render_size // 100)
    for agent_index, (position, goal) in enumerate(
        zip(positions, scenario.goals, strict=True)
    ):
        goal_center = (
            round(goal[1] * scale),
            round(goal[0] * scale),
        )
        cv2.rectangle(
            frame,
            (
                goal_center[0] - radius,
                goal_center[1] - radius,
            ),
            (
                goal_center[0] + radius,
                goal_center[1] + radius,
            ),
            _COLORS[agent_index],
            max(2, radius // 3),
        )
        center = (
            round(int(position[1]) * scale),
            round(int(position[0]) * scale),
        )
        cv2.circle(
            frame,
            center,
            radius,
            _COLORS[agent_index],
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(agent_index),
            (center[0] - radius // 3, center[1] + radius // 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, render_size / 1800),
            (255, 255, 255),
            max(1, radius // 4),
            cv2.LINE_AA,
        )

    title_height = max(36, render_size // 18)
    titled = np.full(
        (render_size + title_height, render_size, 3),
        248,
        dtype=np.uint8,
    )
    titled[title_height:] = frame
    cv2.putText(
        titled,
        (
            f"{label} | {scenario.width}x{scenario.height} | "
            f"seed {seed} | t={step}/{final_step}"
        ),
        (10, title_height * 2 // 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.42, render_size / 1500),
        (30, 41, 59),
        max(1, render_size // 700),
        cv2.LINE_AA,
    )
    return titled


def _open_writer(
    path: Path,
    suffix: str,
    fps: int,
):
    if suffix == "gif":
        return imageio.get_writer(
            path,
            mode="I",
            duration=1000.0 / fps,
            loop=0,
        )
    return imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )


def _prepare_frame(
    frame: UInt8Image,
    suffix: str,
) -> UInt8Image:
    if suffix != "mp4":
        return frame
    height_padding = frame.shape[0] % 2
    width_padding = frame.shape[1] % 2
    if not height_padding and not width_padding:
        return frame
    return np.pad(
        frame,
        (
            (0, height_padding),
            (0, width_padding),
            (0, 0),
        ),
        mode="edge",
    )


if __name__ == "__main__":
    main()

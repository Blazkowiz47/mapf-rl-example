"""Episode metrics specific to procedural pathfinding."""

from __future__ import annotations

import numpy as np
from dl_core.core import EpisodeRecord, EpisodeResult, register_episode_manager
from dl_robotics import RoboticsEpisodeManager


@register_episode_manager("pathfinding")
class PathfindingEpisodeManager(RoboticsEpisodeManager):
    """Add shortest-path and route-efficiency metrics to robotics summaries."""

    def summarize_episode(
        self,
        record: EpisodeRecord,
        result: EpisodeResult,
        **statistics: float,
    ) -> dict[str, float]:
        """Return generic, robotics, and pathfinding episode metrics."""
        return self._summarize_episode(record, result, **statistics)

    def _summarize_episode(
        self,
        record: EpisodeRecord,
        result: EpisodeResult,
        **statistics: float,
    ) -> dict[str, float]:
        metrics = super().summarize_episode(
            record,
            result,
            **statistics,
        )
        final_info = result.final_info
        values: dict[str, float] = {}
        for info_key, metric_key in (
            (
                "shortest_path_length",
                "pathfinding/expected_shortest_path",
            ),
            ("path_length", "pathfinding/total_steps_taken"),
            ("excess_path_length", "pathfinding/excess_path_length"),
            ("distance_to_goal", "pathfinding/distance_to_goal"),
        ):
            value = final_info.get(info_key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values[metric_key] = float(value)
        metrics.update(values)
        expected = values.get("pathfinding/expected_shortest_path")
        total = values.get("pathfinding/total_steps_taken")
        remaining = values.get("pathfinding/distance_to_goal")
        if expected is not None and total is not None:
            metrics["pathfinding/step_difference"] = total - expected
        if (
            expected is not None
            and total is not None
            and remaining is not None
        ):
            required_travel = total + remaining
            metrics["pathfinding/path_efficiency"] = (
                min(1.0, expected / required_travel)
                if required_travel > 0
                else float(result.terminated)
            )
        return metrics


__all__ = ["PathfindingEpisodeManager"]

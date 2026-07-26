"""The small deterministic MAPF task used by the integration test."""

from dl_robotics import GridScenario


def make_two_agent_crossing_scenario() -> GridScenario:
    """Create two actors exchanging opposite corners around fixed walls."""
    return GridScenario(
        name="two_agent_crossing",
        width=5,
        height=5,
        starts=((0, 0), (4, 4)),
        goals=((4, 4), (0, 0)),
        walls=((1, 2), (3, 2)),
        max_steps=12,
    )

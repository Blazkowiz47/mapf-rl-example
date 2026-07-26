"""Tests that demonstrate swapping interaction policies by configuration."""

import numpy as np
from dl_robotics import GridScenario, GridWorldBatch, make_interaction_rule

import rules  # noqa: F401


def test_example_rules_resolve_the_same_conflict_differently() -> None:
    """Rule names should change actor conflicts without changing the world."""
    scenario = GridScenario(
        width=3,
        height=1,
        starts=((0, 0), (0, 2)),
        goals=((0, 2), (0, 0)),
    )
    actions = np.asarray([[2, 4]], dtype=np.int32)

    exclusive_world = GridWorldBatch(
        scenario,
        interaction_rule=make_interaction_rule("example_exclusive_cell"),
    )
    exclusive_events = exclusive_world.step(actions)
    assert exclusive_world.positions.tolist() == [[[0, 0], [0, 2]]]
    assert exclusive_events.actor_collisions.tolist() == [2]

    priority_world = GridWorldBatch(
        scenario,
        interaction_rule=make_interaction_rule("lowest_index_priority"),
    )
    priority_events = priority_world.step(actions)
    assert priority_world.positions.tolist() == [[[0, 1], [0, 2]]]
    assert priority_events.actor_collisions.tolist() == [1]

    ghost_world = GridWorldBatch(
        scenario,
        interaction_rule=make_interaction_rule("ghost_actors"),
    )
    ghost_events = ghost_world.step(actions)
    assert ghost_world.positions.tolist() == [[[0, 1], [0, 1]]]
    assert ghost_events.actor_collisions.tolist() == [0]

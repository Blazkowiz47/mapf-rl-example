"""Tests for the tabular MAPF observation example."""

import numpy as np
from dl_robotics import GridWorldBatch
from gymnasium.spaces import Discrete

from observation_builders import TabularMAPFObservationBuilder
from scenarios import make_two_agent_crossing_scenario


def test_tabular_builder_encodes_ordered_joint_positions() -> None:
    scenario = make_two_agent_crossing_scenario()
    world = GridWorldBatch(scenario, num_worlds=2)
    builder = TabularMAPFObservationBuilder()

    observation_space = builder.observation_space(scenario)
    initial_states = builder.build(world)
    world.positions[1] = np.asarray(((2, 1), (3, 4)), dtype=np.int32)
    changed_states = builder.build(world)

    assert isinstance(observation_space, Discrete)
    assert observation_space.n == 625
    assert initial_states.dtype == np.int64
    assert initial_states.tolist() == [600, 600]
    assert changed_states.tolist() == [600, 11 + 25 * 19]
    assert all(observation_space.contains(state) for state in changed_states)

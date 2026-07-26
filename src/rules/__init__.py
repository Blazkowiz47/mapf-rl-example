"""Example actor interaction rules."""

from .exclusive_cells import ExampleExclusiveCellRule
from .ghost_actors import GhostActorsRule
from .lowest_index_priority import LowestIndexPriorityRule

__all__ = [
    "ExampleExclusiveCellRule",
    "GhostActorsRule",
    "LowestIndexPriorityRule",
]

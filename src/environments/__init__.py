"""Local robotics environments."""

import observation_builders

from .procedural_pathfinding import ProceduralPathfindingEnv

__all__ = ["ProceduralPathfindingEnv", "observation_builders"]

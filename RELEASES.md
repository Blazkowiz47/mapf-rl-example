# MAPF RL Example Release History

The main README shows only the latest release. This page preserves the
release-by-release changes.

## 0.2.0

- local code follows the component layout for environments, models, callbacks,
  episode managers, interaction rules, and scenarios
- three YAML-selectable rules demonstrate exclusive cells, index priority, and
  intentionally non-physical pass-through actors
- classical report and visualization entry points live under `scripts/`, with
  script flow kept directly in `main()`
- dependency floors require `deep-learning-core>=0.0.28,<0.1` and
  `deep-learning-robotics>=0.0.3,<0.1`

## 0.1.0

- introduced centralized DQN training for a two-agent MAPF crossing task
- added A*, Dijkstra, BFS, and DFS baselines plus collision-free simultaneous
  visualizations
- added procedural single-agent RGB pathfinding with a ViT-B/16 Q-network
- connected episode-quality metrics and sampled model diagnostics to W&B

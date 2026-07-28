# MAPF RL Example Release History

The main README shows only the latest release. This page preserves the
release-by-release changes.

## 0.8.0

- a compact semantic-grid MAPF configuration demonstrates recurrent Dreamer
  collection, sequence replay, world-model learning, and latent imagination
- Dreamer persists optimization metrics, evaluation trajectories, GIFs, and a
  replay-inclusive final checkpoint
- all main trainer presets now use 100,000 transitions, progress display, and
  numbered checkpoints every 25,000 transitions
- the example requires `deep-learning-core>=0.0.35,<0.1`

## 0.7.0

- standalone MAPF configurations now demonstrate tabular Q-learning, DQN, PPO,
  and SAC while keeping trainer-specific state and action adapters separate
- a custom continuous point-mass environment demonstrates configurable time
  steps, acceleration control, velocity control, and swappable dynamics rules
- focused tests verify tabular state encoding, continuous MAPF action
  quantization, exact kinematics, and short trainer lifecycles

## 0.6.0

- a local registered observation builder demonstrates complete control over
  model and replay pixels with hollow blue goals and solid red actors
- the active compiled run uses 256 asynchronous environments, a 256-sample
  replay batch, and a 256-transition update interval
- the example requires `deep-learning-core>=0.0.33,<0.1` and
  `deep-learning-robotics>=0.0.4,<0.1`

## 0.5.0

- reproducible configs compare the 32-environment pipelined baseline, a larger
  replay batch, and selected-model PyTorch compilation
- compiled ViT learning keeps variable DQN inference and mutable W&B hooks out
  of the learner graph while retaining sampled weight histograms
- phase timings make environment, replay, and learner throughput comparisons
  visible in W&B
- the example requires `deep-learning-core>=0.0.32,<0.1`

## 0.4.0

- the ViT-DQN reference run balances 20 environment lanes over two
  inference-only actor copies and separate CUDA streams
- actor snapshots refresh every 25 optimizer steps, with policy version and lag
  available through the existing W&B update metrics
- the example requires `deep-learning-core>=0.0.31,<0.1`

## 0.3.0

- the ViT-DQN reference run uses 20 asynchronous environment processes and a
  two-million-transition budget
- numbered checkpoints are saved every 100,000 transitions, and training
  progress is visible in the terminal
- the example requires `deep-learning-core>=0.0.30,<0.1`

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

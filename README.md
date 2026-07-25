# MAPF RL Example

This is a standalone consumer project showing how
`deep-learning-robotics` environments plug into the vector-aware trainers in
`deep-learning-core`.

The example uses centralized DQN for two actors that must exchange opposite
corners of a 5×5 grid. Four worlds collect experience in parallel, while a
separate scalar world performs deterministic evaluation. The robotics episode
manager records MAPF metrics, the full evaluation trajectory, and an animated
GIF.

## Run

```bash
uv sync --extra dev
uv run dl-run --config configs/mapf_dqn.yaml --validate-only
uv run dl-run --config configs/mapf_dqn.yaml
uv run pytest
```

The short 64-transition configuration is an integration example, not a
convergence benchmark. Increase `trainer.dqn.total_timesteps` for a meaningful
learning experiment.

The CLI run writes artifacts beneath
`artifacts/sweeps/mapf_dqn/mapf_dqn/`, including:

- checkpoints and normal dl-core run metadata
- `final/metrics/episodes_robotics.jsonl`
- captured evaluation trajectories under `final/episodes/evaluation/*.npz`
- matching evaluation animations under `final/episodes/evaluation/*.gif`

## Sweep

Preview or run the included learning-rate/collision-penalty sweep:

```bash
uv run dl-sweep experiments/mapf_sweep.yaml --preview
uv run dl-sweep experiments/mapf_sweep.yaml
```

The collision penalty is varied only for training; the scalar evaluation
environment keeps one fixed reward definition so returns remain comparable
between runs. To sweep wall layouts, maximum makespan, or start/goal
assignments, update the matching scenario fields in both environment blocks.
Keep the grid dimensions and agent count fixed within one sweep when sharing
one neural-network shape.

## Design Boundaries

The joint action contains `5 ** num_agents` discrete choices, which is useful
for small cooperative MAPF research and direct DQN/PPO integration. It is not
the intended scaling path for large fleets; that will require decentralized or
parameter-shared multi-agent policies.

# MAPF RL Example

This is a standalone consumer project showing how
`deep-learning-robotics` environments plug into the vector-aware trainers in
`deep-learning-core`.

The example uses centralized DQN for two actors that must exchange opposite
corners of a 5×5 grid. Four worlds collect experience in parallel, while a
separate scalar world performs deterministic evaluation. The robotics episode
manager records MAPF metrics, the full evaluation trajectory, and an animated
GIF.

The repository also contains `ProceduralPathfindingEnv`, a single-agent
Gymnasium environment for the ViT pathfinding example. Every reset replaces the
previous maze with a seeded 1000×1000 task containing one red agent and one blue
goal. `get_grid_rgb()` and `render()` expose the current grid as an RGB
`uint8` matrix, while the normal observation is resized to 256×256 for
memory-bounded replay and ViT input.

```python
from pathfinding_environment import ProceduralPathfindingEnv

environment = ProceduralPathfindingEnv()
observation, info = environment.reset(seed=2026)
full_grid = environment.get_grid_rgb()  # [1000, 1000, 3]
```

The `vit_b_16_q_network` model accepts the resized HWC `uint8` observation,
normalizes it with ImageNet statistics, and emits four unnormalized Q-values
for up, down, left, and right. It starts from ImageNet-1K weights adapted to
the 256×256 positional embedding, fine-tunes the final two transformer blocks,
and leaves the earlier encoder frozen.

## Run

```bash
uv sync --extra dev
uv run dl-run --config configs/mapf_dqn.yaml --validate-only
uv run dl-run --config configs/mapf_dqn.yaml
uv run dl-run --config configs/pathfinding_vit_dqn.yaml --validate-only
uv run pytest
```

The short 64-transition configuration is an integration example, not a
convergence benchmark. Increase `trainer.dqn.total_timesteps` for a meaningful
learning experiment. `pathfinding_vit_dqn.yaml` is the separate long-running
GPU configuration: 32 environments collect procedural tasks in parallel, the
replay buffer retains 256×256 RGB observations, and DQN is budgeted for the
requested two billion environment transitions. The default 4096-transition
replay buffer uses about 1.5 GiB of host RAM for current and next `uint8`
observations. The batch size of 128 is a portable starting point; benchmark a
run-specific override before increasing it for a larger accelerator.

The long run uses `deep-learning-wandb` through the local `sampled_wandb`
callback. It retains the integration's normal configuration, run metadata,
evaluation metrics, and `global_step` semantics while sampling training
episodes and DQN updates so a two-billion-step job does not produce millions
of nearly adjacent API calls. Set `WANDB_API_KEY` in the shell before
launching; `.env.example` documents the expected variable but is not loaded
automatically.

Evaluate a saved checkpoint on three held-out procedural mazes and export both
GIF and MP4 trajectories:

```bash
uv run python scripts/evaluate_pathfinding_checkpoint.py \
  --checkpoint artifacts/runs/pathfinding_vit_2b/final/checkpoints/latest.pth
```

The evaluator loads only the online ViT weights, performs no pretrained-weight
download, selects actions deterministically, and reports returns, success,
distance, path length, shortest-path lower bound, and collisions in
`artifacts/pathfinding_evaluation/evaluation.json`. Its default media is
512×512 for manageable files; pass `--render-size 1000` to preserve the full
logical grid resolution. It records every fourth environment frame by default
to bound encoder memory and adjusts playback FPS to preserve elapsed-time
semantics; use `--frame-stride 1` for every step.

The training CLI writes artifacts beneath
`artifacts/sweeps/mapf_dqn/mapf_dqn/`, including:

- checkpoints and normal dl-core run metadata
- `final/metrics/episodes_robotics.jsonl`
- captured evaluation trajectories under `final/episodes/evaluation/*.npz`
- matching evaluation animations under `final/episodes/evaluation/*.gif`

## Classical Baselines

Print the deterministic A*, Dijkstra, BFS, and DFS paths for every agent:

```bash
uv run python src/mapf_baselines.py
```

The report includes exact per-agent shortest move counts plus static makespan
and sum-of-costs lower bounds. A*, Dijkstra, and BFS all establish the optimal
single-agent costs; DFS is included as a deterministic reachability/debugging
route and is not treated as optimal. The integration test verifies that the
hand-authored collision-free schedule reaches both agents' goals with no
collisions while matching both exact lower bounds.

These paths ignore other moving actors. Independently optimal single-agent
paths can conflict, so use them as RL baselines and lower bounds rather than as
a complete MAPF solver.

## Visualize Classical Plans

Generate a seeded 2000×2000 logical maze with random wall segments, three
random actors, three seeded reachable goals, separate A*/Dijkstra/BFS/DFS
animations, and a combined 2×2 comparison:

```bash
uv run python scripts/visualize_classical_planners.py
```

GIFs are written to `artifacts/classical_planners/` by default. Generate MP4s
as well, select another seed, or control the downsampled render:

```bash
uv run python scripts/visualize_classical_planners.py \
  --format both \
  --output-dir artifacts/classical_planners \
  --grid-size 2000 \
  --render-size 400 \
  --max-frames 60 \
  --seed 2026 \
  --fps 6
```

Every algorithm plans one route per actor. A deterministic reservation pass
then assigns the smallest safe start delay and rejects vertex conflicts and
edge swaps. The resulting joint actions are executed simultaneously through
`GridWorldBatch`; media generation fails if any actor collision, wall
collision, boundary collision, or incomplete goal remains. The colored lines
show complete routes, numbered circles are current actors, and matching
outlined cells are their goals.

The logical occupancy map remains 2000×2000, while OpenCV downsamples its
static maze layer for display. Frames are written incrementally with ImageIO,
so the script does not retain hundreds of full-resolution frames in memory.
This visualizes execution of returned paths, not search-frontier expansion.
Start-delay reservations make this generated example safe, but they are still
a limited coordination strategy rather than a complete MAPF solver.

Goal selection samples reachable points from deterministic DFS traversals, so
it is reproducible and cannot place an unreachable target, but it is not a
spatially uniform sample. The default four-million-cell run can peak near
1.2 GiB because the package's current pure-Python BFS and Dijkstra
implementations retain large search dictionaries. Use `--grid-size 512` for a
lighter teaching run.

![A 2x2 animated comparison of A-star, Dijkstra, BFS, and DFS routes](docs/media/classical_planners/comparison.gif)

Individual previews: [A*](docs/media/classical_planners/astar.gif),
[Dijkstra](docs/media/classical_planners/dijkstra.gif),
[BFS](docs/media/classical_planners/bfs.gif), and
[DFS](docs/media/classical_planners/dfs.gif).

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

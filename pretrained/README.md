# Trained RL Models

These files contain the inference state from the seeded 100,000-transition
runs documented in the project README. Each use case has its own model file:

- `mapf_q_learning_100k.pt`
- `mapf_dqn_100k.pt`
- `mapf_ppo_100k.pt`
- `mapf_sac_100k.pt`
- `point_mass_acceleration_sac_100k.pt`
- `point_mass_velocity_ppo_100k.pt`
- `mapf_dreamer_100k.pt`

Load an artifact without enabling arbitrary Python object deserialization:

```python
import torch

artifact = torch.load(
    "pretrained/mapf_dqn_100k.pt",
    map_location="cpu",
    weights_only=True,
)
online_q_network.load_state_dict(
    artifact["policy_state_dicts"]["online"]
)
```

For Q-learning, `policy_state_dicts["q_table"]["q_table"]` is the learned
table. For DQN, PPO, and SAC, load the state named in `policy_components` into
the matching project model. Dreamer inference needs both `world_model` and
`actor`. `config_path` identifies the matching environment and model
configuration, while `checkpoint` identifies the fixed-seed evaluation winner.

The exports omit replay buffers, optimizers, metric history, and random-number
state. They are intended for inference and evaluation rather than exact
training resumption. After retraining, compare every numbered checkpoint before
exporting:

```bash
uv run python scripts/evaluate_rl_checkpoints.py
uv run python scripts/export_rl_models.py
```

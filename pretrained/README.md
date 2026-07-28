# Trained RL Models

These files contain the inference state from the seeded 100,000-transition
runs documented in the project README. Each use case has its own model file:

- `mapf_q_learning_100k.pt`
- `mapf_dqn_100k.pt`
- `mapf_ppo_100k.pt`
- `mapf_sac_100k.pt`
- `point_mass_acceleration_sac_100k.pt`
- `point_mass_velocity_ppo_100k.pt`

Load an artifact without enabling arbitrary Python object deserialization:

```python
import torch

artifact = torch.load(
    "pretrained/mapf_dqn_100k.pt",
    map_location="cpu",
    weights_only=True,
)
online_q_network.load_state_dict(artifact["policy_state_dict"])
```

For Q-learning, `policy_state_dict["q_table"]` is the learned table. For DQN,
PPO, and SAC, load `policy_state_dict` into the component named by
`policy_component`. `config_path` identifies the matching environment and
model configuration.

The exports omit replay buffers, optimizers, metric history, and random-number
state. They are intended for inference and evaluation rather than exact
training resumption. Run `uv run python scripts/export_rl_models.py` after
retraining to refresh all six files.

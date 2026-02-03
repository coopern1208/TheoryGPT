# GRPO Trainer for TheoryGPT

## Overview

This directory contains the implementation of **GRPO (Group Relative Policy Optimization)** for training TheoryGPT to generate physics theories with reward-based optimization.

### What is GRPO?

GRPO is a reinforcement learning algorithm that:
1. **Generates groups of completions** for each prompt (or from scratch)
2. **Computes rewards** for each completion using physics-based reward functions
3. **Calculates group-relative advantages** by comparing rewards within each group
4. **Updates the policy** to increase the probability of higher-reward completions

### Key Features

- ✅ **Grammar-constrained generation** using TheoryEnvironment
- ✅ **Group-relative advantage estimation** for stable training
- ✅ **KL divergence penalty** to prevent policy from deviating too far from reference
- ✅ **PPO-style clipping** for robust policy updates
- ✅ **Entropy bonus** to encourage exploration
- ✅ **Curriculum learning** support for progressive training
- ✅ **Prioritized replay buffer** option for experience replay
- ✅ **Automatic checkpointing** and logging

---

## Architecture

### Components

1. **GRPOTrainer** (`grpo_trainer.py`)
   - Main training class
   - Handles generation, reward computation, and policy updates
   - Manages checkpointing and logging

2. **TheoryEnvironment** (`environment.py`)
   - RL environment for physics theory generation
   - Enforces grammar constraints
   - Manages state and valid actions

3. **ReplayBuffer** (`replay_buffer.py`)
   - Stores successful trajectories
   - Supports weighted sampling and prioritization

4. **Curriculum** (`curriculum.py`)
   - Manages progressive training phases
   - Adjusts loss weights based on success rate

---

## Installation

The GRPO trainer uses the same dependencies as the main project:

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure all dependencies are installed
pip install torch numpy tqdm
```

---

## Quick Start

### 1. Basic Training (from scratch)

```bash
python training/train_grpo.py \
    --num_episodes 1000 \
    --group_size 64 \
    --learning_rate 1e-4 \
    --temperature 1.0 \
    --log_interval 10 \
    --save_interval 100
```

### 2. Training from Pretrained Model

```bash
python training/train_grpo.py \
    --checkpoint checkpoints/pretrain_20260121_024027/best_model.pt \
    --num_episodes 1000 \
    --group_size 64 \
    --learning_rate 1e-4
```

### 3. Resume Training from GRPO Checkpoint

```bash
python training/train_grpo.py \
    --resume checkpoints/grpo_20260128_123456/checkpoint_episode_500.pt \
    --num_episodes 1000
```

### 4. Evaluation Only

```bash
python training/train_grpo.py \
    --checkpoint checkpoints/grpo_20260128_123456/best_model.pt \
    --eval_only \
    --eval_episodes 100
```

### 5. SLURM Cluster Training

```bash
sbatch training/sbatch_grpo.slurm
```

---

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 64 | Number of completions per prompt |
| `learning_rate` | 1e-4 | Learning rate for optimizer |
| `kl_coef` | 0.05 | KL divergence penalty coefficient |
| `clip_range` | 0.2 | PPO clipping range |
| `entropy_coef` | 0.01 | Entropy bonus coefficient |
| `temperature` | 1.0 | Sampling temperature (higher = more random) |

### Modifying Config

Edit `config.py` to change default parameters:

```python
# In config.py
@dataclass
class Config:
    # GRPO parameters
    RL_LEARNING_RATE: float = 1e-4
    RL_BATCH_SIZE: int = 32
    RL_MAX_STEPS: int = 10000
    GRPO_GROUP_SIZE: int = 64
    REPLAY_BUFFER_SIZE: int = 1000
```

---

## Usage Examples

### Example 1: Using GRPOTrainer Programmatically

```python
from architecture.TheoryGPT import TheoryGPT
from rl.grpo_trainer import GRPOTrainer
from config import config

# Initialize model
model = TheoryGPT(config)

# Initialize trainer
trainer = GRPOTrainer(
    model=model,
    config=config,
    use_curriculum=True,
    use_priority_buffer=False,
)

# Train for 1000 episodes
trainer.train(
    num_episodes=1000,
    log_interval=10,
    save_interval=100,
    temperature=1.0,
)

# Evaluate
metrics = trainer.evaluate(num_episodes=100)
print(f"Mean reward: {metrics['mean_reward']:.4f}")
print(f"Success rate: {metrics['success_rate']:.2%}")
```

### Example 2: Custom Hyperparameters

```python
trainer = GRPOTrainer(model=model, config=config)

# Adjust hyperparameters
trainer.kl_coef = 0.1        # Stronger KL penalty
trainer.clip_range = 0.3     # Wider clipping range
trainer.entropy_coef = 0.02  # More exploration
trainer.group_size = 128     # Larger groups

# Train
trainer.train(num_episodes=500)
```

### Example 3: With Prompts

```python
# Define prompts (list of token lists)
prompts = [
    ["BOS", "GAUGE_GROUPS", "NUM_GAUGE", "num_1"],
    ["BOS", "GAUGE_GROUPS", "NUM_GAUGE", "num_2"],
]

# Train with prompts (cycles through them)
trainer.train(
    num_episodes=1000,
    prompts=prompts,
    temperature=1.0,
)
```

---

## Training Workflow

### 1. Generation Phase

For each training episode:
- Sample `group_size` completions from the policy
- Each completion follows grammar constraints from `TheoryEnvironment`
- Tokens are sampled using temperature-controlled sampling

### 2. Reward Computation

For each completion:
- Build physics model from generated tokens
- Compute reward using `reward.all_rewards()`:
  - **Anomaly cancellation** (gauge anomalies should be zero)
  - **Particle masses** (charged exotics should be heavy)
  - **Theory length** (penalty for overly long theories)

### 3. Advantage Calculation

Within each group:
- Compute group mean reward: `μ = mean(rewards)`
- Compute group std reward: `σ = std(rewards)`
- Advantages: `A_i = (reward_i - μ) / σ`

This makes advantages **relative** within each group, leading to more stable training.

### 4. Policy Update

- Compute log probabilities under current and old policy
- Compute policy ratio: `r = exp(log_prob_new - log_prob_old)`
- Apply PPO clipping: `min(r * A, clip(r, 1-ε, 1+ε) * A)`
- Add KL penalty: `β * KL(π_new || π_ref)`
- Add entropy bonus: `-α * H(π_new)`
- Update parameters via gradient descent

---

## Monitoring Training

### 1. Console Output

During training, you'll see progress bars with real-time metrics:

```
Training: 100%|████████| 1000/1000 [2:34:12<00:00, reward=12.34, success=45.2%, loss=0.1234]
```

### 2. Periodic Logs

Every `log_interval` episodes, detailed statistics are printed:

```
================================================================================
Episode 100 Statistics:
================================================================================
  mean_reward         : 15.2341
  success_rate        : 0.4520
  total_loss          : 0.1234
  policy_loss         : 0.0987
  kl_loss             : 0.0123
  entropy             : 2.3456

Replay Buffer:
  Size: 234/1000
  Mean Reward: 18.45
  Success Rate: 62.34%

Curriculum Phase: 2
================================================================================
```

### 3. Saved Files

Training produces:
- `checkpoints/grpo_YYYYMMDD_HHMMSS/`
  - `checkpoint_episode_N.pt` - Regular checkpoints
  - `best_model.pt` - Best model (highest mean reward)
  - `training_stats.json` - Full training metrics

---

## Checkpoints

### Checkpoint Structure

```python
checkpoint = {
    "episode": 500,
    "global_step": 500,
    "model_state_dict": {...},
    "optimizer_state_dict": {...},
    "best_reward": 25.67,
    "training_stats": {...},
    "config": Config(...),
}
```

### Loading Checkpoints

```python
# Load and resume training
trainer = GRPOTrainer(
    model=model,
    checkpoint_path="checkpoints/grpo_YYYYMMDD_HHMMSS/checkpoint_episode_500.pt"
)
trainer.train(num_episodes=500)  # Continue for 500 more episodes
```

---

## Reward Function

The reward function (`reward/reward.py`) evaluates generated theories based on:

### 1. Anomaly Cancellation
- **Goal**: Gauge anomalies should be zero for consistent quantum field theory
- **Reward**: `log_reward(anomaly_value)` with bonus for exact cancellation

### 2. Particle Masses
- **Goal**: Charged exotic particles should be heavy (above threshold)
- **Reward**: `square_threshold_reward(mass, threshold=100)`

### 3. Theory Length
- **Penalty**: -300 for theories that are too long

### Total Reward
```python
total_reward = anomalies_reward + light_exotics_reward + length_penalty
```

---

## Troubleshooting

### Issue: Low Success Rate

**Symptoms**: Success rate < 10%

**Solutions**:
1. Lower temperature for less random sampling
2. Enable curriculum learning: `--use_curriculum`
3. Increase group size for better advantage estimates
4. Start from pretrained model

### Issue: Policy Collapse

**Symptoms**: Rewards suddenly drop, all completions similar

**Solutions**:
1. Increase KL coefficient: `--kl_coef 0.1`
2. Increase entropy coefficient: `--entropy_coef 0.02`
3. Reduce learning rate: `--learning_rate 5e-5`
4. Use smaller clip range: `--clip_range 0.1`

### Issue: Slow Training

**Symptoms**: Training takes very long per episode

**Solutions**:
1. Reduce group size: `--group_size 32`
2. Use GPU if available
3. Reduce max sequence length in config
4. Use KV cache for generation (already implemented)

### Issue: High Variance in Rewards

**Symptoms**: Rewards fluctuate wildly between episodes

**Solutions**:
1. Increase group size for better statistics
2. Use prioritized replay buffer: `--use_priority_buffer`
3. Adjust reward normalization in advantage computation

---

## Advanced Usage

### Custom Reward Function

To implement a custom reward function:

```python
# In reward/reward.py
def custom_reward(model: dict) -> float:
    # Your custom physics-based reward
    reward = 0.0
    
    # Example: Reward for specific particle content
    for multiplet in model["multiplets"].values():
        if multiplet["type"] == "FERMION":
            reward += 1.0
    
    return reward

# In grpo_trainer.py, modify train_step():
# reward = custom_reward(model_dict)
```

### Custom Advantage Computation

```python
trainer = GRPOTrainer(model=model)

# Override compute_advantages
def custom_advantages(completions):
    rewards = torch.tensor([c["reward"] for c in completions])
    # Custom normalization
    advantages = rewards / (rewards.abs().max() + 1e-8)
    return advantages

# Use in training loop
completions = trainer.generate_group()
advantages = custom_advantages(completions)
loss, metrics = trainer.compute_policy_loss(completions, advantages)
```

---

## Performance Tips

1. **Use pretrained models**: Start from SFT checkpoint for faster convergence
2. **Tune temperature**: Lower (0.7-0.9) for exploitation, higher (1.0-1.2) for exploration
3. **Adjust group size**: Balance between sample efficiency (small) and stability (large)
4. **Monitor KL divergence**: Should stay in range [0.01, 0.1]; adjust `kl_coef` if needed
5. **Use curriculum**: Helps with bootstrapping from easy to hard theories
6. **Save frequently**: Set `--save_interval` low enough to not lose progress

---

## Citation

If you use this GRPO implementation in your research, please cite:

```bibtex
@article{theorygpt2026,
  title={GRPO Training for Physics Theory Generation with TheoryGPT},
  author={Your Name},
  year={2026}
}
```

---

## Support

For questions or issues:
- Check existing logs in `log/` directory
- Review checkpoint statistics in `training_stats.json`
- Consult `config.py` for parameter descriptions
- Open an issue with training logs and config details

---

## Future Improvements

Potential enhancements:
- [ ] Multi-GPU training support
- [ ] Distributed GRPO across multiple nodes
- [ ] Adaptive KL coefficient (increase/decrease based on divergence)
- [ ] Value function baseline for variance reduction
- [ ] Hindsight experience replay for failed trajectories
- [ ] Automatic hyperparameter tuning
- [ ] Real-time visualization dashboard
- [ ] Integration with physics simulation software (SARAH)

---

## License

This code is part of the RL_model_builder_6.0 project.

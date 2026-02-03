# GRPO Trainer - Quick Start Guide

## ⚡ 30-Second Start

```bash
# 1. Test it works
python rl/test_grpo.py

# 2. Train (basic)
python training/train_grpo.py --num_episodes 100

# 3. Train (from pretrained)
python training/train_grpo.py \
    --checkpoint checkpoints/pretrain_20260121_024027/best_model.pt \
    --num_episodes 1000

# 4. Train (cluster)
sbatch training/sbatch_grpo.slurm
```

## 📁 What Was Created

| File | Purpose |
|------|---------|
| `rl/grpo_trainer.py` | Main GRPO implementation |
| `training/train_grpo.py` | Command-line training script |
| `training/sbatch_grpo.slurm` | SLURM batch script |
| `rl/test_grpo.py` | Test suite |
| `rl/README_GRPO.md` | Full documentation |
| `rl/GRPO_SUMMARY.md` | Implementation summary |

## 🎯 Common Commands

### Basic Training
```bash
python training/train_grpo.py --num_episodes 1000
```

### With Custom Parameters
```bash
python training/train_grpo.py \
    --num_episodes 1000 \
    --group_size 64 \
    --learning_rate 1e-4 \
    --temperature 1.0 \
    --kl_coef 0.05 \
    --clip_range 0.2
```

### From Pretrained Checkpoint
```bash
python training/train_grpo.py \
    --checkpoint checkpoints/pretrain_20260121_024027/best_model.pt \
    --num_episodes 1000
```

### Resume Training
```bash
python training/train_grpo.py \
    --resume checkpoints/grpo_20260128_123456/checkpoint_episode_500.pt \
    --num_episodes 500
```

### Evaluation Only
```bash
python training/train_grpo.py \
    --checkpoint checkpoints/grpo_20260128_123456/best_model.pt \
    --eval_only
```

## 🔧 Key Hyperparameters

| Parameter | Default | Description | Typical Range |
|-----------|---------|-------------|---------------|
| `group_size` | 64 | Completions per prompt | 32-128 |
| `learning_rate` | 1e-4 | Optimizer learning rate | 5e-5 to 1e-3 |
| `temperature` | 1.0 | Sampling temperature | 0.7-1.2 |
| `kl_coef` | 0.05 | KL penalty weight | 0.01-0.2 |
| `clip_range` | 0.2 | PPO clipping | 0.1-0.3 |
| `entropy_coef` | 0.01 | Entropy bonus | 0.005-0.02 |

## 📊 What to Expect

### Training Progress

| Phase | Episodes | Success Rate | Mean Reward |
|-------|----------|--------------|-------------|
| Exploration | 0-100 | 10-30% | -50 to -10 |
| Learning | 100-500 | 30-60% | -10 to +10 |
| Optimization | 500+ | 60-80% | +10 to +25 |

### Output Files

```
checkpoints/grpo_YYYYMMDD_HHMMSS/
├── checkpoint_episode_100.pt
├── checkpoint_episode_200.pt
├── ...
├── best_model.pt
└── training_stats.json
```

## 🐛 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Low success rate | `--use_curriculum --temperature 0.8` |
| Policy collapse | `--kl_coef 0.1 --entropy_coef 0.02` |
| Out of memory | `--group_size 32` |
| Slow training | Reduce group size or use GPU |
| High variance | Increase group size |

## 📚 More Information

- **Full Documentation**: See `rl/README_GRPO.md`
- **Implementation Details**: See `rl/GRPO_SUMMARY.md`
- **Test Suite**: Run `python rl/test_grpo.py`

## 🎓 Recommended Workflow

1. **Test**: `python rl/test_grpo.py`
2. **Small-scale**: Train 100 episodes to verify
3. **Tune**: Adjust hyperparameters based on results
4. **Scale up**: Full training 1000+ episodes
5. **Evaluate**: Compare against baselines

## 💡 Pro Tips

- ✅ Start from pretrained model for faster convergence
- ✅ Use `--use_curriculum` for bootstrapping
- ✅ Monitor KL divergence (keep around 0.01-0.1)
- ✅ Lower temperature (0.7-0.9) for exploitation
- ✅ Higher temperature (1.0-1.2) for exploration
- ✅ Save frequently with `--save_interval 50`

## 🚀 Example: Full Training Pipeline

```bash
# Step 1: Test installation
python rl/test_grpo.py

# Step 2: Small test run (10 episodes)
python training/train_grpo.py --num_episodes 10

# Step 3: Load pretrained and train
python training/train_grpo.py \
    --checkpoint checkpoints/pretrain_20260121_024027/best_model.pt \
    --num_episodes 1000 \
    --group_size 64 \
    --learning_rate 1e-4 \
    --use_curriculum \
    --log_interval 10 \
    --save_interval 100

# Step 4: Evaluate
python training/train_grpo.py \
    --checkpoint checkpoints/grpo_20260128_123456/best_model.pt \
    --eval_only \
    --eval_episodes 100
```

## ⚙️ SLURM Cluster Usage

Edit `training/sbatch_grpo.slurm` to set your checkpoint path:

```bash
# In sbatch_grpo.slurm, modify:
CHECKPOINT="checkpoints/pretrain_20260121_024027/best_model.pt"
```

Then submit:

```bash
sbatch training/sbatch_grpo.slurm
```

Monitor progress:

```bash
# Check output
tail -f log/output_JOBID.txt

# Check errors
tail -f log/error_JOBID.txt
```

## 🎯 Ready to Go!

Your GRPO trainer is ready. Start with:

```bash
python training/train_grpo.py --num_episodes 1000
```

Good luck with your training! 🚀

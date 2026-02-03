"""
Training script for GRPO (Group Relative Policy Optimization) on TheoryGPT

This script trains TheoryGPT using GRPO to generate physics theories
with reward-based optimization and grammar constraints.
"""

import torch
import os
from datetime import datetime

from architecture.TheoryGPT import TheoryGPT
from rl.grpo_trainer import GRPOTrainer
from config import config
from grammar.parser import read_model_txt

def main():
    """Main training function for GRPO."""
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    # Pretrained model path (set to None to train from scratch)
    PRETRAINED_MODEL_PATH = 'checkpoints/pretrain_20260130_004801/best_model.pt'
    
    # Resume from checkpoint (set to None to start fresh)
    RESUME_CHECKPOINT_PATH = None  # e.g., "checkpoints/grpo_20260130_010151/checkpoint_episode_500.pt"
    
    # ========================================================================
    # Setup
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("GRPO Training for TheoryGPT")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print(f"  - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"\nTraining Parameters:")
    print(f"  - Random seed: {config.RL_RANDOM_SEED}")
    print(f"  - Episodes: {config.RL_NUM_EPISODES}")
    print(f"  - Group size: {config.GRPO_GROUP_SIZE}")
    print(f"  - Learning rate: {config.RL_LEARNING_RATE}")
    print(f"  - Temperature: {config.RL_TEMPERATURE}")
    print(f"\nGRPO Hyperparameters:")
    print(f"  - KL coefficient: {config.RL_KL_COEF}")
    print(f"  - Clip range: {config.RL_CLIP_RANGE}")
    print(f"  - Entropy coefficient: {config.RL_ENTROPY_COEF_INITIAL}")
    print(f"  - Entropy coefficient final: {config.RL_ENTROPY_COEF_FINAL}")
    print(f"  - Entropy decay schedule: {config.RL_ENTROPY_DECAY_SCHEDULE}")
    print(f"\nOptions:")
    print(f"  - Curriculum learning: {config.RL_USE_CURRICULUM}")
    print(f"  - Pretrained model: {PRETRAINED_MODEL_PATH if PRETRAINED_MODEL_PATH else 'None (training from scratch)'}")
    print(f"  - Resume checkpoint: {RESUME_CHECKPOINT_PATH if RESUME_CHECKPOINT_PATH else 'None (starting fresh)'}")
    print("=" * 80 + "\n")
    
    # ========================================================================
    # Initialize Model
    # ========================================================================
    
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"Loading pretrained model from: {PRETRAINED_MODEL_PATH}")
        model = TheoryGPT(config)
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        print("Successfully loaded pretrained model!")
    else:
        print("Initializing new model from scratch...")
        model = TheoryGPT(config)
    
    # ========================================================================
    # Initialize GRPO Trainer
    # ========================================================================
    
    trainer = GRPOTrainer(
        model=model,
        config=config,
        reference_model=None, 
        use_curriculum=config.RL_USE_CURRICULUM,
        checkpoint_path=RESUME_CHECKPOINT_PATH,
    )
    
    # ========================================================================
    # Training
    # ========================================================================
    
    print("\nStarting GRPO training...\n")
    
    try:
        trainer.train(
            num_episodes=config.RL_NUM_EPISODES,
            log_interval=config.RL_LOG_INTERVAL,
            save_interval=config.RL_SAVE_INTERVAL,
            temperature=config.RL_TEMPERATURE,
            prompts=None,  # No prompts - start from scratch
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving checkpoint before exit...")
        trainer.save_checkpoint(trainer.episode, is_best=False)
        print("Checkpoint saved. Exiting...")
        return
    
    # ========================================================================
    # Final Evaluation
    # ========================================================================
    
    print("\nRunning final evaluation...\n")
    sm_prompt = read_model_txt("dataset/prompts/SM_prompt.txt")
    eval_metrics = trainer.evaluate(
        num_episodes=config.RL_VAL_EPISODES,
        temperature=config.RL_TEMPERATURE,
        prompts=[sm_prompt],  # Wrap in list - prompts expects List[List[str]]
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nFinal Evaluation Results:")
    for key, value in eval_metrics.items():
        print(f"  {key:20s}: {value:.4f}")
    print(f"\nCheckpoint directory: {trainer.checkpoint_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

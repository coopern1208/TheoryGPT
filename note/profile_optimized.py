#!/usr/bin/env python3
"""
Profile the OPTIMIZED generation to measure improvements.
Compare with baseline in log/output_238389.txt
"""

import torch
from architecture.TheoryGPT import TheoryGPT
from rl.grpo_trainer import GRPOTrainer
from config import config

def main():
    print("="*80)
    print("OPTIMIZED GRPO Generation Profiling")
    print("="*80)
    print("\nThis will profile the VECTORIZED batch generation.")
    print("Compare results with baseline in log/output_238389.txt")
    print("="*80 + "\n")
    
    # Initialize model
    print("Initializing model...")
    model = TheoryGPT(config)
    
    # Initialize trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        config=config,
        use_curriculum=True
    )
    
    # Profile with different batch sizes
    batch_sizes = [4, 8, 16]
    
    print("\n" + "#"*80)
    print("# BASELINE RESULTS (from log/output_238389.txt)")
    print("#"*80)
    print("\nBatch Size  | Time/Step | forward_pass | masking_sampling | per_seq_processing")
    print("-"*80)
    print("     4      |   6.34ms  |    32.7%     |      7.8%        |      11.2%")
    print("     8      |   7.70ms  |    26.9%     |     10.7%        |      15.1%")
    print("    16      |  11.03ms  |    18.7%     |     14.7%        |      20.6%")
    print("="*80 + "\n")
    
    for batch_size in batch_sizes:
        print(f"\n\n{'#'*80}")
        print(f"# OPTIMIZED PROFILING WITH BATCH SIZE = {batch_size}")
        print(f"{'#'*80}\n")
        
        try:
            trainer.profile_generation(num_batches=3, batch_size=batch_size)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️  Batch size {batch_size} caused OOM. Skipping larger batches.")
                break
            else:
                raise
    
    print("\n" + "="*80)
    print("PROFILING SESSION COMPLETE")
    print("="*80)
    print("\nExpected Improvements:")
    print("  ✓ Time per step: 2-3x faster (11ms → 4-5ms at batch_size=16)")
    print("  ✓ masking_sampling: Should drop from ~15% to ~3%")
    print("  ✓ per_sequence_processing: Should drop from ~21% to ~8%")
    print("  ✓ forward_pass percentage: Should INCREASE (becomes dominant bottleneck)")
    print("\nIf forward_pass becomes >50%, next optimization: Enable KV cache!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

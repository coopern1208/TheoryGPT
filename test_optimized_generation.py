#!/usr/bin/env python3
"""
Test script to verify the optimized batch generation produces correct results
and measure speedup.
"""

import torch
import time
from architecture.TheoryGPT import TheoryGPT
from rl.grpo_trainer import GRPOTrainer
from config import config

def test_correctness():
    """Verify that optimized version produces valid outputs"""
    print("="*80)
    print("CORRECTNESS TEST")
    print("="*80)
    
    # Initialize
    model = TheoryGPT(config)
    trainer = GRPOTrainer(model=model, config=config, use_curriculum=True)
    
    # Generate a batch
    print("\nGenerating batch of 4 sequences...")
    results = trainer.generate_completion_batch(batch_size=4, temperature=1.0)
    
    # Verify structure
    assert len(results) == 4, "Should return 4 results"
    
    for i, (tokens, log_probs, token_ids, info, model_dict) in enumerate(results):
        print(f"\nSequence {i+1}:")
        print(f"  - Tokens length: {len(tokens)}")
        print(f"  - Log probs shape: {log_probs.shape}")
        print(f"  - Token IDs shape: {token_ids.shape}")
        print(f"  - Has model_dict: {model_dict is not None}")
        print(f"  - Info keys: {list(info.keys())}")
        
        # Verify consistency
        assert len(token_ids) == len(log_probs), "Token IDs and log probs should match"
        assert log_probs.device.type == 'cuda', "Log probs should be on GPU"
        assert token_ids.device.type == 'cuda', "Token IDs should be on GPU"
    
    print("\n✅ All correctness checks passed!")
    return trainer

def benchmark_speed(trainer, batch_sizes=[4, 8, 16, 32]):
    """Benchmark speed with different batch sizes"""
    print("\n" + "="*80)
    print("SPEED BENCHMARK")
    print("="*80)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Warmup
        _ = trainer.generate_completion_batch(batch_size=batch_size)
        
        # Benchmark
        num_runs = 3
        times = []
        
        for run in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            results = trainer.generate_completion_batch(batch_size=batch_size)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            # Count total steps
            total_steps = sum(len(r[2]) for r in results)  # r[2] is token_ids
            
            print(f"  Run {run+1}: {elapsed:.3f}s ({total_steps} total tokens, "
                  f"{total_steps/elapsed:.1f} tokens/sec, "
                  f"{elapsed*1000/total_steps:.2f}ms per token)")
        
        avg_time = sum(times) / len(times)
        print(f"  Average: {avg_time:.3f}s")

def main():
    print("Testing optimized batch generation...\n")
    
    # Test correctness
    trainer = test_correctness()
    
    # Benchmark speed
    benchmark_speed(trainer, batch_sizes=[4, 8, 16, 32])
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nTo compare with baseline, check log/output_238389.txt")
    print("Look for improvements in:")
    print("  - masking_sampling: Should decrease from ~15% to ~3%")
    print("  - per_sequence_processing: Should decrease from ~22% to ~8%")
    print("  - Overall time per step: Should decrease from ~11ms to ~4-5ms")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

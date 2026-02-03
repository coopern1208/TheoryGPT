import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from contextlib import contextmanager
import copy

from architecture.TheoryGPT import TheoryGPT
from rl.environment import TheoryEnvironment
from rl.curriculum import Curriculum
from reward.reward import all_rewards
from config import config, Config
import grammar.vocab as vocab

@contextmanager
def timer(name, stats_dict):
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if name not in stats_dict:
        stats_dict[name] = []
    stats_dict[name].append(elapsed)

class GRPOTrainer:
    def __init__(self,
                 model: TheoryGPT,
                 config: Config = config,
                 reference_model: Optional[TheoryGPT] = None,
                 use_curriculum: bool = True,
                 checkpoint_path: Optional[str] = None,
                 seed: Optional[int] = None
                 ):

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_device = torch.device("cpu")
        
        # Set random seed for reproducibility
        self.seed = seed if seed is not None else config.RL_RANDOM_SEED
        self.set_seed(self.seed)
        
        # Models
        self.model = model.to(self.device)
        if reference_model is None:
            # Create a frozen copy of the initial model for KL penalty
            self.reference_model = copy.deepcopy(model).to(self.device)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        else:
            self.reference_model = reference_model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.RL_LEARNING_RATE,
                                           betas=(0.9, 0.999),
                                           eps=1e-8,
                                           weight_decay=0.01
                                           )
        
        # Environment and replay buffer
        self.curriculum = Curriculum() if use_curriculum else None
        self.env = TheoryEnvironment(config, curriculum=self.curriculum)

        # GRPO hyperparameters
        self.group_size = config.GRPO_GROUP_SIZE
        self.kl_coef = config.RL_KL_COEF
        self.clip_range = config.RL_CLIP_RANGE
        
        # Entropy coefficient with decay
        self.entropy_coef_initial = config.RL_ENTROPY_COEF_INITIAL
        self.entropy_coef_final = config.RL_ENTROPY_COEF_FINAL
        self.entropy_decay_schedule = config.RL_ENTROPY_DECAY_SCHEDULE
        self.entropy_coef = self.entropy_coef_initial  # Start with initial value
        
        # Training state
        self.global_step = 0
        self.episode = 0
        self.best_reward = float('-inf')
        
        # Logging
        self.training_stats = defaultdict(list)
        
        # Setup checkpoint directory
        self.checkpoint_dir = os.path.join(
            config.CHECKPOINT_DIR,
            f"grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        print(f"Initialized GRPO Trainer:")
        print(f"  - Device: {self.device}")
        print(f"  - Group size: {self.group_size}")
        print(f"  - KL coefficient: {self.kl_coef}")
        print(f"  - Clip range: {self.clip_range}")
        print(f"  - Random seed: {self.seed}")
        print(f"  - Checkpoint dir: {self.checkpoint_dir}")
    
    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility across all random number generators.
        
        Args:
            seed: Random seed to use
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            # Make CUDA operations deterministic (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"Set random seed to {seed} for reproducibility")
    
    def generate_completion(self,
                            prompt: Optional[List[str]] = None,
                            max_length: int = None,
                            temperature: float = 1.0,
                            top_k: int = 0,
                            top_p: float = 1.0,
                            ) -> Tuple[List[str], torch.Tensor, torch.Tensor, Dict]:
        """
        Optimized generation with minimal CPU-GPU transfers.
        Environment stays on CPU, model on GPU.
        """
        self.model.eval()
        
        if max_length is None:
            max_length = self.config.MAX_LENGTH
        
        # Reset environment (CPU operation)
        obs = self.env.reset(prompt=prompt)
        
        tokens = obs["sequence"].copy()  # Start with BOS (or prompt)
        log_probs_list = []  # Keep on GPU
        token_ids_list = []  # Keep as Python ints for speed
        
        # Encode initial sequence once and keep on GPU
        input_ids = torch.tensor(
            vocab.encode(obs["sequence"]),
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # [1, seq_len]
        
        done = False
        step = 0
        
        # Pre-allocate mask tensor on GPU (reuse for all steps)
        vocab_size = len(vocab.GRAMMAR_TOKENS)
        mask_template = torch.full((vocab_size,), float('-inf'), device=self.device)
        
        with torch.no_grad():
            while not done and step < max_length:
                # Get valid tokens from environment (CPU operation)
                valid_token_ids = obs["valid_token_ids"]
                
                if len(valid_token_ids) == 0:
                    break
                
                # Forward pass (GPU operation)
                outputs = self.model(input_ids, use_cache=False)
                logits = outputs.logits[0, -1, :]  # [vocab_size]
                
                # Apply grammar mask (GPU operation)
                mask = mask_template.clone()
                mask[valid_token_ids] = 0
                logits = logits + mask
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = float('-inf')
                
                # Sample token (GPU operation)
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                
                # Store log probability (keep on GPU)
                log_prob = torch.log(probs[token_id] + 1e-10)
                log_probs_list.append(log_prob)
                token_ids_list.append(token_id)
                
                # Decode token for environment (single CPU operation)
                token = vocab.decode([token_id])[0]
                
                # Take step in environment (CPU operation)
                obs, reward, done, info = self.env.step(token)
                tokens.append(token)
                
                # Append new token to input_ids (GPU operation, no CPU transfer)
                input_ids = torch.cat([
                    input_ids,
                    torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                ], dim=1)
                
                step += 1
        
        # Convert to tensors (keep log_probs on GPU)
        log_probs = torch.stack(log_probs_list) if log_probs_list else torch.tensor([], device=self.device)
        token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=self.device)
        
        return tokens, log_probs, token_ids, info
    
    def generate_completion_batch(self,
                                 batch_size: int,
                                 prompt: Optional[List[str]] = None,
                                 max_length: int = None,
                                 temperature: float = 1.0,
                                 top_k: int = 0,
                                 top_p: float = 1.0,
                                 ) -> List[Tuple[List[str], torch.Tensor, torch.Tensor, Dict]]:
        """
        Optimized vectorized generation with batched forward passes.
        Each sequence has its own environment for grammar constraints.
        
        Key optimizations:
        - Vectorized masking and sampling (all sequences processed in parallel)
        - Batched CPU-GPU transfers
        - Reduced per-sequence overhead
        """
        self.model.eval()
        
        if max_length is None:
            max_length = self.config.MAX_LENGTH
        
        # Create separate environments for each batch item
        envs = [TheoryEnvironment(self.config, curriculum=self.curriculum) for _ in range(batch_size)]
        
        # Initialize batch state
        batch_tokens = []
        batch_log_probs_list = []
        batch_token_ids_list = []
        batch_done = []
        batch_info = []
        batch_input_ids = []
        
        # Reset all environments
        for env in envs:
            obs = env.reset(prompt=prompt)
            batch_tokens.append(obs["sequence"].copy())
            batch_log_probs_list.append([])
            batch_token_ids_list.append([])
            batch_done.append(False)
            batch_info.append({})
            
            # Encode initial sequence
            input_ids = torch.tensor(
                vocab.encode(obs["sequence"]),
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)
            batch_input_ids.append(input_ids)
        
        # Pre-allocate mask template on GPU
        vocab_size = len(vocab.GRAMMAR_TOKENS)
        mask_template = torch.full((vocab_size,), float('-inf'), device=self.device)
        
        step = 0
        with torch.no_grad():
            while not all(batch_done) and step < max_length:
                # Get active (not done) indices
                active_indices = [i for i, done in enumerate(batch_done) if not done]
                if not active_indices:
                    break
                
                num_active = len(active_indices)
                
                # Pad sequences to the same length for batching
                max_seq_len = max(batch_input_ids[i].size(1) for i in active_indices)
                padded_batch = []
                attention_mask = []
                
                for i in active_indices:
                    input_ids = batch_input_ids[i]
                    seq_len = input_ids.size(1)
                    pad_len = max_seq_len - seq_len
                    
                    if pad_len > 0:
                        # Pad on the left (causal attention)
                        padded = F.pad(input_ids, (pad_len, 0), value=self.config.PAD_TOKEN_ID)
                        mask = torch.cat([
                            torch.zeros(1, pad_len, device=self.device),
                            torch.ones(1, seq_len, device=self.device)
                        ], dim=1)
                    else:
                        padded = input_ids
                        mask = torch.ones(1, seq_len, device=self.device)
                    
                    padded_batch.append(padded)
                    attention_mask.append(mask)
                
                # Stack into batch [num_active, seq_len]
                batched_input_ids = torch.cat(padded_batch, dim=0)
                batched_attention_mask = torch.cat(attention_mask, dim=0)
                
                # Single batched forward pass (GPU operation)
                outputs = self.model(batched_input_ids, attention_mask=batched_attention_mask, use_cache=False)
                logits = outputs.logits[:, -1, :]  # [num_active, vocab_size]
                
                # ==== VECTORIZED OPERATIONS START ====
                
                # Get all valid token masks for all sequences at once
                batch_masks = torch.full((num_active, vocab_size), float('-inf'), device=self.device)
                sequences_to_skip = []
                
                for batch_idx, i in enumerate(active_indices):
                    obs = envs[i]._get_observation()
                    valid_token_ids = obs["valid_token_ids"]
                    
                    if len(valid_token_ids) == 0:
                        batch_done[i] = True
                        sequences_to_skip.append(batch_idx)
                    else:
                        batch_masks[batch_idx, valid_token_ids] = 0
                
                # Remove sequences that should be skipped
                if sequences_to_skip:
                    keep_mask = torch.ones(num_active, dtype=torch.bool, device=self.device)
                    keep_mask[sequences_to_skip] = False
                    logits = logits[keep_mask]
                    batch_masks = batch_masks[keep_mask]
                    active_indices = [idx for i, idx in enumerate(active_indices) if i not in sequences_to_skip]
                    num_active = len(active_indices)
                
                if num_active == 0:
                    break
                
                # Apply grammar masks to entire batch
                logits = logits + batch_masks
                
                # Apply temperature to entire batch
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering to entire batch
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    min_values = values[:, -1:].expand_as(logits)
                    logits = torch.where(logits < min_values, 
                                        torch.full_like(logits, float('-inf')), 
                                        logits)
                
                # Apply top-p filtering to entire batch
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # Scatter sorted tensors back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample tokens for entire batch at once
                probs = F.softmax(logits, dim=-1)
                sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [num_active]
                
                # Compute log probabilities for sampled tokens
                log_probs = torch.log(probs[torch.arange(num_active, device=self.device), sampled_token_ids] + 1e-10)
                
                # Transfer token IDs to CPU once for all sequences
                token_ids_cpu = sampled_token_ids.cpu().tolist()
                
                # ==== VECTORIZED OPERATIONS END ====
                
                # Decode tokens and update environments (still sequential, but minimized)
                tokens_decoded = vocab.decode(token_ids_cpu)
                
                for batch_idx, i in enumerate(active_indices):
                    token_id = token_ids_cpu[batch_idx]
                    token = tokens_decoded[batch_idx]
                    
                    # Store log probability and token ID
                    batch_log_probs_list[i].append(log_probs[batch_idx])
                    batch_token_ids_list[i].append(token_id)
                    
                    # Take step in environment (CPU operation)
                    obs, reward, done, info = envs[i].step(token)
                    batch_tokens[i].append(token)
                    batch_done[i] = done
                    batch_info[i] = info
                    
                    # Append new token to input_ids
                    batch_input_ids[i] = torch.cat([
                        batch_input_ids[i],
                        torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                    ], dim=1)
                
                step += 1
        
        # Convert to the same format as generate_completion, including model_dict
        results = []
        for i in range(batch_size):
            log_probs = torch.stack(batch_log_probs_list[i]) if batch_log_probs_list[i] else torch.tensor([], device=self.device)
            token_ids = torch.tensor(batch_token_ids_list[i], dtype=torch.long, device=self.device)
            # Get model dict from environment
            model_dict = envs[i]._get_full_model()
            results.append((batch_tokens[i], log_probs, token_ids, batch_info[i], model_dict))
        
        return results
    
    def generate_completion_batch_profiled(self,
                                 batch_size: int,
                                 prompt: Optional[List[str]] = None,
                                 max_length: int = None,
                                 temperature: float = 1.0,
                                 top_k: int = 0,
                                 top_p: float = 1.0,
                                 ) -> List[Tuple[List[str], torch.Tensor, torch.Tensor, Dict]]:
        """
        Profiled version of generate_completion_batch to identify bottlenecks.
        """
        timing_stats = {}
        
        self.model.eval()
        
        if max_length is None:
            max_length = self.config.MAX_LENGTH
        
        with timer("env_initialization", timing_stats):
            # Create separate environments for each batch item
            envs = [TheoryEnvironment(self.config, curriculum=self.curriculum) for _ in range(batch_size)]
            
            # Initialize batch state
            batch_tokens = []
            batch_log_probs_list = []
            batch_token_ids_list = []
            batch_done = []
            batch_info = []
            batch_input_ids = []
            
            # Reset all environments
            for env in envs:
                obs = env.reset(prompt=prompt)
                batch_tokens.append(obs["sequence"].copy())
                batch_log_probs_list.append([])
                batch_token_ids_list.append([])
                batch_done.append(False)
                batch_info.append({})
                
                # Encode initial sequence
                input_ids = torch.tensor(
                    vocab.encode(obs["sequence"]),
                    dtype=torch.long,
                    device=self.device
                ).unsqueeze(0)
                batch_input_ids.append(input_ids)
        
        # Pre-allocate mask template on GPU
        vocab_size = len(vocab.GRAMMAR_TOKENS)
        mask_template = torch.full((vocab_size,), float('-inf'), device=self.device)
        
        step = 0
        with torch.no_grad():
            while not all(batch_done) and step < max_length:
                with timer("step_total", timing_stats):
                    # Get active (not done) indices
                    active_indices = [i for i, done in enumerate(batch_done) if not done]
                    if not active_indices:
                        break
                    
                    num_active = len(active_indices)
                    
                    # Pad sequences to the same length for batching
                    with timer("padding", timing_stats):
                        max_seq_len = max(batch_input_ids[i].size(1) for i in active_indices)
                        padded_batch = []
                        attention_mask = []
                        
                        for i in active_indices:
                            input_ids = batch_input_ids[i]
                            seq_len = input_ids.size(1)
                            pad_len = max_seq_len - seq_len
                            
                            if pad_len > 0:
                                padded = F.pad(input_ids, (pad_len, 0), value=self.config.PAD_TOKEN_ID)
                                mask = torch.cat([
                                    torch.zeros(1, pad_len, device=self.device),
                                    torch.ones(1, seq_len, device=self.device)
                                ], dim=1)
                            else:
                                padded = input_ids
                                mask = torch.ones(1, seq_len, device=self.device)
                            
                            padded_batch.append(padded)
                            attention_mask.append(mask)
                        
                        batched_input_ids = torch.cat(padded_batch, dim=0)
                        batched_attention_mask = torch.cat(attention_mask, dim=0)
                    
                    # Single batched forward pass (GPU operation)
                    with timer("forward_pass", timing_stats):
                        outputs = self.model(batched_input_ids, attention_mask=batched_attention_mask, use_cache=False)
                        logits = outputs.logits[:, -1, :]  # [num_active, vocab_size]
                    
                    with timer("vectorized_operations", timing_stats):
                        # Get all valid token masks for all sequences at once
                        batch_masks = torch.full((num_active, vocab_size), float('-inf'), device=self.device)
                        sequences_to_skip = []
                        
                        with timer("env_observation", timing_stats):
                            for batch_idx, i in enumerate(active_indices):
                                obs = envs[i]._get_observation()
                                valid_token_ids = obs["valid_token_ids"]
                                
                                if len(valid_token_ids) == 0:
                                    batch_done[i] = True
                                    sequences_to_skip.append(batch_idx)
                                else:
                                    batch_masks[batch_idx, valid_token_ids] = 0
                        
                        # Remove sequences that should be skipped
                        if sequences_to_skip:
                            keep_mask = torch.ones(num_active, dtype=torch.bool, device=self.device)
                            keep_mask[sequences_to_skip] = False
                            logits = logits[keep_mask]
                            batch_masks = batch_masks[keep_mask]
                            active_indices = [idx for i, idx in enumerate(active_indices) if i not in sequences_to_skip]
                            num_active = len(active_indices)
                        
                        if num_active == 0:
                            break
                        
                        with timer("masking_sampling", timing_stats):
                            # Apply grammar masks to entire batch
                            logits = logits + batch_masks
                            
                            # Apply temperature to entire batch
                            if temperature != 1.0:
                                logits = logits / temperature
                            
                            # Apply top-k filtering to entire batch
                            if top_k > 0:
                                values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                                min_values = values[:, -1:].expand_as(logits)
                                logits = torch.where(logits < min_values, 
                                                    torch.full_like(logits, float('-inf')), 
                                                    logits)
                            
                            # Apply top-p filtering to entire batch
                            if top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                                sorted_indices_to_remove[:, 0] = 0
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                logits = logits.masked_fill(indices_to_remove, float('-inf'))
                            
                            # Sample tokens for entire batch at once
                            probs = F.softmax(logits, dim=-1)
                            sampled_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
                            
                            # Compute log probabilities for sampled tokens
                            log_probs = torch.log(probs[torch.arange(num_active, device=self.device), sampled_token_ids] + 1e-10)
                        
                        with timer("cpu_gpu_sync", timing_stats):
                            # Transfer token IDs to CPU once for all sequences
                            token_ids_cpu = sampled_token_ids.cpu().tolist()
                            # Decode tokens
                            tokens_decoded = vocab.decode(token_ids_cpu)
                    
                    with timer("env_step", timing_stats):
                        # Update environments
                        for batch_idx, i in enumerate(active_indices):
                            token_id = token_ids_cpu[batch_idx]
                            token = tokens_decoded[batch_idx]
                            
                            # Store log probability and token ID
                            batch_log_probs_list[i].append(log_probs[batch_idx])
                            batch_token_ids_list[i].append(token_id)
                            
                            # Take step in environment (CPU operation)
                            obs, reward, done, info = envs[i].step(token)
                            batch_tokens[i].append(token)
                            batch_done[i] = done
                            batch_info[i] = info
                    
                    with timer("tensor_concat", timing_stats):
                        # Append new tokens to input_ids
                        for batch_idx, i in enumerate(active_indices):
                            token_id = token_ids_cpu[batch_idx]
                            batch_input_ids[i] = torch.cat([
                                batch_input_ids[i],
                                torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                            ], dim=1)
                    
                    step += 1
        
        with timer("finalization", timing_stats):
            # Convert to the same format as generate_completion
            results = []
            for i in range(batch_size):
                log_probs = torch.stack(batch_log_probs_list[i]) if batch_log_probs_list[i] else torch.tensor([], device=self.device)
                token_ids = torch.tensor(batch_token_ids_list[i], dtype=torch.long, device=self.device)
                model_dict = envs[i]._get_full_model()
                results.append((batch_tokens[i], log_probs, token_ids, batch_info[i], model_dict))
        
        # Print timing statistics
        print("\n" + "="*80)
        print("PROFILING RESULTS (per-step averages)")
        print("="*80)
        
        for key in sorted(timing_stats.keys()):
            times = timing_stats[key]
            avg_time = np.mean(times) * 1000  # Convert to ms
            total_time = np.sum(times) * 1000
            percentage = (np.sum(times) / np.sum([np.sum(v) for v in timing_stats.values()])) * 100
            print(f"{key:30s}: {avg_time:8.2f}ms avg, {total_time:10.2f}ms total ({percentage:5.1f}%)")
        
        total_time = np.sum([np.sum(v) for v in timing_stats.values()])
        print(f"\n{'Total time':30s}: {total_time*1000:10.2f}ms")
        print(f"{'Steps':30s}: {step}")
        print(f"{'Time per step':30s}: {(total_time/step)*1000 if step > 0 else 0:8.2f}ms")
        print("="*80 + "\n")
        
        return results
    
    def check_gpu_utilization(self):
        """Print GPU utilization stats"""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return
        
        print("\n" + "="*80)
        print("GPU UTILIZATION")
        print("="*80)
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"  Free: {(props.total_memory - torch.cuda.memory_reserved(i)) / 1e9:.2f} GB")
            print(f"  Utilization: {(torch.cuda.memory_allocated(i) / props.total_memory * 100):.1f}%")
        
        print("="*80 + "\n")
    
    def profile_generation(self, num_batches=5, batch_size=None):
        """Profile the generation process to identify bottlenecks"""
        if batch_size is None:
            batch_size = self.config.GRPO_GENERATION_BATCH_SIZE
        
        print(f"\nProfiling generation with batch_size={batch_size}")
        print(f"Running {num_batches} batches...\n")
        
        # Check GPU utilization before
        self.check_gpu_utilization()
        
        # Warmup
        print("Warmup run...")
        _ = self.generate_completion_batch(batch_size=batch_size)
        
        # Profile
        for i in range(num_batches):
            print(f"\n{'='*80}")
            print(f"Batch {i+1}/{num_batches}")
            print(f"{'='*80}")
            _ = self.generate_completion_batch_profiled(batch_size=batch_size)
        
        # Check GPU utilization after
        self.check_gpu_utilization()
        
        print("\n" + "="*80)
        print("PROFILING COMPLETE")
        print("="*80)
        print("\nInterpretation Guide:")
        print("  - forward_pass >60%: GPU bottleneck → Enable KV cache, Flash Attention")
        print("  - env_step >40%: CPU bottleneck → Optimize environment logic")
        print("  - vectorized_operations >20%: Still some overhead → Further optimization needed")
        print("  - cpu_gpu_sync >10%: Transfer overhead → Already optimized")
        print("="*80 + "\n")
    
    def update_entropy_coef(self, current_episode: int, total_episodes: int):
        if total_episodes <= 1:
            return
        
        progress = current_episode / total_episodes
        
        if self.entropy_decay_schedule == "linear":
            self.entropy_coef = self.entropy_coef_initial + progress * (
                self.entropy_coef_final - self.entropy_coef_initial
            )
        elif self.entropy_decay_schedule == "exponential":
            decay_rate = np.log(self.entropy_coef_final / self.entropy_coef_initial)
            self.entropy_coef = self.entropy_coef_initial * np.exp(decay_rate * progress)
        
        elif self.entropy_decay_schedule == "cosine":
            self.entropy_coef = self.entropy_coef_final + 0.5 * (
                self.entropy_coef_initial - self.entropy_coef_final
            ) * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown decay schedule: {self.entropy_decay_schedule}")
    
    def generate_group(self,
                       prompt: Optional[List[str]] = None,
                       num_samples: int = None,
                       temperature: float = 1.0,
                       batch_size: int = None
                       ) -> List[Dict[str, Any]]:

        if num_samples is None:
            num_samples = self.group_size
        
        if batch_size is None:
            batch_size = min(self.config.GRPO_GENERATION_BATCH_SIZE, num_samples)
        
        completions = []
        
        # Process in batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # Generate batch
            batch_results = self.generate_completion_batch(
                batch_size=current_batch_size,
                prompt=prompt,
                temperature=temperature
            )
            
            # Compute rewards for each completion in batch
            for tokens, log_probs, token_ids, info, model_dict in batch_results:
                # Compute reward
                reward_dict = all_rewards(model_dict, device=self.env_device)
                reward = reward_dict["total_reward"]
                
                completions.append({
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "log_probs": log_probs,
                    "reward": reward,
                    "info": info,
                    "model_dict": model_dict,
                })
        
        return completions
    
    def compute_advantages(self,
                           completions: List[Dict[str, Any]],
                           method: str = "group_relative"
                           ) -> torch.Tensor:

        rewards = torch.tensor([c["reward"] for c in completions], dtype=torch.float32)
        
        if method == "group_relative":
            # GRPO: advantages relative to group mean
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            advantages = (rewards - mean_reward) / std_reward
        
        elif method == "normalized":
            # Standardized rewards (z-score normalization)
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            advantages = (rewards - mean_reward) / std_reward
        
        elif method == "raw":
            # Raw rewards as advantages
            advantages = rewards
        
        else:
            raise ValueError(f"Unknown advantage method: {method}")
        
        return advantages.to(self.device)
    
    def compute_policy_loss(self,
                            completions: List[Dict[str, Any]],
                            advantages: torch.Tensor,
                            ) -> Tuple[torch.Tensor, Dict[str, float]]:

        self.model.train()
        
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_entropy = 0.0
        num_tokens = 0
        
        for completion, advantage in zip(completions, advantages):
            if len(completion["token_ids"]) == 0:
                continue
            
            # Prepare full sequence (including BOS and generated tokens)
            full_tokens = completion["tokens"]
            input_ids = torch.tensor(
                vocab.encode(full_tokens[:-1]),  # All but last token
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
            
            target_ids = completion["token_ids"].to(self.device)  # Generated tokens only
            old_log_probs = completion["log_probs"].to(self.device)
            
            # Forward pass with current policy
            outputs = self.model(input_ids, use_cache=False)
            logits = outputs.logits[0, -(len(target_ids)):, :]  # [num_gen_tokens, vocab_size]
            
            # Compute log probabilities for generated tokens
            log_probs = F.log_softmax(logits, dim=-1)
            new_log_probs = log_probs[range(len(target_ids)), target_ids]
            
            # Compute policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO-style clipped objective
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            policy_loss_unclipped = -advantage * ratio
            policy_loss_clipped = -advantage * clipped_ratio
            policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
            
            # KL divergence with reference policy (true KL divergence over full distribution)
            # KL(π_new || π_ref) = Σ π_new(a) * log(π_new(a) / π_ref(a))
            # This is guaranteed to be non-negative
            with torch.no_grad():
                ref_outputs = self.reference_model(input_ids, use_cache=False)
                ref_logits = ref_outputs.logits[0, -(len(target_ids)):, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
            # Compute probabilities for current policy
            probs = F.softmax(logits, dim=-1)
            
            # True KL divergence: sum over all actions, then average over tokens
            kl_div = (probs * (log_probs - ref_log_probs)).sum(dim=-1).mean()
            
            # Entropy bonus (encourage exploration) - reuse probs from above
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            total_policy_loss += policy_loss
            total_kl_loss += kl_div
            total_entropy += entropy
            num_tokens += len(target_ids)
        
        # Average over completions
        num_completions = len([c for c in completions if len(c["token_ids"]) > 0])
        if num_completions == 0:
            return torch.tensor(0.0, device=self.device), {}
        
        avg_policy_loss = total_policy_loss / num_completions
        avg_kl_loss = total_kl_loss / num_completions
        avg_entropy = total_entropy / num_completions
        
        # Total loss
        total_loss = (
            avg_policy_loss +
            self.kl_coef * avg_kl_loss -
            self.entropy_coef * avg_entropy
        )
        
        metrics = {"policy_loss": avg_policy_loss.item(),
                   "kl_loss": avg_kl_loss.item(),
                   "entropy": avg_entropy.item(),
                   "total_loss": total_loss.item(),
                   "num_tokens": num_tokens
                   }
        
        return total_loss, metrics
    
    def train_step(self,
                   prompt: Optional[List[str]] = None,
                   temperature: float = 1.0,
                   ) -> Dict[str, float]:

        # Generate group of completions
        completions = self.generate_group(prompt=prompt, temperature=temperature)
        
        # Compute advantages
        advantages = self.compute_advantages(completions, method="group_relative")
        
        # Compute loss
        loss, loss_metrics = self.compute_policy_loss(completions, advantages)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute reward statistics
        rewards = [c["reward"] for c in completions]
        success_count = sum(1 for c in completions if c["reward"] >= 0)
        
        metrics = {**loss_metrics,
                   "mean_reward": np.mean(rewards),
                   "std_reward": np.std(rewards),
                   "min_reward": np.min(rewards),
                   "max_reward": np.max(rewards),
                   "success_rate": success_count / len(completions),
                   "num_completions": len(completions)
                   }
        
        return metrics
    
    def train(self,
              num_episodes: int = 1000,
              log_interval: int = 10,
              save_interval: int = 100,
              temperature: float = 1.0,
              prompts: Optional[List[List[str]]] = None
              ):
              
        print(f"\nStarting GRPO Training for {num_episodes} episodes...")
        print("=" * 80)
        print(f"Entropy decay: {self.entropy_coef_initial:.4f} → {self.entropy_coef_final:.4f} ({self.entropy_decay_schedule})")
        print("=" * 80)
        
        pbar = tqdm(range(num_episodes), desc="Training")
        
        for episode in pbar:
            self.episode = episode
            
            # Update entropy coefficient with decay
            self.update_entropy_coef(episode, num_episodes)
            
            # Select prompt (cycle through if provided)
            prompt = None
            if prompts is not None and len(prompts) > 0:
                prompt = prompts[episode % len(prompts)]
            
            # Training step
            metrics = self.train_step(prompt=prompt, temperature=temperature)
            
            # Add current entropy coefficient to metrics
            metrics['entropy_coef'] = self.entropy_coef
            
            # Log metrics
            for key, value in metrics.items():
                self.training_stats[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({"reward": f"{metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f}",
                               "success": f"{metrics['success_rate']:.2%}",
                               "loss": f"{metrics['total_loss']:.4f}",
                               "entropy": f"{metrics['entropy']:.4f}",
                               "KL_loss": f"{metrics['kl_loss']:.4f}"
                               })
            
            # Periodic logging
            if (episode + 1) % log_interval == 0:
                self._log_stats(episode + 1)
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self.save_checkpoint(episode + 1)
            
            # Save best model
            if metrics['mean_reward'] > self.best_reward:
                self.best_reward = metrics['mean_reward']
                self.save_checkpoint(episode + 1, is_best=True)
            
            self.global_step += 1
        
        print("\nTraining complete!")
        print("=" * 80)
        
        self.save_checkpoint(num_episodes, is_best=False)
        self.save_training_stats()
    
    def _log_stats(self, episode: int):
        recent_window = 10
        
        print(f"\n{'='*80}")
        print(f"Episode {episode} Statistics:")
        print(f"{'='*80}")
        
        # Compute recent averages
        for key in ["mean_reward", "success_rate", "total_loss", "policy_loss", "kl_loss", "entropy", "entropy_coef"]:
            if key in self.training_stats and len(self.training_stats[key]) > 0:
                recent_values = self.training_stats[key][-recent_window:]
                avg_value = np.mean(recent_values)
                print(f"  {key:20s}: {avg_value:.4f}")
        
        # Curriculum phase
        if self.curriculum:
            print(f"\nCurriculum Phase: {self.curriculum.phase}")
        
        print(f"{'='*80}\n")
    
    def save_checkpoint(self, episode: int, is_best: bool = False):
        checkpoint = {
            "episode": episode,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_reward": self.best_reward,
            "training_stats": dict(self.training_stats),
            "config": self.config,
            "seed": self.seed,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }
        
        # Save regular checkpoint
        if not is_best:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_episode_{episode}.pt")
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")
        
        # Save best model
        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, path)
            print(f"Saved best model: {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode = checkpoint["episode"]
        self.global_step = checkpoint["global_step"]
        self.best_reward = checkpoint["best_reward"]
        self.training_stats = defaultdict(list, checkpoint["training_stats"])
        
        # Restore random seed and RNG states for reproducibility
        if "seed" in checkpoint:
            self.seed = checkpoint["seed"]
        
        if "rng_state" in checkpoint:
            rng_state = checkpoint["rng_state"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            print(f"  Restored RNG states for reproducibility")
        
        print(f"Loaded checkpoint from: {path}")
        print(f"  Episode: {self.episode}")
        print(f"  Best reward: {self.best_reward:.4f}")
        print(f"  Random seed: {self.seed}")
    
    def save_training_stats(self):
        stats_path = os.path.join(self.checkpoint_dir, "training_stats.json")
        
        # Convert to serializable format
        stats_dict = {key: [float(v) for v in values] for key, values in self.training_stats.items()}
        
        with open(stats_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Saved training stats: {stats_path}")
    
    def evaluate(
        self,
        num_episodes: int = 100,
        temperature: float = 1.0,
        prompts: Optional[List[List[str]]] = None,
    ) -> Dict[str, float]:
        print(f"\nEvaluating model for {num_episodes} episodes...")
        
        self.model.eval()
        
        collected_rewards = []
        all_success = []
        all_lengths = []
        
        with torch.no_grad():
            for episode in tqdm(range(num_episodes), desc="Evaluating"):
                prompt = None
                if prompts is not None and len(prompts) > 0:
                    prompt = prompts[episode % len(prompts)]
                
                tokens, log_probs, token_ids, info = self.generate_completion(
                    prompt=prompt,
                    temperature=temperature
                )
                
                # Compute reward
                model_dict = self.env._get_full_model()
                reward_dict = all_rewards(model_dict, device=self.env_device)
                reward = reward_dict["total_reward"]
                
                collected_rewards.append(reward)
                all_success.append(info.get("success", False))
                all_lengths.append(len(tokens))
        
        metrics = {
            "mean_reward": np.mean(collected_rewards),
            "std_reward": np.std(collected_rewards),
            "min_reward": np.min(collected_rewards),
            "max_reward": np.max(collected_rewards),
            "success_rate": np.mean(all_success),
            "mean_length": np.mean(all_lengths),
        }
        
        print("\nEvaluation Results:")
        print("=" * 80)
        for key, value in metrics.items():
            print(f"  {key:20s}: {value:.4f}")
        print("=" * 80)
        
        return metrics


def main():
    # Initialize model
    model = TheoryGPT(config)
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        config=config,
        use_curriculum=True
        )
    
    # Train
    trainer.train(
        num_episodes=1000,
        log_interval=10,
        save_interval=100,
        temperature=1.0,
    )
    
    # Evaluate
    eval_metrics = trainer.evaluate(num_episodes=100)
    
    print("\nFinal Evaluation:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()

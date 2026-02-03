import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import os
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np

from architecture.TheoryGPT import TheoryGPT
from config import config
import grammar.vocab as vocab


class KLSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with valid tokens"""
    def __init__(self, sequences, valid_tokens_list, max_length=None):
        """
        Args:
            sequences: List of tokenized sequences (list of integers)
            valid_tokens_list: List of valid token lists for each position in each sequence
            max_length: Maximum sequence length (pads or truncates)
        """
        self.sequences = sequences
        self.valid_tokens_list = valid_tokens_list
        self.max_length = max_length or config.MAX_LENGTH
        self.pad_token_id = config.PAD_TOKEN_ID
        self.vocab_size = len(vocab.GRAMMAR_TOKENS)
        
        assert len(sequences) == len(valid_tokens_list), \
            "Number of sequences and valid_tokens must match"
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        valid_tokens = self.valid_tokens_list[idx]
        
        # Truncate if too long
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
            valid_tokens = valid_tokens[:self.max_length]
        
        # Store original length before padding
        orig_len = len(seq)
        
        # Pad sequence if too short
        if len(seq) < self.max_length:
            seq = seq + [self.pad_token_id] * (self.max_length - len(seq))
            # Pad valid_tokens with empty lists
            valid_tokens = valid_tokens + [[]] * (self.max_length - len(valid_tokens))
        
        # Convert to tensor
        input_ids = torch.tensor(seq, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # Create target distribution for KL divergence
        # Shape: (seq_len, vocab_size)
        target_distribution = torch.zeros(self.max_length, self.vocab_size, dtype=torch.float32)
        
        for pos, valid_token_ids in enumerate(valid_tokens):
            if len(valid_token_ids) > 0:
                # Create uniform distribution over valid tokens
                prob = 1.0 / len(valid_token_ids)
                for token_id in valid_token_ids:
                    if 0 <= token_id < self.vocab_size:
                        target_distribution[pos, token_id] = prob
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_distribution': target_distribution,
            'orig_len': orig_len
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    target_distribution = torch.stack([item['target_distribution'] for item in batch])
    orig_lens = torch.tensor([item['orig_len'] for item in batch], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'target_distribution': target_distribution,
        'orig_lens': orig_lens
    }


def compute_kl_loss(model, batch, device, temperature=1.0):
    """Compute KL divergence loss"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    target_distribution = batch['target_distribution'].to(device)
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
    shift_target = target_distribution[:, 1:, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
    
    # Apply temperature scaling FIRST (before masking to avoid -inf/temperature issues)
    shift_logits = shift_logits / temperature
    
    # Identify positions with valid tokens BEFORE any masking
    has_valid_tokens = (shift_target.sum(dim=-1) > 0)  # (batch_size, seq_len-1)
    
    # Mask invalid tokens with large negative value (not -inf to avoid NaN)
    # Only mask at positions that have at least one valid token
    invalid_token_mask = (shift_target == 0)  # True for invalid tokens
    # Create safe mask: only apply where position has valid tokens
    safe_invalid_mask = invalid_token_mask & has_valid_tokens.unsqueeze(-1)
    # Use -1e9 instead of -inf to avoid NaN in softmax
    shift_logits = shift_logits.masked_fill(safe_invalid_mask, -1e9)
    
    # Compute log probabilities from logits
    log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch_size, seq_len-1, vocab_size)
    
    # Cross-entropy between target distribution and predictions
    ce_loss = -(shift_target * log_probs).sum(dim=-1)  # (batch_size, seq_len-1)
    
    # Create mask to ignore padding tokens (shift by 1)
    shift_attention_mask = attention_mask[:, 1:].contiguous()  # (batch_size, seq_len-1)
    
    # Combine masks
    combined_mask = shift_attention_mask.float() * has_valid_tokens.float()
    
    # Mask out invalid positions
    masked_loss = ce_loss * combined_mask
    
    # Compute average loss (add check for zero tokens)
    total_tokens = combined_mask.sum()
    if total_tokens == 0:
        # Handle edge case of no valid tokens in batch
        loss = torch.tensor(0.0, device=device)
    else:
        total_loss = masked_loss.sum()
        loss = total_loss / total_tokens
    
    # Compute additional metrics
    with torch.no_grad():
        # For argmax, use original (unmasked) logits to avoid -inf issues
        predicted_tokens = shift_logits.argmax(dim=-1)  # (batch_size, seq_len-1)
        
        # Check if predicted token is in valid set
        pred_is_valid = torch.gather(shift_target, 2, predicted_tokens.unsqueeze(-1)).squeeze(-1)
        pred_is_valid = (pred_is_valid > 0).float()
        
        # Accuracy: percentage of predictions that are valid
        if total_tokens > 0:
            valid_accuracy = (pred_is_valid * combined_mask).sum() / total_tokens
        else:
            valid_accuracy = torch.tensor(0.0, device=device)
        
        # Perplexity (add safety for exp overflow)
        perplexity = torch.exp(torch.clamp(loss, max=20))  # Clamp to avoid overflow
    
    metrics = {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'valid_accuracy': valid_accuracy.item(),
        'total_tokens': total_tokens.item()
    }
    
    return loss, metrics

def evaluate(model, val_loader, device, temperature=1.0):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_accuracy = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            loss, metrics = compute_kl_loss(model, batch, device, temperature)
            
            num_tokens = metrics['total_tokens']
            total_loss += metrics['loss'] * num_tokens
            total_tokens += num_tokens
            total_accuracy += metrics['valid_accuracy'] * num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_accuracy = total_accuracy / total_tokens if total_tokens > 0 else 0.0
    perplexity = np.exp(avg_loss)
    
    model.train()
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'valid_accuracy': avg_accuracy
    }


def train(model, train_loader, val_loader, num_epochs, learning_rate, device,
          grad_accumulation_steps=1, checkpoint_dir="checkpoints", log_interval=10,
          temperature=1.0, warmup_ratio=0.1):
    """Main training loop for supervised fine-tuning with KL divergence"""
    model.train()
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                           betas=(0.9, 0.95), weight_decay=0.1)
    
    # Learning rate schedule with warmup
    total_steps = len(train_loader) * num_epochs // grad_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item()))
        
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training statistics
    best_val_loss = float('inf')
    training_stats = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'val_perplexities': [],
        'val_accuracies': [],
        'learning_rates': []
    }
    
    print("\n" + "="*80)
    print(f"Starting TheoryGPT Supervised Fine-Tuning with KL Divergence")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"Effective batch size: {train_loader.batch_size * grad_accumulation_steps}")
    print(f"Total optimization steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Temperature: {temperature}")
    print("="*80 + "\n")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        # Progress bar for training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in pbar:
            # Forward and backward pass
            loss, metrics = compute_kl_loss(model, batch, device, temperature)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accumulation_steps
            loss.backward()
            
            epoch_loss += metrics['loss']
            epoch_accuracy += metrics['valid_accuracy']
            num_batches += 1
            
            # Update weights after accumulating gradients
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['valid_accuracy']:.3f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Periodic logging
            if (batch_idx + 1) % (log_interval * grad_accumulation_steps) == 0:
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_accuracy / num_batches
                perplexity = np.exp(avg_loss)
                print(f"\n[Epoch {epoch+1}/{num_epochs}] [Batch {batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                      f"Acc: {avg_acc:.3f} | LR: {current_lr:.2e}")
        
        # Compute epoch statistics
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_train_accuracy = epoch_accuracy / num_batches if num_batches > 0 else 0.0
        train_perplexity = np.exp(avg_train_loss)
        
        # Validation
        print(f"\nRunning validation...")
        val_metrics = evaluate(model, val_loader, device, temperature)
        
        # Save statistics
        training_stats['epochs'].append(epoch + 1)
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(val_metrics['loss'])
        training_stats['val_perplexities'].append(val_metrics['perplexity'])
        training_stats['val_accuracies'].append(val_metrics['valid_accuracy'])
        training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print("\n" + "="*80)
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train PPL: {train_perplexity:.2f} | Train Acc: {avg_train_accuracy:.3f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val PPL:   {val_metrics['perplexity']:.2f} | Val Acc:   {val_metrics['valid_accuracy']:.3f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("="*80 + "\n")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_metrics['loss'],
            'val_perplexity': val_metrics['perplexity'],
            'val_accuracy': val_metrics['valid_accuracy'],
            'config': config.__dict__
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_perplexity': val_metrics['perplexity'],
                'val_accuracy': val_metrics['valid_accuracy'],
                'config': config.__dict__
            }, best_model_path)
            print(f"Best model saved to {best_model_path} (Val Loss: {val_metrics['loss']:.4f})\n")
    
    # Save training statistics
    stats_path = os.path.join(checkpoint_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"Training statistics saved to {stats_path}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80 + "\n")
    
    return training_stats


def load_dataset_from_disk(dataset_path):
    """Load dataset from disk and extract sequences and valid tokens"""
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Extract sequences and valid tokens
    sequences = []
    valid_tokens_list = []
    
    for example in dataset:
        # Extract inputs
        inputs = example['inputs']
        if isinstance(inputs, (list, tuple)):
            seq = list(inputs)
        else:
            seq = inputs.tolist() if hasattr(inputs, 'tolist') else list(inputs)
        sequences.append(seq)
        
        # Extract valid tokens
        valid_tokens = example['valid_tokens']
        if isinstance(valid_tokens, (list, tuple)):
            vt = [list(vt_pos) if isinstance(vt_pos, (list, tuple)) else 
                  (vt_pos.tolist() if hasattr(vt_pos, 'tolist') else list(vt_pos))
                  for vt_pos in valid_tokens]
        else:
            vt = [[]]  # Empty if not available
        valid_tokens_list.append(vt)
    
    print(f"Loaded {len(sequences)} sequences from dataset")
    
    # Print some statistics
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Sequence length stats:")
    print(f"  Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, "
          f"Mean: {sum(seq_lengths)/len(seq_lengths):.1f}")
    
    # Print valid tokens statistics
    avg_valid_per_pos = []
    for vt_seq in valid_tokens_list:
        for vt_pos in vt_seq:
            if len(vt_pos) > 0:
                avg_valid_per_pos.append(len(vt_pos))
    
    if len(avg_valid_per_pos) > 0:
        print(f"Valid tokens per position stats:")
        print(f"  Min: {min(avg_valid_per_pos)}, Max: {max(avg_valid_per_pos)}, "
              f"Mean: {sum(avg_valid_per_pos)/len(avg_valid_per_pos):.1f}")
    
    return sequences, valid_tokens_list


def split_dataset(sequences, valid_tokens_list, val_split=0.1, seed=None):
    """Split dataset into train and validation sets"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Shuffle sequences
    indices = torch.randperm(len(sequences)).tolist()
    split_idx = int(len(sequences) * (1 - val_split))
    
    train_sequences = [sequences[i] for i in indices[:split_idx]]
    val_sequences = [sequences[i] for i in indices[split_idx:]]
    
    train_valid_tokens = [valid_tokens_list[i] for i in indices[:split_idx]]
    val_valid_tokens = [valid_tokens_list[i] for i in indices[split_idx:]]
    
    print(f"Split dataset: {len(train_sequences)} train, {len(val_sequences)} val")
    return train_sequences, val_sequences, train_valid_tokens, val_valid_tokens


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Load pretrained model (optional - can also start from scratch)
    print("Initializing TheoryGPT model...")
    model = TheoryGPT(config)
    
    # Optionally load pretrained weights
    pretrained_path = "checkpoints/pretrain_best/best_model.pt"  # Modify as needed
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained weights loaded successfully!")
    else:
        print("No pretrained weights found. Training from scratch.")
    
    model = model.to(device)
    
    # Load dataset from disk
    dataset_path = "dataset/pretrain_data_100k"  # Modify as needed
    all_sequences, all_valid_tokens = load_dataset_from_disk(dataset_path)
    
    # Split into train and validation
    train_sequences, val_sequences, train_valid_tokens, val_valid_tokens = split_dataset(
        all_sequences,
        all_valid_tokens,
        val_split=config.VAL_SPLIT_SIZE,
        seed=config.VAL_SPLIT_SEED
    )
    
    # Create datasets
    train_dataset = KLSupervisedDataset(
        train_sequences,
        train_valid_tokens,
        max_length=config.MAX_LENGTH
    )
    val_dataset = KLSupervisedDataset(
        val_sequences,
        val_valid_tokens,
        max_length=config.MAX_LENGTH
    )
    
    # Create dataloaders
    batch_size = 8  # Can be adjusted based on memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/sft_kl_{timestamp}"
    
    # Train
    training_stats = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.SFT_NUM_EPOCHS,
        learning_rate=config.SFT_LEARNING_RATE,
        device=device,
        grad_accumulation_steps=config.SFT_GRAD_ACCUMULATION_STEPS,
        checkpoint_dir=checkpoint_dir,
        log_interval=config.SFT_LOG_INTERVAL,
        temperature=config.SFT_TEMPERATURE,
        warmup_ratio=config.SFT_WARMUP_RATIO
    )
    
    return training_stats


if __name__ == "__main__":
    main()

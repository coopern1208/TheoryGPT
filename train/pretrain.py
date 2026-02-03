import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import os
from tqdm import tqdm
import json
from datetime import datetime

from architecture.TheoryGPT import TheoryGPT
from config import config
import grammar.vocab as vocab


class TokenSequenceDataset(Dataset):
    """Dataset for tokenized sequences"""
    def __init__(self, sequences, max_length=None):
        """
        Args:
            sequences: List of tokenized sequences (list of integers)
            max_length: Maximum sequence length (pads or truncates)
        """
        self.sequences = sequences
        self.max_length = max_length or config.MAX_LENGTH
        self.pad_token_id = config.PAD_TOKEN_ID
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Truncate if too long
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]
        
        # Pad if too short
        if len(seq) < self.max_length:
            seq = seq + [self.pad_token_id] * (self.max_length - len(seq))
        
        # Convert to tensor
        input_ids = torch.tensor(seq, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def compute_loss(model, batch, criterion, device):
    """Compute loss for a batch"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
    
    # Shift labels for next-token prediction
    # Predict token at position i from tokens at positions 0...i-1
    shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:].contiguous()  # (batch_size, seq_len-1)
    
    # Reshape for loss computation
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size * (seq_len-1), vocab_size)
    shift_labels = shift_labels.view(-1)  # (batch_size * (seq_len-1),)
    
    # Compute loss (only on non-padded tokens)
    # Create mask to ignore padding tokens in labels
    shift_attention_mask = attention_mask[:, 1:].contiguous().view(-1)  # (batch_size * (seq_len-1),)
    loss = criterion(shift_logits, shift_labels)
    
    # Mask out padding tokens
    loss = loss * shift_attention_mask
    loss = loss.sum() / (shift_attention_mask.sum() + 1e-8)  # Add small epsilon to avoid division by zero
    
    return loss, logits


def evaluate(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = compute_loss(model, batch, criterion, device)
            
            # Count non-padding tokens for accurate perplexity
            attention_mask = batch['attention_mask'].to(device)
            num_tokens = attention_mask[:, 1:].sum().item()  # Exclude first token
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return avg_loss, perplexity


def train(model, train_loader, val_loader, num_epochs, learning_rate, device, 
          grad_accumulation_steps=1, checkpoint_dir="checkpoints", log_interval=10):
    """Main training loop with gradient accumulation and validation"""
    model.train()
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Cosine learning rate schedule with warmup
    total_steps = len(train_loader) * num_epochs // grad_accumulation_steps
    warmup_steps = total_steps // 10  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item()))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=config.PAD_TOKEN_ID)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training statistics
    best_val_loss = float('inf')
    training_stats = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'perplexities': [],
        'learning_rates': []
    }
    
    print("\n" + "="*80)
    print(f"Starting TheoryGPT Pretraining")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"Effective batch size: {config.PRETRAIN_BATCH_SIZE * grad_accumulation_steps}")
    print(f"Total optimization steps: {total_steps}")
    print("="*80 + "\n")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        # Progress bar for training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in pbar:
            # Forward and backward pass
            loss, _ = compute_loss(model, batch, criterion, device)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * grad_accumulation_steps
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
                'loss': f"{loss.item() * grad_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Periodic logging
            if (batch_idx + 1) % (log_interval * grad_accumulation_steps) == 0:
                avg_loss = epoch_loss / num_batches
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                print(f"\n[Epoch {epoch+1}/{num_epochs}] [Batch {batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} | LR: {current_lr:.2e}")
        
        # Compute epoch statistics
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()
        
        # Validation
        print(f"\nRunning validation...")
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)
        
        # Save statistics
        training_stats['epochs'].append(epoch + 1)
        training_stats['train_losses'].append(avg_train_loss)
        training_stats['val_losses'].append(val_loss)
        training_stats['perplexities'].append(val_perplexity)
        training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print("\n" + "="*80)
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Perplexity:   {val_perplexity:.2f}")
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
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'config': config.__dict__
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_perplexity': val_perplexity,
                'config': config.__dict__
            }, best_model_path)
            print(f"Best model saved to {best_model_path} (Val Loss: {val_loss:.4f})\n")
    
    # Save training statistics
    stats_path = os.path.join(checkpoint_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"Training statistics saved to {stats_path}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80 + "\n")

def load_dataset_from_disk(dataset_path):
    """Load dataset from disk and extract sequences"""
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    # Extract sequences from 'inputs' field
    sequences = []
    for example in dataset:
        inputs = example['inputs']
        if isinstance(inputs, (list, tuple)):
            sequences.append(list(inputs))
        else:
            sequences.append(inputs.tolist() if hasattr(inputs, 'tolist') else list(inputs))
    
    print(f"Loaded {len(sequences)} sequences from dataset")
    
    # Print some statistics
    seq_lengths = [len(seq) for seq in sequences]
    print(f"Sequence length stats:")
    print(f"  Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, "
          f"Mean: {sum(seq_lengths)/len(seq_lengths):.1f}")
    
    return sequences


def split_dataset(sequences, val_split=0.1, seed=None):
    """Split dataset into train and validation sets"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Shuffle sequences
    indices = torch.randperm(len(sequences)).tolist()
    split_idx = int(len(sequences) * (1 - val_split))
    
    train_sequences = [sequences[i] for i in indices[:split_idx]]
    val_sequences = [sequences[i] for i in indices[split_idx:]]
    
    print(f"Split dataset: {len(train_sequences)} train, {len(val_sequences)} val")
    return train_sequences, val_sequences


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
    
    # Create model
    print("Initializing TheoryGPT model...")
    model = TheoryGPT(config)
    model = model.to(device)
    
    # Load dataset from disk
    dataset_path = "dataset/pretrain_data_100k"
    all_sequences = load_dataset_from_disk(dataset_path)
    
    # Split into train and validation
    train_sequences, val_sequences = split_dataset(
        all_sequences, 
        val_split=config.VAL_SPLIT_SIZE, 
        seed=config.VAL_SPLIT_SEED
    )
    
    # Create datasets and dataloaders
    train_dataset = TokenSequenceDataset(train_sequences, max_length=config.MAX_LENGTH)
    val_dataset = TokenSequenceDataset(val_sequences, max_length=config.MAX_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/pretrain_{timestamp}"
    
    # Train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.PRETRAIN_NUM_EPOCHS,
        learning_rate=config.PRETRAIN_LEARNING_RATE,
        device=device,
        grad_accumulation_steps=config.PRETRAIN_GRAD_ACCUMULATION_STEPS,
        checkpoint_dir=checkpoint_dir,
        log_interval=config.PRETRAIN_LOG_INTERVAL
    )


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F

from architecture.TheoryGPT import TheoryGPT
from config import config
import grammar.vocab as vocab
from grammar.parser import print_sequence


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = TheoryGPT(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'val_loss' in checkpoint:
        print(f"Checkpoint Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"Checkpoint Val Perplexity: {checkpoint['val_perplexity']:.4f}")
    
    return model


def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocab_size,)
        top_k: keep only top k tokens with highest probability (top-k filtering).
        top_p: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    assert logits.dim() == 1  # batch size 1 for now
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits


def generate_sequence(model, device, prompt=None, max_length=512, temperature=1.0, 
                     top_k=0, top_p=0.9, num_samples=1):
    """
    Generate sequences from the model
    
    Args:
        model: Trained TheoryGPT model
        device: torch device
        prompt: Optional list of token strings to use as prompt (e.g., ['<BOS>', 'H', '='])
        max_length: Maximum sequence length
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_samples: Number of sequences to generate
    
    Returns:
        List of generated sequences (as token ID lists)
    """
    model.eval()
    generated_sequences = []
    
    for _ in range(num_samples):
        # Start with prompt if provided, otherwise just BOS token
        if prompt is not None:
            prompt_ids = encode_prompt(prompt)
            if not prompt_ids:  # If encoding failed, use BOS
                print("Warning: Prompt encoding failed, using BOS token instead")
                input_ids = torch.tensor([[vocab.BOS_TOKEN_ID]], dtype=torch.long, device=device)
            else:
                input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                print(f"Starting generation with prompt: {prompt}")
                print(f"Prompt token IDs: {prompt_ids}")
        else:
            input_ids = torch.tensor([[vocab.BOS_TOKEN_ID]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits
                
                # Get logits for the last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                
                # Sample from the filtered distribution
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we generate EOS token
                if next_token.item() == vocab.EOS_TOKEN_ID:
                    break
        
        generated_sequences.append(input_ids[0].cpu().tolist())
    
    return generated_sequences


def decode_sequence(token_ids):
    """Convert token IDs to token strings"""
    tokens = [vocab.id2token.get(id, f"<UNK_{id}>") for id in token_ids]
    return tokens

def encode_prompt(prompt_tokens):
    """
    Convert token strings to token IDs
    
    Args:
        prompt_tokens: List of token strings (e.g., ['<BOS>', 'H', '='])
    
    Returns:
        List of token IDs
    """
    token_ids = []
    for token in prompt_tokens:
        if token in vocab.token2id:
            token_ids.append(vocab.token2id[token])
        else:
            print(f"Warning: Unknown token '{token}', skipping")
    return token_ids


def main(): 
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_model("checkpoints/pretrain_20260130_004801/best_model.pt", device)
    
    # Example 1: Generate without prompt (default behavior)
    # print("=" * 80)
    # print("Generating without prompt:")
    # print("=" * 80)
    # sequences = generate_sequence(
    #     model=model,
    #     device=device,
    #     max_length=512,
    #     temperature=1.0,
    #     top_k=0,
    #     top_p=0.9,
    #     num_samples=1
    # )
    # print(sequences)
    # token_seq = decode_sequence(sequences[0])
    # print_sequence(token_seq)
    
    # Example 2: Generate with a prompt
    # Uncomment and modify the prompt below to use it
    print("\n" + "=" * 80)
    print("Generating with prompt:")
    print("=" * 80)
    sequences = generate_sequence(
        model=model,
        device=device,
        prompt=['BOS', 'H', '='],  
        max_length=512,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        num_samples=1
    )
    print(sequences)
    token_seq = decode_sequence(sequences[0])
    print_sequence(token_seq)



if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Tuple
from dataclasses import dataclass

from config import Config
import grammar.vocab as vocab

@dataclass
class SimpleCausalLMOutput:
    logits: Optional[torch.FloatTensor] = None    
    hidden_states: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Args:
        dim: Dimension of each attention head
        max_seq_len: Maximum sequence length
        base: Base for the geometric progression (default: 10000)
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin for the maximum sequence length
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin values for efficiency"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim//2]
        # Different from paper, but matches common implementations
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer('cos_cached', emb.cos()[None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, :, :], persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
            seq_len: Sequence length (if None, uses x.shape[1])
        
        Returns:
            cos, sin tensors for rotary embedding
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # Expand cos/sin to match q/k dimensions
    cos = cos.unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = sin.unsqueeze(2)  # [1, seq_len, 1, head_dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Positional Embeddings"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: torch.Tensor = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len, seq_len] or [seq_len, seq_len]
                           True/1 for positions to attend to, False/0 for masked positions
            past_key_value: Optional tuple of (key, value) cached tensors
            use_cache: Whether to return key-value cache
        
        Returns:
            output: Attention output [batch, seq_len, d_model]
            present_key_value: Updated (key, value) cache if use_cache=True, else None
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape to [batch, seq_len, num_heads, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Determine the position offset for RoPE
        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[1]  # Add cached sequence length
        
        # Apply rotary embeddings
        # For queries, use the current positions
        if past_key_value is not None:
            # During generation, q is at position [past_len : past_len + new_len]
            past_len = past_key_value[0].shape[1]
            cos, sin = self.rotary_emb(x, kv_seq_len)
            # Take only the positions for new tokens
            cos_q = cos[:, past_len:past_len + seq_len, :]
            sin_q = sin[:, past_len:past_len + seq_len, :]
            cos_k = cos[:, past_len:past_len + seq_len, :]
            sin_k = sin[:, past_len:past_len + seq_len, :]
        else:
            # First pass, apply RoPE to full sequence
            cos, sin = self.rotary_emb(x, kv_seq_len)
            cos_q = cos[:, :seq_len, :]
            sin_q = sin[:, :seq_len, :]
            cos_k = cos_q
            sin_k = sin_q
        
        # Apply RoPE
        cos_q = cos_q.unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin_q = sin_q.unsqueeze(2)
        cos_k = cos_k.unsqueeze(2)
        sin_k = sin_k.unsqueeze(2)
        
        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q_embed.transpose(1, 2)
        k = k_embed.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Concatenate with past key-values if provided
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)  # Concatenate along seq_len dimension
            v = torch.cat([past_value, v], dim=2)
        
        # Store present key-value for next iteration
        present_key_value = (k, v) if use_cache else None
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            # Mask: 0 or False means masked (set to -inf), 1 or True means keep
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Handle NaN from padding tokens (rows with all -inf become NaN after softmax)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)
        
        return output, present_key_value


class TransformerBlock(nn.Module):
    """Complete transformer block with attention, FFN, residual connections, and layer norm"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout, max_seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout)
        )
    
    def forward(self, 
                x: torch.Tensor, 
                attention_mask: torch.Tensor = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional causal/padding mask
            past_key_value: Optional cached (key, value) from previous step
            use_cache: Whether to return KV cache
        
        Returns:
            x: Output tensor
            present_key_value: Updated cache if use_cache=True, else None
        """
        # Attention with residual connection and pre-norm
        x_norm = self.norm1(x)
        attn_out, present_key_value = self.attention(
            x_norm, 
            attention_mask, 
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        x = x + attn_out
        
        # FFN with residual connection and pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x, present_key_value


class TheoryGPT(nn.Module):
    """Complete trainable transformer model with RoPE"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vocab_size = len(vocab.GRAMMAR_TOKENS)
        
        self.d_model = config.D_MODEL
        self.num_layers = config.NUM_LAYERS
        self.num_heads = config.NHEAD
        self.dim_feedforward = config.DIM_FEEDFORWARD
        self.dropout = config.DROPOUT
        self.max_length = config.MAX_LENGTH
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(self.dropout)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.dim_feedforward,
                dropout=self.dropout,
                max_seq_len=self.max_length
            )
            for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(self.d_model)
        
        # Language model head (output projection)
        # Weight tying: share weights with token embedding
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        print(f"Initialized RLTransformer with RoPE:")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Model dim: {self.d_model}")
        print(f"  - Heads: {self.num_heads}")
        print(f"  - FFN dim: {self.dim_feedforward}")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - Max length: {self.max_length}")
        print(f"  - Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize weights"""
        # Token embedding
        self.token_embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        # All linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            # LayerNorm
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive generation"""
        # Upper triangular mask (True = masked, False = attend)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        # Convert to attention mask format: 1 = attend, 0 = masked
        return (~mask).long()
    
    def prepare_inputs_for_generation(self, 
                                     input_ids: torch.LongTensor,
                                     past_key_values: Optional[Tuple] = None,
                                     attention_mask: Optional[torch.LongTensor] = None,
                                     **kwargs):
        """
        Prepare inputs for generation with KV cache.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            past_key_values: Optional cached key-values from previous generation step
            attention_mask: Optional attention mask
        
        Returns:
            Dictionary of inputs ready for forward pass
        """
        # Only use the last token when cache is available
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
        # Update attention mask to include past sequence length
        if attention_mask is not None and past_key_values is not None:
            # Keep the full attention mask covering all tokens
            pass
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
    
    def forward(self, 
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False) -> SimpleCausalLMOutput:

        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape
        
        # Determine the total sequence length (including cache)
        past_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_length = past_key_values[0][0].shape[2]  # [batch, num_heads, seq_len, head_dim]
        
        total_seq_len = seq_len + past_length
        
        # Create causal mask for autoregressive generation
        # When using cache, we only need to mask for the new tokens attending to all tokens
        if past_length > 0:
            # For cached generation: new tokens can attend to all previous + current tokens
            # Shape: [seq_len, total_seq_len] where seq_len is new tokens, total_seq_len includes cache
            causal_mask = torch.ones(seq_len, total_seq_len, device=self.device, dtype=torch.long)
        else:
            # First pass: standard causal mask
            causal_mask = self._create_causal_mask(seq_len, self.device)
        
        # Combine causal mask with padding mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            # attention_mask is [batch, total_seq_len] where 1 = valid, 0 = pad
            # Expand to [batch, seq_len, total_seq_len] and combine with causal
            if past_length > 0:
                # Attention mask should cover all tokens (past + current)
                padding_mask_2d = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, total_seq_len]
            else:
                padding_mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # [batch, seq_len, seq_len]
            # Combine: both must be 1 to attend
            combined_mask = causal_mask.unsqueeze(0) * padding_mask_2d
        else:
            # Only causal mask
            if past_length > 0:
                combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.embed_dropout(x)
        
        # Pass through transformer blocks with KV cache
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(
                x, 
                attention_mask=combined_mask,
                past_key_value=layer_past,
                use_cache=use_cache
            )
            if use_cache:
                present_key_values.append(present_kv)
        
        # Final layer norm
        hidden_states = self.ln_f(x)
        
        # Language model head (logits)
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        return SimpleCausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=tuple(present_key_values) if use_cache else None
        )

if __name__ == "__main__":
    config = Config()
    model = TheoryGPT(config)
    print(model)
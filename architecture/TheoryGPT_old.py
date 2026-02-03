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
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len, seq_len] or [seq_len, seq_len]
                           True/1 for positions to attend to, False/0 for masked positions
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape to [batch, seq_len, num_heads, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rope(q, k, cos, sin)
        
        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
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
        
        return output


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
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional causal/padding mask
        """
        # Attention with residual connection and pre-norm
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, attention_mask)
        x = x + attn_out
        
        # FFN with residual connection and pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x


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
    
    def forward(self, 
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.LongTensor] = None) -> SimpleCausalLMOutput:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional padding mask [batch_size, seq_len] 
                           (1 for real tokens, 0 for padding)
        
        Returns:
            SimpleCausalLMOutput with logits and hidden_states
        """
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask for autoregressive generation
        causal_mask = self._create_causal_mask(seq_len, self.device)
        
        # Combine causal mask with padding mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            # attention_mask is [batch, seq_len] where 1 = valid, 0 = pad
            # Expand to [batch, seq_len, seq_len] and combine with causal
            padding_mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # [batch, seq_len, seq_len]
            # Combine: both must be 1 to attend
            combined_mask = causal_mask.unsqueeze(0) * padding_mask_2d
        else:
            # Only causal mask
            combined_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.embed_dropout(x)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, attention_mask=combined_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(x)
        
        # Language model head (logits)
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        return SimpleCausalLMOutput(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=None
        )

if __name__ == "__main__":
    config = Config()
    model = TheoryGPT(config)
    print(model)
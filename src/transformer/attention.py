import contextlib
from multiprocessing import context
import torch.nn as nn
import torch

# GPT_CONFIG_124M = {
#     "vocab_size": 50257,     # Vocabulary size
#     "context_length": 1024,  # Context length
#     "emb_dim": 768,          # Embedding dimension
#     "n_heads": 12,           # Number of attention heads
#     "n_layers": 12,          # Number of layers
#     "drop_rate": 0.1,        # Dropout rate
#     "qkv_bias": False        # Query-Key-Value bias
# }

class Transformer(nn.Module):
    """Transformer Architecture"""
    def __init__(self, cfg):
        """Ensures editable configurations"""
        super().__init__()
        
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        # self.out_head = nn.Linear(
        #     cfg["emb_dim"], cfg["vocab_size"], bias=False
        # ) # without weight tying

    def forward(self, in_idx):
        # Input processing
        batch_size, seq_len = in_idx.shape
        tok_embs = self.tok_emb(in_idx)
        pos_embs = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embs + pos_embs
        x = self.dropout_emb(x)

        # Transformer blocks
        x = self.trf_blocks(x)

        # Output processing
        x = self.final_norm(x)

        # logits = self.out_head(x) Without weight tying

        # Use transposed token embedding as output projection
        logits = torch.matmul(x, self.token_emb.weight.T)  # Weight tying!

        return logits


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, cfg):
        super().__init__()
        
        self.dropout_shortcut = nn.Dropout(cfg['drop_rate'])
        self.norm1 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.norm2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.multihead = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], 
            cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"]
            )
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        shortcut = x              # shortcut connection
        x = self.norm1(x)         # Layer Norm 1
        x = self.multihead(x)
        x = self.dropout_shortcut(x)

        x = x + shortcut

        shortcut = x             # Store input 

        x = self.norm2(x)        # Layer norm 2
        x = self.ffn(x)
        x = self.dropout_shortcut(x)

        x = x + shortcut        # add shortcut

        return x


class MultiHeadAttention(nn.Module):
    """Masked Multi-Head Attention"""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.num_heads = num_heads

        self.head_dim = d_out // num_heads
        self.d_in = d_in
        self.d_out = d_out
        self.dropout = dropout

        # Precompute the causal mask and store it as a model state (moves with the model to the GPU/CPU)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length))
        )

        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)

        # Output projections
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, num_tokens, d_in]
            
        Returns:
            context_vec: Output tensor of shape [batch_size, num_tokens, d_out]
        """

        b, num_tokens, d_in = x.shape

        # Compute Q, K, and V matrices
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Split the embedding vector into the num of heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose for parallel head processing
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # Slice the pre-computed mask to match current sequence length
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # [[False, True, True, etc]]

        # Set future positions to negative infinity
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Compute attn weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)

        # Context vectors
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Reshape to concatenate all heads back together
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Final linear projection to mix information from all heads
        context_vec = self.out_proj(context_vec)

        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            self.dropout,
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
            self.dropout
        )

    def forward(self, x):
        return self.layers(x)
    
class LayerNorm(nn.Module):
    """
    Normalize inputs to ensure stable input -> Stable training. 
    Inputs have a mean=0, variance=1
    """
    def __init__(self, emb_dim):
        super().__init__()
        
        self.epsilon = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # Learnable scale
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # Learnable shift
    
    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False) # unbiased=False: Divides by n-1 instead of n
        x_normalized = (x - x_mean)/(torch.sqrt(x_var + self.epsilon))

        return self.scale * x_normalized + self.shift

class GELU(nn.Module):
    """Activation Function: Gaussian Error Linear Unit"""
    def __init__(self):
        super().__init__()


    def forward(self, x):
        """Approximate forward pass calculation of GELU"""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
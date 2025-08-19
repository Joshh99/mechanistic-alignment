import contextlib
from multiprocessing import context
import torch.nn as nn
import torch
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

class ActivationCache:
    def __init__(self):
        self.cache = {}
    
    def __getitem__(self, key):
        return self.cache[key]
    
    def __setitem__(self, key, value):
        self.cache[key] = value
        
    def __contains__(self, key):
        return key in self.cache
    
    def keys(self):
        return self.cache.keys()
        
    def items(self):
        return self.cache.items()

class HookPoint(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.hooks: List[Callable] = []
        
    def add_hook(self, hook: Callable):
        self.hooks.append(hook)
        
    def remove_hooks(self):
        self.hooks = []
        
    def forward(self, x):
        for hook in self.hooks:
            x = hook(x, self.name)
        return x

class Transformer(nn.Module):
    """Transformer Architecture"""
    def __init__(self, cfg):
        """Ensures editable configurations"""
        super().__init__()
        self.cfg = cfg
        
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, layer_idx=i) for i in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        self.hook_embed = HookPoint("hook_embed")
        self.hook_pos_embed = HookPoint("hook_pos_embed")
        self.hook_tokens = HookPoint("hook_tokens")

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        
        tok_embs = self.hook_embed(self.token_emb(in_idx))
        pos_embs = self.hook_pos_embed(self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        ))
        x = tok_embs + pos_embs
        x = self.dropout_emb(x)
        x = self.hook_tokens(x)

        for block in self.trf_blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = torch.matmul(x, self.token_emb.weight.T)

        return logits


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.dropout_shortcut = nn.Dropout(cfg['drop_rate'])
        self.norm1 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.norm2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.multihead = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], 
            cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"]
            )
        self.ffn = FeedForward(cfg)
        
        self.hook_resid_pre = HookPoint(f"blocks.{layer_idx}.hook_resid_pre")
        self.hook_resid_mid = HookPoint(f"blocks.{layer_idx}.hook_resid_mid")
        self.hook_resid_post = HookPoint(f"blocks.{layer_idx}.hook_resid_post")
        self.hook_attn_out = HookPoint(f"blocks.{layer_idx}.hook_attn_out")
        self.hook_mlp_out = HookPoint(f"blocks.{layer_idx}.hook_mlp_out")

    def forward(self, x):
        x = self.hook_resid_pre(x)
        shortcut = x
        x = self.norm1(x)
        
        attn_out = self.multihead(x)
        attn_out = self.hook_attn_out(attn_out)
        attn_out = self.dropout_shortcut(attn_out)
        
        x = shortcut + attn_out
        x = self.hook_resid_mid(x)
        shortcut = x

        x = self.norm2(x)
        mlp_out = self.ffn(x)
        mlp_out = self.hook_mlp_out(mlp_out)
        mlp_out = self.dropout_shortcut(mlp_out)

        x = shortcut + mlp_out
        x = self.hook_resid_post(x)

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
        self.dropout = nn.Dropout(dropout)

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
        
        # Hook points for attention weights  
        self.hook_attn_weights = HookPoint("hook_attn_weights")
        self.hook_z = HookPoint("hook_z")

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
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # keys.shape[-1]=head_dim, √head_dim

        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)
        attn_weights = self.hook_attn_weights(attn_weights)

        # Context vectors
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Reshape to concatenate all heads back together
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.hook_z(context_vec)

        # Final linear projection to mix information from all heads
        context_vec = self.out_proj(context_vec)

        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.dropout_prior = nn.Dropout(cfg["drop_rate"])
        self.dropout_post = nn.Dropout(cfg["drop_rate"])


        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            self.dropout_prior,
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
            self.dropout_post
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


class HookedTransformerWrapper(Transformer):
    """SAELens-compatible wrapper for the custom Transformer"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.acts_to_saes = {}
        self._cache_enabled = False
        self._cache = ActivationCache()
        
    def add_sae(self, sae, hook_name: str):
        """Add SAE to specified activation hook"""
        self.acts_to_saes[hook_name] = sae
        
    def remove_sae(self, hook_name: str):
        """Remove SAE from specified activation hook"""
        if hook_name in self.acts_to_saes:
            del self.acts_to_saes[hook_name]
    
    def clear_saes(self):
        """Remove all SAEs"""
        self.acts_to_saes = {}
    
    def _get_hook_points(self) -> Dict[str, HookPoint]:
        """Get all hook points in the model"""
        hook_points = {}
        
        # Add embedding hooks
        hook_points["hook_embed"] = self.hook_embed
        hook_points["hook_pos_embed"] = self.hook_pos_embed  
        hook_points["hook_tokens"] = self.hook_tokens
        
        # Add transformer block hooks
        for i, block in enumerate(self.trf_blocks):
            hook_points[f"blocks.{i}.hook_resid_pre"] = block.hook_resid_pre
            hook_points[f"blocks.{i}.hook_resid_mid"] = block.hook_resid_mid
            hook_points[f"blocks.{i}.hook_resid_post"] = block.hook_resid_post
            hook_points[f"blocks.{i}.hook_attn_out"] = block.hook_attn_out
            hook_points[f"blocks.{i}.hook_mlp_out"] = block.hook_mlp_out
            hook_points[f"blocks.{i}.attn.hook_attn_weights"] = block.multihead.hook_attn_weights
            hook_points[f"blocks.{i}.attn.hook_z"] = block.multihead.hook_z
            
        return hook_points
    
    def _cache_hook(self, activation, hook_name):
        """Hook function to cache activations"""
        if self._cache_enabled:
            self._cache[hook_name] = activation.detach()
        return activation
    
    def _sae_hook(self, activation, hook_name):
        """Hook function to apply SAE if present"""
        if hook_name in self.acts_to_saes:
            sae = self.acts_to_saes[hook_name]
            return sae(activation)
        return activation
    
    def run_with_cache(self, tokens: torch.Tensor, **kwargs) -> tuple[torch.Tensor, ActivationCache]:
        """Run forward pass with activation caching"""
        self._cache = ActivationCache()
        self._cache_enabled = True
        
        # Add caching hooks to all hook points
        hook_points = self._get_hook_points()
        for name, hook_point in hook_points.items():
            hook_point.add_hook(self._cache_hook)
        
        try:
            logits = self.forward(tokens)
        finally:
            # Clean up hooks
            for hook_point in hook_points.values():
                hook_point.remove_hooks()
            self._cache_enabled = False
        
        return logits, self._cache
    
    def run_with_saes(self, tokens: torch.Tensor, saes: List = None, **kwargs) -> torch.Tensor:
        """Run forward pass with SAEs applied"""
        if saes:
            # Temporarily add SAEs
            temp_saes = {}
            for sae in saes:
                hook_name = sae.cfg.hook_name if hasattr(sae, 'cfg') else getattr(sae, 'hook_name', 'unknown')
                temp_saes[hook_name] = self.acts_to_saes.get(hook_name)
                self.acts_to_saes[hook_name] = sae
        
        # Add SAE hooks
        hook_points = self._get_hook_points()
        for name, hook_point in hook_points.items():
            if name in self.acts_to_saes:
                hook_point.add_hook(self._sae_hook)
        
        try:
            logits = self.forward(tokens)
        finally:
            # Clean up hooks
            for hook_point in hook_points.values():
                hook_point.remove_hooks()
            
            # Restore original SAEs if temporary ones were added
            if saes:
                for sae in saes:
                    hook_name = sae.cfg.hook_name if hasattr(sae, 'cfg') else getattr(sae, 'hook_name', 'unknown')
                    if temp_saes[hook_name] is None:
                        del self.acts_to_saes[hook_name]
                    else:
                        self.acts_to_saes[hook_name] = temp_saes[hook_name]
        
        return logits
    
    def run_with_cache_with_saes(self, tokens: torch.Tensor, saes: List = None, **kwargs) -> tuple[torch.Tensor, ActivationCache]:
        """Run forward pass with both caching and SAEs"""
        self._cache = ActivationCache()
        self._cache_enabled = True
        
        if saes:
            # Temporarily add SAEs
            temp_saes = {}
            for sae in saes:
                hook_name = sae.cfg.hook_name if hasattr(sae, 'cfg') else getattr(sae, 'hook_name', 'unknown')
                temp_saes[hook_name] = self.acts_to_saes.get(hook_name)
                self.acts_to_saes[hook_name] = sae
        
        # Add both caching and SAE hooks
        hook_points = self._get_hook_points()
        for name, hook_point in hook_points.items():
            hook_point.add_hook(self._cache_hook)
            if name in self.acts_to_saes:
                hook_point.add_hook(self._sae_hook)
        
        try:
            logits = self.forward(tokens)
        finally:
            # Clean up hooks
            for hook_point in hook_points.values():
                hook_point.remove_hooks()
            self._cache_enabled = False
            
            # Restore original SAEs if temporary ones were added
            if saes:
                for sae in saes:
                    hook_name = sae.cfg.hook_name if hasattr(sae, 'cfg') else getattr(sae, 'hook_name', 'unknown')
                    if temp_saes[hook_name] is None:
                        del self.acts_to_saes[hook_name]
                    else:
                        self.acts_to_saes[hook_name] = temp_saes[hook_name]
        
        return logits, self._cache
    
    def get_state_dict_mapping(self) -> Dict[str, str]:
        """Map parameter names to HookedTransformer naming conventions"""
        mapping = {
            # Embedding layers
            "token_emb.weight": "embed.W_E",
            "pos_emb.weight": "pos_embed.W_pos",
            
            # Layer normalization
            "final_norm.scale": "ln_final.w",
            "final_norm.shift": "ln_final.b",
        }
        
        # Add transformer block mappings
        for i in range(self.cfg["n_layers"]):
            mapping.update({
                f"trf_blocks.{i}.norm1.scale": f"blocks.{i}.ln1.w",
                f"trf_blocks.{i}.norm1.shift": f"blocks.{i}.ln1.b",
                f"trf_blocks.{i}.norm2.scale": f"blocks.{i}.ln2.w", 
                f"trf_blocks.{i}.norm2.shift": f"blocks.{i}.ln2.b",
                
                # Attention
                f"trf_blocks.{i}.multihead.W_query.weight": f"blocks.{i}.attn.W_Q",
                f"trf_blocks.{i}.multihead.W_key.weight": f"blocks.{i}.attn.W_K",
                f"trf_blocks.{i}.multihead.W_value.weight": f"blocks.{i}.attn.W_V",
                f"trf_blocks.{i}.multihead.out_proj.weight": f"blocks.{i}.attn.W_O",
                
                # MLP/FFN
                f"trf_blocks.{i}.ffn.layers.0.weight": f"blocks.{i}.mlp.W_in",
                f"trf_blocks.{i}.ffn.layers.0.bias": f"blocks.{i}.mlp.b_in",
                f"trf_blocks.{i}.ffn.layers.3.weight": f"blocks.{i}.mlp.W_out",
                f"trf_blocks.{i}.ffn.layers.3.bias": f"blocks.{i}.mlp.b_out",
            })
        
        return mapping